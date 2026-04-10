# GPU Sort Bottleneck Analysis — Experiment Design

## Core Question

What actually limits GPU external sort performance?
- HBM bandwidth (reading/writing data)?
- Compute (key comparisons)?
- Kernel launch overhead?
- PCIe bandwidth (for external sort beyond HBM)?

The answer determines the entire algorithm design.

## Hypothesis

For 100-byte records with 10-byte keys:
- **Merge is bandwidth-bound**: each pass reads+writes all data through HBM.
  Total HBM traffic = 2 * D * num_passes. Minimizing passes matters more than
  maximizing thread count.
- **Run generation is compute-bound**: sorting 512 records in shared memory
  involves O(n log^2 n) comparisons with no HBM traffic (data stays in SMEM).
- **Key comparison is cheap**: comparing 10 bytes is ~2-3 instructions.
  Even at 1B comparisons/sec per SM, 108 SMs = 108B comparisons/sec.
  For 10M records with log2(10M) comparisons each = ~230M comparisons total.
  That's 2ms of compute — negligible vs bandwidth time.

## Experiment 1: Bandwidth Ceiling

**Goal**: Measure the raw HBM bandwidth ceiling for our record size.

**Method**: Simple copy kernel — no sorting, just copy N records from buffer A to buffer B.
This gives the maximum possible throughput for any sort that reads input and writes output.

```
Kernel: memcpy_kernel(src, dst, num_records, record_size)
  tid = blockIdx.x * blockDim.x + threadIdx.x
  copy record[tid] from src to dst using uint4 loads/stores

Measure: GB/s for record sizes 10, 50, 100, 200, 500, 1000 bytes
Compare: against cudaMemcpy (which uses the DMA engine)
```

**Expected**: ~1.5-1.8 TB/s for large transfers (85-90% of peak 2 TB/s on A100).
This is the absolute ceiling for any single-pass sort.

**Why this matters**: If a sort achieves 50% of this bandwidth, the bottleneck is
elsewhere (compute, divergence, uncoalesced access). If it achieves 80%+, the
sort is bandwidth-optimal and can only be improved by reducing passes.

## Experiment 2: Merge Pass Cost

**Goal**: Measure how much each merge pass actually costs.

**Method**: Generate pre-sorted runs, then run merge with 1, 2, 4, 8, 16 passes.
Measure wall clock time per pass.

```
Setup: Sort 10M records, then artificially create:
  - 2 runs (1 merge pass with 2-way)
  - 4 runs (2 passes or 1 pass with 4-way)
  - 8 runs (3 passes with 2-way or 1 pass with 8-way)
  etc.

For each: measure time per pass and total time.
Compute: achieved bandwidth per pass = 2 * data_size / pass_time
```

**Expected**: Each 2-way merge pass should achieve ~80% of bandwidth ceiling.
Total time should be roughly proportional to num_passes.

## Experiment 3: Fanin vs Passes Tradeoff

**Goal**: Find the optimal merge strategy — more parallelism (2-way, more passes)
or less bandwidth (high-way, fewer passes)?

**Method**: Implement both and measure:
- 2-way merge path: all threads active, log2(N) passes
- 4-way merge path: fewer threads per comparison, log4(N) passes  
- 8-way loser tree: 1 thread per tree, log8(N) passes
- 16-way loser tree: 1 thread per tree, log16(N) passes
- Hybrid: 2-way for first passes (many small runs), loser tree for last passes (few large runs)

```
For each strategy, measure:
  - Total wall clock time
  - HBM bandwidth per pass (from NSight Compute or calculated)
  - Number of passes
  - Per-pass time
  - Achieved bandwidth efficiency = (2 * D * passes) / (peak_BW * total_time)
```

**Key prediction**: 
  If bandwidth-bound: high fanin wins (fewer passes = less total traffic).
  If compute-bound: 2-way wins (more parallelism hides latency).
  Reality: probably bandwidth-bound for large records (100B+), 
  compute-bound for small records (<20B).

## Experiment 4: Record Size Sensitivity

**Goal**: Find where the compute/bandwidth crossover happens.

**Method**: Fix total data at 1 GB. Vary record size:
- 10B key only (no value) — 100M records
- 20B (10 key + 10 value) — 50M records  
- 50B (10 key + 40 value) — 20M records
- 100B (10 key + 90 value) — 10M records
- 500B (10 key + 490 value) — 2M records
- 1000B (10 key + 990 value) — 1M records

For each, measure sort time with 2-way merge and 8-way loser tree.

**Expected**: 
- Small records (10-20B): compute-bound, 2-way wins
- Large records (100B+): bandwidth-bound, 8-way wins
- Crossover somewhere around 50-100B

**Why**: With 10B records, each comparison touches 10B of useful data but moves
10B through HBM. The ratio is 1:1 — compute and bandwidth are balanced.
With 100B records, each comparison touches 10B but moves 100B. The ratio is 1:10 —
bandwidth dominates.

## Experiment 5: Run Generation Strategies

**Goal**: Measure whether longer runs (fewer merge passes) justify more expensive
run generation.

**Method**: Compare:
- Block sort (512 records/run, cheap, many runs) → many merge passes
- Block sort with larger blocks (2048/run, uses more SMEM) → fewer passes
- Thrust sort of entire input (1 run, expensive) → 0 merge passes
- CUB radix sort (1 run) → 0 merge passes

```
For 10M records:
  Block-512:  19,531 runs, ~15 merge passes
  Block-2048: 4,883 runs, ~13 merge passes
  Block-8192: 1,221 runs, ~11 merge passes  
  Thrust:     1 run, 0 merge passes
  CUB radix:  1 run, 0 merge passes
```

**Expected**: For data fitting in HBM, CUB radix sort will be fastest because it
avoids ALL merge passes. It's specifically designed for GPU and does ~1.5 TB/s.
The external merge sort only wins when data exceeds HBM.

## Experiment 6: External Sort — PCIe vs NVMe Bottleneck

**Goal**: For data larger than HBM, where is the bottleneck?

**Method**: Sort 160 GB on an 80 GB HBM GPU.
- Phase A: stream 80 GB chunks to GPU, sort each, write back → 2 chunks
- Phase B: merge the 2 sorted chunks with streaming

Measure:
- PCIe H2D transfer time
- GPU sort time per chunk  
- PCIe D2H transfer time
- Merge streaming time

Compare against:
- NVMe direct (GPU Direct Storage) if available
- Pinned vs pageable host memory

**Expected**: PCIe Gen4 x16 = ~25 GB/s. For 160 GB:
- Transfer: 160 GB / 25 GB/s = 6.4 seconds (just moving data!)
- Sort: each 80 GB chunk at 1 TB/s = 0.08s × passes
- Total sort compute: negligible vs transfer

**Conclusion**: External GPU sort is PCIe-bound, not GPU-bound.
The algorithm should minimize data movement, not maximize GPU utilization.

## Experiment 7: Comparison Cost Profiling

**Goal**: Measure actual cycles spent on key comparison vs memory access in merge.

**Method**: NSight Compute profiling of merge kernel.
- `sm__inst_executed` — total instructions
- `dram__bytes_read` + `dram__bytes_write` — HBM traffic  
- `l2__hit_rate` — cache effectiveness
- `sm__warps_active` — actual parallelism achieved

Compare merge-path kernel vs loser-tree kernel.

## What To Build Based On Results

**If bandwidth-bound (likely for 100B records)**:
→ Maximize merge fanin to minimize passes
→ Use k-way merge (loser tree) despite low thread utilization
→ Focus on reducing HBM traffic: OVC prefix truncation, compression
→ For external sort: focus on PCIe/NVMe pipelining

**If compute-bound (likely for small keys)**:
→ 2-way merge path is correct
→ More parallelism helps
→ OVC comparison shortcut helps (fewer instructions per comparison)

**Most likely outcome**: hybrid approach
→ Use CUB radix sort for data fitting in HBM (skip merge entirely)
→ Use high-fanin merge ONLY for external sort (data > HBM)
→ For external sort, overlap PCIe transfer with GPU merge (double buffer)

## Implementation Priority

1. Bandwidth ceiling benchmark (Experiment 1) — 30 min to implement
2. CUB radix sort baseline (Experiment 5 partial) — already available
3. Fanin tradeoff (Experiment 3) — needs both merge implementations
4. Record size sensitivity (Experiment 4) — reuses above
5. External sort PCIe bottleneck (Experiment 6) — separate effort
