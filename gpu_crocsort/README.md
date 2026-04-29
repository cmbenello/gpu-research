# GPU CrocSort — GPU-Accelerated External Merge Sort

A CUDA implementation of external merge sort inspired by [CrocSort](https://github.com/cmbenello/gpu-research/tree/main/crocsort_repo) (VLDB 2026), featuring a **novel shared-memory K-way merge tree** that achieves both high merge fan-in (fewer HBM passes) and full GPU thread utilization.

## Quick Start

```bash
cd gpu_crocsort
bash run.sh
```

This auto-detects your GPU, builds everything, and runs the full benchmark suite.

## Findings Summary (April 2026)

### End-to-end performance — TPC-H lineitem external sort

| GPU | Memory | Largest SF that fits | Best wall time | Throughput |
|---|---|---|---|---|
| Quadro RTX 6000 | 24 GB | SF100 (72 GB) | 3.74 s | ~19 GB/s |
| Quadro P5000 | 16 GB | SF50 (36 GB) | 7.72 s | 4.66 GB/s |
| RTX 2080 | 8 GB | SF20 (14 GB) | 2.76 s | 5.21 GB/s |

The "largest SF that fits" column is the real story: scaling is bounded by GPU memory, not throughput. The merge-phase arena allocates `num_records × 24 bytes` plus key buffers; for SF100 that's ~14 GB just for the arena, which doesn't fit on anything below 24 GB.

### Where the time actually goes

For SF100 on the RTX 6000, the kernel-time breakdown of the full pipeline:

| Phase | % of wall time |
|---|---|
| Key encoding (CPU) | 55% |
| PCIe H→D upload | 29% |
| GPU radix sort kernel | 9% |
| Permutation gather | 7% |

The actual sort is 9%. PCIe + encoding is 84%. This is what motivates the compression work.

### Compression headroom on TPC-H sort key

Per-column FOR + bit-packing ratios from [`scripts/a3_codec_ratios.py`](scripts/a3_codec_ratios.py):

| Column | Raw bytes | FOR+bitpack | Ratio |
|---|---|---|---|
| l_returnflag | 1 | 0.625 | 1.6× |
| l_linestatus | 1 | 0.5 | 2.0× |
| l_shipdate | 4 | 1.5 | 2.67× |
| l_commitdate | 4 | 1.5 | 2.67× |
| l_receiptdate | 4 | 1.5 | 2.67× |
| l_extendedprice | 8 | 3.0 | 2.67× |
| l_discount | 8 | 0.5 | **16×** |
| l_tax | 8 | 0.5 | **16×** |
| l_quantity | 8 | 1.625 | 4.9× |
| l_orderkey | 8 | 3.25 | 2.46× |
| l_partkey | 4 | 2.625 | 1.52× |
| l_suppkey | 4 | 2.125 | 1.88× |
| l_linenumber | 4 | 0.375 | 10.7× |
| **Total** | **66** | **~19.6** | **3.36×** |

Compact-key scan (drop invariant bytes) already gets us 88 → 32. Layering FOR+bitpack on top projects to ~10 byte keys end-to-end.

### GPU codec throughput is far above PCIe

From [`scripts/b_codec_benchmarks.cu`](scripts/b_codec_benchmarks.cu) on the RTX 6000:

| Codec | Width | Decode throughput |
|---|---|---|
| FOR | 1 byte | 466 GB/s |
| FOR | 2 byte | 546 GB/s |
| FOR | 4 byte | 568 GB/s |
| Bit-pack | 8 bits/value | 451 GB/s |
| Bit-pack | 24 bits/value | 562 GB/s |

PCIe 3.0 caps at ~12 GB/s. Decode is ~40-50× faster than data can arrive, so a "compress on host, decode on device" pipeline is cheap.

### Direct sort on smaller keys

From [`scripts/b_codec_benchmarks.cu`](scripts/b_codec_benchmarks.cu), 100 M record sort time as key width drops:

| Key width | Time | Throughput |
|---|---|---|
| 32 B | 21.48 ms | 37.3 GB/s |
| 24 B | 18.35 ms | 43.6 GB/s |
| 20 B | 18.36 ms | 43.6 GB/s |
| 16 B | 12.45 ms | **64.3 GB/s** |
| 8 B | 6.56 ms | **122.0 GB/s** |

Smaller keys are faster to sort (one fewer radix pass per byte). The compression work compounds: smaller keys both move faster across PCIe and sort faster on device.

### K-way merge cost grows fast in K

From [`scripts/d1_merge_profile.py`](scripts/d1_merge_profile.py) (CPU K-way merge, 100 M rows):

| K | Throughput | Slowdown vs K=2 |
|---|---|---|
| 2 | 172 Mrows/s | 1.00× |
| 4 | 194 Mrows/s | 0.89× |
| 8 | 153 Mrows/s | 1.12× |
| 16 | 71 Mrows/s | **2.42×** |

This is the cost the upcoming GPU sample-sort path is meant to eliminate. At K=16 the CPU merge is the dominant phase.

### What this points to

1. The PCIe bottleneck is the dominant cost on every GPU we tested. Compression directly attacks it. The codec building blocks (FOR, bit-pack) are ready in [`scripts/b_codec_benchmarks.cu`](scripts/b_codec_benchmarks.cu).
2. Smaller GPU memory (P5000 16 GB, 2080 8 GB) caps the achievable scale because the merge-phase arena scales with total record count, not chunk size. Compression should also extend the envelope by reducing arena size.
3. K-way merge cost is real and grows with K. Replacing it with GPU sample-sort is the next big architectural change.

Detailed breakdown in [`results/related_accelerators.md`](../results/related_accelerators.md) and [`results/overnight_experiments.md`](../results/overnight_experiments.md).

## What This Is

An external merge sort for GPUs that handles datasets from megabytes to larger-than-HBM. The core pipeline:

```
Input records (host or HBM)
  → Phase 1: Run Generation (parallel block-sort, many concurrent blocks)
  → Phase 2: Multi-pass Merge (K-way merge tree OR 2-way merge path)
  → Sorted output
```

### Key Algorithms Ported from CrocSort

| CrocSort Concept | Rust Source | GPU Implementation |
|---|---|---|
| OVC (Offset-Value Coding) | `src/ovc/offset_value_coding_32.rs` | `include/ovc.cuh` — 32-bit packed encoding, device functions |
| Loser Tree (OVC-aware) | `src/ovc/tree_of_losers_ovc.rs` | `include/loser_tree.cuh` — tournament replay, duplicate shortcut |
| Replacement Selection | `src/replacement_selection/tol_mm_ovc.rs` | `src/run_generation.cu` — block-sort variant (better GPU utilization) |
| Zero-Copy Merge | `src/sort/ovc/merge.rs` | `src/merge.cu` — keys in tree/registers, values streamed |
| Byte-Balanced Partitioning | `src/sort/core/run_format.rs` | `experiments/sample_partition.cu` — sample-based GPU partitioning |
| Multi-Pass Merge | `src/sort/core/engine.rs` | `src/host_sort.cu` — host orchestration with double-buffering |

### The Novel Contribution: Shared-Memory K-Way Merge Tree

Existing GPU merge sorts face a fundamental tradeoff:

- **2-way merge path** (Green et al., 2012): All threads active, but needs log₂(N) passes. Each pass reads+writes all data through HBM. For 10K runs: 14 passes = 28× data movement.
- **K-way loser tree**: Only log_K(N) passes, but 1 thread active per block. For K=8: 5 passes = 10× data movement, but terrible GPU utilization.

Our merge tree gets both:

- **K-way merge (K=8)**: Only 5 passes = 10× data movement
- **Full parallelism**: Decomposes K-way merge as log₂(K) levels of 2-way merge-path *inside shared memory*. Every thread participates at every level. Only the final output touches HBM.

This is the approach described in `src/merge.cu` as `smem_kway_merge_kernel`. No prior work (MMS/Chatterjee 2017, Merge Path/Green 2012, ModernGPU/Baxter 2013) combines high fan-in with full intra-block parallelism via shared-memory merge-path trees.

## Project Structure

```
gpu_crocsort/
├── run.sh                          # One-click build + benchmark
├── Makefile                        # Build targets for everything
├── include/
│   ├── record.cuh                  # Record format, config constants, utilities
│   ├── ovc.cuh                     # OVC encoding/decoding (device + host)
│   └── loser_tree.cuh              # OVC-aware loser tree (device)
├── src/
│   ├── main.cu                     # Entry point, data generation, CLI
│   ├── run_generation.cu           # Phase 1: bitonic sort in shared memory + OVC deltas
│   ├── merge.cu                    # Phase 2: 2-way merge path AND 8-way merge tree
│   ├── host_sort.cu                # Host orchestration, multi-pass merge, verification
│   └── external_sort.cu            # Streaming sort for data > HBM
└── experiments/
    ├── README.md                   # Detailed experiment design + hypotheses
    ├── bottleneck_bench.cu         # HBM bandwidth ceiling, merge pass cost, fanin tradeoff
    ├── memory_layout_bench.cu      # AoS vs SoA vs Key+Index merge bandwidth
    ├── cpu_baseline.cu             # CPU std::sort vs GPU direct comparison
    ├── radix_vs_merge.cu           # CUB radix sort crossover analysis
    ├── kv_separate_bench.cu        # Key-value separation optimization
    ├── profile_sort.cu             # Per-pass profiling, bandwidth efficiency
    ├── sample_partition.cu         # Sample-based boundary computation for K-way merge
    └── test_edge_cases.cu          # 18 adversarial input tests
```

## Building

Requires: CUDA Toolkit 11+ with `nvcc`.

```bash
# Main sort binary
make ARCH=sm_80          # A100
make ARCH=sm_89          # L4/4090
make ARCH=sm_90          # H100

# All experiments
make experiments ARCH=sm_80

# Individual experiments
make bottleneck-nocub    # Bottleneck analysis (no CUB dependency)
make layout-bench        # Memory layout comparison
make cpu-vs-gpu          # CPU vs GPU comparison
make test-edge           # Edge case tests
make profile             # Profiling instrumentation
make external-sort       # External sort (data > HBM)
make radix-vs-merge      # CUB radix sort crossover (needs CUB)
make kv-bench            # Key-value separation benchmark
make sample-part         # Sample-based partitioning test
```

## Running

```bash
# Sort with verification
./gpu_crocsort --num-records 1000000 --verify

# Benchmark
./gpu_crocsort --num-records 10000000

# Test duplicate key handling
./gpu_crocsort --num-records 10000000 --duplicates

# External sort (data larger than GPU memory)
./external_sort --total-gb 2

# Full experiment suite
./bottleneck_bench 10000000      # Where does time go?
./layout_bench                    # Which memory layout is fastest?
./cpu_vs_gpu --num-records 1000000  # Is GPU actually faster?
./radix_vs_merge                  # When to use radix vs merge sort?
./profile_sort 10000000           # Detailed per-pass profiling
./test_edge_cases                 # Adversarial correctness tests
```

## Record Format

Default: 100-byte records matching GenSort format.

| Field | Size | Description |
|---|---|---|
| Key | 10 bytes | Sort key (lexicographic comparison) |
| Value | 90 bytes | Payload (carried along during sort) |

Configurable in `include/record.cuh` via `KEY_SIZE`, `VALUE_SIZE`, `RECORD_SIZE`.

## How It Works

### Phase 1: Run Generation

Each CUDA thread block sorts a chunk of records in shared memory using bitonic sort, then writes the sorted run to HBM with OVC deltas computed in parallel.

- **Block size**: 256 threads
- **Records per block**: 512
- **Parallelism**: 1 block per run. For 10M records: 19,531 concurrent blocks.
- **OVC deltas**: Computed in parallel after sorting. Each thread compares its record's key against the predecessor and encodes the first differing byte position + 2-byte value into a 32-bit OVC.

### Phase 2: Multi-Pass Merge

Two strategies implemented (selectable in `host_sort.cu`):

**Strategy A — 2-Way Merge Path** (`merge_2way_kernel`):
- Standard merge path algorithm. Every thread finds its position via binary search, then sequentially merges ~8 records.
- 256 threads per block, thousands of blocks per pass.
- Needs log₂(N) passes. For 19K runs: 15 passes.

**Strategy B — 8-Way Shared-Memory Merge Tree** (`smem_kway_merge_kernel`):
- Partitions K=8 runs into thousands of partitions using sampling.
- Each block loads its partition into shared memory.
- Merges via 3 levels of 2-way merge-path inside shared memory.
- All 256 threads active at every level. Only final output goes to HBM.
- Needs log₈(N) passes. For 19K runs: 5 passes.

### External Sort (data > HBM)

`src/external_sort.cu` implements host-device streaming:

1. **Chunked run generation**: Stream data from host to GPU in chunks (auto-sized to ~70% of free HBM). Each chunk sorted on GPU, streamed back.
2. **Streaming merge**: Load run heads into GPU, merge on GPU, stream output to host. Double-buffered with pinned memory for max PCIe bandwidth.

## Experiments Explained

### `bottleneck_bench.cu` — Where Does Time Go?

Measures the raw HBM bandwidth ceiling (just copying data), the cost per merge pass, and predicts total sort time for different fan-in values. **Run this first** — it tells you whether your GPU is bandwidth-bound or compute-bound, which determines the entire optimization strategy.

### `memory_layout_bench.cu` — AoS vs SoA

During merge, we compare 10-byte keys but move 100-byte records. Array-of-Structs (current) reads 100B per comparison — wasting 90% of bandwidth on values. Struct-of-Arrays reads only 10B keys during comparison, then gathers values separately. This experiment measures the actual speedup.

### `cpu_baseline.cu` — Is GPU Actually Faster?

Head-to-head comparison: CPU `std::sort`, CPU parallel sort, CPU merge sort, vs GPU CrocSort on the same data. Prints speedup ratios.

### `profile_sort.cu` — Detailed Profiling

Per-pass breakdown: time, bandwidth, blocks, records/sec. Computes traffic efficiency vs theoretical minimum. Automatically classifies bottleneck as bandwidth-limited, compute-limited, or launch-overhead-limited.

### `test_edge_cases.cu` — Adversarial Tests

18 tests: already sorted, reverse sorted, all identical keys, two distinct keys, single record, power-of-2 boundaries, all-zeros, all-0xFF, alternating, 90% duplicate skew, sequential keys. Verifies sorted order AND no data loss (multiset equality).

### `sample_partition.cu` — Proper Partitioning

The naive proportional partition in `host_sort.cu` fails on skewed data. This implements CrocSort-style sample-based partitioning: sample keys from all runs, sort samples, pick boundaries, binary search each run for exact split points. Critical for correctness on non-uniform data.

## Performance Characteristics

### Why GPU Merge Sort Can Beat CPU

| Factor | CPU (32 GB RAM) | GPU (80 GB HBM) |
|---|---|---|
| Memory bandwidth | ~200 GB/s (DDR5) | ~2000 GB/s (HBM) |
| Memory capacity | 32 GB | 80 GB |
| Merge throughput | ~200 GB/s | ~1000+ GB/s |
| Sorting throughput | ~2 GB/s | ~50+ GB/s (radix) |
| Runs from 1GB data | ~16 (with E=2) | ~2000 (block sort) |
| Merge passes (K=8) | ~2 | ~4 |

The GPU's 10x bandwidth advantage means each merge pass is 10x faster. Even with more passes (smaller runs from block-sort vs replacement selection), the per-pass speed advantage dominates for in-HBM data.

For external sort (data on NVMe), both CPU and GPU are NVMe-bound at ~6 GB/s. But the GPU's 80 GB HBM means fewer, larger chunks = fewer disk passes.

### Theoretical HBM Traffic

For N initial runs merged with fan-in K:
```
Total traffic = 2 × data_size × ⌈log_K(N)⌉
```

| Strategy | N=19531 | Passes | Traffic (1 GB data) |
|---|---|---|---|
| 2-way merge path | 19531 | 15 | 30 GB |
| 4-way merge tree | 19531 | 8 | 16 GB |
| 8-way merge tree | 19531 | 5 | 10 GB |
| 16-way merge tree | 19531 | 4 | 8 GB |

Higher fan-in = less total traffic = faster (when bandwidth-bound).

## What's Done

- [x] OVC encoding/decoding on GPU (exact port of CrocSort's OVCU32)
- [x] Loser tree with OVC-aware comparison and duplicate shortcut
- [x] Run generation with parallel bitonic sort + OVC delta computation
- [x] 2-way merge path (all threads active, baseline strategy)
- [x] 8-way shared-memory merge tree (novel, fewer passes + full parallelism)
- [x] Multi-pass merge orchestration with double-buffering
- [x] External sort with host-device streaming and pinned memory
- [x] Bottleneck analysis benchmark suite
- [x] Memory layout comparison (AoS vs SoA vs Key+Index)
- [x] CPU vs GPU comparison benchmark
- [x] CUB radix sort crossover analysis
- [x] Per-pass profiling instrumentation
- [x] Sample-based partitioning for K-way merge
- [x] 18 adversarial correctness tests
- [x] Key-value separation benchmark

## Next Steps

### Immediate (run experiments, measure, decide)

1. **Run `bash run.sh`** on your target GPU. The bottleneck analysis will tell you whether merge is bandwidth-bound or compute-bound on your hardware. This determines everything.

2. **Run `./layout_bench`**. If SoA is 2x+ faster than AoS for merge, switch the main sort pipeline to key-value separated storage. This is potentially the single biggest optimization.

3. **Run `./cpu_vs_gpu`**. Get actual CPU vs GPU numbers on your hardware. If GPU isn't winning, the bottleneck analysis will tell you why.

### Short-Term (code changes based on experiment results)

4. **Integrate sample-based partitioning** into the main K-way merge pipeline. The current `host_sort.cu` uses naive proportional splitting which produces wrong results on skewed data. The `sample_partition.cu` implementation is ready to plug in.

5. **Implement key-value separation in the main pipeline** if `layout_bench` shows it's faster. This means storing keys and values in separate arrays, merging keys only, then scattering values by index. Could be 2-5x faster for the merge phase.

6. **Tune RECORDS_PER_BLOCK**. Currently 512 — increasing to 2048 or 4096 (if shared memory allows) produces fewer runs and fewer merge passes. Each 2x increase saves one merge pass.

7. **Add CUB radix sort as the run generation backend** for in-HBM data. Instead of block-level bitonic sort producing 19K small runs, sort the entire array with CUB producing 1 run. This eliminates the merge phase entirely for in-HBM data. Only use our merge sort for the external sort case.

### Medium-Term (algorithmic improvements)

8. **Increase merge tree fan-in to K=16 or K=32**. Needs more shared memory per block but reduces passes further. K=16 drops from 5 to 4 passes; K=32 drops to 3 passes.

9. **Add streaming prefetch** (`cp.async`) to the merge tree kernel. While one batch is being merged in shared memory, prefetch the next batch from HBM. This can hide memory latency and push bandwidth utilization toward 80%+.

10. **Multi-GPU support** via NCCL. CrocSort's cloud paper shows the M/D ratio (memory/data) governs the performance surface shape, and this ratio is invariant when scaling from 1 to N GPUs. This predicts optimal per-GPU config transfers directly to multi-GPU — a powerful result for auto-tuning.

11. **GPU Direct Storage (GDS)** for the external sort path. Currently external sort goes NVMe → host RAM → PCIe → GPU. GDS skips host RAM, going NVMe → GPU directly. This removes one copy and could improve external sort throughput by 30-50%.

### Long-Term (research directions)

12. **OVC-aware merge path**. The current merge path uses full 10-byte key comparison. If we can maintain OVC state across merge-path partitions, we could compare 4-byte OVCs instead of 10-byte keys for most comparisons — a 2.5x reduction in comparison bandwidth.

13. **Adaptive strategy selection**. Use the analytical model from CrocSort (Eq.5: T_gen × T_merge ≤ E × ρ² × M² / (D × P)) with GPU parameters to automatically choose between radix sort, 2-way merge, and K-way merge based on data size, record size, and GPU specs.

14. **H100-specific optimizations**. SM clusters (4 SMs sharing distributed shared memory = 912KB workspace) for longer replacement selection runs. TMA for bulk HBM-to-shared-memory loading. Hardware async barriers for zero-overhead phase transitions.

15. **Benchmark at scale**. Run on 100GB+ datasets across A100, H100, and multi-GPU configurations. Compare against CPU CrocSort on equivalent hardware. Publish results.

## Research Context

This project is part of a GPU research effort exploring how CrocSort's CPU external merge sort concepts transfer to GPU hardware. The `results/` directory contains 75 research ideas across 15 cycles, 12 CUDA kernel sketches, and a comprehensive analysis of which CrocSort mechanisms map well to GPU and which don't.

Key finding: CrocSort's OVC encoding (32-bit packed offset-value coding) maps perfectly to GPU 32-bit registers, enabling single-instruction comparison shortcuts. The sparse index and byte-balanced partitioning concepts are critical for handling skewed data on GPU. The analytical model (Eq.5) provides a framework for predicting optimal GPU sort configuration.

## References

- CrocSort — VLDB 2026 + Cloud paper (the CPU external sort this builds on)
- Green, McColl, Bader — [Merge Path](https://arxiv.org/abs/1406.2628), 2012 (GPU 2-way merge)
- Chatterjee et al. — [GPU Multiway Mergesort](https://arxiv.org/abs/1702.07961), 2017 (closest prior work)
- Stehle & Jacobsen — [GPU External Sort](https://arxiv.org/abs/1611.01137), SIGMOD 2017
- Baxter — [ModernGPU](https://moderngpu.github.io/mergesort.html), 2013 (GPU merge primitives)
