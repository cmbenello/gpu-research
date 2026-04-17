# GPU CrocSort — End-to-End Architecture

**Status:** Research prototype, verified correct on TPC-H SF10/50/100, NYC Taxi (218M records), GenSort 60GB, random up to 400M records.
**Hardware:** Quadro RTX 6000 (24 GB HBM, Turing sm_75, PCIe 3.0 x16), 48-core Xeon Gold 6248, 192 GB DDR4-2933.
**Branch:** `exp/fixup-fast-comparator`, merged to `main`.

---

## 1. What This System Does

Sorts fixed-size records (key + value) that exceed GPU memory, producing a fully materialized sorted output in host RAM. The GPU handles key sorting; the CPU handles value permutation and tie-breaking. Correctness is verified by dual checks: sortedness AND multiset hash equality.

**Best results (warm median, all PASS verified):**

| Dataset         | Records | Size   | Time    | GB/s | vs DuckDB |
|-----------------|---------|--------|---------|------|-----------|
| TPC-H SF10      | 60M     | 7.2 GB | 1.75s   | 4.1  | 13x       |
| TPC-H SF50      | 300M    | 36 GB  | 6.4s    | 5.7  | ~9x       |
| TPC-H SF100     | 600M    | 72 GB  | 8.0s    | 9.0  | ~25x      |
| NYC Taxi 5yr    | 218M    | 26 GB  | 5.2s    | 5.0  | —         |
| Random 400M     | 400M    | 48 GB  | 7.5s    | 6.4  | —         |

---

## 2. Pipeline Architecture

The system has two paths depending on whether keys fit in GPU memory.

### Path A: Single-Chunk (data ≤ ~24 GB)

Used for SF10, Taxi, small random datasets.

```
Host records ──[strided DMA: keys only]──> GPU
                                            │
                                    CUB LSD radix sort (9 passes × 66B key)
                                            │
                                    Permutation array (4B/record)
                                            │
                              <──[D2H: perm only]──
                                            │
CPU gather: apply perm to original h_data ──> sorted output
```

**Why strided DMA?** Only keys (66B of 120B record) cross PCIe. PCIe amplification = 0.6x instead of 2.0x. The GPU never sees value bytes.

### Path B: Multi-Chunk with OVC Merge (data > ~24 GB)

Used for SF50, SF100, large random datasets.

```
┌─────────────────── Triple-Buffered Pipeline ───────────────────┐
│                                                                │
│  Chunk N-1: CPU gather ◄──┐                                    │
│  Chunk N:   GPU LSD sort  ├── overlapped via CUDA streams      │
│  Chunk N+1: H2D upload  ──┘                                    │
│                                                                │
│  Per chunk: extract 8B OVC prefix + 4B permutation on GPU      │
│  Records stay in host memory (no D2H of sorted records)        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                    All chunks' (prefix, perm) pairs on GPU
                              │
                    CUB radix sort on 8B prefix + 4B perm
                    (4 LSD passes over 32B compact key)
                              │
                    D2H: download final permutation
                              │
                    CPU gather: permute h_data → h_output
                              │
                    CPU fixup: detect + resolve tied 8B-prefix groups
                              │
                    Sorted output (verified)
```

---

## 3. Six Phases in Detail

### Phase 1: Run Generation (GPU) — 35% of SF50 time

Triple-buffered pipeline: while the GPU sorts chunk N, the CPU gathers chunk N-1 and DMA uploads chunk N+1.

Each chunk is sorted via CUB LSD radix sort. With compact keys, only the varying byte positions are extracted (runtime-detected by scanning a 1M stratified sample), reducing from 9 LSD passes (66B) to 4 passes (32B compact key).

**Compact key detection:** At startup, the system samples 1M records (stratified across the file) and identifies which byte positions actually vary. If ≤32 positions vary, it builds a compact key containing only those bytes in source-position order. For >50M records, a full scan replaces sampling (a sampling miss on NYC Taxi 6-month data caused a 19x regression before this fix — commit `366335e`).

**Critical invariant:** Compact keys must preserve source-position byte ordering. Entropy-based reordering (selecting the highest-cardinality bytes regardless of position) breaks lexicographic correctness. This was the biggest correctness bug in the project — see Section 6.

### Phase 2: GPU Merge — 10% of SF50 time

After run generation, all chunks' OVC prefixes and permutation indices are on the GPU. A single CUB radix sort over (prefix, perm) pairs produces a globally-ordered permutation.

- 32B compact key: 4 LSD passes, ~610ms on SF50
- 16B full-key prefix: 2 LSD passes, ~5ms on SF50 (fewer passes but no compact savings)

**CUB satellite workaround:** CUB's `SortPairs` carries only ONE satellite array. For multi-pass LSD on 16B+ keys, we need to reorder multiple arrays (prefix chunks + perm). Solved by: sort(key=prefix2, val=identity) → gather prefix1 and perm via identity index → sort(key=prefix1, val=perm).

### Phase 3: Permutation Download + CPU Gather — 29% of SF50 time

The GPU produces a permutation array (4B per record). Download via `cudaMemcpy` (94ms for 300M records on PCIe 3.0). CPU applies the permutation to scatter-read from the original `h_data` into `h_output`.

Gather rate is DRAM-bandwidth bound: ~18-21 GB/s on DDR4-2933. This is the single most architecture-sensitive phase — on Grace Hopper (GH200) with unified memory, this phase disappears entirely because the GPU writes sorted records directly into host memory during the merge.

`MAP_POPULATE` on the output buffer is mandatory. Without it, page faults during gather cause 2-17x slowdowns (commit `bc864b5`).

### Phase 4: CPU Fixup — 28% of SF50 time (0% on SF100)

When the compact key doesn't capture all varying bytes (e.g., SF50: 61/66 vary, only 32 fit), records with identical compact prefixes may be in wrong order. The fixup phase resolves this.

**Group detection (parallel):** 48 threads each scan their 1/48 slice of the output, comparing the compact-key byte positions of adjacent records. Boundary positions are collected and stitched into a group list. SF50: 3.15M groups, avg 95 records, max 693.

**Per-group sort (work-queue):** An atomic counter dispatches batches of 64 groups to threads. Each thread packs the active bytes from its group's records into a contiguous buffer, sorts with `std::sort` on the packed key, then reorders the full records. The pack step uses `__builtin_bswap64` for the 8B key prefix + `memcmp` for the tail.

**Why this was the biggest optimization:** The original fixup was single-threaded group detection (4.8s) + fixed-chunk dispatch with load imbalance. Parallelizing group detection (→ 600ms) + work-queue dispatch (→ 0.98 efficiency) cut SF50 from 12.44s to 6.56s — a 47% reduction (commit `74c5749`).

### Phase 5: Verification

Dual verification at every run:
1. **Sortedness check:** Linear scan confirming `key[i] <= key[i+1]` for all adjacent pairs.
2. **Multiset hash:** XOR of per-record hashes before and after sort must match — proves no records were lost, duplicated, or corrupted.

Both checks are essential. The entropy-selection bug produced output that passed multiset hash (all records present) but failed sortedness (wrong order). Scripts that only checked one would have missed it.

---

## 4. Compact Key System

The compact key optimization is the core algorithmic contribution for structured data.

### How it works

1. **Detect:** Sample records, identify byte positions with >1 distinct value.
2. **Extract:** Build compact key containing only varying bytes, in source-position order.
3. **Sort:** GPU sorts compact keys (fewer LSD passes = less GPU work + less PCIe).
4. **Fixup:** If compact key < full key width, resolve ties on CPU.

### When it wins

| Scenario | Varying bytes | Compact? | Fixup? | Verdict |
|----------|--------------|----------|--------|---------|
| SF100 (27/66 vary) | 27 | Yes, all fit in 32B | None | Compact wins big |
| SF50 (61/66 vary) | 61 | Yes, but 29 overflow | 3.15M groups | Non-compact is 1.6x faster |
| Random (66/66 vary) | 66 | Auto-disabled | None | N/A |

**The rule:** Compact wins when `varying_bytes ≤ COMPACT_KEY_SIZE` (currently 32). When varying bytes overflow, the fixup cost can exceed the PCIe savings. The system auto-detects and decides at runtime.

### Why source-position order is mandatory

Compact bytes must be ordered by their position in the original key, not by entropy or cardinality. If byte B (not in compact set) differs between two records AND byte P > B (in compact set) also differs, the compact key orders by P while the full key orders by B. This breaks lexicographic correctness. The fixup phase CANNOT recover from this because the records land in different tie groups.

---

## 5. Performance Model

```
T ≈ B / G_bw + N×K / C_pcie + N×R / H_bw + N × f(key) / (cores × L1_bw)
    └───┬───┘   └────┬────┘   └────┬────┘   └──────────┬──────────┘
   run_gen+merge    D2H        gather+scan      fixup inner loop
```

Where: B = total compact key bytes on GPU, G_bw = HBM bandwidth, N = records, K = perm size (4B), C_pcie = PCIe bandwidth, R = record size, H_bw = host DRAM bandwidth, f(key) = comparison cost per record.

**Current hardware bottlenecks (RTX 6000 rig):**
- Run gen: GPU HBM bandwidth (672 GB/s)
- D2H: PCIe 3.0 (12 GB/s) — small fraction of total
- Gather: Host DRAM bandwidth (21 GB/s sustained) — largest single phase
- Fixup: Host compute + L1/L2 (cache-resident)

### Projected performance on other hardware

| Phase | RTX 6000 | H100 PCIe | GH200 |
|:------|--------:|---------:|------:|
| Run gen | 2,200 ms | ~600 ms | ~600 ms |
| GPU merge | 610 ms | ~150 ms | ~150 ms |
| D2H perm | 94 ms | ~15 ms | ~2 ms |
| CPU gather | 1,710 ms | ~450 ms | **0 ms** |
| Fixup | 1,760 ms | ~350 ms | ~450 ms |
| **Total SF50** | **6,370 ms** | **~1,560 ms** | **~1,200 ms** |

GH200 is the killer case: unified memory eliminates the gather phase entirely (GPU writes directly to host memory via NVLink-C2C at 900 GB/s).

---

## 6. What Worked

### Wins that shipped

| Optimization | Impact | Commit |
|-------------|--------|--------|
| **Compact key detection** | SF100: 13.2s → 8.0s (zero fixup) | `8098b49` |
| **Parallel group-detect** | SF50: 12.44s → 6.56s | `74c5749` |
| **Work-queue dispatch** | Load balance 0.70 → 0.98 | `74c5749` |
| **8B uint64 key + memcmp tail** | SF50: 6.56s → 6.21s | `4d56f91` |
| **Fixup early-exit** | SF50: -164ms (skip sorted groups) | `44397b1` |
| **Sampling fix (full scan ≤50M)** | Taxi 6mo: 17.4s → 0.9s | `366335e` |
| **Multiset hash verifier** | Caught entropy bug | `634c137` |
| **Triple-buffered pipeline** | Overlap GPU sort + CPU gather + DMA | — |
| **CUB satellite workaround** | Enable 16B merge with single-satellite CUB | `8098b49` |

### Key architectural decisions

1. **Keys on GPU, values on CPU.** The GPU sorts keys and produces a permutation; values never cross PCIe. This gives 0.6x PCIe amplification vs 2.0x for full-record round-trip.

2. **LSD radix sort, not comparison sort.** CUB's LSD radix sort achieves near-HBM-bandwidth throughput. A comparison-based GPU sort (Thrust) was 13x slower due to warp divergence.

3. **Compact keys over full keys.** For structured data with constant bytes, sorting 32B compact keys (4 LSD passes) instead of 66B full keys (9 passes) cuts GPU sort time and PCIe volume. But only when varying bytes ≤ 32.

4. **CPU fixup over GPU segmented sort.** For 3.15M groups of avg 95 records, CPU `std::sort` on packed cache-resident buffers outperforms any GPU approach (GPU kernel launch overhead > sort time per group).

---

## 7. What Didn't Work

### Failed experiments

| Experiment | Result | Why it failed | Branch |
|-----------|--------|---------------|--------|
| **Entropy byte selection** | INCORRECT OUTPUT | Breaks lex ordering — bytes selected by cardinality, not position | `exp/entropy-selection` |
| **GPU boundary detection** | Net neutral | Scan saved 540→46ms but Pass 4 gathers added +58ms, high variance | `exp/fixup-fast-comparator` |
| **NUMA-pinned gather** | Regressed +400ms | Scatter reads from both sockets regardless of thread pinning | `exp/numa-pin-gather` |
| **mbind(INTERLEAVE)** | 2-17x slower | Dropped MAP_POPULATE, page faults during gather | `exp/mbind-interleave` |
| **Insertion sort (small groups)** | Regressed | std::sort already optimal at n=15-95 | `exp/insertion-sort-small-groups` |
| **Prefetch sweep** | Null | 48 threads + 512-line prefetch already optimal | `exp/prefetch-sweep` |
| **Hybrid 32B extract** | +1.2s regression | CPU extraction slower than GPU path | `exp/hybrid-32b-extract-fast` |
| **Native 32B OVC refactor** | +6.7% regression | Group topology changed (1×290M → 3.15M×95), per-group overhead increased | `exp/native-gpu-32b-ovc` |
| **OVC merge (relative deltas)** | Incorrect | OVC deltas are relative, not comparable across runs | — |
| **Thrust comparison sort** | 13x slower | Warp divergence on variable-length key comparison | — |
| **FORCE_SINGLE_PASS on SF50** | OOM | Need ~32 GB for buffers, only 24 GB HBM | — |

### Lessons learned

1. **Always verify BOTH sortedness AND multiset hash.** The entropy bug passed multiset (records present, just reordered) but failed sortedness. Sweep scripts that only check one metric will miss correctness bugs.

2. **Compact key byte selection is order-sensitive.** You cannot reorder bytes for better discrimination. Source-position order is the only correct choice for lexicographic preservation.

3. **MAP_POPULATE is non-negotiable** for output buffers. Page faults during random-access gather are catastrophic.

4. **Sampling can miss low-entropy bytes.** A 1M stratified sample missed byte 9 on NYC Taxi 6-month data (timestamp high byte barely varies over 6 months). Fix: full scan for ≤50M records.

5. **Run-to-run variance is ~1s on SF50/SF100.** Trust same-session alternating sweeps over cross-session comparisons.

---

## 8. Major Challenges

### Challenge 1: The CUB Single-Satellite Limitation

CUB's `DeviceRadixSort::SortPairs` carries exactly one satellite array alongside keys. For multi-pass LSD merge on 16B+ prefixes, three arrays must move together (prefix1, prefix2, global_perm). Solution: two-pass sort with an identity index as intermediary. This was the key insight enabling the 16B merge path that makes SF100 achievable at 8.0s.

### Challenge 2: GPU Memory Management

With 24 GB HBM, fitting all buffers (keys, alt buffers, permutations, CUB temp) requires careful allocation ordering. The 32B path needs: d_prefix (4.8GB) + d_ovc (4.8GB) + d_perm (2.4GB) + alt buffers + CUB temp ≈ 20 GB. Solved by freeing d_ovc before allocating merge buffers. CUDA memory fragmentation makes this order-dependent — wrong order causes OOM even with sufficient total memory.

### Challenge 3: Fixup Scalability

The 8B prefix creates few, large tie groups (3,817 groups × 157K records on SF100). The 32B compact prefix creates many, small tie groups (3.15M groups × 95 records on SF50). Both require different optimization strategies:
- Large groups: need algorithmic improvement (radix sort, GPU segmented sort)
- Small groups: need dispatch efficiency (work-queue, pre-allocated buffers)

The 32B path with small groups won after parallelizing group detection and using work-queue dispatch.

### Challenge 4: Correctness Infrastructure

Building a verification system that catches subtle ordering bugs while being fast enough to run on every invocation. The dual-check system (sortedness + multiset hash) runs in <100ms on 300M records and has caught every bug introduced during optimization, including the entropy-selection bug that would have been a silent data corruption in production.

### Challenge 5: Compact Key Correctness

Understanding that compact key extraction is NOT a simple "pick the interesting bytes" problem. The bytes must be in source-position order to preserve lexicographic ordering. This is a subtle invariant that is easy to violate (the entropy-selection attempt did) and hard to test for (requires adversarial inputs where the violation manifests).

---

## 9. Source Code Map

| File | Lines | Role |
|------|------:|------|
| `src/external_sort.cu` | ~5000 | Main pipeline: run gen, merge, gather, fixup, verification |
| `src/merge.cu` | ~600 | GPU merge kernels (2-way merge path, 8-way merge tree) |
| `src/host_sort.cu` | ~600 | In-HBM sort orchestration, multi-pass merge |
| `src/run_generation.cu` | ~300 | Block-level bitonic sort + OVC delta encoding |
| `src/main.cu` | ~200 | Entry point, data generation, CLI |
| `include/record.cuh` | ~80 | Record format, compile-time config |
| `include/ovc.cuh` | ~170 | OVC encoding/decoding |
| `include/loser_tree.cuh` | ~165 | OVC-aware loser tree (implemented, not used in current path) |
| `benchmark_suite.cu` | ~1000 | Multi-config benchmark driver |
| `tools/verify_sorted.cpp` | — | Standalone sortedness checker |
| `tools/verify_full.cpp` | — | Standalone sortedness + multiset checker |

---

## 10. Experiment Infrastructure

### Data generators (scripts/)
- `gen_tpch_normalized.py` — TPC-H lineitem to 120B fixed records
- `gen_nyctaxi_normalized.py` — NYC Taxi (single year, year range, or months)
- `gen_random_records.py` — Uniform random records
- `gen_sorted_variants.py` — Sorted, reverse, nearly-sorted, sawtooth
- `gen_duplicate_keys.py` — Controlled duplicate rates
- `gen_varying_bytes.py` — Controlled varying-byte counts for compact key testing

### Benchmark scripts (scripts/)
- `overnight_experiments.sh` — Full suite: TPC-H + Taxi + Random, 3+ warm runs
- `paper_run_scaling.sh` — Scaling curve (1M to 300M)
- `paper_run_recsize.sh` — Record size sensitivity (value = 54B to 512B)
- `paper_run_dupkeys.sh` — Duplicate key behavior
- `paper_run_varying_bytes.sh` — Compact key threshold sweep

### Key experiment results (results/paper/)
- `scaling_curve.md` — Near-linear scaling, throughput increases at large sizes (6→8.6 GB/s)
- `phase_breakdown.md` — Per-phase timing for SF10/50/100
- `compact_ablation.md` — Compact 1.6x slower on SF50 (61/66 vary), wins on SF100 (27/66 vary)
- `record_size_sensitivity.md` — GPU sort time constant regardless of value size
- `input_distributions.md` — Input-oblivious: sorted/reverse/random within ±5%
- `duckdb_baselines.md` — 12-13x faster than DuckDB on SF10

---

## 11. Build and Run

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH

# GenSort format (10B key, 100B record)
make external-sort ARCH=sm_75

# TPC-H format (66B key, 54B value, 120B record)
make external-sort-tpch-compact ARCH=sm_75

# Run with verification
./external_sort_tpch_compact --input /tmp/lineitem_sf50_normalized.bin --verify --runs 3

# Generate test data
python3 scripts/gen_tpch_normalized.py 50   # TPC-H SF50
python3 scripts/gen_nyctaxi_normalized.py 2019-2023  # NYC Taxi 5yr
```

---

## 12. Future Directions

1. **DuckDB integration.** DuckDB already normalizes keys to byte-comparable format. Adding a `PhysicalGPUOrder` operator that calls our sort on normalized buffers would give DuckDB 10x+ sort speedup.

2. **GH200/Grace Hopper port.** Unified memory eliminates the gather phase entirely. Projected SF50 time: ~1.2s (5.2x faster than RTX 6000).

3. **H100 with 80 GB HBM.** The memory gate that forces multi-chunk on SF100 disappears. 32B path becomes viable for SF100, and GPU boundary detection becomes worthwhile (D2H cost drops with PCIe 4/5).

4. **Multi-GPU via NCCL.** CrocSort's M/D ratio invariance predicts that optimal per-GPU config transfers directly to multi-GPU scaling.

5. **GPU Direct Storage.** NVMe → GPU directly (skipping host RAM) for the external sort path. Could improve throughput 30-50%.
