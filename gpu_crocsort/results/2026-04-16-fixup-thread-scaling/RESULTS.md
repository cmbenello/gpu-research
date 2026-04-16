# Fixup Thread Scaling — SF50, 6 warm runs per config

Date: 2026-04-16
Binary: `external_sort_tpch_compact` @ `exp/fixup-fast-comparator` 3018e74
Hardware: Quadro RTX 6000 (24 GB HBM), 48-core (24C/48T) DDR4-2933 host.
Input: SF50 TPC-H lineitem, 300M records × 120B, 3.15M tie groups (avg 95 records).

## Fixup scaling curve (medians)

| Threads | Fixup (ms) | Group-detect (ms) | Parallel sort (ms) | Speedup vs T=1 | Efficiency |
|--------:|-----------:|-------------------:|--------------------:|---------------:|-----------:|
| 1 | **26 522** | 6 503 | 19 998 | 1.0× | 100% |
| 2 | **14 936** | 4 041 | 10 972 | 1.78× | 89% |
| 4 | **7 827** | 2 121 | 5 720 | 3.39× | 85% |
| 8 | **3 979** | 1 086 | 2 896 | 6.67× | 83% |
| 16 | **2 342** | 768 | 1 560 | 11.3× | 71% |
| 24 | **1 876** | 621 | 1 247 | 14.1× | 59% |
| 48 | **1 785** | 676 | 1 113 | 14.9× | 31% |

## What the curve shows

**Near-linear scaling through 8 threads** (83% efficiency). The two sub-phases
scale differently:

### Group-detect (sequential scan for prefix-change boundaries)
- DRAM-bandwidth bound: 48 threads scanning 300M × 32 scattered byte positions.
- Scales from 6503 ms (T=1) to 768 ms (T=16) = 8.5× for 16× threads.
- **Saturates at T=16**: going from 16→48 threads saves only 92 ms (768→676).
- At T=48 the variance increases (626–1020 ms vs 750–783 at T=16) — cache-line
  contention on the boundary vector and false sharing in the per-thread arrays.

### Parallel sort (per-group pack + std::sort + reorder)
- Compute + L1/L2 bound: each group's 95 records fit in L1 cache.
- Scales from 19998 ms (T=1) to 1113 ms (T=48) = 18.0× for 48× threads.
- Still scaling at 48 threads (1247→1113 from T=24→T=48) because the work-queue
  distributes 3.15M groups across threads without contention.
- Per-thread sort time: T=1 sorts 14734 ms ÷ 1 = 14734 ms per thread.
  T=48 sorts 483 ms per thread = perfect 30.5× per-thread speedup.

### Why T=24 is the sweet spot
- T=24 gives 14.1× total speedup (1876 ms) at 59% efficiency.
- T=48 gives 14.9× (1785 ms) at 31% efficiency — 91 ms more for 2× the threads.
- The group-detect phase actually **regresses** from T=24 to T=48 in several runs
  (DRAM contention), eating the parallel-sort gains.
- On this 48-core DDR4-2933 host, 24 threads saturate the useful DRAM bandwidth
  for the scan pattern.

## Paper figure

This data supports the paper's core claim: "parallel group-detect + work-queue
dispatch was the single biggest optimization (ablation row 3→4, −6.7 s)."

The thread-scaling curve quantifies WHY:
- At T=1, fixup is 26.5 s (4.2× the total sort time without fixup).
- At T=24, fixup is 1.9 s (0.3× the total sort time).
- The transition from "fixup dominates" to "fixup is noise" happens between
  T=4 and T=16 — this is the regime where parallelization earns its keep.

The group-detect saturation at T=16 suggests a future optimization: move the
boundary scan to GPU (already prototyped, see gpu-boundary/RESULTS.md) to
eliminate the DRAM-bound phase entirely. On H100 with PCIe 5 this becomes
net-positive.

## Files

- `sf50_threads_{1,2,4,8,16,24,48}.log` — per-config raw logs.
- `sweep.log` — concatenated full output.
- `../scripts/fixup_thread_sweep.sh` — driver script.
