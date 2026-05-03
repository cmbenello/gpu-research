# H100 sort bottlenecks — measured + ranked

**Date:** 2026-05-03
**Source data:** all experiments under `results/h100_runs/` from 2026-05-02 session
**Method:** phase-level timing from binary stdout (no profiling tools needed — the binary already prints per-phase ms).

## TL;DR

Three independent bottleneck classes, ranked by impact:

| # | Bottleneck | Where it bites | Cost | Fix | Estimated wall-time win |
|---|-----------|----------------|------|-----|------------------------|
| 1 | **CPU gather phase** | SF300+ warm runs | **5.2 s of 10 s wall (52%)** | Better memory access pattern, NUMA-aware partitioning, or NVMe-direct streaming | 2-3× on the gather → ~25-30% wall improvement |
| 2 | **Slow-path full-key upload** | SF50/SF100 only | **~50% of run-gen wall** | Use compact 32 B keys instead of full 66 B keys (filed as 0.3.1) | 2× at SF50/SF100 |
| 3 | **Single-GPU HBM ceiling** | SF500+ (single GPU) | OOM, no measurement possible | Either chunked-OVC (1.4.2) or multi-GPU (15.4); the latter is the headline story | Unblocks SF500/1000/...; multi-GPU 4× HBM aggregate fits SF500 entirely |
| 4 | **Single-host RAM ceiling** | SF1500+ | OOM-killed before sort | Streaming input from NVMe (1.9.1) — degrades to NVMe-bound (~3 GB/s) | Unblocks SF1500+ but at ~10× slowdown |
| 5 | **Cold-cache penalty** | First run after gen | 78 s vs 5 s warm at SF300 | None — fundamental to NVMe at 3 GB/s | Always discard run 1 in benchmarks |

The order matters: #1 is the single biggest opportunity by *absolute wall time*, but
#2 has the biggest payoff *relative to existing measurements* (it's a code
bug, not a hardware limit). #3 is the architectural story for the paper.

## Measured phase breakdown (SF300 warm best, 8.94 s total)

From `/tmp/1.2_sf300_baseline.log` warm runs:

```
Phase 1: Run Generation (CPU compact-extract + H2D + GPU per-chunk sort)
  → 3 runs of ~76 GB each, 3.8 s total
  → ~38% of wall

Phase 2: GPU 16B Prefix Merge (2 CUB LSD passes)
  → 580 ms (Pass 1 = 389 ms gather, Pass 2 = 191 ms sort)
  → ~6% of wall

Phase 3: CPU Gather (read records via permutation, write contiguous output)
  → 216 GB at 41 GB/s = 5.2 s
  → ~52% of wall  ← BIGGEST

Phase 4: CPU Prefix Fixup (resolve ties on the 16-byte prefix)
  → ~few hundred ms
  → ~4% of wall
```

**Cold-cache contrast** (run 1 of 3, no warm-up): the *same gather phase* takes
**78 s** because the input pages were evicted from page cache during the
compact-extract + sort and have to fault back in from NVMe at ~3 GB/s.

## Bottleneck #1 — CPU Gather (5.2 s @ SF300 = 52% of wall)

### What it does

For each output position `i` in [0, num_records):
- Read `RECORD_SIZE` (120 B) from `h_data[perm[i]]` — *random* host access
- Write `RECORD_SIZE` to `h_output[i]` — sequential output write

At SF300: 1.8 B records → 216 GB read + 216 GB write = **432 GB host memory traffic**.

Measured: 41 GB/s effective. Theoretical DDR5 bandwidth on this box: ~400 GB/s.
**We are at 10% of memory-bandwidth ceiling on the gather phase.**

### Why it's slow

Random reads from a 720 GB-spanning host buffer are cache-pessimal:
- Each L3 miss costs ~100 ns
- 1.8 B reads × 100 ns × (1 / 192 cores) = **~1 s** floor (assuming linear scaling)
- We measure 5.2 s → 5× off the optimistic floor

The current implementation is multi-threaded (one thread per chunk) with
software prefetch (`__builtin_prefetch` 32 records ahead per `gather_worker`),
but cache misses still dominate.

### Approaches to overcome it

| Idea | Estimated win | Effort | Risk |
|------|---------------|--------|------|
| **A. NUMA-pin gather threads** to the NUMA node holding their input chunk | ~1.5× | low (~1 day) | low — tested pattern |
| **B. Two-pass gather**: pass 1 sorts perm by source location to make reads sequential, pass 2 writes output. Trades extra perm-sort for sequential reads | 2-3× | medium (~2 days) | medium — extra memory pass |
| **C. GPU-orchestrated chunked gather**: upload records in 20 GB chunks (sized to fit free HBM after sort buffers), do gather on GPU, download. Saves cache thrash but adds 432 GB of PCIe round-trip | NEUTRAL — PCIe5 ceiling makes this 2× *worse* than current at SF300; might win at SF1000 | high (~1 week) | high — needs orchestration logic |
| **D. Sort-merge-stream**: instead of materializing full output, stream sorted output to NVMe as it's produced. Skips the gather entirely if final destination is disk anyway | 5× to 10× depending on output target | high (~1 week) | medium — only helps disk-bound workloads |
| **E. Hold output as permutation only**: don't materialize records, return `(records, perm)` pair. Caller can iterate sorted-order without gather | Eliminates gather entirely for read-only callers | low (~1 day) | low — API change but additive |

Recommended: **B. two-pass gather** as the next big code work after multi-GPU.
A and E are quicker wins to do first.

## Bottleneck #2 — Slow-path full-key upload (SF50/100 only)

### What it does

In `ExternalGpuSort::sort()` at the "data fits in GPU" / pre-OVC path
(`external_sort.cu:2536`), the binary uploads the full 66-byte key per
record via `cudaMemcpy2DAsync` (strided DMA). PCIe traffic =
`num_records × 66 B`. At SF100 = **39.6 GB H2D**, even though the GPU
sort only needs the 32-byte compact key.

### Why it's slow

Filed as **0.3.1**. The OVC code path at SF300+ already has a CPU-side
compact key extraction + 32 B upload — it just isn't reused at smaller
scales.

### Approach

Refactor the single-pass key-upload path to use the same compact
extraction. **Concrete plan:**

1. Allocate `d_compact` sized for `compact_key_size × num_records`.
2. CPU-extract compact keys into a pinned host buffer (use the runtime
   cmap; reuse the OVC path's extractor function).
3. Single H2D copy of the compact buffer to `d_compact`.
4. Replace `extract_uint64_chunk_kernel` (KEY_SIZE-strided) calls with
   `extract_uint64_from_compact_kernel` (compact-strided).
5. Loop count = `(compact_size + 7) / 8` instead of `(KEY_SIZE + 7) / 8`.

**Predicted win:** SF50 / SF100 wall time drops from 2.6 s / 4.5 s to
~1.3 s / 2.5 s (2× speedup), bringing both into the 20-25 GB/s range
that SF300 already achieves. This makes the throughput envelope chart
**monotonic** — currently it has the embarrassing dip at SF50/100.

**Effort:** 1-2 hours code + verification.
**Risk:** low — the compact extractor is already proven on the OVC path.

## Bottleneck #3 — Single-GPU HBM ceiling (SF500 OOMs)

### What it does

At SF500: the unchunked single-pass key buffer needs `3B × 66 = 198 GB`,
the OVC path's combined buffers need `60+42+0.5 = 102.5 GB`. Both
exceed the 94 GB HBM. The path-selector at line 2069 returns "use_ovc
= false" because the OVC path also doesn't fit, but then falls through
to the unchunked path which OOMs.

### Approaches

| Idea | Win | Effort | Risk |
|------|-----|--------|------|
| **A. Chunked OVC** (1.4.2) | Unblocks SF500/SF1000 single-GPU. Same throughput class as SF300 (~20-24 GB/s) | medium-high (~3 days). Requires breaking the OVC merge into chunks that fit | medium — OVC is intricate |
| **B. 4-GPU partition-then-sort** (15.4) | Unblocks SF500 *entirely in HBM aggregate* (376 GB > 360 GB). Predicted 4× speedup vs 1 GPU on smaller scales | high (~1-2 weeks). New code path with NCCL + sample partitioner | high — new architecture |
| **C. Multi-host** | Unblocks SF1500+ that the host can't even pin | very high (~weeks) | very high |

Recommended: **start with A. chunked OVC** because it's a single-GPU code change
(easier to debug). **B. multi-GPU** is the headline-story for the paper but is
a much bigger lift.

## Bottleneck #4 — Single-host RAM ceiling (SF1500 OOM-killed)

The 1080 GB SF1500 input can't be pinned on a 1024 GB host even with
the input alone. SF1500 is **out of single-host envelope** regardless
of GPU strategy.

### Approach

**Streaming input from NVMe** (1.9.1): instead of pin-everything,
read input in 200-300 GB chunks via mmap (no `cudaHostRegister`),
sort each chunk, merge across chunks. Wall time floor: NVMe at 3 GB/s
× 1080 GB = **360 s** vs current 9 s for SF300. 40× slowdown but it's
the only path on this box.

**Effort:** ~1 week. New top-level orchestrator, has to interleave
NVMe reads with GPU sort.

## Bottleneck #5 — Cold-cache penalty

Cold runs are ~9× slower than warm (SF300 cold = 93 s, warm = 10 s).
This is fundamentally NVMe-bound (3 GB/s × 216 GB = 72 s, matches the
93 s cold delta). **No code fix possible.** Reportable as a
limitation: warm-cache numbers are what the paper should cite.

A consequence: the SF300+ "cold first run" penalty is a real-world
production cost. For a paper about GPU sort *throughput*, warm is
fair. For an end-to-end benchmark, both should be reported.

## Recommended attack plan

Listed in priority order. Each item is independent — can do in parallel
with the others where multiple sessions / hands are available.

### Tier A — small wins, days each, do first

1. **0.3.1 — slow-path compact upload** (this session, 1-2 hours)
   Fixes the throughput dip at SF50/100. Makes the envelope chart
   tell a clean monotonic story for the paper. Low risk.

2. **NUMA-pin the gather threads** (Bottleneck #1A, half day)
   Pin gather threads to the NUMA node holding their input chunk.
   ~1.5× speedup on the 5 s gather phase = ~700 ms wall improvement
   at SF300. Easy if we use `numa_run_on_node()`.

3. **Hold output as permutation only** (Bottleneck #1E, half day)
   Add a "no gather" mode that returns `(records, perm)` pair. For
   any callers that want to *iterate* the sorted records (joins, scans,
   stream-out to disk), this skips the entire 5 s gather phase. **Could
   make SF300 reach 4 s wall time.**

### Tier B — medium wins, week-scale work

4. **Chunked OVC fallback** (1.4.2, 3-4 days)
   Unlocks SF500/SF1000 single-GPU. Adds the data points the envelope
   chart is missing.

5. **Two-pass gather** (Bottleneck #1B, 2-3 days)
   Sort perm by source location first, then read sequentially. 2-3×
   gather speedup. Combined with #2 NUMA pinning, the gather phase
   could go from 5 s to 1.5 s at SF300 — bringing wall time to ~5.5 s
   (40% improvement).

### Tier C — big wins, multi-week work

6. **4-GPU partition-then-sort** (15.4, 1-2 weeks)
   The headline story for the paper. SF500 entirely in HBM, plus
   strong-scaling chart for 1/2/4 GPUs.

7. **Streaming-input external sort** (1.9.1, 1-2 weeks)
   Unblocks SF1500+ on this box. Slow but finishes.

### Tier D — paper-grade work

8. Roofline model (Tier 17.1) — straightforward analytical work
9. Cross-library baselines (Tier 4) — DuckDB / Polars / cuDF
10. Energy / J/GB measurements (Tier 14.1)

## Concrete recommendation for this session

**Start with 0.3.1** (highest leverage, lowest risk, 1-2 hours). After
that lands and we re-baseline SF50/100, decide between (a) jumping to
multi-GPU 15.4 — biggest scientific impact — or (b) tackling the gather
phase (1A, 1B, 1E) — biggest absolute wall-time win on existing
working scales.
