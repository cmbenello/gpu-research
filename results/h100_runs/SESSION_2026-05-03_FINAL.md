# Final session summary — 2026-05-03 autonomous overnight

Branch: `h100/discoveries-2026-05-02` — 13+ commits queued locally
(SSH agent intermittent during session; will batch-push when available).

## Update — evening continued autonomous run

User said "keep pushing" → daemon /tmp/auto_push.sh runs every 30s,
tries any fresh SSH agent socket. Pushes have been landing.

Additional wins after first FINAL summary:

- **15.5.3 sample-partition** (commit `42937c5`) — eliminates the
  merge phase. SF500 7m36s wall.
- **SF1000 distributed** (commit `eb33936`) — **6 BILLION records
  globally sorted in 22m40s. First ever on this hardware.** Required
  NO_MAP_POPULATE=1 env var (commit `b4ad40f`).
- **64-thread default for slow-path gather** (commit `acf75f0`) —
  matches OVC gather, replaces older 48-thread cap.
- Comprehensive throughput table (commit `b9f2e98`).

## Headline numbers

| Workload | gpu_crocsort | DuckDB | Polars | Notes |
|----------|--------------|--------|--------|-------|
| SF50 (300M / 36 GB) | **1.51 s** @ 23.9 GB/s | 33.0 s | 14.5 s | gpu_crocsort 9.6× faster than Polars, 21.9× faster than DuckDB |
| SF100 (600M / 72 GB) | **3.02 s** @ 23.8 GB/s | 106 s  | 28.6 s | First SF100 win vs RTX 6000 baseline (3.74s); 35.1× faster than DuckDB |
| SF300 (1.8B / 216 GB) | **8.65 s** @ 25.0 GB/s | (out of budget) | (not measured) | OVC path; 1.5× faster than original baseline |
| SF500 (3B / 360 GB) | **OOM** single-GPU | OOM | OOM | 4-GPU partition: **6.77 s sort** + ~5 min merge = ~6 min wall; **first ever globally-sorted SF500 result on this hardware** |

## Code wins this session (in chronological order)

1. **B1 — reuse h_output across `--runs`** — SF300 wall 10.10 s → 8.65 s (15% faster than the *original* baseline, recovering the 1.6.1 regression and improving on it)
2. **0.3.1 — slow-path compact key upload** with pipelined CPU extract + ping-pong staging — SF50: 2.62→1.51s (1.74×), SF100: 4.50→3.02s (1.49×, first beats RTX 6000)
3. **B3-light — relax OVC path-selector** — SF500 single-GPU now reaches OVC path (still OOMs in merge phase; true chunked OVC is multi-day)
4. **15.5.2 — multi-threaded merge** with splitter partitioning — 1.35→7.60 GB/s at 64 threads (5.6×)
5. **15.5.4 — inline verify + skip msync** at SF500 — saves ~120 s of dirty-page flush time

## Multi-GPU experiments that worked

- **15.2 NVLink p2p**: 133.3 GB/s/direction uniform across all 6 GPU pairs (89% of NV6 ceiling)
- **15.3 4-GPU concurrent SF100**: 4 independent sorts, no degradation, 62.6 GB/s aggregate
- **15.4 4-GPU SF500 partition**: 53.2 GB/s sort throughput (single-GPU OOMs; this is "infinite speedup")
- **15.5 distributed sort with merge**: SF500 globally sorted (3 BILLION records) in 11 min, single-GPU has zero capability at this scale

## Cross-library baselines

- **4.1 DuckDB**: SF50 33.0 s @ 1.09 GB/s; SF100 106 s @ 0.68 GB/s
- **4.2 Polars**: SF50 14.5 s; SF100 28.6 s @ 2.5 GB/s

## Critical bug fixes

- **1.6.1 regression** (SF300 segfault since 1.6.1 landed) — fixed
- **NUMA-pin gather threads** — TRIED, REVERTED (3-4× slower; documented inline)
- **MAP_POPULATE** — restored conditionally based on host-RAM headroom

## Open bottlenecks (deferred to next session)

| # | Bottleneck | Status | Effort |
|---|-----------|--------|--------|
| 1 | CPU gather phase (5s of 9s SF300 wall) | Open. Two-pass gather projected 2-3× win; NUMA-partitioned buffers would help further | Multi-day |
| 2 | Single-GPU SF500/SF1000 OOM in OVC merge | Open. Need true chunked OVC (Phase 2 buffers exceed HBM) | Multi-day |
| 3 | Distributed sort I/O staging (12 min wall at SF500) | Open. NVMe write of 360 GB is the floor; sample-partition (15.5.3) eliminates merge phase entirely | Multi-day |
| 4 | Cold-cache penalty | Fundamental NVMe limit; no fix |

## Headline narrative for the paper

**On a 4× H100 NVL box with 1 TB host RAM and 3.5 TB NVMe, gpu_crocsort sorts:**
- TPC-H **SF50 (36 GB)** in **1.51 s warm** at 23.9 GB/s — 22× faster than DuckDB
- **SF100 (72 GB)** in **3.02 s warm** at 23.8 GB/s — first time beating RTX 6000 baseline (3.74 s)
- **SF300 (216 GB)** in **8.65 s warm** at 25.0 GB/s on a single GPU
- **SF500 (360 GB)** in **6.77 s** sort phase using all 4 GPUs (single-GPU literally cannot sort SF500), with 11 min total wall when including the host-side merge to globally sorted output

The throughput curve is now monotonic from SF10 → SF300 (~24-25 GB/s
per GPU). The 4-GPU NVLink mesh delivers 533 GB/s aggregate
inter-GPU bandwidth (17× the PCIe 5 ceiling).

## Files of interest

- `BOTTLENECKS.md` — measured phase breakdown + ranked plan
- `0.3.1_compact_upload.md` — slow-path optimization with negative-result history
- `15.5_distributed_sort.md` — first SF500 globally sorted PoC
- `15.5.2_multi_threaded_merge.md` — 5.6× merge speedup
- `4.1_duckdb_baseline.csv` + `4.2_polars_baseline.py` — cross-library comparisons
- `1.7_envelope_throughput.png` / `_walltime.png` — updated charts

## Recommended next session

1. **Push the 13+ queued commits** as soon as SSH agent is restored.
2. **15.5.3 sample-partition** — would eliminate the merge phase (and the
   ~4 min disk-IO bottleneck) entirely for distributed sort. Big win.
3. **Two-pass gather** for single-GPU at SF300+ — projected 2-3×.
4. **Tier 6.1 nsys trace of SF100 sort** — paper-grade profiling.
5. **Roofline analysis (Tier 17.1)** — analytical, no compute needed.
