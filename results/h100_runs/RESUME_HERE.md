# Resume here — overnight session 2026-05-03

## TL;DR for the morning

**~17 commits are queued locally on `h100/discoveries-2026-05-02`** but
*not yet pushed* — the SSH agent socket was lost ~3 AM and never came
back. As soon as you reconnect SSH (the session re-establishes the
forwarded agent socket), run:

```bash
cd ~/gpu-research && bash /tmp/gitpush.sh push
```

That'll dump all the night's work to GitHub.

## What landed (in 50 words)

- **0.3.1 slow-path compact upload** → SF50 1.7×, SF100 1.5× (first beats RTX 6000)
- **15.2 / 15.3 / 15.4 / 15.5 multi-GPU** → SF500 globally sorted (3 BILLION records, single-GPU OOMs)
- **15.5.2 multi-thread merge** → 5.6× speedup
- **DuckDB + Polars baselines** → gpu_crocsort 9-35× faster
- **17.1 roofline analysis** → 30% of peak; gather is the gap
- **64-thread gather default** → 9% wall improvement at SF300
- Two-pass gather attempted but worse (TLB thrash) — kept opt-in via `TWO_PASS_GATHER=1`

## Headline numbers

```
SF50  (300M / 36 GB):  1.51 s @ 24 GB/s   (gpu_crocsort)
                       14.5 s @ 2.5 GB/s  (Polars)
                       33.0 s @ 1.1 GB/s  (DuckDB)

SF100 (600M / 72 GB):  3.02 s @ 24 GB/s   (gpu_crocsort, beats RTX 6000)
                       28.6 s @ 2.5 GB/s  (Polars)
                       106  s @ 0.7 GB/s  (DuckDB)

SF300 (1.8B / 216 GB): 8.67 s @ 25 GB/s   (single-GPU)

SF500 (3B / 360 GB):   OOM (single-GPU)
                       ~12 min globally sorted across 4 H100 NVLs
                       (NVMe-bound on the merge — multi-day fix to skip)
```

## Detailed reading order

1. `BOTTLENECKS.md` — initial bottleneck survey + ranked plan
2. `TIER_A_RESULTS.md` — early Tier A wins + negative NUMA result
3. `0.3.1_compact_upload.md` — biggest single-GPU win
4. `15.2_nvlink_bandwidth.md` → `15.5_distributed_sort.md` → `15.5.2_multi_threaded_merge.md` — multi-GPU progression
5. `4.1_duckdb_baseline.csv` + `4.2_polars_baseline.py` — cross-library
6. `17.1_roofline.md` — analytical ceiling
7. `SESSION_2026-05-03_FINAL.md` — comprehensive session summary

## Open work for next session

The diminishing-returns cliff is clear. The remaining big optimizations
require multi-day code work:

- **Two-pass gather with huge-page TLB optimization** (predicted 2-3× on
  Phase 3 — currently 11% efficient). 2-3 days.
- **Sample-partition for distributed sort (15.5.3)** — eliminates the
  NVMe-bound merge phase. 2-3 days.
- **Chunked-OVC for single-GPU SF500/SF1000** — 1-2 days.

## Disk + box state

```
/mnt/data: 2.3 TB / 3.5 TB used (70%)
/dev/shm:  empty
GPUs:      all 4 idle
Local commits ahead of remote: 17
```

The chunked TPC-H files (SF10/50/100/300/500/1000/1500) are all on
/mnt/data and can be reused without regen.
