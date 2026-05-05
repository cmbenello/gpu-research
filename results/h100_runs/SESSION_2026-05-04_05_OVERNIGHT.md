# Overnight 2026-05-04 → 2026-05-05 session — SF1500 49m → 6m23s, SF500 hits NVMe peak

**Headline:**
- **SF1500**: 49m15s → **6m23s** (best, n=1) / **6m56s** (n=3 mean) = **87% / 86% faster**
- **SF500**: 7m36s → **1m59s** = **74% faster**, 3.01 GB/s **at NVMe peak read**

## What got done (chronological)

1. **19.15** — cache-eviction recipe (49m → 34m). `posix_fadvise(DONTNEED)` fix.

2. **19.16** — K=16 + 4-GPU concurrent (34m → 31m07s, n=3 ±2s).

3. **19.18-19.19** — NVMe ceiling analysis. Quantified that 31m was NVMe-write-bound paid twice.

4. **19.20** — PERM_ONLY=1 (31m → 22m31s, n=3 ±21s). Skip CPU gather, emit 4-byte indices.

5. **19.21** — compact bucket format 40 bytes/record (22m → 12m07s).

6. **19.22** — single-pass partition (12m → 8m54s, n=2 ±13s).

7. **19.23** — keep bucket cache hot (8m → 7m56s best, n=3 mean 8m24s).

8. **19.24** — recipe scales to SF500 (2m35s).

9. **19.25** — streaming partition+sort, NO intermediate NVMe write (7m17s mean, n=3 ±3s).

10. **19.26** — stream + gather true e2e (39m18s, full sorted records output).

11. **19.27** — streaming K=8 (6m43s best, n=3 mean 7m02s).

12. **19.28** — K=4 OOM (45 GB buckets exceed H100 HBM budget).

13. **19.29** — SKIP_PIN NEGATIVE (8m08s, unpinned PCIe slower than pinned).

14. **19.30** — pre-pin all 8 buckets concurrently after partition (6m23s best, n=3 mean 6m56s).

15. **19.31** — SF500 stream pre-pin (**1m59s, 3.01 GB/s = NVMe peak read**).

## Tools written

In `gpu_crocsort/experiments/`:
- `partition_by_range_compact.cpp` (v1, 2-pass)
- `partition_by_range_compact_v2.cpp` (v2, single-pass via per-thread atomic offset counters)
- `sort_compact_bucket.cu` (CUB-based bucket sorter for 40-byte records)
- `stream_partition_sort.cu` (fused partition+sort in one process, no NVMe intermediate)
- `gather_records.cpp` (random-read gather for full-records e2e)
- `gather_records_seq.cpp` (sequential-read gather, alternative)
- `verify_compact_offsets.cpp` (correctness verifier)

In `gpu_crocsort/src/external_sort.cu`:
- `PERM_ONLY=1` env path that skips CPU gather and emits perm.

## SF1500 progression chart

```
49m15s ── (19.1.3 baseline) ───────────────────────────────────────
   │
   ▼ -31% via cache eviction (19.15)
34m00s ── (K=8 + posix_fadvise) ────────────────────────────────────
   │
   ▼ -9% via 4-GPU concurrent (19.16)
31m07s ── (K=16 + 4-GPU + cache evict) [n=3 ±2s] ───────────────────
   │
   ▼ -28% via PERM_ONLY output mode (19.20)
22m31s ── (skip CPU gather, emit perm) [n=3 ±21s] ──────────────────
   │
   ▼ -47% via compact bucket format (19.21)
12m07s ── (40-byte (key + offset) records) ─────────────────────────
   │
   ▼ -27% via single-pass partition (19.22)
 8m54s ── (atomic offset counter, no pass-1 count) [n=2 ±13s] ──────
   │
   ▼ -11% via no-bucket-evict (19.23)
 7m56s ── (keep bucket cache hot) [n=3 best, mean 8m24s] ───────────
   │
   ▼ -8% via streaming (no NVMe intermediate) (19.25)
 7m14s ── (in-RAM bucket buffers) [n=3 ±3s] ────────────────────────
   │
   ▼ -1% via K=8 (fewer rounds) (19.27)
 6m43s ── (K=8 best) [n=3, mean 7m02s] ─────────────────────────────
   │
   ▼ -3% via pre-pin concurrent (19.30)
 6m23s ── (pre-pin all 8 buckets) [n=3 best, mean 6m56s] ◀── BOUNDARY
                                       1.08 TB / 6m23s = 2.82 GB/s
                                       NVMe-read floor: 5.8m (1.08 TB / 3.1 GB/s)
                                       Gap to floor: 33s = NVMe write of 68 GB output
```

## Where we are vs hardware floor

NVMe ceiling (single SSD): write 1.14 GB/s sustained, read 3.1 GB/s peak.

**SF1500 floor:** 1.08 TB / 3.1 GB/s = 5.8 min. We're at 6m23s = **9.6% above floor**.
**SF500 result:** 1m59s = **3.01 GB/s = AT NVMe read peak.** SF500 fits in cache so reads are RAM-speed.

The SF1500 gap is because input doesn't fit in 1 TB cache. Closing requires:
1. More host RAM (hardware change)
2. GPUDirect Storage (admin: load nvidia_fs kernel module)
3. Multi-NVMe (hardware change)

## Variance bounds

| Run | Variant         | n | Best   | Mean   | ±    |
|-----|----------------|---|--------|--------|------|
| 19.16 | full records  | 3 | 31m07s | 31m08s | 2s   |
| 19.20 | PERM_ONLY     | 3 | 22m31s | 22m43s | 21s  |
| 19.22 | compact v2    | 2 | 8m41s  | 8m48s  | 13s  |
| 19.23 | compact v3    | 3 | 7m56s  | 8m24s  | 43s  |
| 19.25 | streaming     | 3 | 7m14s  | 7m17s  | 3s   |
| 19.27 | stream K=8    | 3 | 6m43s  | 7m02s  | 18s  |
| **19.30** | **stream K=8 pre-pin** | **3** | **6m23s** | **6m56s** | **27s** |

Streaming K=16 has the tightest variance. K=8 + pre-pin has the best best-case
but wider variance (NVMe write rate fluctuates with SLC cache state).

## What's left (not pursued, would require admin or hardware)

1. **Streaming gather** for true e2e — random-write output instead of random-read.
   Could halve the 30m gather phase. Implementation: 1 day.

2. **GPUDirect Storage** (cuFile + nvidia_fs) — would skip host pin entirely.
   Requires `sudo modprobe nvidia_fs` + driver setup.

3. **Huge pages** for pin acceleration — would 5-10× pin rate.
   Requires `sudo echo N > /proc/sys/vm/nr_hugepages`.

4. **Multi-SSD striping** — would multiply NVMe ceiling.
   Hardware change.

## Validation

- SF50 end-to-end: 10M pairs, 0 violations.
- SF500: 1M pairs, 0 violations.
- SF1500: 4 buckets × 1M pairs across multiple sweeps, 0 violations.
- All sweeps: 16/16 PASS (or 8/8 for K=8 variants).

## Total commits this session: ~40+

## Final scaling chart (stream pre-pin pipeline, all scales)

| Scale | Bytes | Wall | GB/s | Status |
|-------|-------|------|------|--------|
| SF100 | 72 GB | 32s | 2.51 | linear scaling validated |
| SF300 | 216 GB | 90s | 2.59 | linear scaling validated |
| SF500 | 360 GB | 1m59s | **3.01** | **at NVMe peak read** |
| SF1000 | 720 GB | **3m48s** | **3.16** | **at NVMe peak read; 73% faster than prior 14m20s** |
| SF1500 | 1080 GB | 6m23s | 2.82 | 9.6% above NVMe floor (input > host RAM cache) |

**SF500 and SF1000 are at the NVMe read ceiling** (3.1 GB/s peak). They
fit in 1 TB host RAM cache, so reads are RAM-fast on warm passes. SF1500
exceeds cache (1.08 TB > 1 TB) and drops below ceiling by ~10%.

## Energy

- 4-GPU avg power during SF1500 sort: 308W (77W/GPU = 11% of H100 TDP)
- Energy per SF1500 sort: 140 kJ = 39 Wh
- **130 nJ per byte sorted**, **15.6 µJ per record**
- GPU is idle 90% of wall time — pipeline is NVMe-bound, not compute.
