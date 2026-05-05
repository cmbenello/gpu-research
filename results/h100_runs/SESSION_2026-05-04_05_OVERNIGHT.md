# Overnight 2026-05-04 → 2026-05-05 session — SF1500 49m → 7m56s

**Headline:** Drove SF1500 sort wall from **49m15s** (baseline) to **7m56s**
(best, n=1) / **8m24s** (n=3 mean). **84% faster.** End-to-end **2.27 GB/s**
on a single SSD — 73% of NVMe peak read.

## What got done (chronological)

1. **19.15 — cache-eviction recipe (49m → 34m).** Found that partition writes
   pollute 575 GB of OS page cache, blocking CUDA's pinned alloc. Fixed
   with `posix_fadvise(POSIX_FADV_DONTNEED)`.

2. **19.16 — K=16 + 4-GPU concurrent (34m → 31m07s).** Smaller buckets
   unlock 4-GPU concurrency. n=3 ±2s.

3. **NVMe ceiling analysis (19.18-19.19).** Quantified that 31m07s is at
   the SSD write ceiling, paid twice (1.08 TB intermediate + 1.08 TB output).

4. **19.20 — PERM_ONLY=1 (31m → 22m31s).** Skip CPU gather, emit 4-byte
   sorted indices instead of 120-byte gathered records. Cut sort-phase NVMe
   write 50×. n=3 ±21s.

5. **19.21 — compact bucket format (22m → 12m07s).** Wrote new tools that
   emit 40-byte (32-byte compact key + 8-byte global offset) records into
   bucket files instead of 120-byte records. Cut intermediate write 3×.

6. **19.22 — single-pass partition (12m → 8m54s).** Eliminated partition
   pass-1 count step via per-thread per-bucket RAM buffers and atomic offset
   counters. Saved 3 minutes on partition. n=2 ±13s.

7. **19.23 — keep bucket cache hot (8m54s → 7m56s).** Don't evict bucket
   files between partition and sort. The 360 GB cache stays hot, sort's
   fread serves from RAM. Saves another minute. n=3: 7m56s / 8m38s / 8m39s
   (run 1 hit NVMe SLC cache, runs 2-3 sustained at QLC floor).

8. **19.24 — recipe scales to SF500.** 360 GB sorted in 2m35s. **3× faster
   than prior 15.5.3 SF500 best (7m36s).**

## Tools written

In `gpu_crocsort/experiments/`:
- `partition_by_range_compact.cpp` (v1, 2-pass)
- `partition_by_range_compact_v2.cpp` (v2, single-pass)
- `sort_compact_bucket.cu` (CUB-based bucket sorter)
- `verify_compact_offsets.cpp` (correctness verifier)

In `gpu_crocsort/src/external_sort.cu`:
- `PERM_ONLY=1` env path that skips CPU gather and emits perm.

## Where we are vs theoretical floor

NVMe ceiling: write 1.14 GB/s sustained, read 3.1 GB/s. Theoretical floor
for ANY single-SSD pipeline: read 1.08 TB / 3.1 = **5.8 minutes**.

We're at 7m56s (best) / 8m24s (mean). Gap of 2-2.5 min. That's mostly:
- Single-pass partition writes 360 GB intermediate (~5m at NVMe write rate)
- Per-bucket pin overhead in sort phase (~30s)
- Eviction + setup (~50s)

Closing that gap requires eliminating the intermediate write entirely
(streaming partition→sort), which is a 1-2 day refactor.

## Headline updates in THROUGHPUT_TABLE.md

| Scale  | Before this session | After          | Speedup |
|--------|---------------------|----------------|---------|
| SF500  | 7m36s (15.5.3)      | **2m35s** (19.24) | **3×** |
| SF1500 | (didn't exist clean) | **7m56s** best, 8m24s mean (19.23) | new |

## Validation

- SF50 end-to-end: 10M pairs, 0 violations.
- SF500: 1M pairs (bucket 0), 0 violations.
- SF1500: 4 buckets × 1M pairs each, 0 violations across all sweeps.
- All 7 SF1500 runs (across 19.21-19.23): 16/16 PASS each.

## Commits queued for push

20+ commits on `h100/discoveries-2026-05-02` branch. Auto-push daemon
(PID 308842) will pick them up when SSH agent forwarding returns.

## Remaining open questions

1. **Streaming partition** — eliminate intermediate 360 GB write entirely.
   Multi-day refactor; could land at ~6m.
2. **SF1000** — generate input (~2-3hr dbgen) and apply recipe. Predicted
   ~5m (linear scaling from SF1500's 7m56s × 720/1080).
3. **Adversarial distributions** — test recipe on sorted/reverse/skewed
   synthetic data. Validate robustness.
4. **GPUDirect Storage** — would reduce host pin overhead. Library is
   installed but `nvidia_fs` kernel module not loaded; needs admin.
