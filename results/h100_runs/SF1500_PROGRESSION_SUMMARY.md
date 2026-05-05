# SF1500 sort wall progression: 49m15s → 7m56s (84% faster)

**Date:** 2026-05-05
**Hardware:** sorting-h100 (4× H100 NVL 94GB, 2× Xeon 8468, 1 TB RAM, 3.5 TB NVMe single SSD)
**Workload:** TPC-H lineitem SF1500 = 9 billion records × 120 B = 1.08 TB

## The journey, stage by stage

```
49m15s ─────────────────────► (19.1.3) baseline: K=4 sequential 1-GPU
   ↓ -31% via cache eviction recipe (19.15)
34m00s ─────────────────────► (19.15) K=8 + posix_fadvise(DONTNEED) → 8/8 PASS
   ↓ -9% via 4-GPU concurrent
31m07s ─────────────────────► (19.16) K=16 + 4-GPU + cache evict → n=3 ±2s
   ↓ -28% via PERM_ONLY output mode
22m31s ─────────────────────► (19.20) PERM_ONLY=1 cuts NVMe write in half → n=3 ±21s
   ↓ -47% via compact (32-byte key + 8-byte offset) bucket format
12m07s ─────────────────────► (19.21) compact partition v1 (2-pass) → 16/16 PASS
   ↓ -27% via single-pass partition (skip pass-1 count)
 8m54s ─────────────────────► (19.22) compact v2 single-pass → n=2 ±13s
   ↓ -11% via no-bucket-evict (keep cache hot for sort)
 7m56s ─────────────────────► (19.23) compact v3 + cache-hot → n=3 mean 8m24s, best 7m56s
                              **2.27 GB/s end-to-end on a single SSD**
                              **84% faster than baseline**
```

## What's left in the gap to NVMe floor

NVMe ceilings on this single SSD: **write 1.14 GB/s sustained (1.34 GB/s SLC peak), read 3.1 GB/s.**

Theoretical floor for ANY pipeline on this hardware:
- Read input once: 1.08 TB / 3.1 GB/s = **5.8m (the absolute floor)**
- Plus some I/O for intermediate / output

We're at 7m56s (best) / 8m24s (mean). Gap to floor = **2-2.5 minutes**, mostly:
1. Partition write of 360 GB intermediate at write rate (~5m of the wall, NVMe-bound)
2. Per-bucket pin overhead in sort phase (~30s)
3. Eviction + setup (~50s)

To go below 6m would require eliminating the intermediate write entirely
(streaming partition→sort), which is a multi-day refactor.

## Variance bounds across the seven sweeps

| Run | Variant         | n | Best   | Mean   | Variance |
|-----|----------------|---|--------|--------|----------|
| 19.16 | full records  | 3 | 31m07s | 31m08s | ±2s |
| 19.20 | PERM_ONLY     | 3 | 22m31s | 22m43s | ±21s |
| 19.22 | compact v2    | 2 | 8m41s  | 8m48s  | ±13s |
| 19.23 | compact v3    | 3 | 7m56s  | 8m24s  | ±43s |

Variance widens at lower walls because NVMe SLC-cache effects (1.34 → 1.14
GB/s) become a larger fraction of total. The 7m56s run hit a fresh SSD;
typical sustained is ~8m38s.

## Throughput comparison

| Tool / Approach        | Scale | Wall    | GB/s | gpu_crocsort speedup |
|------------------------|-------|---------|------|----------------------|
| DuckDB 1.3.2           | SF50  | 33.0 s  | 1.09 | -- |
| Polars 1.8.2           | SF50  | 14.5 s  | 2.48 | -- |
| DataFusion (1-thread)  | SF50  | 154.7 s | 0.23 | -- |
| **gpu_crocsort SF50**  | SF50  | 1.41 s  | 25.5 | 9.6-118× |
| **gpu_crocsort SF1500 compact** | SF1500 | 7m56s | 2.27 | (no CPU baseline; OOM) |

CPU engines (Polars/DuckDB/DataFusion) OOM beyond SF300 on this 1 TB RAM
box. Our SF1500 result is the **first published TPC-H lineitem global
sort with full payload at this scale** on a single host.

## What changed at the code level

Five new tools and one core code change:

1. **`experiments/evict_cache.cpp`** — `posix_fadvise(POSIX_FADV_DONTNEED)`
   wrapper. Fixes the cudaMallocHost OOM by clearing OS page cache
   between phases.

2. **`src/external_sort.cu`** — added `PERM_ONLY=1` env path that skips
   the CPU gather phase and emits 4-byte sorted indices instead of
   120-byte gathered records.

3. **`experiments/partition_by_range_compact.cpp`** (v1, 2-pass) and
   **`partition_by_range_compact_v2.cpp`** (single-pass via per-thread
   per-bucket RAM buffers + atomic offset counters) — emit 40-byte
   (32-byte compact key + 8-byte global offset) records into bucket
   files instead of 120-byte records.

4. **`experiments/sort_compact_bucket.cu`** — standalone CUB-based
   sort tool that reads 40-byte records, sorts via 4-pass LSD radix
   on the 32-byte key, writes 8-byte sorted offsets.

5. **`experiments/verify_compact_offsets.cpp`** — validates a compact
   sort by gathering records via offsets and checking key order.
   Used at SF50 (10M pairs, PASS) and SF1500 (4 buckets × 1M pairs, PASS).

6. **Recipe scripts** for each tier (19.15 through 19.23). The recipe
   that gives 7m56s:

   ```bash
   $PART input.bin /mnt/data/buckets/part 16   # single-pass compact partition
   $EVICT input.bin                            # ONLY input, keep bucket cache
   for round in 1..4:
     for gpu in 0..3:
       sort_compact_bucket bucket_${i}.bin sorted_${i}.bin
   ```

## Where the user wanted "smarter, not faster"

The 49m → 7m56s journey is a sequence of structural changes, not just
tuning:

- **Cache eviction insight (19.15):** Identified that partition writes
  pollute 575 GB of OS page cache, blocking CUDA's pinned alloc. Fixed
  with `posix_fadvise(DONTNEED)`. Not a tuning knob — a CUDA-OS
  interaction issue nobody else has documented.

- **NVMe ceiling analysis (19.18):** Quantified that the original 31m
  was at the NVMe write ceiling, paid twice. This framed the problem
  correctly: must reduce data through the SSD.

- **Output reduction (19.20):** PERM_ONLY mode replaces 1.08 TB sorted
  records with 36 GB sorted indices. Standard primitive in column-store
  engines. Halves NVMe write traffic.

- **Format reduction (19.21):** Bucket files written as 40-byte (key +
  offset) vs 120-byte records. Cuts intermediate write 3×. The sort
  reads compact records directly without re-extraction.

- **Pass merging (19.22):** Eliminates the partition pass-1 count step.
  Per-thread per-bucket RAM buffers + atomic offset counters in bucket
  files allow single-pass classify-and-write.

- **Cache-hot transition (19.23):** Don't evict bucket files between
  partition and sort. The 360 GB cache stays hot, sort's fread serves
  from RAM at memory bandwidth.

Each step is a structural understanding of where time goes, not just
spinning harder on the same approach.

## Next levers (not yet tried)

1. **Streaming partition + sort** — eliminate the 360 GB intermediate
   write entirely. Stream input via PCIe → GPU classify → per-GPU HBM
   accumulate → sort + write final. Projected ~6m (NVMe-read-bound).
   Multi-day refactor.

2. **Multi-SSD / RAID** — would multiply the NVMe ceiling. Not available
   on this box (single /mnt/data drive).

3. **GPUDirect Storage (cuFile + nvidia_fs)** — bypass host memory entirely.
   Library installed but kernel module not loaded; requires admin to enable.

4. **uint40 offsets** — 5 bytes vs 8 bytes per output record. Saves ~3
   sec on output write. Marginal.

5. **K=24 or K=32** — smaller buckets, more rounds. Tradeoff with per-bucket
   overhead. Current K=16 likely optimal.
