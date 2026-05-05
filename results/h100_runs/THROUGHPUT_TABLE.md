# Comprehensive throughput table — current best (2026-05-03 evening)

All measurements on `sorting-h100`: 4× NVIDIA H100 NVL (94 GB HBM each),
1 TB host RAM, 3.5 TB NVMe. TPC-H lineitem normalized format (120 B records,
66 B sort key, 26 varying bytes mapped to 32 B compact key).

## Single-GPU sort (warm best)

**As of 2026-05-04: all numbers below are with `numactl --cpunodebind=0
--preferred=0` wrap (17.3.2.7 finding — 1.71×-1.96× wall reduction).**

| Scale | Records  | Bytes  | Wall (best warm) | Effective GB/s | PCIe GB | Notes |
|-------|----------|--------|-------|----------------|---------|-------|
| SF10  | 60 M     | 7.2 GB | 0.35 s | 20.4 | 14.4 | fast path |
| SF50  | 300 M    | 36 GB  | **1.41 s** | **25.5** | 10.8 | numactl + post-0.3.1 |
| SF100 | 600 M    | 72 GB  | **2.95 s** | **24.4** | 21.6 | numactl + post-0.3.1; first beats RTX 6000 |
| SF300 | 1.8 B    | 216 GB | **6.55 s** | **33.0** | 64.8 | numactl --preferred + OVC; **highest single-GPU throughput** (median 6553 ms across 4 warm runs, triple-validated) |
| SF500 | 3 B      | 360 GB | OOM    | —    | —    | exceeds 94 GB HBM |

Pre-numactl (2026-05-03 evening): SF50 1.51, SF100 3.02, SF300 8.65 s.
Numactl wall variance is also dramatically tighter (74 ms range at
SF300 vs 7265 ms range without). See `17.3.2.2_numactl_headlines.md`
and `17.3.2.7_preferred_vs_membind.md`.

**Note: `--preferred=0` is strictly preferred over `--membind=0`** — at
SF50/SF100 they perform identically; at SF300 `--preferred` is 13%
faster because output buffer spills to node 1, getting 2× write
bandwidth from both nodes' DDR5 channels. Strict --membind also
intermittently OOMs at SF300 when node 0's free memory is tight.

## Multi-GPU 4×H100 sort (warm best)

### 15.4 — position-partitioned (each GPU sorts 1/4, NOT globally sorted)

| Scale | Bytes  | Wall (sort phase) | Aggregate GB/s |
|-------|--------|-------------------|----------------|
| SF500 | 360 GB | **6.77 s**        | **53.2**       |

### 15.5 — distributed sort with k-way merge (globally sorted)

| Scale | Bytes  | Wall (incl. merge) | End-to-end GB/s |
|-------|--------|--------------------|-----------------|
| SF500 | 360 GB | ~12 min            | 0.50 |

### 15.5.3 — sample-partition + paired-concurrent sort (globally sorted, NO merge)

| Scale  | Bytes   | Wall   | End-to-end GB/s | Notes |
|--------|---------|--------|-----------------|-------|
| SF500  | 360 GB  | **7m36s** | 0.79 | partition (66 s warm) + 2 pairs (180+200 s) |
| SF1000 | 720 GB  | **14m20s** | 0.84 | first ever globally sorted SF1000. Pre-partitioned sort phase only: **6m14s** (18.5c). 2-GPU one-per-node beats 4-GPU paired by avoiding contention. |
| **SF1000 STREAM PRE-PIN** | **720 GB** | **3m48s** | **3.16** | **stream pre-pin at SF1000** (19.36). Partition: 2m44s. Pre-pin: 24s. Sort: 40s. **At NVMe read peak (3.1 GB/s). 73% faster than 14m20s prior best.** Validated 1M pairs PASS. |
| **SF500 COMPACT** | **360 GB** | **2m35s** | **2.32** | compact pipeline (19.24). Partition: 1m28s. Sort: 1m02s. |
| **SF500 STREAM PRE-PIN** | **360 GB** | **1m59s** | **3.01** | **streaming compact + pre-pin at SF500** (19.31). Partition (in-RAM): 1m20s. Pre-pin: 14s. Sort: 26s. **At NVMe read peak (3.1 GB/s). 4× faster than 15.5.3's 7m36s; 24% faster than 19.24.** Validated 1M pairs PASS. |
| **SF300 STREAM PRE-PIN** | **216 GB** | **1m30s** | **2.59** | **stream pre-pin at SF300** (19.32). Linear scaling vs SF1500. PASS. |
| **SF100 STREAM PRE-PIN** | **72 GB** | **32s** | **2.51** | **stream pre-pin at SF100** (19.32). 31.6s with full file I/O. PASS. |
| **SF1500** | **1080 GB** | **31m07s** | **0.58** | **full records output**. K=16 partition + 4-GPU concurrent + posix_fadvise(DONTNEED) cache eviction (19.16, n=3 ±2s). **37% faster** than 19.1.3 baseline (49m15s). **Largest published TPC-H lineitem global sort with full payload.** |
| **SF1500 PERM** | **1080 GB** | **22m31s** | **0.80** | **perm-only output (34 GB, 4-byte indices)**. PERM_ONLY=1 + K=16 4-GPU recipe (19.20). **27% faster** than full-records mode. **n=3: 22m31s / 22m45s / 22m52s, ±21s variance.** NVMe write halved by skipping the 1.08 TB sorted-records emit. Verified correct at SF50. **54% faster than 49m15s 19.1.3 baseline.** |
| **SF1500 COMPACT v1** | **1080 GB** | **12m07s** | **1.49** | compact pipeline 40-byte buckets + 8-byte sorted offsets (19.21). Partition v1 (2-pass): 7m32s. Sort phase 3m03s. |
| **SF1500 COMPACT v2** | **1080 GB** | **8m54s** | **2.02** | single-pass compact partition (19.22). Partition: 4m33s. Sort: 3m09s. n=2: 8m54s, 8m41s = ±13s. |
| **SF1500 COMPACT v3** | **1080 GB** | **7m56s** (best) **8m24s** (mean) | **2.27** (best) **2.14** (mean) | **single-pass compact + no-bucket-evict** (19.23). Keeps 360 GB bucket cache hot between partition and sort → fread serves from RAM, per-round 38s vs 47s. **n=3: 7m56s / 8m38s / 8m39s. Run 1 hit NVMe SLC cache (1.34 GB/s write); runs 2-3 sustained at NVMe write floor (1.14 GB/s). 83-84% faster than 49m baseline.** Validated at SF1500 (16/16 PASS, 4M pairs). |
| **SF1500 STREAM K=16** | **1080 GB** | **7m14s** (best) **7m17s** (mean) | **2.49** (best) **2.48** (mean) | **streaming partition+sort, NO intermediate NVMe write** (19.25). Pinned host bucket buffers (lazy-pinned per bucket); GPU sorts directly from RAM; outputs 68 GB sorted offsets. **n=3: 7m14s / 7m17s / 7m19s, ±3s — tightest variance.** |
| **SF1500 STREAM K=8** | **1080 GB** | **6m43s** (best) **7m02s** (mean) | **2.68** (best) **2.56** (mean) | streaming with K=8, lazy pin (19.27). n=3: 6m43s / 7m05s / 7m18s. |
| **SF1500 STREAM K=8 PRE-PIN** | **1080 GB** | **6m23s** (best) **6m56s** (mean) | **2.82** (best) **2.59** (mean) | **streaming with K=8 + pre-pin all 8 bucket buffers concurrently after partition** (19.30). Pre-pin saves ~50s on sort phase by overlapping cudaHostRegister across buckets. malloc h_offsets (vs cudaMallocHost) avoids pinned-mem fragmentation OOM. **n=3: 6m23s / 7m07s / 7m18s. 87% faster than 49m baseline (best); 86% mean.** Validated at SF1500 (1M pairs PASS). |
| **SF1500 STREAM E2E** | **1080 GB** | **39m18s** | **0.46** | **stream + gather (full sorted records output, equivalent to 49m baseline output)** (19.26). Stream: 7m31s. Gather (random-read input.bin + sequential-write 1.08 TB sorted records): 30m53s. **20% faster than 49m baseline on apples-to-apples sorted-records output.** Verified 0 violations across 1M records. Gather is bottlenecked by cache miss on input.bin (1.08 TB > 1 TB host RAM). |

## CPU baselines (1 process, 192-core box)

| Tool             | Scale | Wall   | Effective GB/s | gpu_crocsort speedup |
|------------------|-------|--------|----------------|----------------------|
| DuckDB 1.3.2     | SF50  | 33.0 s | 1.09 | 21.9× |
| DuckDB 1.3.2     | SF100 | 106 s  | 0.68 | 35.1× |
| Polars 1.8.2     | SF50  | 14.5 s | 2.48 | 9.6× |
| Polars 1.8.2     | SF100 | 28.6 s | 2.52 | 9.5× |

## NVLink p2p bandwidth (15.2)

- Pair-wise (any of 12 pairs): 133.3 GB/s/direction (89% of NV6 ceiling)
- Aggregate inter-GPU: 533 GB/s (17× PCIe5 ceiling)
- Intra-GPU memcpy: 994 GB/s (HBM3, ~1 TB/s effective per direction)

## Roofline efficiency (single-GPU SF300, post all wins)

| Phase                  | Theoretical min | Measured | Efficiency |
|------------------------|-----------------|----------|------------|
| Phase 1 PCIe upload    | 1.8 s           | ~3.0 s   | 60% |
| Phase 1 CPU extract    | 0.2 s           | ~0.7 s   | 28% |
| Phase 2 GPU merge      | 0.05 s          | 0.6 s    | 8% |
| Phase 3 CPU gather     | 0.5 s           | 4.5 s    | 11%  ← biggest gap |
| Phase 4 fixup          | 0.0 s           | 0.0 s    | n/a |
| **Total**              | **~2.6 s**      | **8.65 s** | **30%** |

## Current biggest unfixed bottleneck

**Phase 3 CPU gather at 11% memory-bandwidth efficiency.** Two-pass
gather attempt (sequential reads, scattered NT writes) was *slower*
than the original (random reads, sequential NT writes) due to TLB
thrash on a 216 GB output buffer. To reach 50% efficiency would
require huge-page-aware buffer placement + NUMA-partitioned output.
Multi-day refactor; deferred.
