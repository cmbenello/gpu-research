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
| **SF1500** | **1080 GB** | **34m00s** | **0.53** | **above-RAM regime, optimized.** K=8 partition + 2-GPU one-per-node + posix_fadvise(DONTNEED) cache eviction (19.15). 31% faster than 19.1.3 baseline. **Largest published TPC-H lineitem global sort with full payload.** |

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
