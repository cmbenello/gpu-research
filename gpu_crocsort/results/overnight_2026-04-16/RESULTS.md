# Overnight Experiment Results — 2026-04-16

Hardware: Quadro RTX 6000 (24 GB, PCIe 3.0), 187 GB RAM, HDD storage.
Binary: `external_sort_tpch_compact` on branch `exp/fixup-fast-comparator`.
All timings: warm median of 3 runs (ms). Records: 120B (66B key + 54B value).

## Summary Table

| Dataset             | Records  | Size (GB) | Chunks | Vary | Compact | Total (ms) | GB/s  | Notes                          |
|---------------------|----------|-----------|--------|------|---------|------------|-------|--------------------------------|
| TPC-H SF10          | 60.0M    | 7.20      | 1      | 61   | ON      | 1,754      | 4.10  |                                |
| TPC-H SF50          | 300.0M   | 36.00     | 2      | 61   | ON      | 7,153      | 5.03  | fixup: 4,942ms                 |
| TPC-H SF50          | 300.0M   | 36.00     | 9      | 61   | OFF     | 7,681      | 4.69  | fixup: 4,487ms                 |
| TPC-H SF100         | 600.0M   | 72.00     | 4      | 27   | ON      | 8,022      | 8.98  | no fixup (all fit 32B)         |
| Taxi 1mo            | 3.1M     | 0.37      | 1      | 60   | ON      | 136        | 2.71  | single-chunk, hybrid OK        |
| Taxi 1mo            | 3.1M     | 0.37      | 1      | 60   | OFF     | 133        | 2.77  |                                |
| Taxi 6mo            | 19.5M    | 2.34      | 1      | 58   | ON      | 17,370     | 0.13  | **BUG: sample misses byte 9** |
| Taxi 6mo            | 19.5M    | 2.34      | 1      | 58   | OFF     | 17,716     | 0.13  | same bug in hybrid retry       |
| Taxi 12mo           | 38.3M    | 4.60      | 1      | 60   | ON      | 1,393      | 3.30  | fixup: 1813 groups, 496ms      |
| Random 60M          | 60.0M    | 7.20      | 1      | 66   | ON      | 1,765      | 4.08  | no ties (full random)          |
| Random 60M          | 60.0M    | 7.20      | 1      | 66   | OFF     | 1,777      | 4.05  |                                |
| Random 300M         | 300.0M   | 36.00     | —      | 66   | ON      | CRASH      | —     | **BUG: CUDA invalid argument** |
| Random 300M         | 300.0M   | 36.00     | 9      | 66   | OFF     | 6,645      | 5.42  |                                |
| SF50 + ADV_ZERO16   | 300.0M   | 36.00     | 2      | —    | ON      | 7,426      | 4.85  | fixup: 5,261ms                 |
| SF50 + ADV_ZERO32   | 300.0M   | 36.00     | 2      | —    | ON      | 8,140      | 4.42  | fixup: 5,918ms                 |

## Bugs Discovered

### 1. Compact sampling misses rare byte variations (taxi 6mo)

The stratified 1M-record sample misses byte positions 8-9 in the taxi 6mo
dataset. These are the high bytes of `pickup_datetime` (big-endian int64 +
2^62 bias) — with only 6 months of timestamps, the variation is extremely
sparse. The sample detects 58/66 varying but misses bytes 8 and 9.

When the compact sort runs, it discovers byte 9 actually varies, triggering
`[Hybrid] Sample missed varying byte 9 on the fast path`. The fallback
does a full CPU `std::sort` on 19.5M records with 66-byte key comparisons,
taking ~17s instead of ~1.4s.

Taxi 12mo (12 months of timestamps) has enough byte-9 spread that the
sample catches it. Result: 60/66 varying detected, single-chunk fixup
handles 1,813 groups in 496ms, total 1,393ms.

**Fix needed:** Either increase sample density for single-chunk data, or
do a full-scan for byte variation when data fits in memory (cheap: ~30ms
for 19.5M records with SIMD).

### 2. Multi-chunk hybrid fallback uses wrong buffer sizes (random 300M)

When compact upload is used for multi-chunk data (300M random records) and
the compact sort fails verification (299.99M exceptions — all records
differ at byte 64+), the hybrid retry attempts full-key upload through
compact-sized GPU buffers. `cudaMemcpyAsync` at line 1267 fails with
"invalid argument" because the buffer is too small.

The first attempt allocates compact-sized buffers (9.6 GB PCIe for 36 GB
data). When verification fails and the retry starts, it reuses these same
buffers but tries to upload full-size records.

**Fix needed:** Reallocate GPU buffers to full-key size on hybrid retry,
or force compact OFF when 66/66 bytes vary (the detection already says
"too many varying bytes" but doesn't actually disable compact upload).

### 3. Compact upload still used when "no compression benefit" detected

CompactDetect reports "66/66 = 100% varying. Decision: too many varying
bytes — no compression benefit, use full upload" but the code still
proceeds with compact upload (COMPACT UPLOAD: 170M rec/chunk). The
detection decision isn't propagated to the upload path.

## Key Findings

### Compact key effectiveness by dataset

| Dataset    | Compact ON | Compact OFF | Delta | Cause                        |
|------------|------------|-------------|-------|------------------------------|
| TPC-H SF50 | 7,153ms   | 7,681ms     | -7%   | PCIe 9.6→36 GB saves time   |
| Random 60M | 1,765ms   | 1,777ms     | 0%    | All bytes vary, no benefit   |
| Taxi 1mo   | 136ms     | 133ms       | +2%   | Noise, single-chunk          |

Compact key wins when varying positions ≤32 (SF100: 27/66 → zero fixup)
or when PCIe bandwidth savings outweigh fixup cost (SF50: 61/66 but saves
26.4 GB PCIe transfer).

### Adversarial zeroing (SF50)

Zeroing key bytes hurts total time because it creates more 32B prefix ties:

| Variant  | GPU (ms) | Fixup (ms) | Total (ms) | vs baseline |
|----------|----------|------------|------------|-------------|
| baseline | 2,201    | 4,942      | 7,153      | —           |
| zero16   | 2,164    | 5,261      | 7,426      | +4%         |
| zero32   | 2,254    | 5,918      | 8,140      | +14%        |

GPU time barely changes (compact adapts), but fixup grows because zeroed
bytes create more tied groups that need CPU resolution.

### Throughput scaling

| Data Size | Dataset      | Total (ms) | GB/s  |
|-----------|--------------|------------|-------|
| 0.37 GB   | Taxi 1mo     | 136        | 2.71  |
| 4.60 GB   | Taxi 12mo    | 1,393      | 3.30  |
| 7.20 GB   | TPC-H SF10   | 1,754      | 4.10  |
| 7.20 GB   | Random 60M   | 1,765      | 4.08  |
| 36.00 GB  | TPC-H SF50   | 7,153      | 5.03  |
| 36.00 GB  | Random 300M  | 6,645      | 5.42  |
| 72.00 GB  | TPC-H SF100  | 8,022      | 8.98  |

Throughput improves with data size due to amortized overhead and better
pipeline utilization. SF100's exceptional 8.98 GB/s comes from zero fixup
(27/66 varying fits in 32B compact prefix).

### Thread scaling on taxi 6mo (meaningless)

All thread configs (1, 4, 8, 24) produced ~17.3-17.7s because the
single-chunk hybrid fallback uses CPU std::sort, not the parallel fixup
thread pool. Thread scaling is only meaningful on multi-chunk datasets.
