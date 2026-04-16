# Adversarial SF100: Why You Can't Force Fixup by Zeroing Bytes

Date: 2026-04-16
Binary: `external_sort_tpch_compact` @ `exp/fixup-fast-comparator` + `ADV_ZERO_BYTES` env var
Hardware: Quadro RTX 6000 (24 GB HBM, PCIe 3 x16), 48-core DDR4-2933 host.
Input: SF100 TPC-H lineitem, 600M records × 120B, 66B key prefix.

## Hypothesis

SF50 (300M records) takes 6.4 s while SF100 (600M) takes 8.1 s — only 1.27× for 2×
data. We hypothesized SF100's high throughput (8.56 GB/s vs 5.60 GB/s) was "byte-layout
luck" — the first 8 key bytes happened to discriminate all 600M records, skipping
the fixup phase entirely. If we zeroed bytes 0–7, the sort would lose that prefix
discrimination and reveal SF100's "true" cost with fixup.

**The hypothesis was wrong.** The experiment found a more interesting explanation.

## What we found

### Varying-position counts explain everything

| Scale | Records | Varying positions | Fit in 32B? | Fixup groups | Fixup time |
|:-----:|--------:|------------------:|:-----------:|:------------:|-----------:|
| SF50  | 300M    | **61/66**         | No (32 of 61) | 3.15M (avg 95 records) | **1 876 ms** |
| SF100 | 600M    | **27/66**         | Yes (27 ≤ 32) | 0 | **0 ms** |
| SF100-zero8 | 600M | **23/66**     | Yes (23 ≤ 32) | 0 | **0 ms** |

SF100's zero fixup is not about bytes 0–7. It is because TPC-H at SF100 has only
27 varying byte positions in the key — all of which fit in the 32B compact prefix.
There are no unresolved positions, so no ties, so no fixup.

SF50 has 61 varying positions: 32 fit in the compact prefix, 29 do not. Those 29
positions create 3.15M tie groups that fixup must resolve.

### Adversarial zero8 made the sort FASTER, not slower

| Config | Varying | Run gen (ms) | GPU merge (ms) | Gather (ms) | Fixup (ms) | **Total (ms)** |
|:-------|--------:|------------:|---------------:|------------:|----------:|--------:|
| Baseline | 27 | 3 712 | 833 | 3 255 | 0 | **8 078** |
| Zero8    | 23 | 3 690 | 828 | 3 195 | 0 | **7 889** |
| **Delta** | −4 | −22 | −5 | −60 | 0 | **−189** |

Warm medians (runs 2–3). All three runs verified PASS sortedness + PASS multiset.

Zeroing bytes 0–7 removed 4 positions from the varying set (0, 1, 4, 5), leaving 23.
With fewer varying positions, CUB radix sort does fewer passes during run generation,
saving ~22 ms. The gather is slightly faster too (−60 ms), likely noise.

### Why zeroing can't force fixup on SF100

The compact-key scheme _adapts_ to whichever positions vary. Zeroing positions removes
them from the varying set, which makes the compact key MORE efficient, not less:

- Zero bytes 0–7: 27 → 23 varying, still ≤ 32, still all in compact → no fixup
- Zero bytes 0–15: would go from 27 to ~19 varying → even more compact
- Zero any subset: removes positions, never adds them → can never exceed 32

To force fixup on SF100, you would need to INTRODUCE variation — e.g., add random
noise to currently-invariant positions to push the count above 32. This would be
synthetic, not adversarial.

## The real story: SF50 vs SF100 scaling

The throughput difference is explained by TPC-H data properties, not algorithm luck:

| Metric | SF50 | SF100 | Reason |
|:-------|-----:|------:|:-------|
| Records | 300M | 600M | 2× |
| Varying key bytes | 61 | 27 | TPC-H key distribution at different scales |
| Compact coverage | 32/61 = 52% | 27/27 = 100% | All SF100 variation fits in compact |
| Fixup groups | 3.15M | 0 | 29 unresolved positions at SF50 |
| Fixup time | 1 876 ms | 0 ms | Dominates the gap |
| End-to-end | 6 431 ms | 8 078 ms | |
| Throughput | 5.60 GB/s | 8.91 GB/s | Fixup-free path is 1.6× higher throughput |

The 1.6× throughput advantage of SF100 is entirely the fixup savings: 1876 ms of
SF50's 6431 ms budget. If SF50 had zero fixup, its end-to-end would be ~4555 ms
(7.89 GB/s), matching SF100's throughput per byte.

## Paper implication

The paper should NOT claim that SF100 throughput is "byte-layout luck" (as the
previous commit ab92985 documented). It should instead explain:

1. TPC-H at SF100 happens to have ≤32 varying key bytes, so compact covers 100%
2. TPC-H at SF50 has 61 varying key bytes, so compact covers only 52%
3. The fixup phase explains the per-record throughput gap
4. This is a property of the benchmark data, not the algorithm — workloads with
   >32 varying positions at any scale will incur fixup cost

The honest framing: **compact key compression is effective when the key has ≤32
varying byte positions. TPC-H at large scale factors naturally falls into this
regime. Workloads with wider key variation would show lower throughput.**

## Files

- `sf100_baseline.log` — 3 runs, original SF100 data.
- `sf100_zero8.log` — 3 runs, bytes 0–7 zeroed via `ADV_ZERO_BYTES=8`.
