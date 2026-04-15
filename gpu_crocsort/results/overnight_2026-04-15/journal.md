# Overnight research journal — 2026-04-15
Started UTC: Wed Apr 15 05:54:21 UTC 2026
Baseline branch: runtime-compact-map-wip @ ff244d8


## Baseline (runtime-compact-map-wip @ ff244d8)

5 runs each, --verify on (sortedness + multiset), all PASS.

| SF | run1 (cold) | run2 | run3 | run4 | run5 | warm median | warm stdev |
|---:|------------:|-----:|-----:|-----:|-----:|------------:|-----------:|
| SF10 | 1730 | 1731 | 1780 | 1755 | 1783 | 1755 | 25 ms |
| SF50 | 17672 | 12042 | 11881 | 11885 | 11964 | 11924 | 76 ms |
| SF100| 11621 |  7952 |  8011 |  7891 |  8046 |  7981 | 67 ms |

Warm = excluding the cold first run after binary was rebuilt. Cold is dominated by CUDA context init + page cache warm-up.

Throughput (warm median, GB/s sort-only): SF10 4.10, SF50 3.02, SF100 9.00.

Headline numbers for tonight's deltas:
- SF10  baseline = **1.755 s**
- SF50  baseline = **11.92 s**
- SF100 baseline = **7.98 s**

## Experiment: prefetch-sweep (exp/prefetch-sweep)

Sweep gather prefetch distance 128..1536 on SF100. Suggestive: 256 → 3597 ms gather (vs 512 baseline 3839 ms, -6%). Variance per-run is high; need focused 5-run validation.


### prefetch-validate: NULL — 512 marginally better on median, within noise

## Experiment: threads-sweep (NULL — 48 already optimal)

8/16/24 = bandwidth-starved. 48 = best at 3679ms gather. 64 = oversubscribes (3719ms).


## Experiment: hybrid-32b-extract-fast (REGRESSION)

Optimized CPU extract (1 read/rec instead of 2) + GPU gather of pfx3 (instead of CPU re-extract). Extension overhead 9s → 6.6s. BUT: SF50 still regresses by 1.2s vs baseline because 32B compact doesn't include orderkey (bytes 51-57). Fixup only saves 0.9s.

Lesson: extending the prefix to more BYTES doesn't help when the discriminating bytes are outside the source-position-first-N window. Pivot to entropy-based byte SELECTION.


## Experiment: entropy-selection (WIN!)

Pick top 32 byte positions by sample distinct-value count (entropy proxy), put them in compact KEY (sorted by position). Remaining varying bytes go in map[32+] for verification. SF50: 12.16s → 10.55s (-1.6s, -13%) with fixup 7.05s → 5.39s. SF10/SF100 unaffected. Mark COMPACT_SELECT=entropy as default-recommended.

## Experiment: insertion-sort-small-groups (NULL — reverted)

Tried insertion sort on tied groups with count ≤ 32. Regressed fixup 5.4s → 8s. O(n²) × 20ns/cmp > O(n log n) × 20ns/cmp + std::sort overhead at n=15. Reverted.
