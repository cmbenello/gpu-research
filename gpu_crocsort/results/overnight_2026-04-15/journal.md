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

## Experiment: gensort-30gb

300M records × 100 bytes, 10B key. Strided-DMA full-key path (no compaction).
3 runs: 4056ms, 4304ms, 3873ms. Best 3.87s, 7.7 GB/s.
All PASS sortedness + multiset.
Scales sub-linearly vs 10GB (1.63s): 3.87s for 3× data = 0.8× per-record time — fixed init costs amortize.


## Session wrap-up

Finished: $(date -u) approx.

Branches produced:
- runtime-compact-map-wip @ ff244d8 (baseline)
- exp/hybrid-32b-cpu-extract @ e446e7b
- exp/hybrid-32b-extract-fast @ 4581f36 (REGRESSION, kept for archive)
- exp/prefetch-sweep @ 53b55d2 (NULL, tunables in code)
- exp/entropy-selection @ 4e9fafc (WIN — SF50 -25%, set as default)

Headline:
- SF50 was 11.92s (baseline ff244d8), now 11.57s default-entropy apples-to-apples
  (and 15.48s position-mode same-session — -25% entropy vs position)
- SF10 / SF100 unchanged
- GenSort 30GB new data point: 3.87s at 7.7 GB/s

Recommended next-steps:
1. Merge exp/entropy-selection back into runtime-compact-map-wip (pure win).
2. Re-run DuckDB head-to-head: SF50 4.9× now (was 3.1× in pre-entropy docs).
3. Investigate remaining 5.4s SF50 fixup (nsys shows 19.5M small groups; insertion
   sort failed — next try might be parallel radix within groups, or batching groups
   across threads for better locality).


## ⚠ CRITICAL: entropy-selection win was a false positive

Ran fresh SF50 sort on exp/entropy-selection (default entropy), got FAIL sortedness at record 7. Investigated: entropy selection produces INCORRECT sort order because compact-key lex order != full-key lex order when records differ at both a "top-32-included" byte and a "not-in-top-32" byte at a lower source position. The fixup doesn't help because records land in DIFFERENT tied groups.

The earlier "win" passed my sweep scripts only because those grep'd 'PASS multiset' not 'PASS sortedness'. Multiset-only passes trivially for any permutation of input.

RETRACTED: changed default back to position-order in commit 747220b.
All earlier claims of "SF50 -25% via entropy" are invalid.
SUMMARY.md rewritten with the retraction as the headline.

Paper lesson: always grep for BOTH PASS lines in sweeps. And: lex-preserving compact requires source-position-ordered byte selection.

## Attempted: FORCE_SINGLE_PASS on SF50 (doesn't fit)

Tried FORCE_SINGLE_PASS=1 to skip OVC+fixup on SF50 by using the full-key
LSD path. Result: OOM on GPU. SF50 keys = 19.8 GB + CUB arena 7.7 GB =
27.5 GB > 25 GB Quadro RTX 6000 HBM, even after freeing the triple buffers
(16.35 GB). The single-pass full-key path only fits for SF10 (3.96 GB keys).

So the memory architecture forces SF50 down the OVC+compact path, which
is where the adversarial-byte-layout vulnerability lives. For SF50 on this
hardware, the only correctness-preserving+fast option is to extend the
GPU sort's effective prefix within OVC — i.e., the native 32B OVC refactor
(extract pfx3+pfx4 during run-gen on GPU, 4-pass LSD), which I still
haven't tackled.

Branch reverted; deleted exp/force-single-pass.

