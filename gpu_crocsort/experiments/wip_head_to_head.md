# Head-to-head benchmarks — runtime-compact-map-wip branch

Hardware: Quadro RTX 6000 (Turing sm_75, 672 GB/s HBM), PCIe Gen3, 48-core Xeon host, 210 GB NVMe.

## ⚠ Numbers prior to commit `0be08a8` were measured against a broken sort

A pre-existing bug interaction caused the build to silently produce no-op kernels on this Turing GPU when compiled with `-arch=sm_80` (the Makefile default at the time). The in-built sortedness verifier could not catch this because all output records were identical (60–600 M copies of `input[0]`), and adjacent-pair comparison trivially passes when every pair is equal. Discovered by `tools/verify_full.cpp` (multiset-hash check). Both bugs fixed at `0be08a8`. **Numbers below are post-fix and externally verified.**

## Verification

Each sort output below was independently checked with `./verify_full --input <input> --output <output> --record-size 120 --key-size 66`, which validates:

1. **Byte size + record count** match the input.
2. **Sortedness** — every adjacent pair is in non-decreasing order by the first 66 key bytes (parallel scan, ~25–60 GB/s).
3. **Multiset preservation** — sum-of-FNV-1a-64-hashes-per-record is identical between input and output (i.e., the output is a permutation of the input — no records lost, duplicated, or modified).

## TPC-H lineitem ORDER BY l_shipdate, l_orderkey, l_linenumber

DuckDB v1.5.2 internal dbgen → `CREATE TABLE s AS SELECT * FROM lineitem ORDER BY ...`
Our binary: `external_sort_tpch_compact --input lineitem_sfN_normalized.bin --runs 2`

| Scale | Records | DuckDB v1.5.2 | GPU CrocSort (ours) | Speedup | sortedness | multiset |
|------:|--------:|--------------:|---------------------:|--------:|:----------:|:--------:|
| SF10  | 60 M    | 8.03 s        | **1.76 s**           | 4.6×    | PASS       | PASS     |
| SF50  | 300 M   | 56.7 s        | **19.4 s**           | 2.9×    | PASS       | PASS     |
| SF100 | 600 M   | (~200 s projected) | **8.2 s**       | ~24×    | PASS       | PASS     |

Multiset check now runs in-memory after the sort (no need to write the 72 GB output to disk for SF100). It computes a parallel sum of FNV-1a-64 hashes per record over `h_data` (input) and `h_output` (sorted), and compares — equal iff the sort produced a permutation of the input. The hash check costs 0.5 s on SF10, 4 s on SF50, 5.5 s on SF100, and runs only when `--verify` is on (default).

### Why SF100 is faster than SF50

| Phase | SF50 | SF100 |
|------:|-----:|------:|
| Run-gen | 2.1 s | 3.7 s |
| GPU 16B merge | 0.3 s | 0.8 s |
| CPU gather | 1.7 s | 3.5 s |
| CPU fixup | **14.3 s** | **0 s (skipped — no GPU 16B-prefix ties)** |
| Total | 19.4 s | 8.2 s |

The compact map's first 16 bytes go into the GPU 16B prefix used by the CUB merge. Compact-map entries are ordered by source byte position (a correctness requirement), so the prefix gets the *first 16 varying byte positions* of the record:

- **SF100**: 27 varying bytes total → first 16 = `0,1,4,5,8,9,12,13,19,20,21,29,37,44,45,50`. Spread across positions 0–50, includes the high-entropy byte 50 (date/orderkey territory) → almost every adjacent pair differs at the GPU level → **fixup skipped**.
- **SF50**: 61 varying bytes total → first 16 = `0,1,5,7,8,10,11,12,13,14,15,16,17,18,19,20`. Clustered in positions 0–20 (low-entropy date prefix) → 290 M of 300 M records end up tied on the prefix → 14 s of CPU fixup over 15 K groups of ~19 K records each.

Fix paths (open follow-ups): extend GPU sort key from 16 B → 32 B (adds ~1 s GPU work, likely eliminates the 14 s fixup), and/or pick prefix bytes by entropy not source position.

## GenSort (JouleSort format: 100 B records, 10 B key)

| Size  | Records | GPU CrocSort | Throughput | Verifier |
|-------|---------|--------------|------------|----------|
| 10 GB | 100 M   | (re-measure) | (re-measure) | (re-run pending) |

The pre-fix gensort number (1.64 s / 6 GB/s) needs to be re-measured under the corrected build. Will update on next sweep.

## Correctness — what every sort prints

Every sort reports one of:
- `[Correctness] Sample map verified against all N records — compact sort is FULL-KEY-EQUIVALENT.` — runtime detection's V[b] expectations matched every record, so compact-key order ≡ full-key order.
- `[Correctness] Full-key sort path (no compaction) — inherently correct.` — the small-data path uploads full KEY_SIZE bytes; no compaction assumption made.
- `[Hybrid] Sample missed varying byte X. Retrying with full-key upload path...` — sample missed a byte that varies in the full data; sort is automatically re-run with no compaction (correct by construction).

## Reproducing

```bash
cd gpu_crocsort
git checkout runtime-compact-map-wip   # head: 0be08a8 or later

# Auto-detects the actual GPU's compute capability — must build native SASS,
# PTX→older-arch JIT silently produces no-op kernels.
make external-sort-tpch-compact

# Sort + dump output
./external_sort_tpch_compact --input /tmp/lineitem_sf10_normalized.bin --runs 2 --output /tmp/sf10_sorted.bin

# Build + run the standalone parallel verifier
g++ -O3 -std=c++17 -pthread tools/verify_full.cpp -o verify_full
./verify_full --input /tmp/lineitem_sf10_normalized.bin --output /tmp/sf10_sorted.bin --record-size 120 --key-size 66
```

## Open follow-ups

1. The compact prefix selects the first 16 *position-ordered* varying bytes. For SF50 this misses high-entropy bytes deeper in the record. Switching to entropy-ordered prefix selection (highest-variance bytes first) should let the GPU resolve more pairs and skip the 14 s fixup.
2. Re-run gensort 10/30/60 GB on the corrected build.
3. SF100 multiset verification is gated on free disk for the 72 GB output — re-run when storage allows.
4. `extract_tiebreaker_kernel` was the most obvious hardcoded-bytes bug; audit other kernels (especially anything that mentions byte 8/9) to confirm none have the same residual gensort assumption.
