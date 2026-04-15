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

| Scale | Records | DuckDB v1.5.2 | GPU CrocSort (ours) | Speedup | verify_full |
|------:|--------:|--------------:|---------------------:|--------:|:-----------:|
| SF10  | 60 M    | 8.03 s        | **1.76 s**           | 4.6×    | ALL PASS    |
| SF50  | 300 M   | 56.7 s        | **19.4 s**           | 2.9×    | ALL PASS    |
| SF100 | 600 M   | (~200 s projected) | **8.2 s**       | ~24×    | sortedness PASS, multiset not run (output too large for free disk) |

SF50 is dominated by CPU fixup (14 s out of 19 s) because the runtime-detected compact prefix only captures 16 of 61 varying bytes for the SF50 dataset, producing many ties on the GPU 16B prefix that need full-key resolution on CPU. SF100 has a more discriminating prefix (16 of 27 varying bytes including unique date/orderkey positions) so most pairs resolve at the GPU level — fixup is skipped.

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
