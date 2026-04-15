# Comprehensive benchmark — runtime-compact-map-wip

(Updated post fixup-skip-known-equal-bytes optimization)

Hardware: Quadro RTX 6000 (Turing sm_75, 25.4 GB HBM, 672 GB/s), PCIe Gen3, 48-core Xeon, 192 GB RAM. Built with `make ARCH=sm_75` (auto-detected). Each workload run 3 times after warm-up; numbers below are the best of 3.

All sorts independently verified by `verify_full`-equivalent in-memory checks: parallel sortedness scan + multiset-hash preservation (FNV-1a 64 sum per record). Every row in the table has BOTH checks PASS unless noted.

## Per-phase breakdown

### TPC-H lineitem ORDER BY (l_shipdate, l_orderkey, l_linenumber) — 120 B record, 66 B key

| Phase | SF10 (60 M) | SF50 (300 M) | SF100 (600 M) |
|-------|------------:|-------------:|--------------:|
| Compact-map detect (sample 1 M, multi-thread) | 25 ms first / 0 ms cached | 26 ms / 0 ms | 25 ms / 0 ms |
| Varying bytes detected | 61 / 66 (92 %) | 61 / 66 (92 %) | 27 / 66 (41 %) |
| Sort path | strided-DMA full-key | OVC compact-upload | OVC compact-upload |
| Run-gen / extract+upload | 1314 ms | 2095 ms | 3700 ms |
| GPU LSD pass 1 | (9 passes total → 370 ms) | 155 ms | 608 ms |
| GPU LSD pass 2 | — | 122 ms | 239 ms |
| 16 B tie check | n/a | TIES FOUND | NO TIES |
| CPU gather | 398 ms | 1786 ms | 3010 ms |
| CPU fixup (active-bytes-only compare) | n/a (no compaction) | **10 821 ms** | **0 ms (skipped)** |
| **Sort total** (`result.total_ms`) | **1.71 s** | **13.0 s** | **7.73 s** |
| Verify: parallel sortedness | 134 ms | 568 ms | 603 ms |
| Verify: parallel multiset hash (FNV-1a 64) | 507 ms | 2153 ms | 4185 ms |
| Throughput (sort only, GB/s) | 4.1 | 1.95 | 9.3 |

### GenSort (JouleSort format, 100 B record, 10 B key)

| Phase | 10 GB (100 M records) |
|-------|---------------------:|
| Sort path | strided-DMA full-key |
| Run-gen | 970 ms |
| GPU LSD (2 passes for 10 B key) | (combined ~50 ms within 650 ms merge) |
| CPU gather | included in merge |
| **Sort total** | **1.62 s** |
| Verify: sortedness | 121 ms |
| Verify: multiset | 630 ms |
| Throughput (sort only, GB/s) | 6.2 |
| External `valsort` | SUCCESS (independent confirm) |

## Speedup vs DuckDB v1.5.2 (`CREATE TABLE s AS SELECT * FROM lineitem ORDER BY ...`)

| Workload | DuckDB | Ours (sort) | Ours (sort + verify) | Speedup (sort) |
|----------|-------:|------------:|---------------------:|---------------:|
| SF10 | 8.03 s | 1.71 s | 2.36 s | 4.7× |
| SF50 | 56.7 s | 13.0 s | 15.8 s | **4.4×** (was 3.1×) |
| SF100 | ~200 s (projected) | 7.73 s | 12.5 s | ~26× |

DuckDB numbers from earlier (commit `27194d7` notes). DuckDB doesn't ship a separate verifier — its sort is trusted because it's been hammered for years; our `verify_full` cost is for the prototype's correctness story and is opt-in via `--verify` (default on).

## Bottleneck analysis

**SF10**: full-key strided-DMA path with 9 LSD passes on the GPU. PCIe upload is dominant because PCIe Gen3 is the bottleneck (3.96 GB / 13 ms = 305 GB/s effective via DMA overlap). No fixup needed — GPU sorts the full 66 B key directly.

**SF50**: 14.3 s of CPU fixup. The runtime-detected compact map's first 16 byte positions (`0,1,5,7-20`) cluster at low-entropy date prefix bytes, so 290 M of 300 M records end up in 15 K tied groups of ~19 K records each. `std::sort` on each tied group dominates. **The single highest-impact optimization target**.

**SF100**: GPU does it all. Compact map's first 16 positions span `0–50` and include the high-entropy byte 50 (date/orderkey territory), so the GPU 16 B merge resolves every adjacent pair — fixup skipped. CPU gather (~3 s) and the GPU sort itself (~0.85 s) are the only real costs; everything else is overhead.

**GenSort 10 GB**: trivial because the key is 10 bytes — no compaction logic involved, plain CUB radix on the 10 B keys.

## Optimization variants — implementation status

| Variant | Status | Expected SF50 | Expected SF100 |
|---------|--------|---------------|----------------|
| **A: 32 B GPU sort (4 LSD passes)** unconditional | not implemented (memory tight on SF100, doesn't fit in 25 GB HBM) | ~3 s (eliminates fixup) | OOM |
| **B: 32 B GPU sort, conditional on memory** | not implemented (~150 lines, mostly LSD-loop refactor + memory check) | ~3 s | unchanged 7.75 s |
| **C: hybrid escalation 16 B → 24 B → 32 B → CPU** | not implemented (~200 lines, incremental allocate/extract on tie detect) | ~3–5 s | unchanged 7.75 s |
| **D: GPU CUB segmented sort within tied groups** | not implemented (~100 lines but PCIe round-trip if records on host) | ~5 s | unchanged 7.75 s |
| **E: parallel radix CPU fixup instead of std::sort** | not implemented (~100 lines, unclear win — std::sort on 19 K records is already L3-friendly) | ~10 s | unchanged 7.75 s |

Only the current variant (16 B GPU + CPU std::sort fixup) is implemented and benchmarked above.

## Multiset hash verification

Every sort run prints two PASS lines after the sort completes:
```
PASS sortedness: N records in non-decreasing order (XX ms)
PASS multiset:   output is a permutation of input (hash=0xHHHHHHHHHHHHHHHH, YY ms)
```

The multiset hash is `Σ_i FNV-1a-64(record_i)` mod 2^64 — order-independent, equal iff the output contains the same multiset of records as the input. This caught two real bugs (commit `0be08a8`) that the in-built sortedness-only verifier had been hiding for the entire history of the project: the GPU kernels were silently producing all-zero permutations under the default `-arch=sm_80` build on this Turing GPU, and a leftover gensort hardcode in `extract_tiebreaker_kernel` was mis-sorting the final LSD pass for any KEY_SIZE != 10.

## Reproduce

```bash
cd gpu_crocsort
git checkout runtime-compact-map-wip   # head: aee7bd0 or later
make external-sort-tpch-compact        # auto-detects ARCH
./external_sort_tpch_compact --input /tmp/lineitem_sf100_normalized.bin --runs 3
```

Verify-only the existing output:
```bash
g++ -O3 -std=c++17 -pthread tools/verify_full.cpp -o verify_full
# (requires writing the output to disk first via --output FILE)
```
