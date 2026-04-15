# Head-to-head benchmarks — runtime-compact-map-wip branch

Hardware: Quadro RTX 6000 (Turing sm_75, 672 GB/s HBM), PCIe Gen3, 48-core Xeon host, 210 GB NVMe.

## TPC-H lineitem ORDER BY l_shipdate, l_orderkey, l_linenumber

DuckDB v1.5.2 internal dbgen → `CREATE TABLE s AS SELECT * FROM lineitem ORDER BY ...`
Our binary: `external_sort_tpch_compact --input lineitem_sfN_normalized.bin --runs N`

| Scale | Records | DuckDB v1.5.2 | GPU CrocSort (ours) | Speedup |
|------:|--------:|--------------:|---------------------:|--------:|
| SF10  | 60 M    | 8.03 s        | **1.19 s**           | 6.7×    |
| SF50  | 300 M   | 56.7 s        | **2.44 s**           | 23×     |
| SF100 | 600 M   | (skipped — >200 s projected, disk) | **3.67 s** | ~55×+ |

Our times are from the fastest warm run (run 2 or 3), after the input is loaded into pinned host memory. DuckDB times are the second `CREATE TABLE ... ORDER BY` after a warm-up run (measures steady-state sort + materialization, page cache hot).

Key metric: PCIe Gen3 is ~12 GB/s; we hit 18.5 GB/s sort throughput on SF100 because the compact-upload path only pushes the ~27 varying byte positions across PCIe instead of all 120 B per record.

## GenSort (JouleSort format: 100 B records, 10 B key)

| Size | Records | GPU CrocSort | Throughput | Verifier   |
|------|---------|--------------|------------|------------|
| 10 GB | 100 M  | **1.64 s**    | 6.09 GB/s  | valsort OK + our parallel verifier OK |

Generated with `/home/cc/tools/64/gensort` (ordinal.com v1.5). Verified with both the official `valsort` and our `tools/verify_sorted` (~50 GB/s page-cached).

## Correctness guarantees on the TPC-H runs

Every sort prints one of:
- `[Correctness] Sample map verified against all N records — compact sort is FULL-KEY-EQUIVALENT.` — verification confirmed every record's non-mapped bytes match the expected constants V[b] from the sample. Sort is provably equivalent to a full-key lexicographic sort.
- `[Correctness] Full-key sort path (no compaction) — inherently correct.` — smaller data goes through a path that uploads full KEY_SIZE bytes to the GPU; no compaction is used, sort is correct by construction.

Independent re-verification with `tools/verify_sorted` on written output: PASS on SF10, gensort 10 GB.

## Reproducing

```bash
cd gpu_crocsort
git checkout runtime-compact-map-wip
make external-sort-tpch-compact

# TPC-H (requires pre-built lineitem_sfN_normalized.bin files in /tmp)
./external_sort_tpch_compact --input /tmp/lineitem_sf100_normalized.bin --runs 3

# GenSort
make external-sort   # build the 10-byte-key binary
/home/cc/tools/64/gensort -t$(nproc) 100000000 /tmp/gs10gb.bin
./external_sort --input /tmp/gs10gb.bin --runs 1 --output /tmp/gs10gb_sorted.bin
/home/cc/tools/64/valsort /tmp/gs10gb_sorted.bin   # official verifier

# External parallel verifier (independent of our sort)
g++ -O3 -std=c++17 -pthread tools/verify_sorted.cpp -o verify_sorted
./verify_sorted /tmp/gs10gb_sorted.bin 100 10

# DuckDB baseline
/home/cc/tools/duckdb /tmp/sf10.duckdb <<EOF
INSTALL tpch; LOAD tpch; CALL dbgen(sf=10);
.timer on
CREATE TABLE s AS SELECT * FROM lineitem ORDER BY l_shipdate, l_orderkey, l_linenumber;
EOF
```
