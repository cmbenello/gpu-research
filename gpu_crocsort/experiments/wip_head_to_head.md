# Head-to-head benchmarks — runtime-compact-map-wip (position-order, correctness-verified)

Hardware: Quadro RTX 6000 (Turing sm_75, 25.4 GB HBM, 672 GB/s), PCIe Gen3, 48-core Xeon, 192 GB RAM. Built via `make` (auto-detects sm_75).

## ⚠ Retraction

An earlier version of this file claimed entropy-based byte selection gave SF50 4.9× DuckDB and -25% vs position. **That claim was against BROKEN sort output** — entropy mode produces records in wrong order (see `results/overnight_2026-04-15/SUMMARY.md` for the counter-example). Position-order is the correct default; entropy remains as `COMPACT_SELECT=entropy` opt-in for the paper's negative-result section only.

## TPC-H lineitem ORDER BY (l_shipdate, l_orderkey, l_linenumber) — position-order default

| Scale | Records | DuckDB v1.5.2 | GPU CrocSort (position) | Speedup | sortedness | multiset |
|------:|--------:|--------------:|------------------------:|--------:|:----------:|:--------:|
| SF10  | 60 M    | 8.03 s        | **1.74 s**              | 4.6×    | PASS       | PASS     |
| SF50  | 300 M   | 56.7 s        | **11.9 s** (±0.08)      | 4.8×    | PASS       | PASS     |
| SF100 | 600 M   | ~200 s proj   | **7.98 s** (±0.07)      | ~25×    | PASS       | PASS     |

SF50 warm median and stdev from the pre-experiment baseline capture (5 runs, verified). SF100 likewise from baseline. Numbers match the `ff244d8`-era commit baseline before any overnight experiments.

## GenSort (JouleSort format, 100 B records, 10 B key)

| Size  | Records | GPU CrocSort | Throughput | Verifier |
|-------|---------|--------------|------------|----------|
| 10 GB | 100 M   | **1.63 s**   | 6.1 GB/s   | valsort OK + verify_full OK |
| 30 GB | 300 M   | **3.87 s**   | 7.7 GB/s   | valsort OK + verify_full in-memory OK |

GenSort uses the full-key strided-DMA path — no compaction involved.

## Verification

Every row in the tables above passes BOTH the parallel sortedness scan AND the parallel multiset-hash preservation (FNV-1a-64 sum per record, order-independent). The multiset-hash check is what caught the entropy correctness bug during the overnight session.

`tools/verify_full.cpp` is the standalone external verifier; the same two checks run in-memory after every `--verify` sort (default on).

## Reproduce

```bash
cd gpu_crocsort && make external-sort-tpch-compact
./external_sort_tpch_compact --input /tmp/lineitem_sf100_normalized.bin --runs 5
```
