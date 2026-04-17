# DuckDB Baseline Comparisons

**DuckDB mode:** SF10 = COPY TO Parquet; SF50 = CREATE TABLE AS SELECT ... ORDER BY
**DuckDB version:** v1.2.2 (via Python duckdb package)
**GPU binary:** `external_sort_tpch_compact` on branch `exp/fixup-fast-comparator`
**CPU:** 2x Xeon Gold 6248 (40 cores / 80 threads), 192 GB DDR4
**GPU:** Quadro RTX 6000 (24 GB HBM, PCIe 3.0 x16)

## SF10: 59,986,052 records, 7.20 GB

| System          | Input       | Median (ms) | Min    | Max    |
|-----------------|-------------|-------------|--------|--------|
| DuckDB (Parquet)| original    | 23,553      | 20,346 | 29,550 |
| DuckDB (Parquet)| sorted      | 20,685      | 20,675 | 21,515 |
| DuckDB (Parquet)| reverse     | 20,032*     | —      | —      |
| GPU sort        | original    | 1,808       | 1,704  | 1,814  |
| GPU sort        | sorted      | 1,660       | 1,650  | 1,693  |
| GPU sort        | reverse     | 1,689       | 1,656  | 1,734  |

*DuckDB reverse from separate clean run.

### Speedup (SF10)

| Input     | DuckDB  | GPU    | Speedup |
|-----------|---------|--------|---------|
| original  | 23,553  | 1,808  | 13.0x   |
| sorted    | 20,685  | 1,660  | 12.5x   |
| reverse   | 20,032  | 1,689  | 11.9x   |

DuckDB benefits from sorted input (~12% faster) while GPU sort is
input-oblivious. The GPU advantage is 12-13x across all distributions.

## SF50: 300,005,811 records, 36.0 GB

| System              | Median (ms) | Min    | Max    |
|---------------------|-------------|--------|--------|
| DuckDB (CTAS)       | 66,611      | 63,365 | 67,705 |
| GPU sort (compact)  | 6,585       | 6,429  | 6,764  |
| GPU sort (no-compact)| 4,169      | 4,153  | 4,180  |

### Speedup (SF50)

| GPU config   | DuckDB  | GPU     | Speedup |
|--------------|---------|---------|---------|
| compact T=48 | 66,611  | 6,585   | 10.1x   |
| compact T=32 | 66,611  | 6,949   | 9.6x    |
| no-compact   | 66,611  | 4,169   | 16.0x   |

DuckDB benchmark: `CREATE TABLE sorted AS SELECT * FROM lineitem ORDER BY ...`
using native TPC-H extension (`CALL dbgen(sf=50)`), 13 sort columns,
database on /dev/shm, temp on /dev/shm. GPU compact from fixup thread
scaling sweep; no-compact from compact ablation experiment.

## SF100: ~600M records, 68.4 GB

| System          | Median (ms) | Source                    |
|-----------------|-------------|---------------------------|
| DuckDB (Parquet)| ~200,000*   | estimated (>3 min)        |
| GPU sort        | 7,820       | measured (6 runs avg)     |

*SF100 DuckDB estimated. GPU sort from `results/2026-04-16-adversarial-sf100/`.
SF100 compact key covers all 27/66 varying bytes — no fixup needed.

## DuckDB stream mode (informational)

For SF10, DuckDB `fetchmany(100000)` streaming mode measured 58,780ms
(median of 3), 2.5x slower than Parquet mode. Python per-batch call overhead
dominates. Parquet mode is the fairer comparison: both systems produce
materialized sorted output.

## Notes

- **SF10:** DuckDB uses FixedSizeBinary(66) key + FixedSizeBinary(54) payload
  registered as Arrow table, sorted via COPY TO Parquet. Arrow table build
  time (3.7-4.3s) excluded from sort time.
- **SF50:** DuckDB uses native TPC-H extension (CALL dbgen), sorted via
  CREATE TABLE AS SELECT ... ORDER BY (13 columns). This is a fair
  comparison: both DuckDB and GPU sort produce materialized sorted output.
  Database and temp files on /dev/shm (tmpfs).
- DuckDB numbers vary with system load — cleanest numbers collected
  with no competing processes
- SF50 GPU sort (compact) uses 61/66 varying bytes with parallel fixup;
  no-compact uses full 66B key with zero fixup
