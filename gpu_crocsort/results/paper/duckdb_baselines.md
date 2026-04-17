# DuckDB Baseline Comparisons

**DuckDB mode:** COPY ... TO Parquet (UNCOMPRESSED) — sort + materialized output
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

## SF50: ~300M records, 34.2 GB

| System          | Median (ms) | Source       |
|-----------------|-------------|--------------|
| DuckDB (Parquet)| ~57,000*    | extrapolated |
| GPU sort        | 6,210       | measured     |

*SF50 DuckDB not run in clean conditions due to memory competition.
GPU sort from `results/2026-04-15-arch-analysis/`.

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

- DuckDB uses FixedSizeBinary(66) key + FixedSizeBinary(54) payload
  registered as Arrow table, sorted via ORDER BY key
- DuckDB numbers vary with system load — cleanest SF10 numbers collected
  with no competing processes on /dev/shm
- Arrow table build time (3.7-4.3s) excluded from DuckDB sort time
  (COPY timing starts after table is registered)
