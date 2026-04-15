# Head-to-head benchmarks — runtime-compact-map-wip + entropy default

Hardware: Quadro RTX 6000 (Turing sm_75, 25.4 GB HBM, 672 GB/s), PCIe Gen3, 48-core Xeon, 192 GB RAM. Built via `make ARCH=sm_75` (auto-detected).

All sorts independently verified by in-memory parallel sortedness scan + multiset-hash preservation (FNV-1a-64 sum per record). Every number in the table has BOTH checks PASS.

## TPC-H lineitem ORDER BY (l_shipdate, l_orderkey, l_linenumber)

Same-session alternating 5-run sweep (`position entropy position entropy ...`), warm median reported. Format: `ours (sort-only)` — verify time is additional (~0.6 s SF10, ~2.7 s SF50, ~4.8 s SF100).

| Scale | Records | DuckDB v1.5.2 | GPU CrocSort position-order | **GPU CrocSort entropy (DEFAULT)** | Speedup vs DuckDB |
|------:|--------:|--------------:|----------------------------:|-----------------------------------:|------------------:|
| SF10  | 60 M    | 8.03 s        | 1.74 s                       | **1.74 s**                         | 4.6×              |
| SF50  | 300 M   | 56.7 s        | 15.48 s (stdev ±1.1 s)       | **11.57 s** (stdev ±0.6 s)         | **4.9×**          |
| SF100 | 600 M   | (~200 s proj) | 8.43 s                       | **8.52 s** (within noise)          | ~24×              |

### Why entropy changed SF50 so much

Runtime byte-position detection originally placed the first 32 *position-ordered* varying bytes into the GPU compact key. For SF50 that's bytes 0–36 (date prefix + header) — l_orderkey at bytes 51–57 stays outside the prefix. Every adjacent pair sharing that span ends up in tied 16B-prefix groups → 14 s of CPU fixup.

Entropy mode tracks per-byte sample distinct-value count (256-bit bitmap, ~5 ms), ranks candidates descending, and picks the top 32 by distinct count (sorted by source position within the selected set to preserve lex-compatibility). For SF50 entropy lands bytes `8, 11, 36–65` in the compact key — now the prefix discriminates on orderkey territory → fewer pairs survive to CPU fixup.

SF10 uses the full-key no-compaction path and doesn't trigger the entropy branch (ncand ≤ 32). SF100's 27 candidates also don't trigger it. Only SF50-like datasets with > 32 varying bytes are affected.

## GenSort (JouleSort format, 100 B records, 10 B key)

| Size  | Records | GPU CrocSort | Throughput | Verifier |
|-------|---------|--------------|------------|----------|
| 10 GB | 100 M   | **1.63 s**    | 6.1 GB/s   | valsort OK + verify_full OK |
| 30 GB | 300 M   | **3.87 s**    | 7.7 GB/s   | valsort OK + verify_full in-memory OK |

GenSort uses the strided-DMA full-key path (KEY_SIZE=10), no compaction — unaffected by entropy selection.

## Reproduce

```bash
cd gpu_crocsort
git checkout runtime-compact-map-wip     # production baseline
git checkout exp/entropy-selection       # entropy-default branch (SF50 -25%)
make external-sort-tpch-compact          # auto-detects ARCH

# Entropy is the default now; explicitly toggle with:
#   COMPACT_SELECT=position ./external_sort_tpch_compact ...  # old behavior
#   COMPACT_SELECT=entropy  ./external_sort_tpch_compact ...  # new default (redundant)

./external_sort_tpch_compact --input /tmp/lineitem_sf100_normalized.bin --runs 5
```

## Full overnight research log

`results/overnight_2026-04-15/SUMMARY.md` — top 3 wins, top 3 dead ends, footnotes.
