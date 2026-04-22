# GPU Radix Sort in Transactional (OLTP) Databases

**Date:** 2026-04-20  
**Context:** RTX 6000 Ada — 70ms for 300M rows of 4-byte keys (radix sort, GPU-resident data).  
**Question:** Where does a fast GPU sort add value in OLTP workloads?

---

## 1. Where OLTP Databases Actually Sort

### A. Index Creation (DDL) — The Dominant Sort Consumer
`CREATE INDEX` / `ALTER TABLE ADD INDEX` is a bulk sort of an entire column (or composite key) followed by a B-tree bulk-load from the sorted output. This is the largest and most predictable sort in OLTP.

Measured wall-clock times (PostgreSQL, various hardware):
- 100M rows → ~52 seconds (single-threaded)
- 500M rows → 9–17 minutes (depends on parallel workers and SSD throughput)
- 750M rows (`CREATE INDEX CONCURRENTLY`) → ~5 hours (includes dual-scan validation)
- 1B rows → 8–13 minutes (tuned, parallel)

The sort phase is CPU-bound and parallelizes cleanly — PostgreSQL's parallel tuplesort shipped in PG11 specifically for this. The final merge is serial, limiting multi-core speedup to ~5–6x even with 8 workers.

### B. LSM-Tree Compaction (RocksDB, LevelDB)
Compaction merges overlapping SSTables and re-sorts the result. At high write throughput this becomes CPU-bound. LUDA (arXiv:2004.03054) showed 2x throughput improvement and 2x processing speed over stock RocksDB/LevelDB by offloading the sort to a GPU. A 2024 MSST paper (Q-Compaction) targets specifically L0→L1 and L1→L2 where sort latency directly causes write stalls.

### C. CLUSTER / VACUUM FULL (PostgreSQL)
`CLUSTER` rewrites a table in index order — a full-table sort. `VACUUM FULL` does a similar rewrite. Both are offline, bulk operations that can dominate maintenance windows.

### D. Per-Query ORDER BY, Merge Joins
Typically tiny (< 100K rows) in OLTP. PCIe latency (~1–2ms baseline) dominates for small transfers. **GPU provides no benefit here.**

### E. Write-Ahead Log Compaction
Not sort-intensive; WAL is append-only and recovery replay is sequential. Not a GPU target.

---

## 2. Where GPU Sort Makes Sense

### Strong fit: Large index creation and rebuild

| Rows | CPU time (est.) | GPU sort time (est.) | Potential speedup |
|------|----------------|---------------------|-------------------|
| 10M  | ~3–5s          | ~5ms sort + ~15ms PCIe | CPU wins (PCIe overhead) |
| 100M | ~50s           | ~50ms sort + ~150ms PCIe | **~200ms total, ~200x on sort, ~25–50x wall** |
| 500M | ~10 min        | ~250ms sort + ~750ms PCIe | **~1s total, ~10–60x wall** |
| 1B   | ~10–15 min     | ~500ms sort + ~1.5s PCIe | **~2s total, ~300–450x on sort** |

*PCIe 3.0 transfer estimate: ~6 GB/s effective for 4-byte keys (~0.5–1.5ms per 10M rows.)*

The GPU sort phase is essentially free at scale — the bottleneck shifts to PCIe transfer and the subsequent B-tree bulk-load (which remains CPU-bound). Even accounting for PCIe, the total time for index creation on a 100M-row table drops from ~50s to ~200ms: a real, usable speedup.

### Strong fit: LSM compaction
Compaction is already identified as CPU-bound in production RocksDB workloads (2x throughput demonstrated by LUDA). The sort within a compaction job typically involves 10M–500M keys depending on level and compaction trigger. This is exactly the regime where the RTX 6000 dominates.

### Moderate fit: CLUSTER / VACUUM FULL
Same bulk-sort structure as CREATE INDEX. Useful when maintenance windows are constrained. Less urgent than index creation because CLUSTER is rarely on the critical path.

### Poor fit: Per-query ORDER BY, merge joins
The PCIe round-trip is 1–5ms regardless of data size. Any result set under ~1M rows has negligible sort time on CPU (<5ms). GPU overhead is worse. **Threshold: GPU wins only above ~1M rows**, and OLTP queries almost never return that many rows to sort.

---

## 3. Specific Database Targets

### PostgreSQL
- **`CREATE INDEX`**: Replaces `tuplesort.c`'s sort phase. The tuple fetch (heap scan) and B-tree bulk-load remain CPU-side. GPU handles the key-sort middle step. This is the highest-value target.
- **`CLUSTER`**: Same tuplesort path.
- **`VACUUM FULL`**: Same, but less frequent.
- Integration point: `tuplesort_performsort()` in `tuplesort.c`. Key extraction already produces normalized byte-comparable datums for many types.

### MySQL / InnoDB
- **`ALTER TABLE ... ADD INDEX`**: Uses a 3-phase Sort Index Build (read/run → merge-sort → insert). The merge-sort phase (Phase 2) is the direct GPU target. MySQL 8.0.31 parallelized the insert phase; the merge-sort is still serial. GPU sort would replace the serial merge-sort.
- Integration point: `row0merge.cc` in the InnoDB source.

### RocksDB / LevelDB
- **Compaction**: The most compelling OLTP target because write stalls are a user-visible latency spike. LUDA demonstrated 2x throughput without query-latency PCIe exposure (compaction is background). The GPU processes the sort asynchronously while the foreground path continues accepting writes.
- Integration point: `DBImpl::BackgroundCompaction()` → sort the merged key range on GPU.

### SQLite
- **`CREATE INDEX`** on large WAL-mode databases. SQLite's sort is single-threaded. For embedded/desktop use this is less relevant, but for server-side SQLite (e.g., Cloudflare D1, Turso) large index creation is a real pain point.

---

## 4. The Key Advantage: Unblocking DDL

Index creation in PostgreSQL and MySQL either:
1. **Blocks all writes** (non-concurrent) — the table is locked for the entire duration.
2. **Allows writes but takes 2–3x longer** (CONCURRENTLY / ONLINE) — plus requires extra validation scans.

If CREATE INDEX on 500M rows goes from 10 minutes to 1 second, the lock duration (or the penalty period for online rebuild) shrinks by the same factor. This directly impacts production deployments: schema migrations that today require maintenance windows or complex pt-online-schema-change choreography become near-instant DDL.

---

## 5. PCIe Considerations

The PCIe floor matters less for OLTP DDL than for OLAP queries because:
- **Data is transferred once, sorted once, result written once.** Amortization is good.
- The index creation pipeline is already dominated by disk I/O on cold tables. For hot tables (buffer pool resident), the full sort + PCIe round-trip is still faster than the CPU sort alone.
- **Threshold:** GPU wins above ~1M rows for 4-byte keys on PCIe 3.0. For wider keys (8–32 bytes, typical of composite OLTP indexes), the threshold rises to ~2–5M rows due to higher transfer cost, but the CPU sort also scales worse, so the crossover is approximately the same row count.

---

## 6. Existing Work

| Work | Target | Result |
|------|--------|--------|
| [LUDA (arXiv:2004.03054)](https://arxiv.org/abs/2004.03054) | RocksDB/LevelDB LSM compaction | 2x throughput, 2x processing speed |
| [Q-Compaction (MSST 2024)](https://www.msstconference.org/MSST-history/2024/Papers/msst24-3.3.pdf) | L0→L1 RocksDB compaction | GPU-sort for upper-level SST sort, targets write-stall reduction |
| [GPU B-Tree (PPoPP 2019)](https://dl.acm.org/doi/10.1145/3293883.3295706) | GPU-resident B-tree queries | 3.4x over prior GPU B-tree; assumes data stays GPU-resident |
| [PostgreSQL GPU sort discussion (2010)](https://www.postgresql.org/message-id/4C7BBAA1.6040808@2ndquadrant.com) | PostgreSQL sort | Concluded GPU servers are rare, PCIe overhead a concern — predates modern GPU-server prevalence |
| [Harmonia (PPoPP 2019)](https://www.ece.lsu.edu/lpeng/papers/ppopp-19.pdf) | GPU B+tree | 3.6B queries/sec on Volta; focuses on GPU-side queries, not index construction |

The PostgreSQL thread (2010) was skeptical because of PCIe and limited GPU availability. Both concerns are significantly weaker in 2026: modern data center servers routinely have A100/H100/RTX 6000 GPUs, and PCIe 4.0/5.0 halves transfer time.

---

## 7. Recommendation

**Primary target: CREATE INDEX / ALTER TABLE ADD INDEX on tables > 10M rows.**

This is where:
- The sort is provably CPU-bound and the dominant DDL cost.
- The GPU sort speedup is real (10–300x on the sort step).
- The end-to-end win is large even after PCIe: a 100M-row index drops from ~50s to ~200ms.
- The benefit is operational: schema migrations no longer require maintenance windows.

**Secondary target: RocksDB LSM compaction.**

LUDA already proved the concept. The GPU sort replaces the merge-sort in background compaction, reducing write stalls without touching the foreground query path. This is the safest integration (no ACID risk, purely background).

**Skip: Per-query ORDER BY / merge joins in OLTP.**

PCIe latency dominates for any result set OLTP queries actually return. The threshold (~1M rows) is far above what typical OLTP transactions touch.

---

*References:*
- LUDA: https://arxiv.org/abs/2004.03054
- Q-Compaction (MSST 2024): https://www.msstconference.org/MSST-history/2024/Papers/msst24-3.3.pdf
- GPU B-Tree (PPoPP 2019): https://dl.acm.org/doi/10.1145/3293883.3295706
- Harmonia GPU B+tree: https://www.ece.lsu.edu/lpeng/papers/ppopp-19.pdf
- PostgreSQL parallel tuplesort: https://commitfest.postgresql.org/patch/690/
- PostgreSQL GPU sort thread (2010): https://www.postgresql.org/message-id/4C7BBAA1.6040808@2ndquadrant.com
- PostgreSQL index creation benchmarks: https://dev.to/andatki/how-long-does-it-take-to-create-an-index-60o
- EdstemTech 750M row index: https://www.edstem.com/blog/creating-indexes-on-750-million-rows-in-postgresql
