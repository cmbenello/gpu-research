#!/usr/bin/env python3
"""4.1 — DuckDB ORDER BY baseline at SF50/SF100/SF300.

Compares to gpu_crocsort warm best:
  SF50  : 1.51 s @ 23.9 GB/s  (post-0.3.1)
  SF100 : 3.02 s @ 23.8 GB/s  (post-0.3.1)
  SF300 : 8.65 s @ 25.0 GB/s  (warm best from 1.2)

DuckDB approach: read /mnt/data/lineitem_sf<N>.bin as 120 B normalized
records, decode the relevant key fields, ORDER BY them, time the
sort. Use duckdb.connect() with explicit threads/memory.

We can't directly load the binary 120 B record into duckdb because it's
a custom format. Two options:
  (A) Run tpchgen-cli to produce a parquet, then ORDER BY via duckdb.
  (B) Use duckdb's tpch extension to generate inline.

(B) is simplest. duckdb tpch.dbgen + ORDER BY l_returnflag, l_linestatus,
l_shipdate (the canonical sort key) gets us a comparable baseline.
"""
import duckdb, time, sys, os

def bench_sf(con, sf):
    print(f"\n=== SF{sf} ===")
    print(f"  Generating TPC-H data via tpch.dbgen ...")
    t0 = time.time()
    con.execute(f"CALL dbgen(sf={sf})")
    gen_s = time.time() - t0
    nrows = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
    print(f"  Generated {nrows:,} rows in {gen_s:.1f}s")

    # Warm-up: ORDER BY a single column first to warm the buffers
    con.execute("SELECT COUNT(*) FROM lineitem").fetchall()

    # 5 timed sorts, take warm best
    SQL = """
    CREATE OR REPLACE TABLE sorted AS
    SELECT * FROM lineitem
    ORDER BY l_returnflag, l_linestatus, l_shipdate, l_commitdate,
             l_receiptdate, l_extendedprice, l_discount, l_tax,
             l_quantity, l_orderkey, l_partkey, l_suppkey, l_linenumber
    """
    times = []
    for run in range(5):
        # Drop the sorted table between runs to make timings comparable
        con.execute("DROP TABLE IF EXISTS sorted")
        t0 = time.time()
        con.execute(SQL)
        ms = (time.time() - t0) * 1000.0
        times.append(ms)
        print(f"  Run {run+1}: {ms:.0f} ms")

    warm_best = min(times[1:]) if len(times) > 1 else times[0]
    # Approx GB/s: TPC-H lineitem ~ 6 M rows per SF × ~140 B ≈ 840 MB / SF
    # For comparison with our 120 B normalized: GB = sf * 6_001_215 * 120 / 1e9
    gb = sf * 6001215 * 120 / 1e9
    print(f"  Warm best: {warm_best:.0f} ms, throughput {gb / (warm_best/1000):.2f} GB/s "
          f"(120B-normalized comparable basis)")
    print(f"  CSV,duckdb_sort,sf={sf},rows={nrows},warm_best_ms={warm_best:.1f},gb_per_s={gb/(warm_best/1000):.2f}")

    con.execute("DROP TABLE IF EXISTS lineitem")
    con.execute("DROP TABLE IF EXISTS sorted")
    return warm_best

def main():
    print(f"DuckDB version: {duckdb.__version__}")
    # Use a disk-backed db on /mnt/data so SF300 doesn't blow out memory.
    db_path = "/mnt/data/duckdb_4.1.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute("SET threads=192")
    con.execute("SET memory_limit='800GB'")
    con.execute("PRAGMA enable_progress_bar=false")

    for sf in [int(x) for x in sys.argv[1:] or ["50", "100"]]:
        bench_sf(con, sf)

    con.close()
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    main()
