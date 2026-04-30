#!/usr/bin/env python3
"""CPU baselines for TPC-H lineitem ORDER BY.
Runs DuckDB and Polars on the same TPC-H scale factors that gpu_crocsort
sorts, so comparisons are apples-to-apples per machine.
"""
import argparse, csv, gc, os, sys, time, statistics

SORT_COLS = [
    "l_returnflag", "l_linestatus",
    "l_shipdate", "l_commitdate", "l_receiptdate",
    "l_extendedprice", "l_discount", "l_tax", "l_quantity",
    "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
]

def fmt_time(s):
    return f"{s:7.2f}s" if s >= 1.0 else f"{s*1000:7.1f}ms"

def bench_duckdb(sf, runs):
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    n = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
    order = ", ".join(SORT_COLS)
    times = []
    for r in range(runs):
        con.execute("DROP TABLE IF EXISTS sorted;")
        gc.collect()
        t0 = time.time()
        con.execute(f"CREATE TABLE sorted AS SELECT * FROM lineitem ORDER BY {order};")
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  duckdb SF{sf} run {r+1}/{runs}: {fmt_time(elapsed)} ({n:,} rows)")
    return n, times

def bench_polars(sf, runs):
    try:
        import polars as pl
    except ImportError:
        print("  polars not installed, skipping")
        return None, []
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    arrow_table = con.execute("SELECT * FROM lineitem").arrow()
    df = pl.from_arrow(arrow_table)
    n = len(df)
    times = []
    for r in range(runs):
        gc.collect()
        t0 = time.time()
        out = df.sort(SORT_COLS)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  polars SF{sf} run {r+1}/{runs}: {fmt_time(elapsed)} ({n:,} rows)")
        del out
    return n, times

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="10,20,50",
                    help="comma-separated TPC-H scale factors")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--engines", default="duckdb,polars")
    ap.add_argument("--out", default="baseline_results.csv")
    args = ap.parse_args()

    scales = [int(x) for x in args.scales.split(",")]
    engines = args.engines.split(",")

    rows = []
    for engine in engines:
        for sf in scales:
            print(f"\n=== {engine} SF{sf} ===")
            try:
                if engine == "duckdb":
                    n, times = bench_duckdb(sf, args.runs)
                elif engine == "polars":
                    n, times = bench_polars(sf, args.runs)
                else:
                    print(f"  unknown engine: {engine}")
                    continue
            except MemoryError as e:
                print(f"  OOM: {e}")
                continue
            except Exception as e:
                print(f"  FAILED: {e}")
                continue
            if not times:
                continue
            best = min(times)
            median = statistics.median(times)
            data_gb = n * 88 / 1e9   # approx
            print(f"  best: {fmt_time(best)}  median: {fmt_time(median)}  rate: {n/best/1e6:.2f} Mrows/s")
            rows.append({
                "engine": engine, "sf": sf, "n_rows": n,
                "data_gb_approx": round(data_gb, 2),
                "best_s": round(best, 4),
                "median_s": round(median, 4),
                "all_runs": ",".join(f"{t:.4f}" for t in times),
                "rate_Mrows_s": round(n/best/1e6, 2),
            })

    # write CSV
    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nResults → {args.out}")

if __name__ == "__main__":
    main()
