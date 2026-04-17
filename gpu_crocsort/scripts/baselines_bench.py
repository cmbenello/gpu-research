#!/usr/bin/env python3
"""Baseline sort benchmark: DuckDB + Polars vs our GPU sort.

Reads a TPC-H normalized binary (120B record, 66B key prefix) using mmap,
wraps it as a pyarrow FixedSizeBinary table, sorts by the key, and reports
warm wall-clock for each engine.

Output is written to /dev/shm to avoid touching the main disk.
"""
import os, sys, time, hashlib
sys.path.insert(0, '/dev/shm/pylib')

import numpy as np
import pyarrow as pa

RECORD = 120
KEY = 66

def load_binary(path):
    mm = np.memmap(path, dtype=np.uint8, mode='r')
    n = mm.size // RECORD
    recs = mm[: n * RECORD].reshape(n, RECORD)
    return recs, n

def to_arrow(recs):
    # FixedSizeBinary needs contiguous memory; copy is ~3.4 GB for SF10 and is done once outside the timed region.
    keys = np.ascontiguousarray(recs[:, :KEY]).reshape(-1)
    payloads = np.ascontiguousarray(recs[:, KEY:]).reshape(-1)
    key_arr = pa.FixedSizeBinaryArray.from_buffers(pa.binary(KEY), len(recs), [None, pa.py_buffer(keys)])
    pay_arr = pa.FixedSizeBinaryArray.from_buffers(pa.binary(RECORD - KEY), len(recs), [None, pa.py_buffer(payloads)])
    return pa.table({'key': key_arr, 'payload': pay_arr})

def bench_duckdb(table, n, mode='parquet'):
    import duckdb
    con = duckdb.connect()
    con.register('lineitem', table)
    # Warm JIT with one row
    con.execute("SELECT key FROM lineitem LIMIT 1").fetchall()
    if mode == 'stream':
        # Sort + streaming fetch: measures sort + output without Arrow table overhead
        t0 = time.monotonic()
        result = con.execute("SELECT key, payload FROM lineitem ORDER BY key")
        count = 0
        while True:
            batch = result.fetchmany(100000)
            if not batch:
                break
            count += len(batch)
        elapsed = (time.monotonic() - t0) * 1000.0
        con.close()
        return elapsed, f"stream:{count}"
    elif mode == 'arrow':
        # Sort + full Arrow table materialization
        t0 = time.monotonic()
        result = con.execute("SELECT key, payload FROM lineitem ORDER BY key").fetch_arrow_table()
        elapsed = (time.monotonic() - t0) * 1000.0
        con.close()
        return elapsed, f"arrow:{len(result)}"
    else:
        # Full: sort + Parquet write (original behavior)
        out = f'/dev/shm/baselines/duckdb_out_{n}.parquet'
        if os.path.exists(out): os.remove(out)
        t0 = time.monotonic()
        con.execute(f"COPY (SELECT key, payload FROM lineitem ORDER BY key) TO '{out}' (FORMAT PARQUET, COMPRESSION UNCOMPRESSED)")
        elapsed = (time.monotonic() - t0) * 1000.0
        con.close()
        return elapsed, out

def bench_polars(table, n):
    import polars as pl
    df = pl.from_arrow(table)
    t0 = time.monotonic()
    sorted_df = df.sort('key')
    out = f'/dev/shm/baselines/polars_out_{n}.parquet'
    sorted_df.write_parquet(out, compression='uncompressed')
    elapsed = (time.monotonic() - t0) * 1000.0
    return elapsed, out

def main():
    if len(sys.argv) < 3:
        print("usage: bench.py <binary> <runs>")
        sys.exit(1)
    path = sys.argv[1]
    runs = int(sys.argv[2])
    engine = sys.argv[3] if len(sys.argv) > 3 else 'both'

    print(f"[load] {path}")
    t0 = time.monotonic()
    recs, n = load_binary(path)
    print(f"[load] {n:,} records in {(time.monotonic()-t0)*1000:.0f} ms (mmap)")

    print(f"[arrow] building fixed-size-binary table")
    t0 = time.monotonic()
    table = to_arrow(recs)
    print(f"[arrow] built in {(time.monotonic()-t0)*1000:.0f} ms")

    duckdb_modes = []
    if engine in ('duckdb', 'both'):
        duckdb_modes = ['parquet', 'stream']
    elif engine == 'duckdb_sort':
        duckdb_modes = ['stream']
    elif engine == 'duckdb_parquet':
        duckdb_modes = ['parquet']

    for mode in duckdb_modes:
        label = f'duckdb ({mode})'
        print(f"\n[{label}] warm runs...")
        times = []
        for i in range(runs):
            ms, out = bench_duckdb(table, n, mode=mode)
            print(f"  run {i+1}: {ms:.0f} ms")
            times.append(ms)
        print(f"[{label}] median={sorted(times)[len(times)//2]:.0f} ms  min={min(times):.0f}  max={max(times):.0f}")

    if engine in ('polars', 'both'):
        print(f"\n[polars] warm runs...")
        times = []
        for i in range(runs):
            ms, out = bench_polars(table, n)
            print(f"  run {i+1}: {ms:.0f} ms")
            times.append(ms)
        print(f"[polars] median={sorted(times)[len(times)//2]:.0f} ms  min={min(times):.0f}  max={max(times):.0f}")

if __name__ == '__main__':
    main()
