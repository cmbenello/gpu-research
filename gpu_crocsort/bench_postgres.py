#!/usr/bin/env python3
"""
bench_postgres.py — Benchmark GPU sort vs SQLite CREATE INDEX on int32 data.

PostgreSQL is not installed on this machine, so we use SQLite as a proxy
for a traditional RDBMS index build.  We also collect GPU sort timings
and compare at several row counts.
"""

import sys
import os
import time
import sqlite3
import statistics
import tempfile
import numpy as np
import pyarrow as pa

sys.path.insert(0, '/home/cc/gpu-research/gpu_crocsort')
import gpu_sort

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIZES         = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]
REPEATS       = 3   # number of timed trials per (engine, size)
RNG_SEED      = 42
SQLITE_DB     = '/tmp/bench_sort.db'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human(n):
    for unit, div in [("M", 1_000_000), ("K", 1_000)]:
        if n >= div:
            return f"{n/div:.0f}{unit}"
    return str(n)

def median(xs):
    return statistics.median(xs)

def gen_int32(n, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(-2**31, 2**31, size=n, dtype=np.int32)

# ---------------------------------------------------------------------------
# SQLite benchmark — CREATE INDEX on a column of int32 values
# ---------------------------------------------------------------------------
def bench_sqlite_one(n, db_path):
    """
    1. Create a fresh DB with a table of n int32 rows.
    2. Time CREATE INDEX (exclude INSERT time).
    Returns list of elapsed seconds (REPEATS trials).
    """
    data = gen_int32(n)

    if os.path.exists(db_path):
        os.unlink(db_path)

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA page_size=65536")
    con.execute("PRAGMA cache_size=-262144")  # 256 MB

    # Create table and bulk-insert once
    con.execute("CREATE TABLE t (v INTEGER NOT NULL)")
    con.execute("BEGIN")
    con.executemany("INSERT INTO t VALUES (?)", ((int(x),) for x in data))
    con.execute("COMMIT")
    con.execute("VACUUM")  # compact
    con.close()

    times = []
    for _ in range(REPEATS):
        con = sqlite3.connect(db_path)
        con.execute("PRAGMA cache_size=-262144")
        try:
            con.execute("DROP INDEX IF EXISTS idx_v")
        except Exception:
            pass
        t0 = time.perf_counter()
        con.execute("CREATE INDEX idx_v ON t(v)")
        con.commit()
        t1 = time.perf_counter()
        con.close()
        times.append(t1 - t0)

    return times


# ---------------------------------------------------------------------------
# GPU sort benchmark — sort a pa.Array of int32 values (sort_indices only)
# ---------------------------------------------------------------------------
def bench_gpu_one(n):
    """
    Sort a pyarrow int32 array of n random values.
    Returns list of elapsed seconds (REPEATS trials).
    """
    data = gen_int32(n)
    arr  = pa.array(data, type=pa.int32())

    times = []
    # Warm-up (not counted)
    _ = gpu_sort.sort_indices(arr)

    for _ in range(REPEATS):
        arr = pa.array(gen_int32(n, seed=RNG_SEED + _), type=pa.int32())
        t0 = time.perf_counter()
        _ = gpu_sort.sort_indices(arr)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


# ---------------------------------------------------------------------------
# NumPy sort benchmark — np.argsort as a CPU baseline
# ---------------------------------------------------------------------------
def bench_numpy_one(n):
    """
    np.argsort on int32 array of n values.
    Returns list of elapsed seconds (REPEATS trials).
    """
    times = []
    # Warm-up
    _ = np.argsort(gen_int32(1000))

    for i in range(REPEATS):
        data = gen_int32(n, seed=RNG_SEED + i)
        t0 = time.perf_counter()
        _ = np.argsort(data, kind='stable')
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    gpu_available = gpu_sort.is_available()
    print(f"GPU available: {gpu_available}")
    if not gpu_available:
        print("WARNING: GPU not found — GPU timings will be fake/zero.")

    results = []  # list of dicts

    for n in SIZES:
        print(f"\n--- {human(n)} rows ({n:,}) ---")

        # SQLite
        try:
            sq_times = bench_sqlite_one(n, SQLITE_DB)
            sq_med = median(sq_times)
            print(f"  SQLite  CREATE INDEX: {sq_times}  median={sq_med:.3f}s")
        except Exception as e:
            print(f"  SQLite  FAILED: {e}")
            sq_med = None

        # GPU sort
        if gpu_available:
            try:
                gpu_times = bench_gpu_one(n)
                gpu_med = median(gpu_times)
                print(f"  GPU sort (sort_indices): {[f'{x:.3f}' for x in gpu_times]}  median={gpu_med:.3f}s")
            except Exception as e:
                print(f"  GPU sort FAILED: {e}")
                gpu_med = None
        else:
            gpu_med = None

        # NumPy argsort (CPU reference)
        try:
            np_times = bench_numpy_one(n)
            np_med = median(np_times)
            print(f"  NumPy   argsort:      {[f'{x:.3f}' for x in np_times]}  median={np_med:.3f}s")
        except Exception as e:
            print(f"  NumPy   FAILED: {e}")
            np_med = None

        results.append({
            'n':       n,
            'sqlite':  sq_med,
            'gpu':     gpu_med,
            'numpy':   np_med,
        })

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "="*78)
    print("RESULTS SUMMARY — median of 3 trials, sorting random int32 values")
    print("="*78)
    hdr = f"{'Rows':>10}  {'SQLite INDEX':>14}  {'GPU sort':>12}  {'NumPy sort':>12}  {'GPU/SQLite':>12}  {'GPU/NumPy':>10}"
    print(hdr)
    print("-"*78)
    for r in results:
        n       = r['n']
        sq      = r['sqlite']
        gpu     = r['gpu']
        np_     = r['numpy']
        sq_str  = f"{sq:.3f}s" if sq  is not None else "  N/A"
        gpu_str = f"{gpu:.3f}s" if gpu is not None else "  N/A"
        np_str  = f"{np_:.3f}s" if np_ is not None else "  N/A"
        if sq is not None and gpu is not None:
            ratio_sq = f"{sq/gpu:.2f}x"
        else:
            ratio_sq = "  N/A"
        if np_ is not None and gpu is not None:
            ratio_np = f"{np_/gpu:.2f}x"
        else:
            ratio_np = "  N/A"
        print(f"{n:>10,}  {sq_str:>14}  {gpu_str:>12}  {np_str:>12}  {ratio_sq:>12}  {ratio_np:>10}")
    print("="*78)

    # PostgreSQL note
    print("""
NOTES
-----
PostgreSQL is not installed on this machine.  SQLite's B-tree index build
is used as a proxy for a row-store RDBMS.  SQLite's sorter uses an internal
merge-sort similar to PostgreSQL's btree build, so the per-row cost should
be representative (PostgreSQL is typically 1.5-3x faster due to its
multi-pass external-merge btree builder and better OS page cache use).

GPU sort here is sort_indices only (produces a permutation array) — it does
NOT include writing a persistent on-disk B-tree structure.  The comparison
is therefore:
  • SQLite : full disk-persisted index (read data + sort + write B-tree)
  • GPU    : in-memory sort (read data + GPU sort + download permutation)
  • NumPy  : in-memory CPU sort (reference)

A fair "index build" GPU comparison would add the B-tree write cost, which
would widen the gap further in GPU's favour for in-memory use cases and
narrow it for on-disk use cases.
""")

if __name__ == '__main__':
    main()
