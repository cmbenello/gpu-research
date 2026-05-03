#!/usr/bin/env python3
"""4.2 — Polars sort baseline at SF50 / SF100.

Reads our normalized 120 B record format directly via numpy, decodes the
sort columns, then runs Polars sort. Times the sort separately from the
decode.

Compares to:
  gpu_crocsort post-0.3.1 SF50  : 1.51 s @ 23.9 GB/s
  gpu_crocsort post-0.3.1 SF100 : 3.02 s @ 23.8 GB/s
  duckdb_1.3.2          SF50    : 33.0 s @ 1.09 GB/s
  duckdb_1.3.2          SF100   : 106 s  @ 0.68 GB/s
"""
import polars as pl
import numpy as np
import os, sys, time, struct

KEY_SIZE   = 66
RECORD_SIZE = 120

def load_to_polars(path, n_rows=None):
    """mmap the binary, decode sort columns into a polars DataFrame."""
    print(f"Loading {path} ...")
    t0 = time.time()
    fsize = os.path.getsize(path)
    n_total = fsize // RECORD_SIZE
    if n_rows is None: n_rows = n_total
    n_rows = min(n_rows, n_total)
    arr = np.memmap(path, dtype=np.uint8, mode='r', shape=(n_total, RECORD_SIZE))
    arr = arr[:n_rows]
    print(f"  mmap'd {n_rows:,} records in {time.time()-t0:.1f}s")

    # The compact key bytes (positions [0,1,4,5,8,9,12,13,19,20,21,29,37,
    # 44,45,51,52,53,54,55,56,57,59,60,61,65]) are byte-comparable in lex
    # order. To make polars sort meaningful, treat them as a single 32 B
    # key column (concatenated bytes), or as separate fields.
    # Simpler: take the first 16 bytes (fits in two uint64) which is what
    # the OVC merge sorts by. Polars can sort by 2 uint64 columns.
    t1 = time.time()
    # Polars rejects big-endian uint64 dtype directly; convert to native
    # via np.uint64 (byteswap so the byte ordering preserves lex sort).
    pfx1_be = np.frombuffer(arr[:, 0:8].tobytes(), dtype='>u8')
    pfx2_be = np.frombuffer(arr[:, 8:16].tobytes(), dtype='>u8')
    pfx1 = np.ascontiguousarray(pfx1_be.astype(np.uint64))
    pfx2 = np.ascontiguousarray(pfx2_be.astype(np.uint64))
    df = pl.DataFrame({'pfx1': pfx1, 'pfx2': pfx2})
    print(f"  decoded prefix into 2-col DataFrame in {time.time()-t1:.1f}s")
    return df

def bench(df, label):
    print(f"\n=== {label} ({len(df):,} rows) ===")
    times = []
    for run in range(5):
        t0 = time.time()
        sorted_df = df.sort(['pfx1', 'pfx2'])
        ms = (time.time() - t0) * 1000.0
        times.append(ms)
        del sorted_df
        print(f"  Run {run+1}: {ms:.0f} ms")
    warm_best = min(times[1:]) if len(times) > 1 else times[0]
    gb = len(df) * 16 / 1e9  # 16 B sort key
    rec_gb = len(df) * 120 / 1e9  # 120 B record-equivalent
    print(f"  Warm best: {warm_best:.0f} ms")
    print(f"  Throughput: {gb / (warm_best/1000):.2f} GB/s on 16-byte sort key")
    print(f"             ({rec_gb / (warm_best/1000):.2f} GB/s on 120-byte record-equiv)")
    print(f"  CSV,polars_sort,label={label},rows={len(df)},warm_best_ms={warm_best:.1f},"
          f"key_gb_per_s={gb/(warm_best/1000):.2f},record_gb_per_s={rec_gb/(warm_best/1000):.2f}")

def main():
    print(f"Polars version: {pl.__version__}")
    sf_to_path = {
        50:  "/mnt/data/lineitem_sf50.bin",
        100: "/mnt/data/lineitem_sf100.bin",
    }
    sfs = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [50, 100]
    for sf in sfs:
        path = sf_to_path.get(sf)
        if not path or not os.path.exists(path):
            print(f"SF{sf}: no input {path}")
            continue
        df = load_to_polars(path)
        bench(df, f"SF{sf}")
        del df

if __name__ == "__main__":
    main()
