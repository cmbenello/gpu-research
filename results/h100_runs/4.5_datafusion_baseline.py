#!/usr/bin/env python3
"""4.5 — DataFusion (Apache Arrow Acero) sort baseline at SF50/SF100.

Adds a 4th CPU baseline alongside DuckDB (4.1) and Polars (4.2).
Same approach as 4.2: decode 16 B sort prefix into 2 uint64 columns,
then sort.
"""
import datafusion as df
import pyarrow as pa
import numpy as np
import os, sys, time

KEY_SIZE   = 66
RECORD_SIZE = 120

def load_to_arrow(path, n_rows=None):
    print(f"Loading {path} ...")
    t0 = time.time()
    fsize = os.path.getsize(path)
    n_total = fsize // RECORD_SIZE
    if n_rows is None: n_rows = n_total
    n_rows = min(n_rows, n_total)
    arr = np.memmap(path, dtype=np.uint8, mode='r', shape=(n_total, RECORD_SIZE))
    arr = arr[:n_rows]
    print(f"  mmap'd {n_rows:,} records in {time.time()-t0:.1f}s")

    t1 = time.time()
    pfx1_be = np.frombuffer(arr[:, 0:8].tobytes(), dtype='>u8')
    pfx2_be = np.frombuffer(arr[:, 8:16].tobytes(), dtype='>u8')
    pfx1 = np.ascontiguousarray(pfx1_be.astype(np.uint64))
    pfx2 = np.ascontiguousarray(pfx2_be.astype(np.uint64))
    table = pa.table({'pfx1': pfx1, 'pfx2': pfx2})
    print(f"  decoded prefix into Arrow Table in {time.time()-t1:.1f}s")
    return table

def bench(ctx, table, label):
    print(f"\n=== {label} ({table.num_rows:,} rows) ===")
    ctx.deregister_table('t')
    ctx.register_record_batches('t', [table.to_batches()])

    times = []
    for run in range(5):
        t0 = time.time()
        # Use SQL for explicit ORDER BY
        result = ctx.sql("SELECT * FROM t ORDER BY pfx1, pfx2").collect()
        ms = (time.time() - t0) * 1000.0
        times.append(ms)
        del result
        print(f"  Run {run+1}: {ms:.0f} ms")
    warm_best = min(times[1:]) if len(times) > 1 else times[0]
    gb = table.num_rows * 16 / 1e9
    rec_gb = table.num_rows * 120 / 1e9
    print(f"  Warm best: {warm_best:.0f} ms")
    print(f"  Throughput: {gb / (warm_best/1000):.2f} GB/s on 16-byte sort key")
    print(f"             ({rec_gb / (warm_best/1000):.2f} GB/s on 120-byte record-equiv)")
    print(f"  CSV,datafusion_sort,label={label},rows={table.num_rows},warm_best_ms={warm_best:.1f},"
          f"key_gb_per_s={gb/(warm_best/1000):.2f},record_gb_per_s={rec_gb/(warm_best/1000):.2f}")

def main():
    print(f"DataFusion version: {df.__version__}")
    ctx = df.SessionContext()

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
        table = load_to_arrow(path)
        bench(ctx, table, f"SF{sf}")
        del table

if __name__ == "__main__":
    main()
