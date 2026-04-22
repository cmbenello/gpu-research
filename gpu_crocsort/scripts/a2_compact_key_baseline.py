#!/usr/bin/env python3
"""A2: Compact-key scan effectiveness baseline.

For each TPC-H scale factor, compute:
- Bytes per key before compact-key scan
- Bytes per key after compact-key scan
- Ratio
- Which byte positions vary

Output: results/overnight/a2_compact_key.csv
"""
import os, sys, struct, time
import numpy as np

try:
    import duckdb
except ImportError:
    print("pip install duckdb"); sys.exit(1)

OUT = "results/overnight/a2_compact_key.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

KEY_SIZE = 66  # total key bytes in the normalized format

def encode_all_keys(df):
    """Encode all sort columns into a single KEY_SIZE-byte normalized key per row."""
    n = len(df)
    keys = np.zeros((n, KEY_SIZE), dtype=np.uint8)
    offset = 0

    # l_returnflag (1B)
    keys[:, offset] = np.frombuffer(np.array(df['l_returnflag'].values, dtype='S1').tobytes(), dtype=np.uint8)
    offset += 1

    # l_linestatus (1B)
    keys[:, offset] = np.frombuffer(np.array(df['l_linestatus'].values, dtype='S1').tobytes(), dtype=np.uint8)
    offset += 1

    # Dates: shipdate, commitdate, receiptdate (4B each)
    import pandas as pd
    epoch_ts = pd.Timestamp('1970-01-01')
    for col in ['l_shipdate', 'l_commitdate', 'l_receiptdate']:
        ts = pd.to_datetime(df[col].values)
        days = ((ts - epoch_ts).days).values.astype(np.uint32)
        be = days.byteswap()
        keys[:, offset:offset+4] = be.view(np.uint8).reshape(n, 4)
        offset += 4

    # Decimals: extendedprice, discount, tax, quantity (8B each)
    for col in ['l_extendedprice', 'l_discount', 'l_tax', 'l_quantity']:
        raw = (np.asarray(df[col].values, dtype=np.float64) * 100).round().astype(np.int64)
        flipped = raw.view(np.uint64) ^ np.uint64(0x8000000000000000)
        be = flipped.byteswap()
        keys[:, offset:offset+8] = be.view(np.uint8).reshape(n, 8)
        offset += 8

    # Integers: orderkey(8B), partkey(4B), suppkey(4B), linenumber(4B)
    raw = np.asarray(df['l_orderkey'].values, dtype=np.uint64).byteswap()
    keys[:, offset:offset+8] = raw.view(np.uint8).reshape(n, 8)
    offset += 8

    for col in ['l_partkey', 'l_suppkey', 'l_linenumber']:
        raw = np.asarray(df[col].values, dtype=np.uint32).byteswap()
        keys[:, offset:offset+4] = raw.view(np.uint8).reshape(n, 4)
        offset += 4

    return keys

def compact_key_scan(keys):
    """Simulate compact-key scan: find byte positions that vary across dataset."""
    n, key_size = keys.shape
    # A byte position varies if not all rows have the same value
    # Sample-based approach (same as CrocSort): check first row vs all
    # Actually CrocSort scans all rows. Use vectorized approach.
    varying = []
    for b in range(key_size):
        col = keys[:, b]
        if col.min() != col.max():
            varying.append(b)
    return varying

def main():
    results = []

    for sf in [10, 50, 100]:
        db_path = f"/tmp/tpch_sf{sf}.db"
        if not os.path.exists(db_path):
            print(f"  SKIP SF{sf}: {db_path} not found")
            continue

        print(f"\n=== TPC-H SF{sf} ===")
        t0 = time.time()
        con = duckdb.connect(db_path, read_only=True)

        # For large SFs, sample to keep memory manageable
        sample_size = min(10_000_000, 60_000_000 * sf // 10)
        col_list = "l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, l_extendedprice, l_discount, l_tax, l_quantity, l_orderkey, l_partkey, l_suppkey, l_linenumber"

        total_rows = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
        print(f"  Total rows: {total_rows:,}")

        if total_rows > sample_size:
            print(f"  Sampling {sample_size:,} rows for byte-level analysis...")
            df = con.execute(f"SELECT {col_list} FROM lineitem USING SAMPLE {sample_size}").fetchdf()
        else:
            df = con.execute(f"SELECT {col_list} FROM lineitem").fetchdf()

        n = len(df)
        print(f"  Fetched {n:,} rows in {time.time()-t0:.1f}s")

        # Encode to normalized keys
        t1 = time.time()
        print(f"  Encoding keys...")
        keys = encode_all_keys(df)
        encode_s = time.time() - t1
        print(f"  Encoded in {encode_s:.1f}s")

        # Run compact-key scan
        t2 = time.time()
        varying = compact_key_scan(keys)
        scan_s = time.time() - t2

        raw_bytes = KEY_SIZE
        compact_bytes = len(varying)
        ratio = raw_bytes / compact_bytes if compact_bytes > 0 else float('inf')

        print(f"  Raw key: {raw_bytes}B, Compact key: {compact_bytes}B, Ratio: {ratio:.1f}x")
        print(f"  Varying positions: {varying}")
        print(f"  Scan time: {scan_s*1000:.0f}ms")

        results.append({
            'scale_factor': sf,
            'total_rows': total_rows,
            'sample_rows': n,
            'raw_key_bytes': raw_bytes,
            'compact_key_bytes': compact_bytes,
            'ratio': round(ratio, 2),
            'varying_positions': str(varying),
            'n_varying': compact_bytes,
            'scan_time_ms': round(scan_s * 1000, 1),
            'encode_time_s': round(encode_s, 1),
        })

        con.close()

    # Write CSV
    if results:
        import csv
        with open(OUT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {len(results)} rows to {OUT}")

if __name__ == '__main__':
    main()
