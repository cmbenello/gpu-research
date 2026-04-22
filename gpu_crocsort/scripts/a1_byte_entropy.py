#!/usr/bin/env python3
"""A1: Per-column byte entropy for TPC-H lineitem sort keys.

For every sort key column at SF10/50/100:
- Compute Shannon entropy per byte position
- Report bytes-per-key after perfect order-preserving encoder
- Output: results/overnight/a1_byte_entropy.csv
"""
import os, sys, struct, math, time
import numpy as np

try:
    import duckdb
except ImportError:
    print("pip install duckdb"); sys.exit(1)

OUT = "results/overnight/a1_byte_entropy.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# TPC-H lineitem sort key columns and their encoding to normalized bytes
# Same encoding as gen_tpch_normalized.py
SORT_COLS = [
    ("l_returnflag",   "char",    1),   # 1 byte ASCII
    ("l_linestatus",   "char",    1),   # 1 byte ASCII
    ("l_shipdate",     "date",    4),   # 4 bytes BE uint32 (days since epoch)
    ("l_commitdate",   "date",    4),
    ("l_receiptdate",  "date",    4),
    ("l_extendedprice","decimal", 8),   # 8 bytes BE int64 + offset
    ("l_discount",     "decimal", 8),
    ("l_tax",          "decimal", 8),
    ("l_quantity",     "decimal", 8),
    ("l_orderkey",     "bigint",  8),   # 8 bytes BE uint64
    ("l_partkey",      "int",     4),   # 4 bytes BE uint32
    ("l_suppkey",      "int",     4),
    ("l_linenumber",   "int",     4),
]

def to_big_endian_bytes(arr, width):
    """Convert integer array to big-endian byte representation (vectorized)."""
    n = len(arr)
    # byteswap to big-endian (x86 is little-endian)
    be = arr.byteswap()
    return be.view(np.uint8).reshape(n, width)

def encode_column(col_name, col_type, values):
    """Encode column values to normalized byte-comparable representation."""
    n = len(values)
    if col_type == "char":
        return np.frombuffer(np.array(values, dtype='S1').tobytes(), dtype=np.uint8).reshape(n, 1)
    elif col_type == "date":
        import pandas as pd
        ts = pd.to_datetime(values)
        epoch = pd.Timestamp('1970-01-01')
        days = ((ts - epoch).days).values.astype(np.uint32)
        return to_big_endian_bytes(days, 4)
    elif col_type == "decimal":
        raw = (np.asarray(values, dtype=np.float64) * 100).round().astype(np.int64)
        flipped = raw.view(np.uint64) ^ np.uint64(0x8000000000000000)
        return to_big_endian_bytes(flipped, 8)
    elif col_type == "bigint":
        raw = np.asarray(values, dtype=np.uint64)
        return to_big_endian_bytes(raw, 8)
    elif col_type == "int":
        raw = np.asarray(values, dtype=np.uint32)
        return to_big_endian_bytes(raw, 4)

def byte_entropy(byte_array):
    """Shannon entropy of a byte column (0-8 bits)."""
    counts = np.bincount(byte_array, minlength=256)
    probs = counts[counts > 0] / len(byte_array)
    return -np.sum(probs * np.log2(probs))

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

        # Fetch sort columns (sample 10M for SF50/100 — entropy converges quickly)
        col_list = ", ".join(c[0] for c in SORT_COLS)
        total_rows = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
        sample_size = min(10_000_000, total_rows)
        print(f"  Fetching columns ({sample_size:,} of {total_rows:,} rows): {col_list}")
        if total_rows > sample_size:
            df = con.execute(f"SELECT {col_list} FROM lineitem USING SAMPLE {sample_size}").fetchdf()
        else:
            df = con.execute(f"SELECT {col_list} FROM lineitem").fetchdf()
        n = len(df)
        print(f"  {n:,} rows fetched in {time.time()-t0:.1f}s")

        for col_name, col_type, n_bytes in SORT_COLS:
            t1 = time.time()
            values = df[col_name].values

            # Encode to normalized bytes
            encoded = encode_column(col_name, col_type, values)

            # Compute per-byte-position entropy
            for byte_pos in range(n_bytes):
                ent = byte_entropy(encoded[:, byte_pos])
                n_distinct = len(np.unique(encoded[:, byte_pos]))
                results.append({
                    'scale_factor': sf,
                    'column': col_name,
                    'col_type': col_type,
                    'total_bytes': n_bytes,
                    'byte_position': byte_pos,
                    'entropy_bits': round(ent, 4),
                    'n_distinct': n_distinct,
                    'n_rows': n,
                })

            col_entropy = sum(r['entropy_bits'] for r in results[-n_bytes:])
            print(f"  {col_name}: {n_bytes}B, total entropy = {col_entropy:.2f} bits ({col_entropy/8:.2f} bytes)")

        con.close()
        print(f"  SF{sf} done in {time.time()-t0:.1f}s")

    # Write CSV
    if results:
        import csv
        with open(OUT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {len(results)} rows to {OUT}")

        # Summary: bytes per key after perfect encoder
        print("\n=== Summary: bytes per key with perfect order-preserving encoder ===")
        for sf in [10, 50, 100]:
            sf_rows = [r for r in results if r['scale_factor'] == sf]
            if not sf_rows:
                continue
            total_entropy_bits = sum(r['entropy_bits'] for r in sf_rows)
            total_raw_bytes = 66  # key size
            print(f"  SF{sf}: {total_entropy_bits:.1f} bits = {total_entropy_bits/8:.1f} bytes (raw: {total_raw_bytes}B, ratio: {total_raw_bytes/(total_entropy_bits/8):.1f}x)")

if __name__ == '__main__':
    main()
