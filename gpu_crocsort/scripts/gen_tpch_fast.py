#!/usr/bin/env python3
"""Fast TPC-H lineitem normalized binary generator using vectorized numpy.

Usage: python3 scripts/gen_tpch_fast.py SF [output_file]
"""
import sys, os, struct, time
import numpy as np

try:
    import duckdb
except ImportError:
    print("pip install duckdb"); sys.exit(1)

KEY_SIZE = 66
VALUE_SIZE = 54
RECORD_SIZE = KEY_SIZE + VALUE_SIZE

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gen_tpch_fast.py SF [output_file]")
        sys.exit(1)

    sf = int(sys.argv[1])
    outpath = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/lineitem_sf{sf}_normalized.bin"

    db_path = f"/tmp/tpch_sf{sf}.db"
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)

    import pandas as pd

    print(f"Generating TPC-H SF{sf} normalized data → {outpath}")
    t0 = time.time()

    con = duckdb.connect(db_path, read_only=True)
    total = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
    print(f"  Total rows: {total:,}")

    # Process in chunks to limit memory
    CHUNK = 5_000_000
    epoch_ts = pd.Timestamp('1970-01-01')

    with open(outpath, 'wb') as f:
        for offset in range(0, total, CHUNK):
            n = min(CHUNK, total - offset)
            df = con.execute(f"""
                SELECT l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate,
                       l_extendedprice, l_discount, l_tax, l_quantity,
                       l_orderkey, l_partkey, l_suppkey, l_linenumber
                FROM lineitem LIMIT {n} OFFSET {offset}
            """).fetchdf()

            actual_n = len(df)
            records = np.zeros((actual_n, RECORD_SIZE), dtype=np.uint8)

            pos = 0
            # l_returnflag (1B)
            records[:, pos] = np.frombuffer(np.array(df['l_returnflag'].values, dtype='S1').tobytes(), dtype=np.uint8)
            pos += 1
            # l_linestatus (1B)
            records[:, pos] = np.frombuffer(np.array(df['l_linestatus'].values, dtype='S1').tobytes(), dtype=np.uint8)
            pos += 1

            # Dates (4B each, big-endian uint32)
            for col in ['l_shipdate', 'l_commitdate', 'l_receiptdate']:
                ts = pd.to_datetime(df[col].values)
                days = ((ts - epoch_ts).days).values.astype(np.uint32).byteswap()
                records[:, pos:pos+4] = days.view(np.uint8).reshape(actual_n, 4)
                pos += 4

            # Decimals (8B each, sign-flipped big-endian int64)
            for col in ['l_extendedprice', 'l_discount', 'l_tax', 'l_quantity']:
                raw = (np.asarray(df[col].values, dtype=np.float64) * 100).round().astype(np.int64)
                flipped = (raw.view(np.uint64) ^ np.uint64(0x8000000000000000)).byteswap()
                records[:, pos:pos+8] = flipped.view(np.uint8).reshape(actual_n, 8)
                pos += 8

            # l_orderkey (8B big-endian uint64)
            raw = np.asarray(df['l_orderkey'].values, dtype=np.uint64).byteswap()
            records[:, pos:pos+8] = raw.view(np.uint8).reshape(actual_n, 8)
            pos += 8

            # l_partkey, l_suppkey, l_linenumber (4B each)
            for col in ['l_partkey', 'l_suppkey', 'l_linenumber']:
                raw = np.asarray(df[col].values, dtype=np.uint32).byteswap()
                records[:, pos:pos+4] = raw.view(np.uint8).reshape(actual_n, 4)
                pos += 4

            f.write(records.tobytes())
            elapsed = time.time() - t0
            pct = (offset + actual_n) / total * 100
            rate = (offset + actual_n) / elapsed
            eta = (total - offset - actual_n) / rate if rate > 0 else 0
            print(f"  {offset + actual_n:>12,} / {total:,} ({pct:.0f}%) — {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    file_size = os.path.getsize(outpath)
    elapsed = time.time() - t0
    print(f"\nDone: {file_size / 1e9:.1f} GB in {elapsed:.0f}s ({file_size / elapsed / 1e6:.0f} MB/s)")

if __name__ == '__main__':
    main()
