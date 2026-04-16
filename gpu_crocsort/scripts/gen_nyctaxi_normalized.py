#!/usr/bin/env python3
"""Generate normalized NYC Yellow Taxi binary data for external sort benchmarks.

Usage: python3 gen_nyctaxi_normalized.py MONTHS [output_file]
  MONTHS = number of 2023 months to include (1-12)
  output_file = defaults to /dev/shm/nyctaxi_{MONTHS}mo_normalized.bin

Downloads parquet from NYC TLC, normalizes to 120-byte fixed-width records.
Key layout matches the TPC-H binary format (66B key + 54B value = 120B record)
so the same external_sort_tpch_compact binary can sort it.

Key layout (66 bytes):
  [0]     VendorID        (1 byte, uint8)
  [1]     payment_type    (1 byte, uint8)
  [2:4]   PULocationID    (2 bytes, big-endian uint16)
  [4:6]   DOLocationID    (2 bytes, big-endian uint16)
  [6]     passenger_count (1 byte, uint8, clamped 0-255)
  [7]     RatecodeID      (1 byte, uint8)
  [8:16]  pickup_datetime (8 bytes, microseconds since epoch, big-endian int64 + 2^62)
  [16:24] dropoff_datetime (8 bytes, microseconds since epoch, big-endian int64 + 2^62)
  [24:32] fare_amount     (8 bytes, cents as big-endian int64 + 2^62)
  [32:40] tip_amount      (8 bytes, cents as big-endian int64 + 2^62)
  [40:48] total_amount    (8 bytes, cents as big-endian int64 + 2^62)
  [48:56] trip_distance   (8 bytes, mm as big-endian int64 + 2^62)
  [56:64] tolls_amount    (8 bytes, cents as big-endian int64 + 2^62)
  [64]    store_and_fwd_flag (1 byte, 'Y'=1, else 0)
  [65]    padding         (1 byte, zero)

Value layout (54 bytes): zero-padded.

Design rationale: key starts with low-cardinality columns (vendor, payment,
location IDs — ~4000 combinations) to create many tie groups, followed by
high-cardinality timestamps and amounts. This stresses the compact-key fixup
path differently from TPC-H.
"""

import sys
import struct
import time
import os
import numpy as np

RECORD_SIZE = 120
KEY_SIZE = 66
VALUE_SIZE = 54
OFFSET_I64 = 2**62

MONTHS_2023 = [
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-05.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-06.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-07.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-08.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-09.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-10.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-12.parquet",
]

def ts_to_micros(ts):
    """Convert a duckdb timestamp to microseconds since epoch."""
    if hasattr(ts, 'timestamp'):
        return int(ts.timestamp() * 1_000_000)
    return int(ts)

def to_cents(val):
    """Convert a decimal/float to integer cents."""
    return int(round(float(val) * 100))

def to_mm(val):
    """Convert miles to millimeters (arbitrary but gives integer precision)."""
    return int(round(float(val) * 1_609_344))  # 1 mile = 1,609,344 mm

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gen_nyctaxi_normalized.py MONTHS [output_file]")
        print("  MONTHS = 1-12 (months of 2023 yellow taxi data)")
        sys.exit(1)

    months = int(sys.argv[1])
    if months < 1 or months > 12:
        print("MONTHS must be 1-12")
        sys.exit(1)

    outpath = sys.argv[2] if len(sys.argv) > 2 else f"/dev/shm/nyctaxi_{months}mo_normalized.bin"

    try:
        import duckdb
    except ImportError:
        os.system("pip3 install duckdb")
        import duckdb

    con = duckdb.connect()

    urls = MONTHS_2023[:months]
    print(f"Loading {months} month(s) of NYC Yellow Taxi 2023 data...")

    # DuckDB can read parquet directly from URLs
    url_list = ", ".join(f"'{u}'" for u in urls)
    query = f"""
        SELECT
            COALESCE(VendorID, 0)::INT AS VendorID,
            COALESCE(payment_type, 0)::INT AS payment_type,
            COALESCE(PULocationID, 0)::INT AS PULocationID,
            COALESCE(DOLocationID, 0)::INT AS DOLocationID,
            COALESCE(passenger_count, 0)::INT AS passenger_count,
            COALESCE(RatecodeID, 0)::INT AS RatecodeID,
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            COALESCE(fare_amount, 0.0) AS fare_amount,
            COALESCE(tip_amount, 0.0) AS tip_amount,
            COALESCE(total_amount, 0.0) AS total_amount,
            COALESCE(trip_distance, 0.0) AS trip_distance,
            COALESCE(tolls_amount, 0.0) AS tolls_amount,
            COALESCE(store_and_fwd_flag, 'N') AS store_and_fwd_flag
        FROM read_parquet([{url_list}])
        WHERE tpep_pickup_datetime IS NOT NULL
          AND tpep_dropoff_datetime IS NOT NULL
    """

    t0 = time.time()
    print("  Querying (downloads parquet from NYC TLC)...")
    result = con.execute(query)

    # Count rows
    nrows = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
    print(f"  {nrows:,} rows in {time.time()-t0:.1f}s")

    # Write in chunks
    CHUNK = 2_000_000
    print(f"Writing {outpath} ({nrows * RECORD_SIZE / 1e9:.2f} GB)...")

    with open(outpath, 'wb') as f:
        buf = bytearray(RECORD_SIZE)
        written = 0
        t1 = time.time()

        offset = 0
        while offset < nrows:
            limit = min(CHUNK, nrows - offset)
            rows = con.execute(f"""
                SELECT * FROM ({query}) LIMIT {limit} OFFSET {offset}
            """).fetchall()
            offset += len(rows)

            for row in rows:
                (vendor_id, payment_type, pu_loc, do_loc,
                 passenger_count, ratecode_id,
                 pickup_dt, dropoff_dt,
                 fare, tip, total, distance, tolls,
                 store_fwd) = row

                for i in range(RECORD_SIZE):
                    buf[i] = 0

                off = 0
                buf[off] = min(255, max(0, int(vendor_id))); off += 1
                buf[off] = min(255, max(0, int(payment_type))); off += 1
                struct.pack_into('>H', buf, off, min(65535, max(0, int(pu_loc)))); off += 2
                struct.pack_into('>H', buf, off, min(65535, max(0, int(do_loc)))); off += 2
                buf[off] = min(255, max(0, int(passenger_count))); off += 1
                buf[off] = min(255, max(0, int(ratecode_id))); off += 1

                struct.pack_into('>q', buf, off, ts_to_micros(pickup_dt) + OFFSET_I64); off += 8
                struct.pack_into('>q', buf, off, ts_to_micros(dropoff_dt) + OFFSET_I64); off += 8
                struct.pack_into('>q', buf, off, to_cents(fare) + OFFSET_I64); off += 8
                struct.pack_into('>q', buf, off, to_cents(tip) + OFFSET_I64); off += 8
                struct.pack_into('>q', buf, off, to_cents(total) + OFFSET_I64); off += 8
                struct.pack_into('>q', buf, off, to_mm(distance) + OFFSET_I64); off += 8
                struct.pack_into('>q', buf, off, to_cents(tolls) + OFFSET_I64); off += 8
                buf[off] = 1 if (isinstance(store_fwd, str) and store_fwd == 'Y') else 0; off += 1

                f.write(buf)
                written += 1
                if written % 5_000_000 == 0:
                    elapsed = time.time() - t1
                    rate = written / elapsed
                    eta = (nrows - written) / rate
                    print(f"  {written/1e6:.0f}M / {nrows/1e6:.0f}M ({100*written/nrows:.0f}%) "
                          f"{written*RECORD_SIZE/1e9:.1f} GB, ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    size_gb = os.path.getsize(outpath) / 1e9
    print(f"\nDone: {nrows:,} records, {size_gb:.2f} GB, {elapsed:.1f}s total")
    print(f"Output: {outpath}")

if __name__ == "__main__":
    main()
