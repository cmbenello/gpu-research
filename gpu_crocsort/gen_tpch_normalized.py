#!/usr/bin/env python3
"""Generate normalized TPC-H lineitem binary data for external sort benchmarks.

Usage: python3 gen_tpch_normalized.py SF [output_file]
  SF = scale factor (10, 50, 100)
  output_file = defaults to /tmp/lineitem_sfN_normalized.bin

Output format: fixed 120-byte records (88B key + 32B value)
Key encodes all 9 sort columns as a normalized binary string where
memcmp gives correct multi-column ascending ordering.

Key layout (88 bytes):
  [0]    l_returnflag  (1 byte, ASCII)
  [1]    l_linestatus  (1 byte, ASCII)
  [2:6]  l_shipdate    (4 bytes, days since epoch, big-endian uint32)
  [6:10] l_commitdate  (4 bytes, days since epoch, big-endian uint32)
  [10:14] l_receiptdate (4 bytes, days since epoch, big-endian uint32)
  [14:22] l_extendedprice (8 bytes, cents as big-endian int64 + offset)
  [22:30] l_discount    (8 bytes, hundredths as big-endian int64 + offset)
  [30:38] l_tax         (8 bytes, hundredths as big-endian int64 + offset)
  [38:46] l_quantity    (8 bytes, big-endian int64 + offset)
  [46:54] l_orderkey    (8 bytes, big-endian uint64)
  [54:58] l_partkey     (4 bytes, big-endian uint32)
  [58:62] l_suppkey     (4 bytes, big-endian uint32)
  [62:66] l_linenumber  (4 bytes, big-endian uint32)
  [66:88] padding       (22 bytes, zero)

Value layout (32 bytes): zero-padded (could store l_comment or shipinstruct)
"""

import sys
import struct
import time
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gen_tpch_normalized.py SF [output_file]")
        sys.exit(1)

    sf = int(sys.argv[1])
    outpath = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/lineitem_sf{sf}_normalized.bin"

    print(f"Generating TPC-H SF{sf} lineitem data...")
    t0 = time.time()

    try:
        import duckdb
    except ImportError:
        print("Installing duckdb...")
        os.system("pip3 install duckdb")
        import duckdb

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")

    # Fetch lineitem with sort columns
    print("Querying lineitem table...")
    result = con.execute("""
        SELECT l_returnflag, l_linestatus,
               l_shipdate, l_commitdate, l_receiptdate,
               l_extendedprice, l_discount, l_tax, l_quantity,
               l_orderkey, l_partkey, l_suppkey, l_linenumber
        FROM lineitem
    """).fetchall()

    nrows = len(result)
    elapsed = time.time() - t0
    print(f"  {nrows:,} rows in {elapsed:.1f}s")

    # Epoch for dates
    from datetime import date
    epoch = date(1970, 1, 1)

    # Normalize to 120-byte records
    print(f"Writing {outpath}...")
    KEY_SIZE = 88
    VALUE_SIZE = 32
    RECORD_SIZE = KEY_SIZE + VALUE_SIZE

    with open(outpath, 'wb') as f:
        buf = bytearray(RECORD_SIZE)
        written = 0
        t1 = time.time()
        for row in result:
            (returnflag, linestatus, shipdate, commitdate, receiptdate,
             extprice, discount, tax, quantity,
             orderkey, partkey, suppkey, linenumber) = row

            # Clear buffer
            for i in range(RECORD_SIZE):
                buf[i] = 0

            # Key: normalize each field to binary where memcmp = correct order
            off = 0

            # [0] returnflag: 1 byte ASCII
            buf[off] = ord(returnflag) if isinstance(returnflag, str) else returnflag
            off += 1

            # [1] linestatus: 1 byte ASCII
            buf[off] = ord(linestatus) if isinstance(linestatus, str) else linestatus
            off += 1

            # [2:6] shipdate: days since epoch, big-endian uint32
            days = (shipdate - epoch).days if hasattr(shipdate, 'year') else int(shipdate)
            struct.pack_into('>I', buf, off, days + 2**31)  # offset to make unsigned
            off += 4

            # [6:10] commitdate
            days = (commitdate - epoch).days if hasattr(commitdate, 'year') else int(commitdate)
            struct.pack_into('>I', buf, off, days + 2**31)
            off += 4

            # [10:14] receiptdate
            days = (receiptdate - epoch).days if hasattr(receiptdate, 'year') else int(receiptdate)
            struct.pack_into('>I', buf, off, days + 2**31)
            off += 4

            # [14:22] extendedprice: cents, big-endian int64 + offset
            cents = int(round(float(extprice) * 100))
            struct.pack_into('>q', buf, off, cents + 2**62)
            off += 8

            # [22:30] discount
            hundredths = int(round(float(discount) * 100))
            struct.pack_into('>q', buf, off, hundredths + 2**62)
            off += 8

            # [30:38] tax
            hundredths = int(round(float(tax) * 100))
            struct.pack_into('>q', buf, off, hundredths + 2**62)
            off += 8

            # [38:46] quantity
            qty = int(round(float(quantity) * 100))
            struct.pack_into('>q', buf, off, qty + 2**62)
            off += 8

            # [46:54] orderkey
            struct.pack_into('>Q', buf, off, int(orderkey))
            off += 8

            # [54:58] partkey
            struct.pack_into('>I', buf, off, int(partkey))
            off += 4

            # [58:62] suppkey
            struct.pack_into('>I', buf, off, int(suppkey))
            off += 4

            # [62:66] linenumber
            struct.pack_into('>I', buf, off, int(linenumber))
            off += 4

            # [66:88] padding (already zeroed)
            # [88:120] value (already zeroed)

            f.write(buf)
            written += 1
            if written % 10000000 == 0:
                elapsed = time.time() - t1
                rate = written / elapsed
                eta = (nrows - written) / rate
                print(f"  {written/1e6:.0f}M / {nrows/1e6:.0f}M ({100*written/nrows:.0f}%) "
                      f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    size_gb = os.path.getsize(outpath) / 1e9
    print(f"\nDone: {nrows:,} records, {size_gb:.2f} GB, {elapsed:.1f}s total")
    print(f"Output: {outpath}")

if __name__ == "__main__":
    main()
