#!/usr/bin/env python3
"""A3: Codec compression ratios on TPC-H sort keys (no GPU work).

Codecs evaluated:
- FOR (frame of reference): subtract min, store as minimal-width integer
- FOR + bit-packing: same but pack bits tightly
- Dictionary encoding (full alphabet)
- Mostly-order-preserving dictionary (top-N then escape)

Output: results/overnight/a3_codec_ratios.csv
"""
import os, sys, struct, time, math
import numpy as np

try:
    import duckdb
except ImportError:
    print("pip install duckdb"); sys.exit(1)

OUT = "results/overnight/a3_codec_ratios.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Sort columns with their types
SORT_COLS = [
    ("l_returnflag",   "char",    1),
    ("l_linestatus",   "char",    1),
    ("l_shipdate",     "date",    4),
    ("l_commitdate",   "date",    4),
    ("l_receiptdate",  "date",    4),
    ("l_extendedprice","decimal", 8),
    ("l_discount",     "decimal", 8),
    ("l_tax",          "decimal", 8),
    ("l_quantity",     "decimal", 8),
    ("l_orderkey",     "bigint",  8),
    ("l_partkey",      "int",     4),
    ("l_suppkey",      "int",     4),
    ("l_linenumber",   "int",     4),
]

def to_uint_array(df, col_name, col_type):
    """Convert column to unsigned integer array for codec evaluation."""
    vals = df[col_name].values
    n = len(vals)
    if col_type == "char":
        return np.array([ord(str(v)[0]) if v else 0 for v in vals], dtype=np.uint64)
    elif col_type == "date":
        import pandas as pd
        ts = pd.to_datetime(vals)
        epoch = pd.Timestamp('1970-01-01')
        return ((ts - epoch).days).values.astype(np.uint64)
    elif col_type == "decimal":
        raw = (np.asarray(vals, dtype=np.float64) * 100).round().astype(np.int64)
        flipped = raw.view(np.uint64) ^ np.uint64(0x8000000000000000)
        return flipped
    elif col_type == "bigint":
        return np.array(vals, dtype=np.uint64)
    elif col_type == "int":
        return np.array(vals, dtype=np.uint64)

def codec_for(values):
    """FOR: subtract min, report bytes needed for the range."""
    vmin = values.min()
    vmax = values.max()
    range_val = int(vmax) - int(vmin)
    if range_val == 0:
        bits_needed = 0
    else:
        bits_needed = int(math.ceil(math.log2(range_val + 1)))
    bytes_needed = max(1, (bits_needed + 7) // 8)
    return bytes_needed, bits_needed

def codec_for_bitpack(values):
    """FOR + bit-packing: same as FOR but report exact bits."""
    vmin = values.min()
    range_val = int(values.max()) - int(vmin)
    if range_val == 0:
        bits_needed = 0
    else:
        bits_needed = int(math.ceil(math.log2(range_val + 1)))
    # Bit-packed size = ceil(n * bits / 8)
    bytes_per_value = bits_needed / 8.0
    return bytes_per_value, bits_needed

def codec_dictionary(values):
    """Dictionary encoding: map each unique value to a code."""
    n_distinct = len(np.unique(values))
    if n_distinct <= 1:
        bits_needed = 0
    else:
        bits_needed = int(math.ceil(math.log2(n_distinct)))
    bytes_needed = max(1, (bits_needed + 7) // 8)
    return bytes_needed, bits_needed, n_distinct

def codec_mostly_op_dict(values, top_n=254):
    """Mostly-order-preserving dictionary: top-N frequent values get 1-byte codes,
    rest get escape + original value."""
    unique, counts = np.unique(values, return_counts=True)
    n_distinct = len(unique)

    if n_distinct <= top_n:
        # All fit in dictionary — same as regular dict
        bits = int(math.ceil(math.log2(max(n_distinct, 2))))
        return max(1, (bits + 7) // 8), n_distinct, 0, 1.0

    # Top-N by frequency
    top_idx = np.argsort(-counts)[:top_n]
    top_set = set(unique[top_idx])
    n = len(values)
    n_escape = sum(1 for v in values if v not in top_set)
    coverage = 1 - n_escape / n

    # Size: 1 byte per dictionary hit, (1 + orig_bytes) per escape
    orig_bytes = 8  # worst case uint64
    avg_bytes = coverage * 1.0 + (1 - coverage) * (1 + orig_bytes)
    return avg_bytes, n_distinct, n_escape, coverage

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

        # Sample for analysis (full scan too slow for SF100)
        sample_size = min(10_000_000, 60_000_000 * sf // 10)
        total_rows = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
        col_list = ", ".join(c[0] for c in SORT_COLS)

        if total_rows > sample_size:
            print(f"  Sampling {sample_size:,} of {total_rows:,} rows")
            df = con.execute(f"SELECT {col_list} FROM lineitem USING SAMPLE {sample_size}").fetchdf()
        else:
            df = con.execute(f"SELECT {col_list} FROM lineitem").fetchdf()

        n = len(df)
        print(f"  {n:,} rows loaded in {time.time()-t0:.1f}s")

        for col_name, col_type, raw_bytes in SORT_COLS:
            t1 = time.time()
            values = to_uint_array(df, col_name, col_type)

            # FOR
            for_bytes, for_bits = codec_for(values)
            results.append({
                'scale_factor': sf, 'column': col_name, 'codec': 'FOR',
                'raw_bytes': raw_bytes, 'compressed_bytes': for_bytes,
                'ratio': round(raw_bytes / max(for_bytes, 0.125), 2),
                'bits_per_value': for_bits, 'n_distinct': int(len(np.unique(values))),
                'notes': f'range={int(values.max())-int(values.min())}'
            })

            # FOR + bit-pack
            bp_bytes, bp_bits = codec_for_bitpack(values)
            results.append({
                'scale_factor': sf, 'column': col_name, 'codec': 'FOR+bitpack',
                'raw_bytes': raw_bytes, 'compressed_bytes': round(bp_bytes, 3),
                'ratio': round(raw_bytes / max(bp_bytes, 0.01), 2),
                'bits_per_value': bp_bits, 'n_distinct': int(len(np.unique(values))),
                'notes': ''
            })

            # Dictionary
            dict_bytes, dict_bits, n_dict = codec_dictionary(values)
            results.append({
                'scale_factor': sf, 'column': col_name, 'codec': 'dictionary',
                'raw_bytes': raw_bytes, 'compressed_bytes': dict_bytes,
                'ratio': round(raw_bytes / max(dict_bytes, 0.125), 2),
                'bits_per_value': dict_bits, 'n_distinct': n_dict,
                'notes': f'{n_dict} distinct values'
            })

            # Mostly-order-preserving dictionary
            mop_bytes, mop_distinct, mop_escape, mop_coverage = codec_mostly_op_dict(values)
            results.append({
                'scale_factor': sf, 'column': col_name, 'codec': 'mostly_op_dict',
                'raw_bytes': raw_bytes, 'compressed_bytes': round(mop_bytes, 3),
                'ratio': round(raw_bytes / max(mop_bytes, 0.01), 2),
                'bits_per_value': round(mop_bytes * 8, 1), 'n_distinct': mop_distinct,
                'notes': f'coverage={mop_coverage:.1%}, escapes={mop_escape}'
            })

            elapsed = time.time() - t1
            print(f"  {col_name}: FOR={for_bytes}B bitpack={bp_bits}b dict={dict_bytes}B mop={mop_bytes:.1f}B ({elapsed:.1f}s)")

        con.close()

    # Write CSV
    if results:
        import csv
        with open(OUT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {len(results)} rows to {OUT}")

        # Summary: total compressed key size per codec
        print("\n=== Summary: total key bytes by codec ===")
        for sf in [10, 50, 100]:
            sf_rows = [r for r in results if r['scale_factor'] == sf]
            if not sf_rows:
                continue
            print(f"  SF{sf}:")
            for codec in ['FOR', 'FOR+bitpack', 'dictionary', 'mostly_op_dict']:
                codec_rows = [r for r in sf_rows if r['codec'] == codec]
                total = sum(float(r['compressed_bytes']) for r in codec_rows)
                raw = sum(r['raw_bytes'] for r in codec_rows)
                print(f"    {codec:20s}: {total:.1f}B / {raw}B = {raw/total:.1f}x compression")

if __name__ == '__main__':
    main()
