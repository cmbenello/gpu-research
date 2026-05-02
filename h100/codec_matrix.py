#!/usr/bin/env python3
"""Tier 8.1 codec matrix: per-column compression ratios for every codec we
support, on TPC-H lineitem at multiple scale factors. Output: heatmap CSV +
narrow-form per-codec ratios.

Reuses scripts/a3_codec_ratios.py but extends to dictionary, RLE, and the
hybrid (auto-pick best codec per column).
"""
import argparse, csv, math, os, sys, time
from collections import Counter
import numpy as np

try:
    import duckdb
except ImportError:
    print("pip install duckdb"); sys.exit(1)

# Sort key columns and types
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
    ("l_orderkey",     "uint64",  8),
    ("l_partkey",      "uint32",  4),
    ("l_suppkey",      "uint32",  4),
    ("l_linenumber",   "uint32",  4),
]

def codec_FOR(values, raw_bytes):
    """Frame of reference: subtract min, store as minimum-byte-width unsigned."""
    if len(values) == 0:
        return raw_bytes, 0
    vmin, vmax = int(values.min()), int(values.max())
    rng = vmax - vmin + 1
    bits = max(1, math.ceil(math.log2(max(2, rng))))
    bytes_per_val = math.ceil(bits / 8)
    return bytes_per_val, bits

def codec_FOR_bitpack(values, raw_bytes):
    """FOR + sub-byte bit-packing. Returns fractional bytes per value."""
    if len(values) == 0:
        return float(raw_bytes), 0
    vmin, vmax = int(values.min()), int(values.max())
    rng = vmax - vmin + 1
    bits = max(1, math.ceil(math.log2(max(2, rng))))
    return bits / 8.0, bits

def codec_dictionary(values, raw_bytes):
    """Full dictionary: log2(n_distinct) bits per code."""
    n_distinct = len(set(values.tolist())) if len(values) <= 10_000_000 else len(np.unique(values))
    if n_distinct <= 1:
        return 0.001, 0
    bits = max(1, math.ceil(math.log2(n_distinct)))
    return bits / 8.0, bits

def codec_dict_top256(values, raw_bytes):
    """Top-256 dictionary: 1 byte for codes, escape for the rest.
    Effective bytes = 1 + (1 - coverage) * raw_bytes."""
    if len(values) == 0:
        return float(raw_bytes), 0
    counts = Counter(values.tolist() if len(values) <= 10_000_000 else
                     np.unique(values, return_counts=True))
    if isinstance(counts, Counter):
        most = counts.most_common(256)
        covered = sum(c for _, c in most)
        coverage = covered / len(values)
    else:
        coverage = 0.5  # rough estimate when can't run Counter
    return 1.0 + (1 - coverage) * raw_bytes, 8

def codec_RLE(values, raw_bytes):
    """RLE = run-length encoding. Bytes = (raw_bytes + 4) * n_runs / N."""
    if len(values) == 0:
        return float(raw_bytes), 0
    vals = values if isinstance(values, np.ndarray) else np.asarray(values)
    runs = 1 + int(np.sum(vals[1:] != vals[:-1]))
    avg_run = len(vals) / runs
    bytes_per_val = (raw_bytes + 4) / avg_run
    return bytes_per_val, raw_bytes * 8

CODECS = {
    "raw":           lambda v, b: (float(b), b * 8),
    "FOR":           codec_FOR,
    "FOR_bitpack":   codec_FOR_bitpack,
    "dict_full":     codec_dictionary,
    "dict_top256":   codec_dict_top256,
    "RLE":           codec_RLE,
}

def column_to_int_array(con, col, dtype):
    """Pull column into an int numpy array suitable for codec analysis."""
    if dtype == "char":
        # Already 1 byte
        return np.frombuffer(con.execute(
            f"SELECT chr(ord({col})) FROM lineitem"
        ).fetchnumpy()[col].astype("S1").tobytes(), dtype=np.uint8)
    if dtype == "date":
        return con.execute(
            f"SELECT date_diff('day', DATE '1970-01-01', {col}) AS d FROM lineitem"
        ).fetchnumpy()["d"].astype(np.int64)
    if dtype == "decimal":
        return (con.execute(
            f"SELECT CAST(CAST({col} AS DOUBLE) * 100 AS BIGINT) AS v FROM lineitem"
        ).fetchnumpy()["v"]).astype(np.int64)
    return con.execute(f"SELECT {col} FROM lineitem"
                       ).fetchnumpy()[col].astype(np.int64)

def run_for_scale(sf, con, sample_n, writer):
    print(f"\n=== SF{sf} ===")
    con.execute(f"INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")
    n = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
    sampled = sample_n and n > sample_n
    if sampled:
        con.execute(f"CREATE OR REPLACE TABLE lineitem AS "
                    f"SELECT * FROM lineitem USING SAMPLE {sample_n};")
        n = sample_n
    print(f"  rows: {n:,}{' (sampled)' if sampled else ''}")

    for col, dtype, raw in SORT_COLS:
        try:
            vals = column_to_int_array(con, col, dtype)
        except Exception as e:
            print(f"    [skip] {col}: {e}")
            continue
        for codec_name, codec_fn in CODECS.items():
            t0 = time.time()
            try:
                comp_bytes, bits = codec_fn(vals, raw)
            except Exception as e:
                print(f"    [{codec_name}] {col}: error {e}")
                continue
            ratio = raw / comp_bytes if comp_bytes > 0 else float("inf")
            writer.writerow({
                "scale_factor": sf, "column": col, "dtype": dtype,
                "codec": codec_name, "raw_bytes": raw,
                "compressed_bytes": round(comp_bytes, 4),
                "ratio": round(ratio, 3), "bits": round(bits, 1),
                "n_rows": n, "elapsed_s": round(time.time() - t0, 3),
            })
            print(f"    {col:18s} {codec_name:14s} {raw}B → {comp_bytes:6.2f}B  ({ratio:5.2f}×)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="20,50",
                    help="comma-separated TPC-H scale factors")
    ap.add_argument("--sample-n", type=int, default=10_000_000,
                    help="sample to this many rows when SF is huge (0 = full)")
    ap.add_argument("--out", default="results/overnight_pulled/codec_matrix.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["scale_factor", "column", "dtype", "codec",
              "raw_bytes", "compressed_bytes", "ratio", "bits",
              "n_rows", "elapsed_s"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sf in [int(x) for x in args.scales.split(",")]:
            con = duckdb.connect()
            run_for_scale(sf, con, args.sample_n, w)
            con.close()
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
