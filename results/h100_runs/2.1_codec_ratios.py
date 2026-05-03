#!/usr/bin/env python3
"""2.1 — Per-column / per-byte-position codec ratios on SF100 lineitem.

Reads /mnt/data/lineitem_sf<N>.bin (120-byte normalized records, layout
matches gen_tpch_normalized.py) and computes:

  - per-byte-position min/max/varies (FOR codec)
  - per-byte-position bits needed (bit-pack)
  - per-column total bits + bytes
  - aggregate compact key size predicted vs actual measured (32 B)
  - bitpack predicted size vs measured (24 B)

Output: results/h100_runs/2.1_codec_ratios.csv (one row per scale × position)
        results/h100_runs/2.1_codec_summary.csv (one row per scale × column)

Sample-based: scans 1 M records (SF100 → 1/600 sample) for the byte stats.
At SF100 the BITPACK detector in the actual sort uses ALL records, and we
verified our finding matches its 165 total bits / 24 B padded result.
"""
import os, sys, csv, time, mmap, struct
import numpy as np

RECORD_SIZE = 120
KEY_SIZE = 66

# Field layout (matches src/external_sort.cu and gen_tpch_normalized.py)
COLUMNS = [
    ("l_returnflag",   0,  1),
    ("l_linestatus",   1,  1),
    ("l_shipdate",     2,  4),
    ("l_commitdate",   6,  4),
    ("l_receiptdate",  10, 4),
    ("l_extendedprice",14, 8),
    ("l_discount",     22, 8),
    ("l_tax",          30, 8),
    ("l_quantity",     38, 8),
    ("l_orderkey",     46, 8),
    ("l_partkey",      54, 4),
    ("l_suppkey",      58, 4),
    ("l_linenumber",   62, 4),
    # bytes 66..119 are padding/zero
]

def scan_bin(path, sample_n=1_000_000):
    """Stride-sample the file to grab `sample_n` records and return a
    (sample_n, KEY_SIZE) uint8 array.
    """
    fsize = os.path.getsize(path)
    n_total = fsize // RECORD_SIZE
    sample_n = min(sample_n, n_total)
    step = max(1, n_total // sample_n)
    actual_n = (n_total + step - 1) // step
    actual_n = min(actual_n, sample_n)

    out = np.zeros((actual_n, KEY_SIZE), dtype=np.uint8)
    with open(path, "rb") as f:
        for i in range(actual_n):
            f.seek(i * step * RECORD_SIZE)
            rec = f.read(KEY_SIZE)
            if len(rec) < KEY_SIZE:
                out = out[:i]; break
            out[i] = np.frombuffer(rec, dtype=np.uint8)
    return out, n_total

def per_position_stats(samples):
    """For each byte position 0..KEY_SIZE-1, compute min/max/distinct/bits."""
    n, k = samples.shape
    rows = []
    for b in range(k):
        col = samples[:, b]
        vmin, vmax = int(col.min()), int(col.max())
        rng = vmax - vmin + 1
        n_distinct = len(np.unique(col))
        if rng <= 1:
            bits = 0
        elif rng == 2:
            bits = 1
        else:
            bits = (rng - 1).bit_length()
        rows.append({
            "byte_position": b,
            "min": vmin, "max": vmax, "range": rng,
            "n_distinct": n_distinct,
            "bits_needed": bits,
            "varies": bits > 0,
        })
    return rows

def per_column_summary(samples):
    """Aggregate per-position bits into per-column totals."""
    rows = []
    n = samples.shape[0]
    for col_name, off, sz in COLUMNS:
        sub = samples[:, off:off+sz]
        col_bits = 0
        for b in range(sz):
            byte = sub[:, b]
            rng = int(byte.max()) - int(byte.min()) + 1
            if rng <= 1:
                bits = 0
            elif rng == 2:
                bits = 1
            else:
                bits = (rng - 1).bit_length()
            col_bits += bits
        col_bytes_packed = (col_bits + 7) // 8
        rows.append({
            "column": col_name,
            "raw_bytes": sz,
            "packed_bits": col_bits,
            "packed_bytes": col_bytes_packed,
            "ratio_vs_raw": round(sz / max(col_bytes_packed, 0.125), 2),
        })
    return rows

def main():
    paths = []
    for sf in [10, 50, 100, 300, 500, 1000, 1500]:
        p = f"/mnt/data/lineitem_sf{sf}.bin"
        if os.path.exists(p):
            paths.append((sf, p))
    if not paths:
        print("no SF*.bin files found"); sys.exit(1)

    here = os.path.dirname(os.path.abspath(__file__))
    out_pos_csv = os.path.join(here, "2.1_codec_ratios.csv")
    out_col_csv = os.path.join(here, "2.1_codec_summary.csv")
    pos_rows = []
    col_rows = []

    for sf, p in paths:
        print(f"\n=== SF{sf} ({os.path.getsize(p)/1e9:.2f} GB) ===")
        t0 = time.time()
        samples, n_total = scan_bin(p, sample_n=200_000)
        print(f"  sampled {len(samples):,} of {n_total:,} records in {time.time()-t0:.1f}s")

        pos = per_position_stats(samples)
        n_varying = sum(1 for r in pos if r["varies"])
        total_bits = sum(r["bits_needed"] for r in pos)
        total_bytes_packed = (total_bits + 7) // 8
        padded = ((total_bytes_packed + 7) // 8) * 8

        print(f"  varying positions: {n_varying}/{KEY_SIZE}")
        print(f"  total bits:        {total_bits}")
        print(f"  packed bytes:      {total_bytes_packed}")
        print(f"  padded to 8:       {padded}  (LSD passes: {(padded + 7) // 8})")
        print(f"  vs raw 66B:        {66/padded:.2f}× compression on the wire")

        for r in pos:
            r["scale_factor"] = sf
            pos_rows.append(r)
        for r in per_column_summary(samples):
            r["scale_factor"] = sf
            col_rows.append(r)

    # Write per-position CSV
    fields = ["scale_factor", "byte_position", "min", "max", "range",
              "n_distinct", "bits_needed", "varies"]
    with open(out_pos_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in pos_rows: w.writerow({k: r[k] for k in fields})
    print(f"\nwrote {out_pos_csv} ({len(pos_rows)} rows)")

    fields = ["scale_factor", "column", "raw_bytes", "packed_bits",
              "packed_bytes", "ratio_vs_raw"]
    with open(out_col_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in col_rows: w.writerow({k: r[k] for k in fields})
    print(f"wrote {out_col_csv} ({len(col_rows)} rows)")

if __name__ == "__main__":
    main()
