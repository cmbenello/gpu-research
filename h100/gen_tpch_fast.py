#!/usr/bin/env python3
"""Fast TPC-H lineitem normalized binary generator.

Uses `tpchgen-cli` (Rust, 10-17x faster than DuckDB dbgen) to generate parquet,
then converts to our 120-byte normalized record format.

Falls back to duckdb dbgen if tpchgen-cli isn't installed.

Usage: python3 h100/gen_tpch_fast.py SF [output_path]
"""
import os, sys, struct, time, subprocess, shutil
import numpy as np

KEY_SIZE = 88
VALUE_SIZE = 32
RECORD_SIZE = KEY_SIZE + VALUE_SIZE  # 120

# Date epoch for TPC-H date columns (1970-01-01 = day 0)
EPOCH = "1970-01-01"

def have_tpchgen():
    return shutil.which("tpchgen-cli") is not None

def gen_via_tpchgen(sf, work_dir):
    """Run tpchgen-cli to produce parquet for lineitem only."""
    print(f"  [tpchgen-cli] generating SF{sf} lineitem...")
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    # tpchgen-cli writes one parquet file per table; we only want lineitem
    subprocess.check_call([
        "tpchgen-cli",
        "--scale-factor", str(sf),
        "--output-dir", work_dir,
        "--format", "parquet",
        "--tables", "lineitem",
    ])
    parquet_path = os.path.join(work_dir, "lineitem.parquet")
    if not os.path.exists(parquet_path):
        # Some versions write per-partition parquets
        cands = [f for f in os.listdir(work_dir) if "lineitem" in f and f.endswith(".parquet")]
        if cands:
            parquet_path = os.path.join(work_dir, cands[0])
    print(f"  [tpchgen-cli] done in {time.time()-t0:.1f}s")
    return parquet_path

def encode_lineitem_chunk(df, output_buf, offset):
    """Encode a polars/pyarrow chunk into the 120-byte normalized format.
    Field layout matches gen_tpch_normalized.py."""
    n = len(df)
    # Vectorized encoding — much faster than Python loops
    out = output_buf[offset:offset + n * RECORD_SIZE]
    out_view = np.frombuffer(out, dtype=np.uint8).reshape(n, RECORD_SIZE)

    # Sort key bytes 0-65 (88 actually but compact 66 used)
    rf = np.frombuffer(df["l_returnflag"].cast(str).to_numpy().astype("S1").tobytes(), dtype=np.uint8)
    ls = np.frombuffer(df["l_linestatus"].cast(str).to_numpy().astype("S1").tobytes(), dtype=np.uint8)
    out_view[:, 0] = rf
    out_view[:, 1] = ls

    # Dates as days-since-epoch big-endian uint32 (4 bytes each)
    for col, off in [("l_shipdate", 2), ("l_commitdate", 6), ("l_receiptdate", 10)]:
        days = (df[col].cast("date").dt.days_since_epoch()).to_numpy().astype(">u4")
        out_view[:, off:off+4] = days.view(np.uint8).reshape(n, 4)

    # Decimal columns: extendedprice (8B), discount (8B), tax (8B), quantity (8B)
    OFF_BIAS = 1 << 63
    for col, off, scale in [("l_extendedprice", 14, 100),
                            ("l_discount",      22, 100),
                            ("l_tax",           30, 100),
                            ("l_quantity",      38, 1)]:
        v = (df[col].cast("float64") * scale).to_numpy().astype(np.int64)
        v_unsigned = (v.astype(np.uint64) + OFF_BIAS).astype(">u8")
        out_view[:, off:off+8] = v_unsigned.view(np.uint8).reshape(n, 8)

    # IDs
    for col, off, dtype in [("l_orderkey",   46, ">u8"),
                            ("l_partkey",    54, ">u4"),
                            ("l_suppkey",    58, ">u4"),
                            ("l_linenumber", 62, ">u4")]:
        v = df[col].cast(int).to_numpy().astype(dtype)
        sz = 8 if "8" in dtype else 4
        out_view[:, off:off+sz] = v.view(np.uint8).reshape(n, sz)

    # Padding bytes 66:120 left as zero (out_view[:, 66:120] = 0; already zero from buffer init)

def gen_via_duckdb(sf, output_path):
    """Fallback: original gen_tpch_normalized.py path."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    script = os.path.join(repo, "gpu_crocsort", "gen_tpch_normalized.py")
    print(f"  [duckdb-fallback] running {script} {sf} {output_path}")
    subprocess.check_call(["python3", script, str(sf), output_path])

def main():
    if len(sys.argv) < 2:
        print("usage: gen_tpch_fast.py SF [output_path]")
        sys.exit(1)
    sf = int(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 \
                  else f"/tmp/lineitem_sf{sf}.bin"

    if not have_tpchgen():
        print("[!] tpchgen-cli not found, falling back to slow duckdb path")
        gen_via_duckdb(sf, output_path)
        return

    # Use tpchgen-cli + polars to encode
    try:
        import polars as pl
    except ImportError:
        print("[!] polars not installed, falling back to duckdb")
        gen_via_duckdb(sf, output_path)
        return

    work_dir = output_path + ".tmp_parquet"
    parquet_path = gen_via_tpchgen(sf, work_dir)

    print(f"  [encode] reading parquet + encoding to {output_path}")
    t0 = time.time()
    df = pl.read_parquet(parquet_path)
    n = len(df)
    print(f"  [encode] {n:,} rows, allocating {n * RECORD_SIZE / 1e9:.2f} GB output")

    output_buf = bytearray(n * RECORD_SIZE)
    # Process in chunks for memory friendliness
    CHUNK = 5_000_000
    for off in range(0, n, CHUNK):
        end = min(off + CHUNK, n)
        chunk = df.slice(off, end - off)
        encode_lineitem_chunk(chunk, output_buf, off * RECORD_SIZE)
        if off % (CHUNK * 4) == 0:
            print(f"    {off/n*100:.1f}%")

    print(f"  [encode] writing {output_path}...")
    with open(output_path, "wb") as f:
        f.write(output_buf)

    # Cleanup parquet
    shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.time() - t0
    gb = n * RECORD_SIZE / 1e9
    print(f"  Done: {n:,} records, {gb:.2f} GB, {elapsed:.1f}s ({gb/elapsed:.2f} GB/s encode rate)")

if __name__ == "__main__":
    main()
