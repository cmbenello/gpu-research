#!/usr/bin/env python3
"""Fast TPC-H lineitem normalized binary generator.

Uses tpchgen-cli (Rust, 10-17x faster than DuckDB dbgen) to generate parquet,
then encodes into our 120-byte normalized record format with vectorized numpy.

Falls back to gpu_crocsort/gen_tpch_normalized.py (slower DuckDB dbgen) if
tpchgen-cli or polars isn't available, or if the fast path errors out.

Usage: python3 gen_tpch_fast.py SF [output_path]
"""
import os, sys, struct, time, subprocess, shutil, datetime
import numpy as np

KEY_SIZE   = 88
VALUE_SIZE = 32
RECORD_SIZE = KEY_SIZE + VALUE_SIZE   # 120

# TPC-H date columns are days since 1970-01-01.
EPOCH = datetime.date(1970, 1, 1)
# Biases must match gpu_crocsort/gen_tpch_normalized.py:
DATE_BIAS    = 1 << 31   # 2**31, makes uint32 byte compare match signed-day order
DECIMAL_BIAS = 1 << 62   # 2**62, biased int64 packed as BE — high bit stays clear

def have_tpchgen():
    return shutil.which("tpchgen-cli") is not None

def gen_via_tpchgen(sf, work_dir):
    """Run tpchgen-cli to produce a single lineitem.parquet."""
    print(f"  [tpchgen-cli] generating SF{sf} lineitem...", flush=True)
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    subprocess.check_call([
        "tpchgen-cli",
        "--scale-factor", str(sf),
        "--output-dir", work_dir,
        "--format", "parquet",
        "--tables", "lineitem",
    ])
    print(f"  [tpchgen-cli] done in {time.time()-t0:.1f}s", flush=True)

    # tpchgen-cli writes lineitem.parquet by default in single-file mode
    candidate = os.path.join(work_dir, "lineitem.parquet")
    if os.path.exists(candidate):
        return candidate
    # Fallback: pick the first lineitem*.parquet
    cands = [f for f in os.listdir(work_dir) if "lineitem" in f and f.endswith(".parquet")]
    if not cands:
        raise RuntimeError(f"tpchgen-cli produced no lineitem parquet in {work_dir}")
    return os.path.join(work_dir, cands[0])

def encode_chunk(df_chunk, output_buf, byte_offset):
    """Encode a polars DataFrame chunk into the 120-byte normalized format.

    Field layout (matches gpu_crocsort/gen_tpch_normalized.py):
      0      l_returnflag    (1 byte ASCII)
      1      l_linestatus    (1 byte ASCII)
      2:6    l_shipdate      (4 bytes BE uint32, days since 1970-01-01)
      6:10   l_commitdate    (4 bytes BE uint32)
      10:14  l_receiptdate   (4 bytes BE uint32)
      14:22  l_extendedprice (8 bytes BE int64+bias, value * 100)
      22:30  l_discount      (8 bytes BE int64+bias, value * 100)
      30:38  l_tax           (8 bytes BE int64+bias, value * 100)
      38:46  l_quantity      (8 bytes BE int64+bias)
      46:54  l_orderkey      (8 bytes BE uint64)
      54:58  l_partkey       (4 bytes BE uint32)
      58:62  l_suppkey       (4 bytes BE uint32)
      62:66  l_linenumber    (4 bytes BE uint32)
      66:120 padding (zeros)
    """
    import polars as pl

    n = len(df_chunk)
    out = np.frombuffer(memoryview(output_buf)[byte_offset : byte_offset + n * RECORD_SIZE],
                        dtype=np.uint8).reshape(n, RECORD_SIZE)
    out.fill(0)   # padding bytes 66:120 stay zero

    # --- chars (returnflag, linestatus) ---
    rf = df_chunk["l_returnflag"].to_numpy()
    ls = df_chunk["l_linestatus"].to_numpy()
    # to_numpy() may give object array of str; coerce to bytes
    if rf.dtype == object:
        rf = np.array([s.encode("ascii")[0] for s in rf], dtype=np.uint8)
    else:
        rf = rf.astype("S1").view(np.uint8)
    if ls.dtype == object:
        ls = np.array([s.encode("ascii")[0] for s in ls], dtype=np.uint8)
    else:
        ls = ls.astype("S1").view(np.uint8)
    out[:, 0] = rf
    out[:, 1] = ls

    # --- dates: days_since_epoch + 2**31 packed as big-endian uint32 ---
    for col, off in [("l_shipdate", 2), ("l_commitdate", 6), ("l_receiptdate", 10)]:
        date_vals = df_chunk[col].to_numpy()
        if date_vals.dtype == object:
            days = np.array([(d - EPOCH).days for d in date_vals], dtype=np.int64)
        else:
            days = ((date_vals.astype("datetime64[D]")
                       - np.datetime64("1970-01-01", "D"))
                    .astype(np.int64))
        biased = (days + DATE_BIAS).astype(">u4")
        out[:, off:off+4] = biased.view(np.uint8).reshape(n, 4)

    # --- decimals: (value * scale) + 2**62 packed as big-endian int64 ---
    # Reference uses struct.pack_into('>q', ...) so we keep signed packing.
    for col, off, scale in [
        ("l_extendedprice", 14, 100),
        ("l_discount",      22, 100),
        ("l_tax",           30, 100),
        ("l_quantity",      38, 100),  # match reference: round(qty * 100) — TPC-H has decimal qty
    ]:
        v = df_chunk[col].cast(pl.Float64).to_numpy()
        v_int = (v * scale).round().astype(np.int64)
        biased = (v_int + DECIMAL_BIAS).astype(">i8")  # signed BE
        out[:, off:off+8] = biased.view(np.uint8).reshape(n, 8)

    # --- integer IDs ---
    for col, off, dtype, sz in [
        ("l_orderkey",   46, ">u8", 8),
        ("l_partkey",    54, ">u4", 4),
        ("l_suppkey",    58, ">u4", 4),
        ("l_linenumber", 62, ">u4", 4),
    ]:
        v = df_chunk[col].cast(pl.Int64).to_numpy().astype(dtype)
        out[:, off:off+sz] = v.view(np.uint8).reshape(n, sz)

def gen_via_duckdb_fallback(sf, output_path):
    """Try a few well-known locations for the slow gen_tpch_normalized.py."""
    candidates = [
        os.path.join(os.environ.get("WORK_DIR", ""), "gpu_crocsort/gen_tpch_normalized.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "gpu_crocsort", "gen_tpch_normalized.py"),
        os.path.expanduser("~/gpu-research/gpu_crocsort/gen_tpch_normalized.py"),
        "/mnt/nvme1/cmbenello/gpu-research/gpu_crocsort/gen_tpch_normalized.py",
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"  [duckdb-fallback] running {c} {sf} {output_path}", flush=True)
            subprocess.check_call(["python3", c, str(sf), output_path])
            return
    raise FileNotFoundError(f"No gen_tpch_normalized.py found. Tried: {candidates}")

def main():
    if len(sys.argv) < 2:
        print("usage: gen_tpch_fast.py SF [output_path]")
        sys.exit(1)
    sf = int(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/lineitem_sf{sf}.bin"

    if not have_tpchgen():
        print("[!] tpchgen-cli not on PATH, using duckdb fallback")
        gen_via_duckdb_fallback(sf, output_path)
        return

    try:
        import polars as pl
    except ImportError:
        print("[!] polars not installed, using duckdb fallback")
        gen_via_duckdb_fallback(sf, output_path)
        return

    work_dir = output_path + ".tmp_parquet"
    try:
        parquet_path = gen_via_tpchgen(sf, work_dir)
    except subprocess.CalledProcessError as e:
        print(f"[!] tpchgen-cli failed ({e}); duckdb fallback")
        shutil.rmtree(work_dir, ignore_errors=True)
        gen_via_duckdb_fallback(sf, output_path)
        return

    try:
        print(f"  [encode] reading parquet + encoding to {output_path}", flush=True)
        t0 = time.time()
        df = pl.read_parquet(parquet_path)
        n = len(df)
        gb = n * RECORD_SIZE / 1e9
        print(f"  [encode] {n:,} rows, allocating {gb:.2f} GB output buffer", flush=True)

        # Pre-allocate the entire output buffer (mmap-backed if disk file)
        with open(output_path, "wb") as f:
            f.truncate(n * RECORD_SIZE)

        # Encode in chunks for memory friendliness; mmap the output for in-place writes
        import mmap
        with open(output_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            CHUNK = 5_000_000
            last_print = 0
            for off_rows in range(0, n, CHUNK):
                end = min(off_rows + CHUNK, n)
                chunk = df.slice(off_rows, end - off_rows)
                encode_chunk(chunk, mm, off_rows * RECORD_SIZE)
                pct = off_rows / n * 100
                if pct - last_print >= 10:
                    print(f"    {pct:5.1f}% ({off_rows:,}/{n:,})", flush=True)
                    last_print = pct
            mm.flush(); mm.close()

        elapsed = time.time() - t0
        print(f"  Done: {n:,} records, {gb:.2f} GB, {elapsed:.1f}s "
              f"({gb/elapsed:.2f} GB/s encode rate)", flush=True)
    except Exception as e:
        print(f"[!] fast encode failed ({type(e).__name__}: {e}); duckdb fallback")
        gen_via_duckdb_fallback(sf, output_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
