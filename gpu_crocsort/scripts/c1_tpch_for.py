#!/usr/bin/env python3
"""C1: TPC-H ORDER BY with FOR-encoded keys vs baseline.

Approach: generate normalized binary data with FOR-compressed keys,
run external_sort_tpch_compact on both, compare wall times.

FOR encoding: for each varying byte position in the compact key,
subtract the min value. This reduces the value range and often
lets the compact key scan eliminate more positions.

Output: results/overnight/c1_tpch_for.csv
"""
import os, sys, subprocess, struct, time, csv
import numpy as np

try:
    import duckdb
except ImportError:
    print("pip install duckdb"); sys.exit(1)

OUT = "results/overnight/c1_tpch_for.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

BINARY = "./external_sort_tpch_compact"
RUNS = 3
KEY_SIZE = 66
VALUE_SIZE = 54
RECORD_SIZE = KEY_SIZE + VALUE_SIZE  # 120

def generate_normalized_bin(sf, output_path):
    """Generate normalized binary data from DuckDB (same as gen_tpch_normalized.py)."""
    import datetime
    db_path = f"/tmp/tpch_sf{sf}.db"
    if not os.path.exists(db_path):
        return None

    con = duckdb.connect(db_path, read_only=True)
    col_list = ("l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate, "
                "l_extendedprice, l_discount, l_tax, l_quantity, "
                "l_orderkey, l_partkey, l_suppkey, l_linenumber")

    count = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
    print(f"  SF{sf}: {count:,} rows")

    # Stream in chunks to avoid OOM
    chunk_size = 5_000_000
    epoch = datetime.date(1970, 1, 1)

    with open(output_path, 'wb') as f:
        offset = 0
        while offset < count:
            n = min(chunk_size, count - offset)
            df = con.execute(f"SELECT {col_list} FROM lineitem LIMIT {n} OFFSET {offset}").fetchdf()
            offset += n

            for _, row in df.iterrows():
                key = bytearray(KEY_SIZE)
                # l_returnflag (1B)
                key[0] = ord(str(row['l_returnflag'])[0]) if row['l_returnflag'] else 0
                # l_linestatus (1B)
                key[1] = ord(str(row['l_linestatus'])[0]) if row['l_linestatus'] else 0
                # dates (4B each)
                for i, col in enumerate(['l_shipdate', 'l_commitdate', 'l_receiptdate']):
                    d = row[col]
                    if hasattr(d, 'date'):
                        d = d.date()
                    days = (d - epoch).days if d else 0
                    struct.pack_into('>I', key, 2 + i*4, days & 0xFFFFFFFF)
                # decimals (8B each)
                for i, col in enumerate(['l_extendedprice', 'l_discount', 'l_tax', 'l_quantity']):
                    raw = int(round(float(row[col]) * 100)) if row[col] is not None else 0
                    flipped = np.uint64(raw).view(np.uint64) ^ np.uint64(0x8000000000000000)
                    struct.pack_into('>Q', key, 14 + i*8, int(flipped))
                # integers
                struct.pack_into('>Q', key, 46, int(row['l_orderkey']))
                struct.pack_into('>I', key, 54, int(row['l_partkey']))
                struct.pack_into('>I', key, 58, int(row['l_suppkey']))
                struct.pack_into('>I', key, 62, int(row['l_linenumber']))

                value = bytearray(VALUE_SIZE)
                f.write(bytes(key) + bytes(value))

            print(f"    {offset:,}/{count:,} rows written...")

    con.close()
    return count

def generate_for_bin(input_path, output_path):
    """Apply FOR encoding to the key bytes: subtract per-position min."""
    file_size = os.path.getsize(input_path)
    n_records = file_size // RECORD_SIZE
    print(f"  FOR encoding {n_records:,} records...")

    # First pass: find min per key byte position
    mins = np.full(KEY_SIZE, 255, dtype=np.uint8)
    with open(input_path, 'rb') as f:
        chunk = 1_000_000
        for start in range(0, n_records, chunk):
            n = min(chunk, n_records - start)
            data = np.frombuffer(f.read(n * RECORD_SIZE), dtype=np.uint8).reshape(n, RECORD_SIZE)
            keys = data[:, :KEY_SIZE]
            mins = np.minimum(mins, keys.min(axis=0))

    # Count varying positions before and after FOR
    maxs = np.zeros(KEY_SIZE, dtype=np.uint8)
    with open(input_path, 'rb') as f:
        for start in range(0, n_records, 1_000_000):
            n = min(1_000_000, n_records - start)
            data = np.frombuffer(f.read(n * RECORD_SIZE), dtype=np.uint8).reshape(n, RECORD_SIZE)
            keys = data[:, :KEY_SIZE]
            maxs = np.maximum(maxs, keys.max(axis=0))

    raw_varying = sum(1 for i in range(KEY_SIZE) if mins[i] != maxs[i])
    for_range = maxs.astype(int) - mins.astype(int)
    for_varying = sum(1 for i in range(KEY_SIZE) if for_range[i] > 0)

    print(f"  Raw varying: {raw_varying}, FOR varying: {for_varying} (same — FOR doesn't change this)")
    print(f"  Per-position ranges after FOR: {[for_range[i] for i in range(KEY_SIZE) if for_range[i] > 0]}")

    # Second pass: write FOR-encoded data
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        chunk = 1_000_000
        for start in range(0, n_records, chunk):
            n = min(chunk, n_records - start)
            raw = fin.read(n * RECORD_SIZE)
            data = np.frombuffer(raw, dtype=np.uint8).reshape(n, RECORD_SIZE).copy()
            # Subtract min from each key byte position
            data[:, :KEY_SIZE] = data[:, :KEY_SIZE].astype(np.int16) - mins[:KEY_SIZE].astype(np.int16)
            data[:, :KEY_SIZE] = np.clip(data[:, :KEY_SIZE], 0, 255).astype(np.uint8)
            fout.write(data.tobytes())

    return raw_varying, for_varying, n_records

def run_sort(input_path, label, runs=RUNS):
    """Run external_sort_tpch_compact and parse timing."""
    if not os.path.exists(input_path):
        return None
    if not os.path.exists(BINARY):
        print(f"  ERROR: {BINARY} not found")
        return None

    cmd = [BINARY, '--input', input_path, '--runs', str(runs)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Parse wall times from output
    times = []
    for line in result.stdout.split('\n'):
        if 'Wall time:' in line or 'wall_ms' in line.lower():
            # Try to extract ms
            import re
            match = re.search(r'(\d+\.?\d*)\s*ms', line)
            if match:
                times.append(float(match.group(1)) / 1000.0)
            match = re.search(r'(\d+\.?\d*)\s*s', line)
            if match and not times:
                times.append(float(match.group(1)))

    # Also look for "TOTAL" or "Sort completed" lines
    for line in result.stdout.split('\n'):
        if 'Sort completed' in line or 'Total:' in line:
            import re
            match = re.search(r'(\d+\.?\d*)\s*s', line)
            if match:
                times.append(float(match.group(1)))

    print(f"  {label}: stdout extract = {result.stdout[-500:]}")
    if times:
        median = sorted(times)[len(times)//2]
        return median
    return None

def main():
    results = []

    for sf in [10, 50, 100]:
        db_path = f"/tmp/tpch_sf{sf}.db"
        if not os.path.exists(db_path):
            print(f"SKIP SF{sf}: {db_path} not found")
            continue

        print(f"\n=== SF{sf} ===")
        base_bin = f"/tmp/lineitem_sf{sf}_normalized.bin"
        for_bin = f"/tmp/lineitem_sf{sf}_for.bin"

        # Generate baseline if needed
        if not os.path.exists(base_bin):
            print(f"  Generating baseline binary...")
            t0 = time.time()
            n = generate_normalized_bin(sf, base_bin)
            print(f"  Generated in {time.time()-t0:.0f}s")

        # Generate FOR-encoded version
        print(f"  Generating FOR-encoded binary...")
        t0 = time.time()
        raw_vary, for_vary, n_records = generate_for_bin(base_bin, for_bin)
        print(f"  Generated in {time.time()-t0:.0f}s")

        # Run baseline sort
        baseline_s = run_sort(base_bin, "baseline")

        # Run FOR sort
        for_s = run_sort(for_bin, "FOR")

        speedup = baseline_s / for_s if (baseline_s and for_s and for_s > 0) else None

        results.append({
            'scale_factor': sf,
            'n_records': n_records,
            'baseline_s': baseline_s,
            'with_for_s': for_s,
            'speedup': round(speedup, 2) if speedup else None,
            'raw_varying_bytes': raw_vary,
            'for_varying_bytes': for_vary,
            'pcie_bytes_baseline': raw_vary * n_records,
            'pcie_bytes_with_for': for_vary * n_records,
        })

        # Cleanup FOR file to save disk
        os.remove(for_bin)

    # Write CSV
    if results:
        with open(OUT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote to {OUT}")

if __name__ == '__main__':
    main()
