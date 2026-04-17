#!/usr/bin/env python3
"""Generate 120-byte records with controlled key duplication patterns.

Usage: python3 gen_duplicate_keys.py MODE NUM_RECORDS [OUTPUT_FILE]

Modes:
  pool1000  Sample keys from 1000 unique random 66B keys (avg group = N/1000)
  pool10    Sample keys from 10 unique random 66B keys (avg group = N/10)
  zipf      Zipfian distribution (a=2.0) over 10000 unique random 66B keys

Each record: 66B key (from pool) + 54B zero value = 120B.
Default output: /dev/shm/dupkeys_{MODE}_{N}M_normalized.bin

Uses numpy, seed=42. Processes in 5M-record chunks for memory efficiency.
"""

import sys
import os
import time
import numpy as np

RECORD_SIZE = 120
KEY_SIZE = 66
VALUE_SIZE = 54

MODES = {
    "pool1000": {"pool_size": 1000,  "dist": "uniform"},
    "pool10":   {"pool_size": 10,    "dist": "uniform"},
    "zipf":     {"pool_size": 10000, "dist": "zipf"},
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 gen_duplicate_keys.py MODE NUM_RECORDS [OUTPUT_FILE]")
        print(f"  MODE = {', '.join(MODES.keys())}")
        sys.exit(1)

    mode = sys.argv[1]
    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES.keys())}")
        sys.exit(1)

    nrec = int(sys.argv[2])
    label = f"{nrec // 1_000_000}M" if nrec >= 1_000_000 else str(nrec)
    outpath = (sys.argv[3] if len(sys.argv) > 3
               else f"/dev/shm/dupkeys_{mode}_{label}_normalized.bin")

    cfg = MODES[mode]
    pool_size = cfg["pool_size"]
    dist = cfg["dist"]
    avg_group = nrec / pool_size

    size_gb = nrec * RECORD_SIZE / 1e9
    print(f"Mode: {mode} (pool={pool_size}, dist={dist}, avg group={avg_group:.0f})")
    print(f"Generating {nrec:,} records ({size_gb:.2f} GB)...")

    rng = np.random.default_rng(seed=42)

    # Generate the key pool
    print(f"  Creating pool of {pool_size} unique 66B keys...")
    key_pool = rng.integers(0, 256, size=(pool_size, KEY_SIZE), dtype=np.uint8)

    CHUNK = 5_000_000
    t0 = time.time()

    with open(outpath, 'wb') as f:
        written = 0
        while written < nrec:
            n = min(CHUNK, nrec - written)

            # Pick key indices according to distribution
            if dist == "uniform":
                indices = rng.integers(0, pool_size, size=n)
            else:  # zipf
                raw = rng.zipf(a=2.0, size=n)
                indices = (raw - 1) % pool_size

            # Build records: selected keys + zero values
            keys = key_pool[indices]  # (n, 66)
            values = np.zeros((n, VALUE_SIZE), dtype=np.uint8)
            records = np.hstack([keys, values])
            f.write(records.tobytes())

            written += n
            if written % 5_000_000 == 0 or written == nrec:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (nrec - written) / rate if rate > 0 else 0
                print(f"  {written/1e6:.0f}M / {nrec/1e6:.0f}M "
                      f"({written*RECORD_SIZE/1e9:.1f} GB) "
                      f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    actual_gb = os.path.getsize(outpath) / 1e9
    print(f"\nDone: {nrec:,} records, {actual_gb:.2f} GB, {elapsed:.1f}s")
    print(f"Output: {outpath}")

    # Print distribution stats
    print(f"\nDistribution summary:")
    print(f"  Pool size: {pool_size} unique keys")
    print(f"  Avg duplicates per key: {nrec / pool_size:.0f}")
    if dist == "zipf":
        # Quick sample to show skew
        sample = (rng.zipf(a=2.0, size=min(nrec, 1_000_000)) - 1) % pool_size
        unique, counts = np.unique(sample, return_counts=True)
        top5 = counts[np.argsort(-counts)][:5]
        print(f"  Zipf skew (sample 1M): top-5 key counts = {top5.tolist()}")

if __name__ == "__main__":
    main()
