#!/usr/bin/env python3
"""Generate uniform random 120-byte records (66B key + 54B value).

Usage: python3 gen_random_normalized.py NUM_RECORDS [output_file]
  NUM_RECORDS = number of records (e.g., 60000000 for ~SF10 equivalent)
  output_file = defaults to /dev/shm/random_{N}M_normalized.bin

All 66 key bytes are uniformly random — worst case for compact key
compression (all positions vary, all full-range).
"""

import sys
import os
import time
import numpy as np

RECORD_SIZE = 120
KEY_SIZE = 66
VALUE_SIZE = 54

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gen_random_normalized.py NUM_RECORDS [output_file]")
        sys.exit(1)

    nrec = int(sys.argv[1])
    label = f"{nrec // 1_000_000}M" if nrec >= 1_000_000 else str(nrec)
    outpath = sys.argv[2] if len(sys.argv) > 2 else f"/dev/shm/random_{label}_normalized.bin"

    size_gb = nrec * RECORD_SIZE / 1e9
    print(f"Generating {nrec:,} random records ({size_gb:.2f} GB)...")

    CHUNK = 5_000_000
    rng = np.random.default_rng(seed=42)

    t0 = time.time()
    with open(outpath, 'wb') as f:
        written = 0
        while written < nrec:
            n = min(CHUNK, nrec - written)
            # Random keys (66B) + zero values (54B)
            keys = rng.integers(0, 256, size=(n, KEY_SIZE), dtype=np.uint8)
            values = np.zeros((n, VALUE_SIZE), dtype=np.uint8)
            records = np.hstack([keys, values])
            f.write(records.tobytes())
            written += n
            if written % 10_000_000 == 0 or written == nrec:
                elapsed = time.time() - t0
                rate = written / elapsed
                eta = (nrec - written) / rate if rate > 0 else 0
                print(f"  {written/1e6:.0f}M / {nrec/1e6:.0f}M "
                      f"({written*RECORD_SIZE/1e9:.1f} GB) "
                      f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    actual_gb = os.path.getsize(outpath) / 1e9
    print(f"\nDone: {nrec:,} records, {actual_gb:.2f} GB, {elapsed:.1f}s")
    print(f"Output: {outpath}")

if __name__ == "__main__":
    main()
