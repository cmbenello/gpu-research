#!/usr/bin/env python3
"""Generate 120-byte records with exactly N varying key byte positions.

Usage: python3 gen_varying_bytes.py N_VARY NUM_RECORDS [output_file]
  N_VARY     = number of key byte positions that vary (0..66)
  NUM_RECORDS = number of records
  output_file = defaults to /dev/shm/vary{N_VARY}_{RECORDS}M_normalized.bin

The first N_VARY byte positions of the 66-byte key get uniformly random
values (0-255). The remaining (66 - N_VARY) positions are set to 0x42.
Value bytes (54 bytes) are zero. This lets the compact key detector find
exactly N_VARY varying positions.
"""

import sys
import os
import time
import numpy as np

RECORD_SIZE = 120
KEY_SIZE = 66
VALUE_SIZE = 54
FIXED_BYTE = 0x42

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 gen_varying_bytes.py N_VARY NUM_RECORDS [output_file]")
        sys.exit(1)

    n_vary = int(sys.argv[1])
    if n_vary < 0 or n_vary > KEY_SIZE:
        print(f"Error: N_VARY must be in 0..{KEY_SIZE}, got {n_vary}")
        sys.exit(1)

    nrec = int(sys.argv[2])
    label = f"{nrec // 1_000_000}M" if nrec >= 1_000_000 else str(nrec)
    outpath = (sys.argv[3] if len(sys.argv) > 3
               else f"/dev/shm/vary{n_vary}_{label}_normalized.bin")

    size_gb = nrec * RECORD_SIZE / 1e9
    print(f"Generating {nrec:,} records with {n_vary}/{KEY_SIZE} varying "
          f"key bytes ({size_gb:.2f} GB)...")

    CHUNK = 5_000_000
    rng = np.random.default_rng(seed=42)

    t0 = time.time()
    with open(outpath, 'wb') as f:
        written = 0
        while written < nrec:
            n = min(CHUNK, nrec - written)
            # Start with fixed bytes for entire key
            keys = np.full((n, KEY_SIZE), FIXED_BYTE, dtype=np.uint8)
            # Overwrite first n_vary positions with random values
            if n_vary > 0:
                keys[:, :n_vary] = rng.integers(0, 256,
                                                size=(n, n_vary),
                                                dtype=np.uint8)
            values = np.zeros((n, VALUE_SIZE), dtype=np.uint8)
            records = np.hstack([keys, values])
            f.write(records.tobytes())
            written += n
            if written % 10_000_000 == 0 or written == nrec:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else written
                eta = (nrec - written) / rate if rate > 0 else 0
                print(f"  {written/1e6:.0f}M / {nrec/1e6:.0f}M "
                      f"({written*RECORD_SIZE/1e9:.1f} GB) "
                      f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    actual_gb = os.path.getsize(outpath) / 1e9
    print(f"\nDone: {nrec:,} records, {actual_gb:.2f} GB, {elapsed:.1f}s")
    print(f"Varying positions: 0..{n_vary - 1}" if n_vary > 0
          else "Varying positions: none (all fixed)")
    print(f"Output: {outpath}")

if __name__ == "__main__":
    main()
