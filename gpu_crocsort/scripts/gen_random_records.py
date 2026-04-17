#!/usr/bin/env python3
"""Generate random 120-byte records for scaling experiments.

Usage: python3 gen_random_records.py NUM_RECORDS OUTPUT_FILE [--record-size BYTES]
"""

import sys
import os
import time
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 gen_random_records.py NUM_RECORDS OUTPUT_FILE [--record-size BYTES]")
        sys.exit(1)

    nrec = int(sys.argv[1])
    outpath = sys.argv[2]
    rec_size = 120
    for i, arg in enumerate(sys.argv):
        if arg == '--record-size' and i + 1 < len(sys.argv):
            rec_size = int(sys.argv[i + 1])

    size_gb = nrec * rec_size / 1e9
    print(f"Generating {nrec:,} records × {rec_size}B = {size_gb:.2f} GB...")

    CHUNK = 5_000_000
    rng = np.random.default_rng(seed=42)

    t0 = time.time()
    with open(outpath, 'wb') as f:
        written = 0
        while written < nrec:
            n = min(CHUNK, nrec - written)
            data = rng.integers(0, 256, size=(n, rec_size), dtype=np.uint8)
            f.write(data.tobytes())
            written += n
            if written % 10_000_000 == 0 or written == nrec:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else written
                eta = (nrec - written) / rate if rate > 0 else 0
                print(f"  {written/1e6:.1f}M / {nrec/1e6:.1f}M "
                      f"({written * rec_size / 1e9:.1f} GB) "
                      f"ETA: {eta:.0f}s")

    elapsed = time.time() - t0
    actual_gb = os.path.getsize(outpath) / 1e9
    print(f"Done: {nrec:,} records, {actual_gb:.2f} GB, {elapsed:.1f}s")

if __name__ == "__main__":
    main()
