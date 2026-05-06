#!/bin/bash
# 19.38 — Polars at SF300 (216 GB) for SOTA comparison row.
set -uo pipefail
source ~/gpu-research/h100/env.sh

cat > /tmp/polars_sf300.py << 'PYEOF'
import polars as pl
import numpy as np
import os, sys, time

KEY_SIZE = 66
RECORD_SIZE = 120
PATH = "/mnt/data/lineitem_sf300.bin"

print(f"Polars version: {pl.__version__}")
fsize = os.path.getsize(PATH)
n_total = fsize // RECORD_SIZE
print(f"SF300: {n_total:,} records ({fsize/1e9:.1f} GB)")

t0 = time.time()
arr = np.memmap(PATH, dtype=np.uint8, mode='r', shape=(n_total, RECORD_SIZE))
print(f"mmap: {time.time()-t0:.1f}s")

t1 = time.time()
pfx1_be = np.frombuffer(arr[:, 0:8].tobytes(), dtype='>u8')
pfx2_be = np.frombuffer(arr[:, 8:16].tobytes(), dtype='>u8')
print(f"prefix extract: {time.time()-t1:.1f}s, mem ~{(pfx1_be.nbytes + pfx2_be.nbytes)/1e9:.1f} GB")

pfx1 = np.ascontiguousarray(pfx1_be.astype(np.uint64))
pfx2 = np.ascontiguousarray(pfx2_be.astype(np.uint64))
df = pl.DataFrame({'pfx1': pfx1, 'pfx2': pfx2})
print(f"DataFrame built: {time.time()-t1:.1f}s")

print("Sort 1...")
t0 = time.time()
s1 = df.sort(['pfx1', 'pfx2'])
sort1_s = time.time() - t0
print(f"  sort 1: {sort1_s:.1f}s")
del s1

print("Sort 2 (warm)...")
t0 = time.time()
s2 = df.sort(['pfx1', 'pfx2'])
sort2_s = time.time() - t0
print(f"  sort 2: {sort2_s:.1f}s")

rec_gb = n_total * 120 / 1e9
warm = sort2_s
print(f"\nCSV,polars_sf300,records={n_total},warm_s={warm:.1f},record_gb_per_s={rec_gb/warm:.2f}")
print(f"vs gpu_crocsort SF300 stream pre-pin: 90.1s, 2.59 GB/s")
print(f"Polars {warm:.0f}s vs gpu_crocsort 90s = {warm/90:.1f}x slower")
PYEOF

python3 /tmp/polars_sf300.py 2>&1
