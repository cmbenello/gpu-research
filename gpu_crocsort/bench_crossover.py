"""
bench_crossover.py — find where GPU sort starts beating CPU engines.

Tests numpy.argsort, polars .sort(), and gpu_sort.sort_indices
across a range of sizes with int32 random data.
"""

import sys
import time
import statistics
import numpy as np

# ── GPU sort setup ──────────────────────────────────────────────────────────
sys.path.insert(0, "/home/cc/gpu-research/gpu_crocsort")
import gpu_sort
import pyarrow as pa

# ── Polars ──────────────────────────────────────────────────────────────────
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("WARNING: polars not available, skipping Polars column")

# ── Config ──────────────────────────────────────────────────────────────────
SIZES   = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]
N_RUNS  = 5
RNG     = np.random.default_rng(42)


def median_ms(fn, n_runs):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.median(times)


def bench_numpy(data_np):
    def fn():
        np.argsort(data_np, kind="stable")
    return median_ms(fn, N_RUNS)


def bench_polars(data_np):
    if not HAS_POLARS:
        return None
    s = pl.Series(data_np)
    def fn():
        s.sort()
    return median_ms(fn, N_RUNS)


def bench_gpu(data_np):
    # Encode as PyArrow int32 array (gpu_sort handles the key encoding)
    arr = pa.array(data_np, type=pa.int32())
    def fn():
        gpu_sort.sort_indices(arr)
    return median_ms(fn, N_RUNS)


# ── Warm up GPU (avoid first-call CUDA init penalty in measurements) ────────
print("Warming up GPU...", flush=True)
_warmup = pa.array(RNG.integers(-2**31, 2**31, size=100_000, dtype=np.int32), type=pa.int32())
for _ in range(3):
    gpu_sort.sort_indices(_warmup)
print("GPU warmed up.\n", flush=True)

# ── Header ──────────────────────────────────────────────────────────────────
if HAS_POLARS:
    hdr = f"{'Size':>10}  {'NumPy(ms)':>12}  {'Polars(ms)':>12}  {'GPU(ms)':>12}  {'GPU/NumPy':>10}  {'GPU/Polars':>11}  {'Winner'}"
else:
    hdr = f"{'Size':>10}  {'NumPy(ms)':>12}  {'GPU(ms)':>12}  {'GPU/NumPy':>10}  {'Winner'}"

print(hdr)
print("-" * len(hdr))

results = []

for size in SIZES:
    data = RNG.integers(-2**31, 2**31, size=size, dtype=np.int32)

    np_ms  = bench_numpy(data)
    pol_ms = bench_polars(data)
    gpu_ms = bench_gpu(data)

    ratio_np  = gpu_ms / np_ms
    ratio_pol = (gpu_ms / pol_ms) if pol_ms is not None else None

    if pol_ms is not None:
        winner = "GPU" if gpu_ms < min(np_ms, pol_ms) else ("NumPy" if np_ms < pol_ms else "Polars")
    else:
        winner = "GPU" if gpu_ms < np_ms else "NumPy"

    results.append((size, np_ms, pol_ms, gpu_ms, ratio_np, ratio_pol, winner))

    if HAS_POLARS:
        pol_str  = f"{pol_ms:>12.2f}"
        ratio_pol_str = f"{ratio_pol:>11.2f}x"
        print(f"{size:>10,}  {np_ms:>12.2f}  {pol_str}  {gpu_ms:>12.2f}  {ratio_np:>9.2f}x  {ratio_pol_str}  {winner}")
    else:
        print(f"{size:>10,}  {np_ms:>12.2f}  {gpu_ms:>12.2f}  {ratio_np:>9.2f}x  {winner}")

    sys.stdout.flush()

# ── Summary ─────────────────────────────────────────────────────────────────
print()
print("=== CROSSOVER SUMMARY ===")
gpu_beats_numpy  = [r for r in results if r[3] < r[1]]
gpu_beats_polars = [r for r in results if r[2] is not None and r[3] < r[2]]

if gpu_beats_numpy:
    crossover_np = gpu_beats_numpy[0][0]
    print(f"GPU beats NumPy  starting at size: {crossover_np:,}")
else:
    print("GPU never beats NumPy in tested range")

if HAS_POLARS:
    if gpu_beats_polars:
        crossover_pol = gpu_beats_polars[0][0]
        print(f"GPU beats Polars starting at size: {crossover_pol:,}")
    else:
        print("GPU never beats Polars in tested range")
