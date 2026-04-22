#!/usr/bin/env python3
"""
Comprehensive sorting benchmark: GPU sort vs numpy, pandas, pyarrow.

Tests across data types (int32, int64, float64, nearly-sorted, few-unique)
and data sizes (1M to 300M rows).
"""

import time
import signal
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, "/home/cc/gpu-research/gpu_crocsort")
import gpu_sort

TIMEOUT = 120  # seconds per engine per test

# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Exceeded timeout")

def timed_call(fn, timeout_s=TIMEOUT):
    """Run fn(), return elapsed seconds or None if timeout."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        return elapsed, result
    except TimeoutError:
        return None, None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def gen_int32_random(n):
    return np.random.randint(0, 2**31 - 1, size=n, dtype=np.int32)

def gen_int64_random(n):
    return np.random.randint(0, 2**63 - 1, size=n, dtype=np.int64)

def gen_float64_random(n):
    return np.random.uniform(-1e18, 1e18, size=n).astype(np.float64)

def gen_int32_nearly_sorted(n):
    arr = np.arange(n, dtype=np.int32)
    num_swaps = max(1, n // 100)
    rng = np.random.default_rng(42)
    for _ in range(num_swaps):
        i, j = rng.integers(0, n, size=2)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def gen_int32_few_unique(n):
    return np.random.randint(0, 100, size=n, dtype=np.int32)

# ---------------------------------------------------------------------------
# Engine wrappers — each returns sort indices as numpy int array
# ---------------------------------------------------------------------------
def sort_numpy(np_arr):
    return np.argsort(np_arr, kind='quicksort')

def sort_pandas(np_arr):
    s = pd.Series(np_arr, copy=False)
    return s.sort_values(kind='quicksort').index.to_numpy()

def sort_pyarrow(np_arr):
    pa_arr = pa.array(np_arr)
    return pc.sort_indices(pa_arr).to_numpy()

def sort_gpu(np_arr):
    pa_arr = pa.array(np_arr)
    return gpu_sort.sort_indices(pa_arr).to_numpy()

ENGINES = [
    ("numpy",   sort_numpy),
    ("pandas",  sort_pandas),
    ("pyarrow", sort_pyarrow),
    ("GPU",     sort_gpu),
]

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_indices(data, indices_dict, check_n=1000):
    """Spot-check that all engines agree on the first check_n sorted values."""
    # Use the first available result as reference
    ref_name = None
    ref_sorted = None
    for name in ["numpy", "pyarrow", "GPU", "pandas"]:
        if name in indices_dict and indices_dict[name] is not None:
            idx = indices_dict[name]
            ref_sorted = data[idx[:check_n]]
            ref_name = name
            break
    if ref_name is None:
        return "no results"

    mismatches = []
    for name, idx in indices_dict.items():
        if idx is None or name == ref_name:
            continue
        vals = data[idx[:check_n]]
        if not np.array_equal(ref_sorted, vals):
            mismatches.append(name)
    if mismatches:
        return f"MISMATCH: {mismatches} vs {ref_name}"
    return "OK"

# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def fmt_time(t):
    if t is None:
        return ">120s"
    if t < 0.01:
        return f"{t*1000:.1f}ms"
    return f"{t:.3f}s"

def print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_dtype_benchmark():
    """Test different data types at 100M rows."""
    N = 100_000_000
    print("=" * 80)
    print(f"DATA TYPE BENCHMARK  ({N:,} rows each)")
    print("=" * 80)

    tests = [
        ("int32 random",        gen_int32_random),
        ("int64 random",        gen_int64_random),
        ("float64 random",      gen_float64_random),
        ("int32 nearly-sorted", gen_int32_nearly_sorted),
        ("int32 few-unique",    gen_int32_few_unique),
    ]

    headers = ["Data Type", "numpy", "pandas", "pyarrow", "GPU", "Verify"]
    rows = []

    for test_name, gen_fn in tests:
        print(f"\n--- {test_name} ({N:,} rows) ---")
        print("Generating data...", end=" ", flush=True)
        data = gen_fn(N)
        print("done.")

        times = {}
        indices = {}

        for eng_name, eng_fn in ENGINES:
            # Warmup for GPU only (CUDA context)
            if eng_name == "GPU":
                print(f"  {eng_name}: warmup...", end=" ", flush=True)
                small = data[:1000]
                eng_fn(small)
                print("timed run...", end=" ", flush=True)
            else:
                print(f"  {eng_name}: timed run...", end=" ", flush=True)

            elapsed, result = timed_call(lambda fn=eng_fn, d=data: fn(d))
            times[eng_name] = elapsed
            indices[eng_name] = result
            print(fmt_time(elapsed), flush=True)

        verify = verify_indices(data, indices)
        row = [test_name] + [fmt_time(times.get(e, None)) for e, _ in ENGINES] + [verify]
        rows.append(row)

        # Free memory
        del data, indices
        import gc; gc.collect()

    print("\n")
    print_table(headers, rows)

    # Speedup summary
    print("\nSpeedup vs GPU:")
    for row in rows:
        dtype = row[0]
        gpu_str = row[4]
        if gpu_str == ">120s":
            print(f"  {dtype}: GPU timed out")
            continue
        gpu_t = float(gpu_str.replace("s", "").replace("m", ""))
        if "ms" in row[4]:
            gpu_t /= 1000
        parts = []
        for i, (eng_name, _) in enumerate(ENGINES):
            if eng_name == "GPU":
                continue
            t_str = row[i + 1]
            if t_str == ">120s":
                parts.append(f"{eng_name}: >120s")
            else:
                t = float(t_str.replace("s", "").replace("m", ""))
                if "ms" in t_str:
                    t /= 1000
                ratio = t / gpu_t
                parts.append(f"{eng_name}: {ratio:.2f}x")
        print(f"  {dtype}: {', '.join(parts)}")


def run_scaling_benchmark():
    """Test int32 at different sizes."""
    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000, 300_000_000]
    print("\n\n" + "=" * 80)
    print("SCALING BENCHMARK  (int32 random)")
    print("=" * 80)

    headers = ["Rows", "numpy", "pandas", "pyarrow", "GPU", "Fastest"]
    rows = []

    # Track crossover points
    crossovers = {e: None for e, _ in ENGINES if e != "GPU"}

    for n in sizes:
        print(f"\n--- {n:,} rows ---")
        print("Generating data...", end=" ", flush=True)
        data = gen_int32_random(n)
        print("done.")

        times = {}
        indices = {}

        for eng_name, eng_fn in ENGINES:
            if eng_name == "GPU":
                print(f"  {eng_name}: warmup...", end=" ", flush=True)
                small = data[:1000]
                eng_fn(small)
                print("timed run...", end=" ", flush=True)
            else:
                print(f"  {eng_name}: timed run...", end=" ", flush=True)

            elapsed, result = timed_call(lambda fn=eng_fn, d=data: fn(d))
            times[eng_name] = elapsed
            indices[eng_name] = result
            print(fmt_time(elapsed), flush=True)

        # Determine fastest
        valid = {k: v for k, v in times.items() if v is not None}
        fastest = min(valid, key=valid.get) if valid else "N/A"

        # Check crossovers
        gpu_t = times.get("GPU")
        if gpu_t is not None:
            for eng_name in crossovers:
                cpu_t = times.get(eng_name)
                if cpu_t is not None and gpu_t < cpu_t and crossovers[eng_name] is None:
                    crossovers[eng_name] = n

        row = [f"{n:,}"] + [fmt_time(times.get(e, None)) for e, _ in ENGINES] + [fastest]
        rows.append(row)

        del data, indices
        import gc; gc.collect()

    print("\n")
    print_table(headers, rows)

    # Crossover summary
    print("\nGPU crossover points (where GPU becomes faster):")
    for eng_name, cross_n in crossovers.items():
        if cross_n is not None:
            print(f"  GPU faster than {eng_name} at {cross_n:,} rows")
        else:
            print(f"  GPU never faster than {eng_name} in tested range")


if __name__ == "__main__":
    print(f"GPU available: {gpu_sort.is_available()}")
    np.random.seed(42)

    run_dtype_benchmark()
    run_scaling_benchmark()

    print("\n\nBenchmark complete.")
