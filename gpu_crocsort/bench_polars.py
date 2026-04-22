#!/usr/bin/env python3
"""Benchmark: Polars sort vs PyArrow sort vs GPU sort (ctypes)."""

import ctypes
import time
import sys
import numpy as np

# ---------------------------------------------------------------------------
# GPU sort via ctypes
# ---------------------------------------------------------------------------
LIBGPU = ctypes.CDLL("/home/cc/gpu-research/gpu_crocsort/libgpusort.so")
LIBGPU.gpu_sort_keys.argtypes = [
    ctypes.c_void_p,   # keys
    ctypes.c_uint32,    # key_size
    ctypes.c_uint32,    # key_stride
    ctypes.c_uint64,    # num_records
    ctypes.c_void_p,    # perm_out
]
LIBGPU.gpu_sort_keys.restype = ctypes.c_int


def encode_int32_key(arr: np.ndarray) -> np.ndarray:
    """Convert int32 array to big-endian byte-comparable key bytes."""
    flipped = (arr.view(np.uint32) ^ np.uint32(0x80000000)).astype(">u4")
    return flipped.view(np.uint8)


def encode_int64_key(arr: np.ndarray) -> np.ndarray:
    """Convert int64 array to big-endian byte-comparable key bytes."""
    flipped = (arr.view(np.uint64) ^ np.uint64(0x8000000000000000)).astype(">u8")
    return flipped.view(np.uint8)


def gpu_sort_perm(keys_flat: np.ndarray, key_width: int, n: int) -> np.ndarray:
    """Call gpu_sort_keys and return the permutation array."""
    assert keys_flat.dtype == np.uint8
    assert keys_flat.flags["C_CONTIGUOUS"]
    perm = np.empty(n, dtype=np.uint32)
    rc = LIBGPU.gpu_sort_keys(
        keys_flat.ctypes.data,
        ctypes.c_uint32(key_width),
        ctypes.c_uint32(key_width),
        ctypes.c_uint64(n),
        perm.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"gpu_sort_keys returned {rc}")
    return perm


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def time_fn(fn, warmup=1, repeats=3):
    """Time a function, return median seconds (excluding warmup)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def verify_sorted_keys(name: str, perm: np.ndarray, keys_flat: np.ndarray,
                        key_width: int, n_check: int = 1000) -> bool:
    """Verify that applying perm to keys produces sorted order."""
    n = len(perm)
    keys_2d = keys_flat.reshape(n, key_width)
    # Check first n_check consecutive pairs in sorted output
    check = min(n_check, n - 1)
    # Check head
    for i in range(check):
        k_a = bytes(keys_2d[perm[i]])
        k_b = bytes(keys_2d[perm[i + 1]])
        if k_a > k_b:
            print(f"  WARNING: {name} not sorted at position {i}: {k_a.hex()} > {k_b.hex()}")
            return False
    # Check tail
    for i in range(max(0, n - check - 1), n - 1):
        k_a = bytes(keys_2d[perm[i]])
        k_b = bytes(keys_2d[perm[i + 1]])
        if k_a > k_b:
            print(f"  WARNING: {name} not sorted at tail position {i}: {k_a.hex()} > {k_b.hex()}")
            return False
    return True


def verify_values_match(name_a: str, sorted_keys_a: np.ndarray,
                        name_b: str, sorted_keys_b: np.ndarray,
                        n_check: int = 100) -> bool:
    """Verify that two sorted outputs have the same key values at head/tail."""
    head_ok = np.array_equal(sorted_keys_a[:n_check], sorted_keys_b[:n_check])
    tail_ok = np.array_equal(sorted_keys_a[-n_check:], sorted_keys_b[-n_check:])
    if not (head_ok and tail_ok):
        print(f"  WARNING: {name_a} vs {name_b} sorted values differ!")
        return False
    return True


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------
def run_benchmark(n_rows: int):
    import polars as pl
    import pyarrow as pa
    import pyarrow.compute as pc

    print(f"\n{'='*70}")
    print(f"  Benchmark: {n_rows:,} rows")
    print(f"{'='*70}")

    configs = [
        ("1x int32", 4),
        ("2x int32", 8),
        ("1x int64", 8),
    ]

    results = []

    for label, key_width in configs:
        print(f"\n--- {label} (key_width={key_width}B) ---")

        # Generate data
        rng = np.random.default_rng(42)
        if label == "1x int32":
            col_a = rng.integers(-(2**31), 2**31 - 1, size=n_rows, dtype=np.int32)
            sort_cols = [("a", col_a)]
        elif label == "2x int32":
            col_a = rng.integers(-(2**31), 2**31 - 1, size=n_rows, dtype=np.int32)
            col_b = rng.integers(-(2**31), 2**31 - 1, size=n_rows, dtype=np.int32)
            sort_cols = [("a", col_a), ("b", col_b)]
        elif label == "1x int64":
            col_a = rng.integers(-(2**63), 2**63 - 1, size=n_rows, dtype=np.int64)
            sort_cols = [("a", col_a)]

        # ---- Build Polars DataFrame ----
        df_dict = {name: arr for name, arr in sort_cols}
        df = pl.DataFrame(df_dict)
        col_names = [name for name, _ in sort_cols]

        # ---- Build PyArrow table ----
        pa_arrays = {name: pa.array(arr) for name, arr in sort_cols}
        pa_table = pa.table(pa_arrays)

        # ---- Build GPU key buffer ----
        if label == "1x int32":
            keys_flat = encode_int32_key(col_a)
        elif label == "2x int32":
            k_a = encode_int32_key(col_a).reshape(n_rows, 4)
            k_b = encode_int32_key(col_b).reshape(n_rows, 4)
            keys_flat = np.ascontiguousarray(np.hstack([k_a, k_b]).reshape(-1))
        elif label == "1x int64":
            keys_flat = encode_int64_key(col_a)

        # ---- Polars sort ----
        def polars_sort():
            return df.sort(col_names)

        t_polars = time_fn(polars_sort, warmup=1, repeats=3)
        print(f"  Polars sort:   {t_polars:.3f}s")

        # ---- PyArrow sort ----
        sort_keys = [(name, "ascending") for name in col_names]

        def pyarrow_sort():
            indices = pc.sort_indices(pa_table, sort_keys=sort_keys)
            return pa_table.take(indices)

        t_arrow = time_fn(pyarrow_sort, warmup=1, repeats=3)
        print(f"  PyArrow sort:  {t_arrow:.3f}s")

        # ---- GPU sort ----
        def gpu_sort():
            return gpu_sort_perm(keys_flat, key_width, n_rows)

        t_gpu = time_fn(gpu_sort, warmup=1, repeats=3)
        print(f"  GPU sort:      {t_gpu:.3f}s")

        # ---- Verify ----
        # GPU permutation
        gpu_perm = gpu_sort_perm(keys_flat, key_width, n_rows)
        keys_2d = keys_flat.reshape(n_rows, key_width)

        # Check GPU output is sorted
        v_gpu = verify_sorted_keys("GPU", gpu_perm, keys_flat, key_width)

        # Get Polars sorted keys for value comparison
        polars_sorted = df.sort(col_names)
        if label == "1x int32":
            polars_sorted_keys = encode_int32_key(polars_sorted[col_names[0]].to_numpy())
        elif label == "2x int32":
            pk_a = encode_int32_key(polars_sorted["a"].to_numpy()).reshape(n_rows, 4)
            pk_b = encode_int32_key(polars_sorted["b"].to_numpy()).reshape(n_rows, 4)
            polars_sorted_keys = np.ascontiguousarray(np.hstack([pk_a, pk_b]))
        elif label == "1x int64":
            polars_sorted_keys = encode_int64_key(polars_sorted[col_names[0]].to_numpy()).reshape(n_rows, key_width)

        gpu_sorted_keys = keys_2d[gpu_perm]
        if polars_sorted_keys.ndim == 1:
            polars_sorted_keys = polars_sorted_keys.reshape(n_rows, key_width)

        v_vals = verify_values_match("Polars", polars_sorted_keys, "GPU", gpu_sorted_keys)
        status = "PASS" if (v_gpu and v_vals) else "FAIL"
        print(f"  Verification:  {status}")

        # Speedup
        print(f"  GPU vs Polars: {t_polars/t_gpu:.2f}x")
        print(f"  GPU vs Arrow:  {t_arrow/t_gpu:.2f}x")

        results.append({
            "config": label,
            "n": n_rows,
            "polars_s": t_polars,
            "arrow_s": t_arrow,
            "gpu_s": t_gpu,
            "verified": status,
        })

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"  Summary ({n_rows:,} rows)")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'Polars':>8} {'PyArrow':>8} {'GPU':>8} {'GPU/Pol':>8} {'GPU/Arr':>8} {'Check':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        sp_pol = r["polars_s"] / r["gpu_s"]
        sp_arr = r["arrow_s"] / r["gpu_s"]
        print(
            f"  {r['config']:<12} {r['polars_s']:>7.3f}s {r['arrow_s']:>7.3f}s "
            f"{r['gpu_s']:>7.3f}s {sp_pol:>7.2f}x {sp_arr:>7.2f}x {r['verified']:>6}"
        )
    print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
    run_benchmark(n)
