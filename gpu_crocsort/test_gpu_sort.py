#!/usr/bin/env python3
"""
Test gpu_sort against pyarrow.compute — 10M rows, multi-type, correctness + timing.
"""

import sys
import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Ensure we can import gpu_sort from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_sort

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N = 10_000_000
SEED = 42

def main():
    print(f"=== gpu_sort test — {N:,} rows ===\n")

    if not gpu_sort.is_available():
        print("WARNING: No CUDA GPU detected — gpu_sort will likely fail")

    rng = np.random.default_rng(SEED)

    # Build columns
    col_i32 = pa.array(rng.integers(-2**31, 2**31 - 1, size=N, dtype=np.int32))
    col_i64 = pa.array(rng.integers(-2**63, 2**63 - 1, size=N, dtype=np.int64))
    col_f64 = pa.array(rng.standard_normal(N).astype(np.float64))
    col_u32 = pa.array(rng.integers(0, 2**32 - 1, size=N, dtype=np.uint32))

    table = pa.table({
        "i32": col_i32,
        "i64": col_i64,
        "f64": col_f64,
        "u32": col_u32,
    })

    sort_keys = [("i32", "ascending"), ("i64", "ascending"), ("f64", "ascending")]

    # -----------------------------------------------------------------------
    # 1. Single-column sort (int32)
    # -----------------------------------------------------------------------
    print("--- Single-column sort (int32, 10M) ---")

    t0 = time.perf_counter()
    gpu_idx = gpu_sort.sort_indices(col_i32, verbose=True)
    t_gpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    pa_idx = pc.sort_indices(col_i32)
    t_pa = time.perf_counter() - t0

    gpu_np = gpu_idx.to_numpy()
    pa_np = pa_idx.to_numpy().astype(np.uint32)
    # They might not be identical (different tie-breaking) — compare sorted values
    gpu_vals = col_i32.to_numpy()[gpu_np]
    pa_vals = col_i32.to_numpy()[pa_np]
    match = np.array_equal(gpu_vals, pa_vals)
    print(f"  GPU: {t_gpu*1000:.1f} ms | PyArrow: {t_pa*1000:.1f} ms | "
          f"speedup: {t_pa/t_gpu:.2f}x | match: {match}")
    assert match, "Single-column sort mismatch!"

    # -----------------------------------------------------------------------
    # 2. Multi-column sort on table
    # -----------------------------------------------------------------------
    print("\n--- Multi-column sort (i32, i64, f64 — 10M) ---")

    t0 = time.perf_counter()
    gpu_idx2 = gpu_sort.sort_indices(table, sort_keys=sort_keys, verbose=True)
    t_gpu2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    pa_idx2 = pc.sort_indices(table, sort_keys=sort_keys)
    t_pa2 = time.perf_counter() - t0

    # Verify by comparing sorted column values
    gpu_np2 = gpu_idx2.to_numpy()
    pa_np2 = pa_idx2.to_numpy().astype(np.uint32)

    ok = True
    for col_name, _ in sort_keys:
        col_np = table.column(col_name).to_numpy()
        if not np.array_equal(col_np[gpu_np2], col_np[pa_np2]):
            # Check if values are equal even if indices differ (ties)
            # Compare all sort key columns together
            ok = False
            break

    if not ok:
        # More thorough check: build composite key and compare
        gpu_composite = np.column_stack([table.column(c).to_numpy()[gpu_np2] for c, _ in sort_keys])
        pa_composite = np.column_stack([table.column(c).to_numpy()[pa_np2] for c, _ in sort_keys])
        ok = np.array_equal(gpu_composite, pa_composite)

    print(f"  GPU: {t_gpu2*1000:.1f} ms | PyArrow: {t_pa2*1000:.1f} ms | "
          f"speedup: {t_pa2/t_gpu2:.2f}x | match: {ok}")
    assert ok, "Multi-column sort mismatch!"

    # -----------------------------------------------------------------------
    # 3. sort_table convenience
    # -----------------------------------------------------------------------
    print("\n--- sort_table (i32 asc, f64 desc — 10M) ---")
    sk2 = [("i32", "ascending"), ("f64", "descending")]

    t0 = time.perf_counter()
    gpu_tbl = gpu_sort.sort_table(table, sk2)
    t_gpu3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    pa_tbl = table.take(pc.sort_indices(table, sort_keys=sk2))
    t_pa3 = time.perf_counter() - t0

    gpu_i32 = gpu_tbl.column("i32").to_numpy()
    pa_i32 = pa_tbl.column("i32").to_numpy()
    match3 = np.array_equal(gpu_i32, pa_i32)
    if match3:
        gpu_f64 = gpu_tbl.column("f64").to_numpy()
        pa_f64 = pa_tbl.column("f64").to_numpy()
        match3 = np.array_equal(gpu_f64, pa_f64)

    print(f"  GPU: {t_gpu3*1000:.1f} ms | PyArrow: {t_pa3*1000:.1f} ms | "
          f"speedup: {t_pa3/t_gpu3:.2f}x | match: {match3}")
    assert match3, "sort_table mismatch!"

    # -----------------------------------------------------------------------
    # 4. sort_array convenience
    # -----------------------------------------------------------------------
    print("\n--- sort_array (float64, descending — 10M) ---")

    t0 = time.perf_counter()
    gpu_sorted = gpu_sort.sort_array(col_f64, descending=True)
    t_gpu4 = time.perf_counter() - t0

    t0 = time.perf_counter()
    pa_sorted = col_f64.take(pc.sort_indices(col_f64, sort_keys=[("x", "descending")]))
    t_pa4 = time.perf_counter() - t0

    match4 = np.array_equal(gpu_sorted.to_numpy(), pa_sorted.to_numpy())
    print(f"  GPU: {t_gpu4*1000:.1f} ms | PyArrow: {t_pa4*1000:.1f} ms | "
          f"speedup: {t_pa4/t_gpu4:.2f}x | match: {match4}")
    assert match4, "sort_array descending mismatch!"

    # -----------------------------------------------------------------------
    # 5. Type coverage
    # -----------------------------------------------------------------------
    print("\n--- Type coverage (1M each) ---")
    n_small = 1_000_000
    types_to_test = [
        ("int8",    pa.int8(),    rng.integers(-128, 127, n_small, dtype=np.int8)),
        ("int16",   pa.int16(),   rng.integers(-32768, 32767, n_small, dtype=np.int16)),
        ("uint8",   pa.uint8(),   rng.integers(0, 255, n_small, dtype=np.uint8)),
        ("uint16",  pa.uint16(),  rng.integers(0, 65535, n_small, dtype=np.uint16)),
        ("uint64",  pa.uint64(),  rng.integers(0, 2**64 - 1, n_small, dtype=np.uint64)),
        ("float32", pa.float32(), rng.standard_normal(n_small).astype(np.float32)),
        ("date32",  pa.date32(),  rng.integers(0, 20000, n_small, dtype=np.int32)),
    ]
    for name, typ, np_data in types_to_test:
        if typ == pa.date32():
            arr = pa.array(np_data.astype("datetime64[D]"), type=pa.date32())
        else:
            arr = pa.array(np_data, type=typ)
        gpu_s = gpu_sort.sort_array(arr)
        pa_s = arr.take(pc.sort_indices(arr))
        ok = np.array_equal(
            gpu_s.to_numpy(zero_copy_only=False),
            pa_s.to_numpy(zero_copy_only=False)
        )
        print(f"  {name:8s}: {'PASS' if ok else 'FAIL'}")
        assert ok, f"Type {name} sort mismatch!"

    print(f"\n=== All tests passed ===")


if __name__ == "__main__":
    main()
