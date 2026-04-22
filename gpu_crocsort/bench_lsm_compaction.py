"""
bench_lsm_compaction.py — Simulated LSM-tree compaction benchmark

Simulates K-way merge-sort of K sorted SSTable runs as done during
RocksDB/LevelDB L0→L1 (or Lx→Ly) compaction.

Three approaches compared:
  1. CPU heapq.merge()   — classic K-way merge, O(N*K log K) comparisons
  2. CPU numpy argsort  — concatenate + sort, O(N*K log(N*K))
  3. GPU radix sort     — concatenate + GPU sort via libgpusort.so

Key insight: GPU radix sort is O(N*K) with a small constant (no log factor),
so the gap widens as K or N grows.

Real RocksDB mapping:
  L0→L1 compaction: 4–8 SSTables of 64–256 MB each → 256 MB – 2 GB per job
  int64 keys (8 bytes each) for simplicity; real SSTables also carry values.
"""

import ctypes
import heapq
import os
import sys
import time
import struct

import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Load GPU sort library
# ---------------------------------------------------------------------------
_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "libgpusort.so")
_lib = ctypes.CDLL(_lib_path)

_lib.gpu_sort_keys.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_void_p,
]
_lib.gpu_sort_keys.restype = ctypes.c_int

_lib.gpu_sort_available.argtypes = []
_lib.gpu_sort_available.restype = ctypes.c_int

class GpuSortTiming(ctypes.Structure):
    _fields_ = [
        ("total_ms",          ctypes.c_double),
        ("upload_ms",         ctypes.c_double),
        ("gpu_sort_ms",       ctypes.c_double),
        ("download_ms",       ctypes.c_double),
        ("gather_ms",         ctypes.c_double),
        ("fixup_ms",          ctypes.c_double),
        ("num_fixup_groups",  ctypes.c_uint64),
        ("num_fixup_records", ctypes.c_uint64),
    ]

_lib.gpu_sort_get_timing.argtypes = [ctypes.POINTER(GpuSortTiming)]
_lib.gpu_sort_get_timing.restype = None


def _encode_int64_to_bytes(arr: np.ndarray) -> np.ndarray:
    """Encode int64 array as 8-byte big-endian keys (sign-bit flipped for sort order)."""
    u = arr.view(np.uint64) ^ np.uint64(0x8000000000000000)
    return u.astype(">u8").view(np.uint8).reshape(-1, 8)


def gpu_sort_int64(data: np.ndarray) -> np.ndarray:
    """Sort int64 array via GPU, return sorted array."""
    keys = np.ascontiguousarray(_encode_int64_to_bytes(data), dtype=np.uint8)
    n = len(data)
    key_size = 8
    key_stride = keys.strides[0]

    perm = np.empty(n, dtype=np.uint32)
    rc = _lib.gpu_sort_keys(
        keys.ctypes.data,
        ctypes.c_uint32(key_size),
        ctypes.c_uint32(key_stride),
        ctypes.c_uint64(n),
        perm.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"gpu_sort_keys failed rc={rc}")

    return data[perm]


def gpu_get_timing() -> GpuSortTiming:
    t = GpuSortTiming()
    _lib.gpu_sort_get_timing(ctypes.byref(t))
    return t


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def generate_sorted_runs(K: int, N: int, rng: np.random.Generator) -> list:
    """Generate K sorted int64 runs of N keys each."""
    runs = []
    for _ in range(K):
        r = rng.integers(-(2**62), 2**62, size=N, dtype=np.int64)
        r.sort()
        runs.append(r)
    return runs


def cpu_heapq_merge(runs: list) -> np.ndarray:
    """K-way merge using heapq.merge (pure Python iterator)."""
    merged = list(heapq.merge(*runs))
    return np.array(merged, dtype=np.int64)


def cpu_numpy_sort(runs: list) -> np.ndarray:
    """Concatenate + numpy argsort."""
    combined = np.concatenate(runs)
    idx = np.argsort(combined, kind="stable")
    return combined[idx]


def gpu_radix_sort(runs: list) -> np.ndarray:
    """Concatenate all runs, GPU radix sort."""
    combined = np.ascontiguousarray(np.concatenate(runs))
    return gpu_sort_int64(combined)


def verify_same(a: np.ndarray, b: np.ndarray, label: str, ref_label: str = "ref") -> bool:
    """Spot-check that two sorted arrays are identical."""
    if len(a) != len(b):
        print(f"  [FAIL] {label} vs {ref_label}: length mismatch {len(a)} vs {len(b)}")
        return False
    if not np.array_equal(a, b):
        mismatch = np.where(a != b)[0]
        print(f"  [FAIL] {label} vs {ref_label}: first mismatch at idx {mismatch[0]}")
        return False
    # Also verify sorted
    if not np.all(a[:-1] <= a[1:]):
        print(f"  [FAIL] {label}: output not sorted!")
        return False
    print(f"  [OK]   {label} matches {ref_label} (spot-checked {len(a):,} elements)")
    return True


def fmt_bytes(n_elements: int) -> str:
    b = n_elements * 8
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    return f"{b/1e6:.0f} MB"


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

CONFIGS = [
    # (K, N_per_run)  total = K*N
    (4,  1_000_000),   #  4M keys  ~  32 MB
    (4,  5_000_000),   # 20M keys  ~ 160 MB
    (4, 10_000_000),   # 40M keys  ~ 320 MB
    (8,  1_000_000),   #  8M keys  ~  64 MB
    (8,  5_000_000),   # 40M keys  ~ 320 MB
    (8, 10_000_000),   # 80M keys  ~ 640 MB
    (16, 1_000_000),   # 16M keys  ~ 128 MB
    (16, 5_000_000),   # 80M keys  ~ 640 MB
    (16,10_000_000),   #160M keys  ~1.28 GB
]

# For K=16, N=10M, heapq.merge will take very long (pure Python). Skip it.
HEAPQ_MAX_TOTAL = 20_000_000   # skip heapq beyond this many total keys

def run_benchmark():
    if _lib.gpu_sort_available() == 0:
        print("ERROR: GPU not available or libgpusort.so not working.")
        sys.exit(1)

    rng = np.random.default_rng(42)

    print("=" * 80)
    print("LSM Compaction Benchmark — GPU radix sort vs CPU merge approaches")
    print("Key type: int64 (8 bytes), simulating LSM key-value store keys")
    print("=" * 80)
    print()
    print(f"{'Config':>20}  {'Total':>8}  {'heapq(s)':>10}  {'numpy(s)':>10}  {'GPU(s)':>8}  {'GPU MB/s':>10}  {'vs heapq':>10}  {'vs numpy':>10}")
    print("-" * 110)

    results = []

    for K, N in CONFIGS:
        total = K * N
        total_bytes = total * 8
        label = f"K={K:2d} N={N//1_000_000}M"

        # Generate runs (not timed)
        runs = generate_sorted_runs(K, N, rng)

        # ----- GPU sort (always run) -----
        # warm-up pass (first GPU call includes JIT/init overhead)
        _ = gpu_radix_sort([runs[0][:1000]])

        t0 = time.perf_counter()
        gpu_result = gpu_radix_sort(runs)
        gpu_time = time.perf_counter() - t0
        gpu_timing = gpu_get_timing()
        gpu_mb_s = total_bytes / 1e6 / gpu_time

        # ----- numpy sort -----
        t0 = time.perf_counter()
        np_result = cpu_numpy_sort(runs)
        np_time = time.perf_counter() - t0

        # ----- heapq merge (skip for very large configs) -----
        if total <= HEAPQ_MAX_TOTAL:
            t0 = time.perf_counter()
            hq_result = cpu_heapq_merge(runs)
            hq_time = time.perf_counter() - t0
            hq_str = f"{hq_time:10.3f}"
            speedup_hq = f"{hq_time/gpu_time:9.1f}x"
        else:
            hq_result = None
            hq_time = None
            hq_str = f"{'(skip)':>10}"
            speedup_hq = f"{'N/A':>10}"

        speedup_np = f"{np_time/gpu_time:9.1f}x"

        print(f"{label:>20}  {fmt_bytes(total):>8}  {hq_str}  {np_time:10.3f}  {gpu_time:8.3f}  {gpu_mb_s:10.0f}  {speedup_hq}  {speedup_np}")

        results.append({
            "K": K, "N": N, "total": total,
            "hq_time": hq_time, "np_time": np_time, "gpu_time": gpu_time,
            "gpu_mb_s": gpu_mb_s,
            "gpu_upload_ms": gpu_timing.upload_ms,
            "gpu_sort_ms": gpu_timing.gpu_sort_ms,
            "gpu_download_ms": gpu_timing.download_ms,
            "gpu_gather_ms": gpu_timing.gather_ms,
        })

        # Verify correctness (vs numpy as reference)
        verify_same(gpu_result, np_result, "GPU", "numpy")
        if hq_result is not None:
            verify_same(hq_result.astype(np.int64), np_result, "heapq", "numpy")

    print()
    print("=" * 80)
    print("GPU timing breakdown for largest config (K=16, N=10M, 160M keys, 1.28 GB)")
    print("=" * 80)
    last = results[-1]
    gpu_t = last["gpu_time"]
    print(f"  Upload to GPU:     {last['gpu_upload_ms']:8.1f} ms  ({last['gpu_upload_ms']/gpu_t/10:.1f}%)")
    print(f"  GPU radix sort:    {last['gpu_sort_ms']:8.1f} ms  ({last['gpu_sort_ms']/gpu_t/10:.1f}%)")
    print(f"  Download result:   {last['gpu_download_ms']:8.1f} ms  ({last['gpu_download_ms']/gpu_t/10:.1f}%)")
    print(f"  Gather/fixup:      {last['gpu_gather_ms']:8.1f} ms  ({last['gpu_gather_ms']/gpu_t/10:.1f}%)")
    print(f"  Total wall-clock:  {gpu_t*1000:8.1f} ms")
    print()

    print("=" * 80)
    print("Summary — speedup of GPU over numpy (CPU concat+sort) by config")
    print("=" * 80)
    print(f"{'K':>4}  {'N':>8}  {'Total data':>12}  {'GPU time':>10}  {'numpy time':>12}  {'Speedup':>10}")
    print("-" * 65)
    for r in results:
        speedup = r["np_time"] / r["gpu_time"]
        print(f"{r['K']:4d}  {r['N']//1_000_000:6d}M  {fmt_bytes(r['total']):>12}  {r['gpu_time']*1000:8.0f} ms  {r['np_time']*1000:10.0f} ms  {speedup:9.2f}x")

    print()
    print("=" * 80)
    print("Real RocksDB L0→L1 compaction context")
    print("=" * 80)
    print("""
  Typical L0→L1 compaction:
    - Merges 4–8 SSTables of 64–256 MB each
    - Total data per compaction job: 256 MB – 2 GB
    - Key type: variable-length (usually 8–32 bytes for user keys)
    - Value size varies; benchmark above measures key-comparison throughput only

  Mapping to this benchmark:
    - K=4,  N=10M  → 40M keys  × 8B = 320 MB  ≈ 4 SSTables of 80MB
    - K=8,  N=10M  → 80M keys  × 8B = 640 MB  ≈ 8 SSTables of 80MB
    - K=16, N=10M  → 160M keys × 8B = 1.28GB  ≈ L1→L2 wide compaction

  Algorithmic complexity:
    - CPU K-way merge (heapq): O(N·K · log K)  — grows with both N and K
    - CPU concat+sort (numpy):  O(N·K · log(N·K)) — worst asymptotically but SIMD-fast
    - GPU radix sort:           O(N·K)           — linear, no log factor

  The GPU advantage grows with K (fan-in) because:
    - CPU merge cost = N·K·log₂(K) comparisons  → doubles each time K doubles
    - GPU cost ≈ constant passes over data       → nearly flat as K grows

  Current bottleneck: PCIe 3.0 upload (≈4 GB/s effective). On NVLink/GH200
  or PCIe 5.0, GPU advantage would be ~3–5× larger.
""")


if __name__ == "__main__":
    run_benchmark()
