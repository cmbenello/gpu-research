import datafusion
import pyarrow as pa
import numpy as np
import time
import ctypes, os

# Load GPU sort
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgpusort.so")
lib = ctypes.CDLL(lib_path)
lib.gpu_sort_keys.restype = ctypes.c_int
lib.gpu_sort_keys.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p]

def gpu_sort_int32(arr_np):
    """Sort int32 array on GPU, return permutation"""
    N = len(arr_np)
    # Encode to byte-comparable: flip sign bit, big-endian
    encoded = ((arr_np.astype(np.int32).view(np.uint32) ^ 0x80000000).byteswap()).view(np.uint8)
    perm = np.empty(N, dtype=np.uint32)
    lib.gpu_sort_keys(encoded.ctypes.data, 4, 4, N, perm.ctypes.data)
    return perm

for N in [10_000_000, 50_000_000, 100_000_000]:
    print(f"\n=== {N//1_000_000}M rows ===")
    data = np.random.randint(-2**31, 2**31-1, N, dtype=np.int32)

    # DataFusion sort
    ctx = datafusion.SessionContext()
    table = pa.table({"key": pa.array(data)})
    df = ctx.from_arrow(table)

    # Warmup
    df.sort(datafusion.col("key")).collect()

    t0 = time.time()
    result = df.sort(datafusion.col("key")).collect()
    t1 = time.time()
    df_time = t1 - t0

    # GPU sort
    # warmup
    gpu_sort_int32(data)

    t0 = time.time()
    perm = gpu_sort_int32(data)
    t1 = time.time()
    gpu_time = t1 - t0

    print(f"  DataFusion: {df_time:.3f}s")
    print(f"  GPU sort:   {gpu_time:.3f}s")
    print(f"  Speedup:    {df_time/gpu_time:.1f}x")
