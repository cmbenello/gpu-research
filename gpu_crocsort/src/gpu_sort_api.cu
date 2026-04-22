// GPU Sort C API implementation
// Wraps CUB radix sort with LSD multi-pass approach for arbitrary key sizes.

#include "gpu_sort_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cub/cub.cuh>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ── GPU Kernels ──────────────────────────────────────────────────

__global__ void api_init_identity(uint32_t* arr, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = (uint32_t)i;
}

__global__ void api_extract_uint64(
    const uint8_t* __restrict__ keys,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ out,
    uint64_t n,
    uint32_t key_stride,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t orig = perm[i];
    const uint8_t* p = keys + (uint64_t)orig * key_stride + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | p[b];
    // Data stays in low bits — CUB sorts on [0, chunk_bytes*8)
    out[i] = v;
}

// GPU-side gather: reorder rows according to permutation
__global__ void api_gather_rows(
    const uint8_t* __restrict__ src,
    const uint32_t* __restrict__ perm,
    uint8_t* __restrict__ dst,
    uint64_t n,
    uint32_t row_size
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t src_idx = perm[i];
    const uint8_t* src_row = src + (uint64_t)src_idx * row_size;
    uint8_t* dst_row = dst + i * row_size;
    // Copy row (use 8-byte copies when possible)
    uint32_t j = 0;
    for (; j + 8 <= row_size; j += 8) {
        *reinterpret_cast<uint64_t*>(dst_row + j) =
            *reinterpret_cast<const uint64_t*>(src_row + j);
    }
    for (; j < row_size; j++) {
        dst_row[j] = src_row[j];
    }
}

// ── Timing state ─────────────────────────────────────────────────

static GpuSortTiming g_last_timing = {};

struct WallTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void begin() { t0 = std::chrono::high_resolution_clock::now(); }
    double end_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ── Persistent GPU workspace (avoids re-alloc per call) ─────────

static uint64_t* s_d_sort_keys = nullptr;
static uint64_t* s_d_sort_keys_alt = nullptr;
static uint32_t* s_d_perm = nullptr;
static uint32_t* s_d_perm_alt = nullptr;
static void*     s_d_temp = nullptr;
static size_t    s_cub_temp_bytes = 0;
static uint64_t  s_workspace_capacity = 0;  // num records allocated for

static void ensure_workspace(uint64_t num_records) {
    if (num_records <= s_workspace_capacity) return;
    // Free old
    if (s_d_sort_keys) cudaFree(s_d_sort_keys);
    if (s_d_sort_keys_alt) cudaFree(s_d_sort_keys_alt);
    if (s_d_perm) cudaFree(s_d_perm);
    if (s_d_perm_alt) cudaFree(s_d_perm_alt);
    if (s_d_temp) cudaFree(s_d_temp);
    // Alloc new (round up to next power of 2 for headroom)
    uint64_t cap = 1;
    while (cap < num_records) cap <<= 1;
    cudaMalloc(&s_d_sort_keys, cap * sizeof(uint64_t));
    cudaMalloc(&s_d_sort_keys_alt, cap * sizeof(uint64_t));
    cudaMalloc(&s_d_perm, cap * sizeof(uint32_t));
    cudaMalloc(&s_d_perm_alt, cap * sizeof(uint32_t));
    // CUB temp
    s_cub_temp_bytes = 0;
    cub::DoubleBuffer<uint64_t> kb(nullptr, nullptr);
    cub::DoubleBuffer<uint32_t> vb(nullptr, nullptr);
    cub::DeviceRadixSort::SortPairs(nullptr, s_cub_temp_bytes, kb, vb, (int)cap, 0, 64);
    cudaMalloc(&s_d_temp, s_cub_temp_bytes);
    s_workspace_capacity = cap;
}

// ── Internal sort (returns d_perm pointer to sorted perm on GPU) ──

static int gpu_sort_internal(
    const uint8_t* d_keys,
    uint32_t key_size,
    uint32_t key_stride,
    uint64_t num_records,
    uint32_t** d_perm_result,
    GpuSortTiming* timing
) {
    int nthreads = 256;
    int nblks = (num_records + nthreads - 1) / nthreads;

    ensure_workspace(num_records);

    WallTimer step_timer; step_timer.begin();

    api_init_identity<<<nblks, nthreads>>>(s_d_perm, num_records);

    uint32_t* perm_in = s_d_perm;
    uint32_t* perm_out_gpu = s_d_perm_alt;

    int num_chunks = (key_size + 7) / 8;
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int byte_offset = chunk * 8;
        int chunk_bytes = std::min(8, (int)key_size - byte_offset);

        api_extract_uint64<<<nblks, nthreads>>>(
            d_keys, perm_in, s_d_sort_keys, num_records,
            key_stride, byte_offset, chunk_bytes);

        cub::DoubleBuffer<uint64_t> kb(s_d_sort_keys, s_d_sort_keys_alt);
        cub::DoubleBuffer<uint32_t> vb(perm_in, perm_out_gpu);
        size_t temp = s_cub_temp_bytes;
        cub::DeviceRadixSort::SortPairs(s_d_temp, temp, kb, vb,
            (int)num_records, 0, chunk_bytes * 8);
        perm_in = vb.Current();
        perm_out_gpu = vb.Alternate();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timing->gpu_sort_ms = step_timer.end_ms();

    *d_perm_result = perm_in;
    return 0;
}

// ── API Implementation ───────────────────────────────────────────

extern "C" int gpu_sort_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

extern "C" int gpu_query_memory(size_t* free_bytes, size_t* total_bytes) {
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" void gpu_sort_get_timing(GpuSortTiming* out) {
    if (out) *out = g_last_timing;
}

extern "C" int gpu_pin_memory(void* ptr, size_t size) {
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" int gpu_unpin_memory(void* ptr) {
    cudaError_t err = cudaHostUnregister(ptr);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" void* gpu_alloc_pinned(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    return (err == cudaSuccess) ? ptr : nullptr;
}

extern "C" void gpu_free_pinned(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

// Cached pinned host alloc — grows but never shrinks, avoids repeated cudaMallocHost.
static void*  s_pinned_cache = nullptr;
static size_t s_pinned_cache_cap = 0;

extern "C" void* gpu_pinned_cache_alloc(size_t bytes) {
    if (bytes <= s_pinned_cache_cap) return s_pinned_cache;
    if (s_pinned_cache) cudaFreeHost(s_pinned_cache);
    s_pinned_cache = nullptr;
    s_pinned_cache_cap = 0;
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) return nullptr;
    s_pinned_cache = ptr;
    s_pinned_cache_cap = bytes;
    return ptr;
}

extern "C" int gpu_sort_keys(
    const uint8_t* keys,
    uint32_t key_size,
    uint32_t key_stride,
    uint64_t num_records,
    uint32_t* perm_out
) {
    if (num_records == 0) return 0;
    if (!keys || !perm_out || key_size == 0) return -1;

    GpuSortTiming timing = {};
    WallTimer total_timer; total_timer.begin();
    WallTimer step_timer;

    // ── Upload keys to GPU ──
    step_timer.begin();
    uint64_t upload_bytes = (uint64_t)num_records * key_stride;
    uint8_t* d_keys;
    CUDA_CHECK(cudaMalloc(&d_keys, upload_bytes));
    CUDA_CHECK(cudaMemcpy(d_keys, keys, upload_bytes, cudaMemcpyHostToDevice));
    timing.upload_ms = step_timer.end_ms();

    // ── GPU sort ──
    uint32_t* d_perm_result;
    int rc = gpu_sort_internal(d_keys, key_size, key_stride, num_records, &d_perm_result, &timing);
    if (rc != 0) { cudaFree(d_keys); return rc; }

    // ── Download permutation ──
    step_timer.begin();
    CUDA_CHECK(cudaMemcpy(perm_out, d_perm_result,
                           num_records * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    timing.download_ms = step_timer.end_ms();

    cudaFree(d_keys);

    timing.gather_ms = 0;
    timing.fixup_ms = 0;
    timing.num_fixup_groups = 0;
    timing.num_fixup_records = 0;
    timing.total_ms = total_timer.end_ms();
    g_last_timing = timing;
    return 0;
}

extern "C" int gpu_sort_and_gather(
    const uint8_t* keys,
    uint32_t key_size,
    uint32_t key_stride,
    const uint8_t* payload,
    uint32_t payload_stride,
    uint64_t num_records,
    uint8_t* sorted_payload
) {
    if (num_records == 0) return 0;
    if (!keys || !payload || !sorted_payload || key_size == 0) return -1;

    GpuSortTiming timing = {};
    WallTimer total_timer; total_timer.begin();
    WallTimer step_timer;

    int nthreads = 256;
    int nblks = (num_records + nthreads - 1) / nthreads;

    // ── Upload keys + payload to GPU ──
    step_timer.begin();
    uint64_t key_bytes = (uint64_t)num_records * key_stride;
    uint64_t payload_bytes = (uint64_t)num_records * payload_stride;

    uint8_t* d_keys;
    uint8_t* d_payload;
    uint8_t* d_sorted_payload;
    CUDA_CHECK(cudaMalloc(&d_keys, key_bytes));
    CUDA_CHECK(cudaMalloc(&d_payload, payload_bytes));
    CUDA_CHECK(cudaMalloc(&d_sorted_payload, payload_bytes));

    // Overlap key and payload uploads using streams
    cudaStream_t stream_key, stream_payload;
    cudaStreamCreate(&stream_key);
    cudaStreamCreate(&stream_payload);
    cudaMemcpyAsync(d_keys, keys, key_bytes, cudaMemcpyHostToDevice, stream_key);
    cudaMemcpyAsync(d_payload, payload, payload_bytes, cudaMemcpyHostToDevice, stream_payload);
    cudaStreamSynchronize(stream_key);
    cudaStreamSynchronize(stream_payload);
    cudaStreamDestroy(stream_key);
    cudaStreamDestroy(stream_payload);
    timing.upload_ms = step_timer.end_ms();

    // ── GPU sort ──
    uint32_t* d_perm_result;
    int rc = gpu_sort_internal(d_keys, key_size, key_stride, num_records, &d_perm_result, &timing);
    if (rc != 0) {
        cudaFree(d_keys); cudaFree(d_payload); cudaFree(d_sorted_payload);
        return rc;
    }

    // ── GPU-side gather: reorder payload using permutation ──
    step_timer.begin();
    api_gather_rows<<<nblks, nthreads>>>(d_payload, d_perm_result, d_sorted_payload,
                                          num_records, payload_stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.gather_ms = step_timer.end_ms();

    // ── Download sorted payload ──
    step_timer.begin();
    CUDA_CHECK(cudaMemcpy(sorted_payload, d_sorted_payload, payload_bytes,
                           cudaMemcpyDeviceToHost));
    timing.download_ms = step_timer.end_ms();

    cudaFree(d_keys);
    cudaFree(d_payload);
    cudaFree(d_sorted_payload);

    timing.fixup_ms = 0;
    timing.num_fixup_groups = 0;
    timing.num_fixup_records = 0;
    timing.total_ms = total_timer.end_ms();
    g_last_timing = timing;
    return 0;
}

// ── Low-level device memory API ─────────────────────────────────

extern "C" void* gpu_device_alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    return (err == cudaSuccess) ? ptr : nullptr;
}

extern "C" void gpu_device_free(void* d_ptr) {
    if (d_ptr) cudaFree(d_ptr);
}

// ── Cached device buffer pool (avoids per-call cudaMalloc/cudaFree) ──
// Up to 4 slots that grow but never shrink.

#define GPU_CACHE_SLOTS 4
static void*  s_cache_ptrs[GPU_CACHE_SLOTS] = {};
static size_t s_cache_caps[GPU_CACHE_SLOTS] = {};

extern "C" void* gpu_cache_alloc(int slot, size_t bytes) {
    if (slot < 0 || slot >= GPU_CACHE_SLOTS) return gpu_device_alloc(bytes);
    if (bytes <= s_cache_caps[slot]) return s_cache_ptrs[slot];
    if (s_cache_ptrs[slot]) cudaFree(s_cache_ptrs[slot]);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) { s_cache_ptrs[slot] = nullptr; s_cache_caps[slot] = 0; return nullptr; }
    s_cache_ptrs[slot] = ptr;
    s_cache_caps[slot] = bytes;
    return ptr;
}

extern "C" void gpu_cache_free_all(void) {
    for (int i = 0; i < GPU_CACHE_SLOTS; i++) {
        if (s_cache_ptrs[i]) { cudaFree(s_cache_ptrs[i]); s_cache_ptrs[i] = nullptr; s_cache_caps[i] = 0; }
    }
}

extern "C" void gpu_memcpy_h2d(void* d_dst, const void* h_src, size_t bytes) {
    cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
}

extern "C" void gpu_memcpy_d2h(void* h_dst, const void* d_src, size_t bytes) {
    cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost);
}

extern "C" int gpu_sort_and_gather_device(
    uint8_t* d_keys, uint32_t key_size, uint32_t key_stride,
    uint8_t* d_payload, uint32_t payload_stride,
    uint64_t num_records,
    uint8_t* d_sorted_payload
) {
    if (num_records == 0) return 0;
    if (!d_keys || !d_payload || !d_sorted_payload || key_size == 0) return -1;

    GpuSortTiming timing = {};
    WallTimer total_timer; total_timer.begin();

    int nthreads = 256;
    int nblks = (num_records + nthreads - 1) / nthreads;

    timing.upload_ms = 0; // caller handled upload

    // ── GPU sort ──
    uint32_t* d_perm_result;
    int rc = gpu_sort_internal(d_keys, key_size, key_stride, num_records, &d_perm_result, &timing);
    if (rc != 0) return rc;

    // ── GPU-side gather ──
    WallTimer step_timer; step_timer.begin();
    api_gather_rows<<<nblks, nthreads>>>(d_payload, d_perm_result, d_sorted_payload,
                                          num_records, payload_stride);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;
    timing.gather_ms = step_timer.end_ms();

    timing.download_ms = 0; // caller handles download
    timing.fixup_ms = 0;
    timing.num_fixup_groups = 0;
    timing.num_fixup_records = 0;
    timing.total_ms = total_timer.end_ms();
    g_last_timing = timing;
    return 0;
}

// ── Permutation-only sort (for columnar mode) ──────────────────

extern "C" int gpu_sort_permutation_device(
    uint8_t* d_keys, uint32_t key_size, uint32_t key_stride,
    uint64_t num_records,
    uint32_t* d_perm_out
) {
    if (num_records == 0) return 0;
    if (!d_keys || !d_perm_out || key_size == 0) return -1;

    GpuSortTiming timing = {};
    WallTimer total_timer; total_timer.begin();
    timing.upload_ms = 0; // caller handled upload

    uint32_t* d_perm_result;
    int rc = gpu_sort_internal(d_keys, key_size, key_stride, num_records, &d_perm_result, &timing);
    if (rc != 0) return rc;

    // Copy permutation to caller's output buffer (device-to-device) if needed
    if (d_perm_result != d_perm_out) {
        cudaError_t err = cudaMemcpy(d_perm_out, d_perm_result,
                                      num_records * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return -1;
    }

    timing.gather_ms = 0;
    timing.download_ms = 0;
    timing.fixup_ms = 0;
    timing.num_fixup_groups = 0;
    timing.num_fixup_records = 0;
    timing.total_ms = total_timer.end_ms();
    g_last_timing = timing;
    return 0;
}

// ── GPU-side gather only (separate from sort) ──────────────────
extern "C" int gpu_gather_device(
    const uint8_t* d_payload, uint32_t payload_stride,
    const uint32_t* d_perm, uint64_t num_records,
    uint8_t* d_sorted_payload
) {
    if (num_records == 0) return 0;
    int nthreads = 256;
    int nblks = (num_records + nthreads - 1) / nthreads;
    api_gather_rows<<<nblks, nthreads>>>(d_payload, d_perm, d_sorted_payload,
                                          num_records, payload_stride);
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

// ── Stream API ──────────────────────────────────────────────────

extern "C" void* gpu_stream_create(void) {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    return (err == cudaSuccess) ? (void*)s : nullptr;
}

extern "C" void gpu_stream_sync(void* stream) {
    cudaStreamSynchronize((cudaStream_t)stream);
}

extern "C" void gpu_stream_destroy(void* stream) {
    if (stream) cudaStreamDestroy((cudaStream_t)stream);
}

extern "C" void gpu_memcpy_h2d_async(void* d_dst, const void* h_src, size_t bytes, void* stream) {
    cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

extern "C" void gpu_memcpy_d2h_async(void* h_dst, const void* d_src, size_t bytes, void* stream) {
    cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

// ── Double-buffered pinned staging (overlaps CPU copy with DMA) ─

static void*        s_pipe_stage[2] = {nullptr, nullptr};
static size_t       s_pipe_cap = 0;
static cudaStream_t s_pipe_streams[2] = {nullptr, nullptr};

static void ensure_pipeline(size_t max_block_size) {
    if (max_block_size <= s_pipe_cap) return;
    for (int i = 0; i < 2; i++) {
        if (s_pipe_stage[i]) cudaFreeHost(s_pipe_stage[i]);
        cudaMallocHost(&s_pipe_stage[i], max_block_size);
    }
    if (!s_pipe_streams[0]) {
        cudaStreamCreate(&s_pipe_streams[0]);
        cudaStreamCreate(&s_pipe_streams[1]);
    }
    s_pipe_cap = max_block_size;
}

extern "C" void gpu_upload_pipelined(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks) {
    if (num_blocks == 0) return;

    size_t max_block = 0;
    for (uint32_t i = 0; i < num_blocks; i++)
        if (src_sizes[i] > max_block) max_block = src_sizes[i];

    ensure_pipeline(max_block);

    size_t offset = 0;
    int cur = 0;

    for (uint32_t i = 0; i < num_blocks; i++) {
        // Wait for previous transfer on this staging buffer to complete
        cudaStreamSynchronize(s_pipe_streams[cur]);
        // CPU memcpy block → pinned staging buffer (fast, host-to-host)
        memcpy(s_pipe_stage[cur], src_ptrs[i], src_sizes[i]);
        // Async DMA from pinned staging → GPU (truly async because staging is pinned)
        cudaMemcpyAsync((uint8_t*)d_dst + offset, s_pipe_stage[cur], src_sizes[i],
                        cudaMemcpyHostToDevice, s_pipe_streams[cur]);
        offset += src_sizes[i];
        cur ^= 1;
    }
    cudaStreamSynchronize(s_pipe_streams[0]);
    cudaStreamSynchronize(s_pipe_streams[1]);
}

// Direct pinned upload: pin each block, queue DMA, unpin after transfer.
// Interleaves pin→DMA→unpin per block so the next pin overlaps with DMA.
extern "C" void gpu_upload_direct_pinned(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks) {
    if (num_blocks == 0) return;

    // Use two streams to overlap pin+queue of next block with DMA of current
    static cudaStream_t s_upload_streams[2] = {nullptr, nullptr};
    if (!s_upload_streams[0]) {
        cudaStreamCreate(&s_upload_streams[0]);
        cudaStreamCreate(&s_upload_streams[1]);
    }

    size_t offset = 0;
    int cur = 0;
    std::vector<void*> pinned_ptrs;
    pinned_ptrs.reserve(num_blocks);

    for (uint32_t i = 0; i < num_blocks; i++) {
        void* ptr = const_cast<void*>(src_ptrs[i]);
        if (cudaHostRegister(ptr, src_sizes[i], cudaHostRegisterDefault) == cudaSuccess) {
            pinned_ptrs.push_back(ptr);
        }
        cudaMemcpyAsync((uint8_t*)d_dst + offset, src_ptrs[i], src_sizes[i],
                        cudaMemcpyHostToDevice, s_upload_streams[cur]);
        offset += src_sizes[i];
        cur ^= 1;
    }
    cudaStreamSynchronize(s_upload_streams[0]);
    cudaStreamSynchronize(s_upload_streams[1]);

    // Unpin all blocks
    for (auto* ptr : pinned_ptrs) {
        cudaHostUnregister(ptr);
    }
}

// Direct pinned download: pin destination, single DMA, unpin.
extern "C" void gpu_download_direct_pinned(void* h_dst, const void* d_src, size_t bytes) {
    if (bytes == 0) return;
    bool pinned = (cudaHostRegister(h_dst, bytes, cudaHostRegisterDefault) == cudaSuccess);
    cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost);
    if (pinned) cudaHostUnregister(h_dst);
}

extern "C" void gpu_download_pipelined(void* h_dst, const void* d_src, size_t bytes, size_t chunk_size) {
    if (bytes == 0) return;
    if (chunk_size == 0) chunk_size = 4 * 1024 * 1024; // 4MB default chunks

    ensure_pipeline(chunk_size);

    size_t offset = 0;
    int cur = 0;
    // First chunk: start async DMA
    size_t first_sz = std::min(chunk_size, bytes);
    cudaMemcpyAsync(s_pipe_stage[0], (const uint8_t*)d_src, first_sz,
                    cudaMemcpyDeviceToHost, s_pipe_streams[0]);
    offset = first_sz;
    cur = 1;

    size_t copy_offset = 0;
    while (copy_offset < bytes) {
        // Start next DMA chunk if there's more data
        size_t next_sz = 0;
        if (offset < bytes) {
            next_sz = std::min(chunk_size, bytes - offset);
            cudaMemcpyAsync(s_pipe_stage[cur], (const uint8_t*)d_src + offset, next_sz,
                            cudaMemcpyDeviceToHost, s_pipe_streams[cur]);
        }
        // Wait for previous chunk and copy to host
        int prev = cur ^ 1;
        cudaStreamSynchronize(s_pipe_streams[prev]);
        size_t prev_sz = std::min(chunk_size, bytes - copy_offset);
        memcpy((uint8_t*)h_dst + copy_offset, s_pipe_stage[prev], prev_sz);
        copy_offset += prev_sz;
        if (next_sz > 0) offset += next_sz;
        cur ^= 1;
    }
}

// ── Chunked pipelined upload: fixed-size pinned staging, double-buffered ──
// Uses small (128MB) pinned staging buffers instead of per-block cudaHostRegister.
// Eliminates pin/unpin overhead by reusing pre-allocated pinned memory.
static void*        s_chunk_stage[2] = {nullptr, nullptr};
static size_t       s_chunk_cap = 0;
static cudaStream_t s_chunk_streams[2] = {nullptr, nullptr};

static void ensure_chunk_pipeline(size_t chunk_size) {
    if (chunk_size <= s_chunk_cap) return;
    for (int i = 0; i < 2; i++) {
        if (s_chunk_stage[i]) cudaFreeHost(s_chunk_stage[i]);
        cudaMallocHost(&s_chunk_stage[i], chunk_size);
    }
    if (!s_chunk_streams[0]) {
        cudaStreamCreate(&s_chunk_streams[0]);
        cudaStreamCreate(&s_chunk_streams[1]);
    }
    s_chunk_cap = chunk_size;
}

extern "C" void gpu_upload_chunked(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks) {
    if (num_blocks == 0) return;

    const size_t CHUNK = 128 * 1024 * 1024; // 128MB staging buffers
    ensure_chunk_pipeline(CHUNK);

    size_t d_offset = 0; // offset into device destination
    int cur = 0;

    for (uint32_t bi = 0; bi < num_blocks; bi++) {
        const uint8_t* src = (const uint8_t*)src_ptrs[bi];
        size_t remaining = src_sizes[bi];
        size_t s_offset = 0;

        while (remaining > 0) {
            size_t chunk_sz = std::min(CHUNK, remaining);

            // Wait for previous DMA on this staging buffer to finish
            cudaStreamSynchronize(s_chunk_streams[cur]);

            // CPU memcpy: source → pinned staging (host bandwidth, ~20 GB/s)
            memcpy(s_chunk_stage[cur], src + s_offset, chunk_sz);

            // Async DMA: pinned staging → GPU (PCIe bandwidth)
            cudaMemcpyAsync((uint8_t*)d_dst + d_offset, s_chunk_stage[cur], chunk_sz,
                            cudaMemcpyHostToDevice, s_chunk_streams[cur]);

            d_offset += chunk_sz;
            s_offset += chunk_sz;
            remaining -= chunk_sz;
            cur ^= 1;
        }
    }
    cudaStreamSynchronize(s_chunk_streams[0]);
    cudaStreamSynchronize(s_chunk_streams[1]);
}

// ── Persistent pinned staging buffer ────────────────────────────

static void*  s_staging_buf = nullptr;
static size_t s_staging_cap = 0;

static void ensure_staging(size_t needed) {
    if (needed <= s_staging_cap) return;
    if (s_staging_buf) cudaFreeHost(s_staging_buf);
    // Round up to 16MB granularity
    size_t cap = ((needed + (16 << 20) - 1) / (16 << 20)) * (16 << 20);
    cudaMallocHost(&s_staging_buf, cap);
    s_staging_cap = cap;
}

extern "C" void gpu_upload_staged(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks) {
    size_t total = 0;
    for (uint32_t i = 0; i < num_blocks; i++) total += src_sizes[i];
    if (total == 0) return;

    ensure_staging(total);

    // Concat into pinned staging buffer (fast CPU memcpy)
    size_t offset = 0;
    for (uint32_t i = 0; i < num_blocks; i++) {
        memcpy((uint8_t*)s_staging_buf + offset, src_ptrs[i], src_sizes[i]);
        offset += src_sizes[i];
    }

    // Single DMA from pinned staging to GPU (true DMA, no internal staging)
    cudaMemcpy(d_dst, s_staging_buf, total, cudaMemcpyHostToDevice);
}
