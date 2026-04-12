// ============================================================================
// GPU External Merge Sort — Triple-Buffered Pipeline + K-Way Streaming Merge
//
// Architecture (Stehle-Jacobsen SIGMOD 2017 pipeline model):
//   Phase 1: Triple-buffered run generation
//     Stream 0: H2D upload chunk i+2
//     Stream 1: GPU sort chunk i+1
//     Stream 2: D2H download chunk i
//     → GPU compute completely hidden behind PCIe transfer
//
//   Phase 2: GPU streaming K-way merge (novel)
//     Stream chunks from K sorted runs through GPU, merge on GPU, stream back.
//     K-way single pass instead of log2(K) cascade passes → K× less PCIe traffic.
//     Cursor-based with boundary detection: only output records guaranteed correct.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -DEXTERNAL_SORT_MAIN -Iinclude \
//        src/external_sort.cu src/run_generation.cu src/merge.cu -o external_sort
// Run:   ./external_sort --total-gb 20
// ============================================================================

#include "record.cuh"
#include "ovc.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <cub/cub.cuh>
#include <sys/mman.h>  // madvise for huge pages

// Forward-declare the in-HBM sort from host_sort.cu
enum MergeStrategy { STRATEGY_2WAY, STRATEGY_KWAY };
void gpu_crocsort_in_hbm(uint8_t* d_data, uint64_t num_records,
                           bool verify, MergeStrategy strategy);

// Forward-declare kernel launchers
struct PairDesc2Way {
    uint64_t a_byte_offset; int a_count;
    uint64_t b_byte_offset; int b_count;
    uint64_t out_byte_offset; int first_block;
};
extern "C" void launch_run_generation(
    const uint8_t*, uint8_t*, uint32_t*, uint64_t,
    SparseEntry*, int*, int, cudaStream_t);
extern "C" void launch_merge_2way(
    const uint8_t*, uint8_t*, const PairDesc2Way*, int, int, cudaStream_t);

// K-way merge tree (from merge.cu + host_sort.cu)
static constexpr int KWAY_K = 8;
struct KWayPartition {
    int      src_rec_start[KWAY_K];
    int      src_rec_count[KWAY_K];
    uint64_t src_byte_off[KWAY_K];
    uint64_t out_byte_offset;
    int      total_records;
};
extern "C" void launch_merge_kway(
    const uint8_t*, uint8_t*, const KWayPartition*, int, int, cudaStream_t);

// Run struct matching host_sort.cu's Run (identical layout)
struct Run { uint64_t byte_offset; uint64_t num_records; };

// Sample-based partitioning (from host_sort.cu, made non-static)
void compute_sample_partitions(
    const uint8_t* d_runs, const std::vector<Run>& group_runs,
    int K, int P, uint64_t out_base_offset,
    std::vector<KWayPartition>& out_partitions);

// ── Timing ──────────────────────────────────────────────────────────

struct WallTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void begin() { t0 = std::chrono::high_resolution_clock::now(); }
    double end_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ── Host-side upper_bound for boundary computation ──────────────────

static uint64_t host_upper_bound(const uint8_t* data, uint64_t n, const uint8_t* target_key) {
    uint64_t lo = 0, hi = n;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (key_compare(data + mid * RECORD_SIZE, target_key, KEY_SIZE) <= 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ── GPU kernels for CUB radix sort pipeline ─────────────────────────

// Extract 8-byte sort key (big-endian) from each record for CUB radix sort
__global__ void extract_sort_keys_kernel(
    const uint8_t* __restrict__ records,
    uint64_t* __restrict__ keys,
    uint32_t* __restrict__ indices,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* rec = records + i * RECORD_SIZE;
    // Big-endian 8-byte key for correct lexicographic ordering
    uint64_t k = 0;
    for (int b = 0; b < 8; b++) k = (k << 8) | rec[b];
    keys[i] = k;
    indices[i] = (uint32_t)i;
}

// Reorder records from src to dst using sorted index array
__global__ void reorder_records_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const uint32_t* __restrict__ sorted_indices,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t src_idx = sorted_indices[i];
    const uint8_t* s = src + (uint64_t)src_idx * RECORD_SIZE;
    uint8_t* d = dst + i * RECORD_SIZE;
    // Copy 100-byte record (25 × uint32)
    for (int b = 0; b < RECORD_SIZE; b += 4) {
        *reinterpret_cast<uint32_t*>(d + b) = *reinterpret_cast<const uint32_t*>(s + b);
    }
}

// Initialize uint32 array to identity [0, 1, 2, ..., N-1]
__global__ void init_identity_kernel(uint32_t* __restrict__ arr, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = (uint32_t)i;
}

// Extract uint64 sort key from KEY_SIZE-stride key buffer (for CUB merge sort)
__global__ void extract_uint64_from_keys_kernel(
    const uint8_t* __restrict__ key_buffer,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* k = key_buffer + i * KEY_SIZE;
    uint64_t v = 0;
    for (int b = 0; b < 8; b++) v = (v << 8) | k[b];
    sort_keys[i] = v;
}

// ── GPU key extraction kernel ────────────────────────────────────────
// Extract KEY_SIZE bytes from each sorted record into a contiguous key array.
// Runs at HBM bandwidth (~672 GB/s), essentially free compared to PCIe.

__global__ void extract_keys_kernel(
    const uint8_t* __restrict__ sorted_records,
    uint8_t* __restrict__ key_buffer,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* src = sorted_records + i * RECORD_SIZE;
    uint8_t* dst = key_buffer + i * KEY_SIZE;
    // Copy 10-byte key
    for (int b = 0; b < KEY_SIZE; b++) dst[b] = src[b];
}

// Pre-allocated workspace for sort_chunk_on_gpu (allocated once, reused per chunk)
struct SortWorkspace {
    // CUB radix sort workspace
    uint64_t* d_keys = nullptr;
    uint64_t* d_keys_alt = nullptr;
    uint32_t* d_indices = nullptr;
    uint32_t* d_indices_alt = nullptr;
    uint64_t capacity = 0;

    void allocate(uint64_t max_records) {
        if (capacity >= max_records) return;
        free();
        capacity = max_records;
        CUDA_CHECK(cudaMalloc(&d_keys, max_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_alt, max_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_indices, max_records * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_indices_alt, max_records * sizeof(uint32_t)));
    }
    void free() {
        if (d_keys) { cudaFree(d_keys); d_keys = nullptr; }
        if (d_keys_alt) { cudaFree(d_keys_alt); d_keys_alt = nullptr; }
        if (d_indices) { cudaFree(d_indices); d_indices = nullptr; }
        if (d_indices_alt) { cudaFree(d_indices_alt); d_indices_alt = nullptr; }
        capacity = 0;
    }
};

// ============================================================================
// External Sort Engine
// ============================================================================

class ExternalGpuSort {
    static constexpr int NBUFS = 3;
    size_t gpu_budget;
    uint64_t buf_records;
    size_t buf_bytes;
    uint8_t* d_buf[NBUFS];
    uint8_t* h_pin[NBUFS];
    cudaStream_t streams[NBUFS];
    cudaEvent_t events[NBUFS];

    // Persistent key buffer
    uint8_t* d_key_buffer;
    uint64_t key_buffer_capacity;
    std::vector<uint64_t> run_key_offsets;

    // Pre-allocated sort workspace (avoids cudaMalloc per chunk)
    SortWorkspace sort_ws;

public:
    struct TimingResult {
        double run_gen_ms, merge_ms, total_ms;
        int num_runs, merge_passes;
        double pcie_h2d_gb, pcie_d2h_gb;
        uint8_t* sorted_output;  // pointer to sorted data (caller must free())
    };

    ExternalGpuSort();
    ~ExternalGpuSort();
    TimingResult sort(uint8_t* h_data, uint64_t num_records);

private:
    struct RunInfo { uint64_t host_offset; uint64_t num_records; };

    void sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_scratch,
                            uint64_t n, cudaStream_t s);

    std::vector<RunInfo> generate_runs_pipelined(
        uint8_t* h_data, uint64_t num_records,
        double& ms, double& h2d, double& d2h);

    uint8_t* streaming_merge(uint8_t* h_data, uint64_t num_records,
                              std::vector<RunInfo>& runs,
                              double& ms, int& passes, double& h2d, double& d2h,
                              uint32_t* h_perm_prealloc);

};

ExternalGpuSort::ExternalGpuSort() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Sort buffers get 65% of GPU memory (3 × ~5.5GB each)
    // Key buffer allocated separately in sort() based on actual data size
    gpu_budget = (size_t)(free_mem * 0.65);
    key_buffer_capacity = 0; // computed dynamically
    d_key_buffer = nullptr; // allocated lazily in sort()
    buf_records = (gpu_budget / NBUFS) / RECORD_SIZE;
    buf_bytes = buf_records * RECORD_SIZE;

    printf("[ExternalSort] GPU: %.2f GB free, budget: %.2f GB\n",
           free_mem/1e9, gpu_budget/1e9);
    printf("[ExternalSort] Triple-buffer: %d × %.2f GB (%.0f M records each)\n",
           NBUFS, buf_bytes/1e9, buf_records/1e6);

    for (int i = 0; i < NBUFS; i++) {
        CUDA_CHECK(cudaMallocHost(&h_pin[i], buf_bytes));
        CUDA_CHECK(cudaMalloc(&d_buf[i], buf_bytes));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&events[i]));
    }
}

ExternalGpuSort::~ExternalGpuSort() {
    for (int i = 0; i < NBUFS; i++) {
        cudaFreeHost(h_pin[i]);
        cudaFree(d_buf[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
    if (d_key_buffer) cudaFree(d_key_buffer);
    sort_ws.free();
}

// Sort a chunk on GPU using CUB radix sort on 8-byte key prefix + record reorder.
// CUB does 8 radix passes at near-HBM bandwidth — replaces bitonic sort + K-way merge.
void ExternalGpuSort::sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_scratch,
                                         uint64_t n, cudaStream_t s) {
    int nthreads = 256;
    int nblks = (n + nthreads - 1) / nthreads;

    // Step 1: Extract big-endian uint64 keys + initialize index array
    extract_sort_keys_kernel<<<nblks, nthreads, 0, s>>>(
        d_in, sort_ws.d_keys, sort_ws.d_indices, n);

    // Step 2: CUB radix sort (key, index) pairs — pre-allocated double buffers
    cub::DoubleBuffer<uint64_t> d_keys_buf(sort_ws.d_keys, sort_ws.d_keys_alt);
    cub::DoubleBuffer<uint32_t> d_idx_buf(sort_ws.d_indices, sort_ws.d_indices_alt);

    size_t temp_bytes = buf_bytes;  // d_scratch is buf_bytes large
    cub::DeviceRadixSort::SortPairs(d_scratch, temp_bytes,
        d_keys_buf, d_idx_buf, (int)n, 0, 64, s);
    CUDA_CHECK(cudaStreamSynchronize(s));  // ensure sort done before reusing d_scratch

    // Step 3: Reorder full records using sorted indices (d_in → d_scratch)
    reorder_records_kernel<<<nblks, nthreads, 0, s>>>(
        d_in, d_scratch, d_idx_buf.Current(), n);

    // Copy sorted result back to d_in
    CUDA_CHECK(cudaMemcpyAsync(d_in, d_scratch, n * RECORD_SIZE,
                                cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// gpu_merge_inplace removed — CUB radix sort replaced bitonic+K-way merge

// ════════════════════════════════════════════════════════════════════
// Phase 1: Triple-Buffered Run Generation
// ════════════════════════════════════════════════════════════════════

std::vector<ExternalGpuSort::RunInfo>
ExternalGpuSort::generate_runs_pipelined(
    uint8_t* h_data, uint64_t num_records,
    double& ms, double& h2d, double& d2h
) {
    std::vector<RunInfo> runs;
    h2d = d2h = 0;
    WallTimer timer; timer.begin();

    // Count total chunks
    int total_chunks = (num_records + buf_records - 1) / buf_records;

    // Pipeline: for chunk i, use buf[i % 3]
    // Stage 0: upload (H2D)
    // Stage 1: sort (GPU compute)  — needs TWO buffers (in + scratch)
    // Stage 2: download (D2H)
    //
    // Since sort needs 2 buffers but we only have 3 total, we can't
    // fully overlap all 3 stages. Instead: overlap upload with download,
    // sort sequentially (it uses 2 of the 3 buffers).

    // Pipelined run generation:
    // - Use 2 main buffers (d_buf[0], d_buf[1]) alternating, d_buf[2] as sort scratch
    // - Overlap: D2H of chunk N on stream[2] runs concurrently with
    //   CPU staging of chunk N+1, then H2D of chunk N+1 on stream[0]
    //   overlaps with final D2H→host memcpy of chunk N

    for (int c = 0; c < total_chunks; c++) {
        uint64_t offset = (uint64_t)c * buf_records;
        uint64_t cur_n = std::min(buf_records, num_records - offset);
        uint64_t cur_bytes = cur_n * RECORD_SIZE;
        int cur = c % 2;
        int scratch = 2;

        // Start H2D upload on stream[0] — runs concurrently with previous D2H on stream[2]
        // (bidirectional PCIe: H2D and D2H use separate DMA engines)
        CUDA_CHECK(cudaMemcpyAsync(d_buf[cur], h_data + offset * RECORD_SIZE, cur_bytes,
                                    cudaMemcpyHostToDevice, streams[0]));
        h2d += cur_bytes;

        // Record event when H2D completes, make sort stream wait for it
        CUDA_CHECK(cudaEventRecord(events[0], streams[0]));
        CUDA_CHECK(cudaStreamWaitEvent(streams[1], events[0], 0));

        // Also wait for previous D2H to finish (it reads d_buf[prev], not d_buf[cur],
        // but the sort needs d_buf[scratch]=d_buf[2] which might still be in use
        // by the previous sort's internal operations)
        CUDA_CHECK(cudaStreamSynchronize(streams[2]));

        // Sort (GPU-side wait for H2D via event — no CPU blocking)
        sort_chunk_on_gpu(d_buf[cur], d_buf[scratch], cur_n, streams[1]);

        // Extract keys to persistent GPU key buffer
        uint64_t key_off = run_key_offsets.empty() ? 0 :
            run_key_offsets.back() + (runs.empty() ? 0 : runs.back().num_records * KEY_SIZE);
        if (d_key_buffer && (key_off + cur_n * KEY_SIZE) <= key_buffer_capacity) {
            int nthreads = 256;
            int nblocks_k = (cur_n + nthreads - 1) / nthreads;
            extract_keys_kernel<<<nblocks_k, nthreads, 0, streams[1]>>>(
                d_buf[cur], d_key_buffer + key_off, cur_n);
            CUDA_CHECK(cudaStreamSynchronize(streams[1]));
            run_key_offsets.push_back(key_off);
        }

        // Download directly to h_data (pinned memory — no intermediate copy)
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset * RECORD_SIZE, d_buf[cur], cur_bytes,
                                    cudaMemcpyDeviceToHost, streams[2]));
        d2h += cur_bytes;

        runs.push_back({offset * RECORD_SIZE, cur_n});
        printf("\r  Run %d/%d: %.1f MB sorted    ", c+1, total_chunks,
               cur_bytes/(1024.0*1024.0));
        fflush(stdout);
    }

    // Finalize: wait for last D2H to complete
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));
    printf("\n");
    ms = timer.end_ms();
    return runs;
}

// ════════════════════════════════════════════════════════════════════
// Phase 2: Key-Only GPU Merge
//
// Instead of sending full 100B records over PCIe, send only 10B keys.
// GPU merges keys and produces a permutation array.
// CPU gathers full records using the permutation.
//
// For 60GB dataset: keys = 6GB (fits in GPU), perm = 2.4GB download.
// Total PCIe: ~8.4GB vs ~120GB with full-record approach. 14x less.
// ════════════════════════════════════════════════════════════════════

// Key-only merge descriptor (must match merge.cu)
struct KeyMergePair {
    uint64_t a_key_offset;
    int      a_count;
    uint64_t b_key_offset;
    int      b_count;
    uint64_t out_key_offset;
    uint64_t out_perm_offset;
    uint64_t a_perm_offset;
    uint64_t b_perm_offset;
    int      first_block;
};

extern "C" void launch_merge_keys_only(
    const uint8_t*, uint8_t*, const uint32_t*, uint32_t*,
    const KeyMergePair*, int, int, cudaStream_t);

uint8_t* ExternalGpuSort::streaming_merge(
    uint8_t* h_data, uint64_t num_records,
    std::vector<RunInfo>& runs,
    double& ms, int& passes, double& h2d, double& d2h,
    uint32_t* h_perm_prealloc
) {
    if (runs.size() <= 1) { ms = 0; passes = 0; h2d = d2h = 0; return nullptr; }

    int K = (int)runs.size();
    uint64_t total_bytes = num_records * RECORD_SIZE;
    uint64_t total_keys_bytes = num_records * KEY_SIZE;
    uint64_t total_perm_bytes = num_records * sizeof(uint32_t);

    // Detailed merge phase profiling
    auto tpoint = [](const char* label) {
        static std::chrono::high_resolution_clock::time_point prev;
        auto now = std::chrono::high_resolution_clock::now();
        if (label[0] != '_')
            printf("    [merge] %-30s %6.0f ms\n", label,
                   std::chrono::duration<double,std::milli>(now - prev).count());
        prev = now;
    };
    tpoint("_start");

    WallTimer timer; timer.begin();

    // Build global index mapping
    std::vector<uint64_t> merge_key_offsets(K);
    std::vector<uint64_t> run_global_base(K);
    uint64_t global_idx = 0;
    for (int r = 0; r < K; r++) {
        run_global_base[r] = global_idx;
        global_idx += runs[r].num_records;
    }

    // ── Step 1+2: Get keys onto GPU ──
    // Check if keys were retained on GPU during run generation
    bool keys_retained = (d_key_buffer != nullptr &&
                          (int)run_key_offsets.size() == K);

    uint8_t *d_keys_in, *d_keys_out;
    uint32_t *d_perm_in, *d_perm_out;
    uint8_t* h_keys = nullptr;  // only allocated if keys not retained
    bool allocated_keys_gpu = false;

    if (keys_retained) {
        printf("  Keys already on GPU from run generation (%.2f GB, saved upload!)\n",
               total_keys_bytes / 1e9);
        // Keys are in d_key_buffer at known offsets — use directly as d_keys_in
        d_keys_in = d_key_buffer;
        for (int r = 0; r < K; r++) merge_key_offsets[r] = run_key_offsets[r];
    } else {
        // Fallback: extract keys on CPU and upload
        printf("  Extracting %lu keys (%.2f GB) on CPU...\n", num_records, total_keys_bytes/1e9);
        WallTimer ext_timer; ext_timer.begin();
        CUDA_CHECK(cudaMallocHost(&h_keys, total_keys_bytes));
        uint64_t key_off = 0;
        for (int r = 0; r < K; r++) {
            merge_key_offsets[r] = key_off;
            const uint8_t* run_data = h_data + runs[r].host_offset;
            uint8_t* key_dst = h_keys + key_off;
            for (uint64_t i = 0; i < runs[r].num_records; i++)
                memcpy(key_dst + i * KEY_SIZE, run_data + i * RECORD_SIZE, KEY_SIZE);
            key_off += runs[r].num_records * KEY_SIZE;
        }
        printf("    Extracted in %.0f ms\n", ext_timer.end_ms());

        printf("  Uploading %.2f GB keys to GPU...\n", total_keys_bytes/1e9);
        CUDA_CHECK(cudaMalloc(&d_keys_in, total_keys_bytes));
        allocated_keys_gpu = true;
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys, total_keys_bytes, cudaMemcpyHostToDevice));
        h2d += total_keys_bytes;
    }

    tpoint("keys ready");

    // ── Step 3: CUB radix sort ALL keys + permutation in ONE pass ──
    passes = 1;
    printf("  CUB radix sort on all %lu keys (single pass)...\n", num_records);

    // Try to reuse sort buffers as merge workspace. Falls back to fresh alloc if too small.
    size_t needed_buf0 = num_records * 2 * sizeof(uint64_t); // keys + keys_alt
    size_t needed_buf1 = num_records * 2 * sizeof(uint32_t) + 256*1024*1024; // perm + perm_alt + temp estimate
    bool reuse_bufs = (d_buf[0] && d_buf[1] && needed_buf0 <= buf_bytes && needed_buf1 <= buf_bytes);

    uint64_t* d_sort_keys; uint64_t* d_sort_keys_alt;
    void* d_temp; size_t cub_temp_bytes;
    uint8_t* d_merge_arena = nullptr;

    if (reuse_bufs) {
        d_sort_keys = (uint64_t*)d_buf[0];
        d_sort_keys_alt = d_sort_keys + num_records;
        d_perm_in = (uint32_t*)d_buf[1];
        d_perm_out = d_perm_in + num_records;
        d_temp = (void*)(d_perm_out + num_records);
        cub_temp_bytes = buf_bytes - num_records * 2 * sizeof(uint32_t);
    } else {
        // Free sort buffers and allocate arena using stream-ordered allocation
        // cudaFreeAsync + cudaMallocAsync is faster than sync free + sync alloc
        for (int i = 0; i < NBUFS; i++) {
            if (d_buf[i]) { cudaFreeAsync(d_buf[i], streams[0]); d_buf[i] = nullptr; }
        }
        // Query CUB temp requirement
        cub::DoubleBuffer<uint64_t> dk(nullptr,nullptr);
        cub::DoubleBuffer<uint32_t> dp(nullptr,nullptr);
        cub_temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, dk, dp, (int)num_records, 0, 64, streams[0]);
        size_t arena_sz = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + cub_temp_bytes;
        CUDA_CHECK(cudaMallocAsync(&d_merge_arena, arena_sz, streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        d_sort_keys = (uint64_t*)d_merge_arena;
        d_sort_keys_alt = d_sort_keys + num_records;
        d_perm_in = (uint32_t*)(d_sort_keys_alt + num_records);
        d_perm_out = d_perm_in + num_records;
        d_temp = (void*)(d_perm_out + num_records);
    }

    tpoint("alloc merge workspace");

    // Init perm + extract keys — both on GPU, overlapped on same stream
    {
        int nt = 256, nb = (num_records + nt - 1) / nt;
        init_identity_kernel<<<nb, nt, 0, streams[0]>>>(d_perm_in, num_records);
        extract_uint64_from_keys_kernel<<<nb, nt, 0, streams[0]>>>(
            d_keys_in, d_sort_keys, num_records);
    }

    // Free key buffers — no longer needed
    if (allocated_keys_gpu && d_keys_in) { CUDA_CHECK(cudaFree(d_keys_in)); d_keys_in = nullptr; }
    if (d_keys_out) { CUDA_CHECK(cudaFree(d_keys_out)); d_keys_out = nullptr; }
    if (d_key_buffer) { cudaFree(d_key_buffer); d_key_buffer = nullptr; }
    tpoint("init perm + extract keys + free bufs");

    // CUB sort (uint64 key, uint32 perm) pairs
    cub::DoubleBuffer<uint64_t> d_sortkey_buf(d_sort_keys, d_sort_keys_alt);
    cub::DoubleBuffer<uint32_t> d_perm_buf(d_perm_in, d_perm_out);

    cub::DeviceRadixSort::SortPairs(d_temp, cub_temp_bytes,
        d_sortkey_buf, d_perm_buf, (int)num_records, 0, 64, streams[0]);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    tpoint("CUB radix sort done");

    d_perm_in = d_perm_buf.Current();
    printf("  Downloading permutation (%.2f GB)...\n", total_perm_bytes/1e9);
    uint32_t* h_perm = h_perm_prealloc;  // pre-allocated in sort() before run gen
    if (!h_perm) { CUDA_CHECK(cudaMallocHost(&h_perm, total_perm_bytes)); } // fallback
    CUDA_CHECK(cudaMemcpy(h_perm, d_perm_in, total_perm_bytes, cudaMemcpyDeviceToHost));
    d2h += total_perm_bytes;
    tpoint("perm downloaded");

    // Free merge workspace (async to avoid blocking)
    if (d_merge_arena) { cudaFreeAsync(d_merge_arena, streams[0]); }
    for (int i = 0; i < NBUFS; i++) {
        if (d_buf[i]) { cudaFreeAsync(d_buf[i], streams[0]); d_buf[i] = nullptr; }
    }
    if (h_keys) CUDA_CHECK(cudaFreeHost(h_keys));
    tpoint("freed GPU memory");

    // ── Step 5: Multi-threaded CPU value gather using permutation ──
    int num_threads = std::min(48, (int)std::thread::hardware_concurrency());
    printf("  CPU value gather (%.2f GB, %d threads)...\n", total_bytes/1e9, num_threads);
    WallTimer gather_timer; gather_timer.begin();

    // Use regular malloc for output buffer — pinned memory (cudaMallocHost) has
    // slower write performance due to write-combining, which hurts the scatter pattern
    uint8_t* h_output = (uint8_t*)malloc(total_bytes);
    if (!h_output) { fprintf(stderr, "malloc failed for gather output\n"); ms = timer.end_ms(); return nullptr; }
    madvise(h_output, total_bytes, MADV_HUGEPAGE);

    // Pre-compute run lookup table for O(1) run_id lookup instead of O(K) scan
    // run_global_base is sorted, so we can use it directly
    // But for speed, build a direct lookup: global_idx → (run_id, idx_in_run)
    // For K <= 64, binary search is fast enough

    // Pre-compute source pointers for all records in a block, then copy.
    // This separates the "find source" phase (random permutation reads) from
    // the "copy data" phase (random source reads + sequential writes).
    // Deeper prefetch pipeline hides more DRAM latency.
    auto gather_worker = [&](uint64_t start, uint64_t end) {
        constexpr int BLOCK = 256;  // process 256 records at a time
        const uint8_t* src_ptrs[BLOCK];

        for (uint64_t base = start; base < end; base += BLOCK) {
            int count = std::min((uint64_t)BLOCK, end - base);

            // Phase 1: Resolve all source pointers + prefetch
            for (int j = 0; j < count; j++) {
                uint32_t src_global = h_perm[base + j];
                int run_id = K - 1;
                for (int r = 0; r < K - 1; r++) {
                    if (src_global < run_global_base[r+1]) { run_id = r; break; }
                }
                uint64_t idx_in_run = src_global - run_global_base[run_id];
                src_ptrs[j] = h_data + runs[run_id].host_offset + idx_in_run * RECORD_SIZE;
                __builtin_prefetch(src_ptrs[j], 0, 0);  // prefetch source record
            }

            // Phase 2: Copy all records (sources should be in cache from prefetch)
            for (int j = 0; j < count; j++) {
                memcpy(h_output + (base + j) * RECORD_SIZE, src_ptrs[j], RECORD_SIZE);
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    uint64_t chunk_sz = (num_records + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        uint64_t start = (uint64_t)t * chunk_sz;
        uint64_t end = std::min(start + chunk_sz, num_records);
        if (start < end) {
            threads.emplace_back(gather_worker, start, end);
        }
    }
    for (auto& t : threads) t.join();

    double gather_ms = gather_timer.end_ms();
    printf("    Gathered in %.0f ms (%.2f GB/s)\n",
           gather_ms, total_bytes / (gather_ms * 1e6));

    CUDA_CHECK(cudaFreeHost(h_perm));
    ms = timer.end_ms();
    return h_output;  // caller owns this buffer (allocated with malloc)
}

// ════════════════════════════════════════════════════════════════════
// Main Entry Point
// ════════════════════════════════════════════════════════════════════

ExternalGpuSort::TimingResult ExternalGpuSort::sort(uint8_t* h_data, uint64_t num_records) {
    TimingResult r = {};
    if (num_records <= 1) return r;
    uint64_t total_bytes = num_records * RECORD_SIZE;
    printf("[ExternalSort] Sorting %lu records (%.2f GB)\n\n", num_records, total_bytes/1e9);

    // Fast path: fits in one buffer
    if (num_records <= buf_records) {
        printf("  Data fits in GPU — single-chunk sort\n");
        sort_ws.allocate(num_records);
        WallTimer t; t.begin();
        CUDA_CHECK(cudaMemcpy(d_buf[0], h_data, total_bytes, cudaMemcpyHostToDevice));
        sort_chunk_on_gpu(d_buf[0], d_buf[1], num_records, streams[0]);
        CUDA_CHECK(cudaMemcpy(h_data, d_buf[0], total_bytes, cudaMemcpyDeviceToHost));
        r.total_ms = r.run_gen_ms = t.end_ms();
        r.num_runs = 1;
        r.pcie_h2d_gb = r.pcie_d2h_gb = total_bytes / 1e9;
        r.sorted_output = nullptr; // sorted in-place in h_data
        return r;
    }

    // Allocate persistent key buffer sized to actual data
    // Keys = 10% of record data, so they always fit if we have enough GPU memory
    uint64_t total_keys_bytes = num_records * KEY_SIZE;
    size_t free_after_bufs, dummy;
    CUDA_CHECK(cudaMemGetInfo(&free_after_bufs, &dummy));
    // Need: key_buffer + later merge workspace (keys_out + perm_in + perm_out = keys + 2*perm)
    // But merge workspace is allocated AFTER freeing sort buffers, so we have plenty
    if (total_keys_bytes < free_after_bufs * 0.8) {
        CUDA_CHECK(cudaMalloc(&d_key_buffer, total_keys_bytes));
        key_buffer_capacity = total_keys_bytes;
        printf("  Key retention: %.2f GB GPU key buffer allocated\n", total_keys_bytes/1e9);
    } else {
        printf("  Key retention: keys too large (%.2f GB), will extract on CPU\n",
               total_keys_bytes/1e9);
    }
    run_key_offsets.clear();

    // Pre-allocate sort workspace and host permutation buffer
    sort_ws.allocate(buf_records);
    // Pre-allocate pinned host perm buffer NOW to avoid 1s cudaMallocHost during merge
    uint32_t* h_perm_prealloc = nullptr;
    uint64_t total_perm_bytes_prealloc = num_records * sizeof(uint32_t);
    cudaMallocHost(&h_perm_prealloc, total_perm_bytes_prealloc); // non-critical if fails

    printf("\n== Phase 1: Run Generation (pipelined) ==\n");
    double rg_h2d = 0, rg_d2h = 0;
    auto runs = generate_runs_pipelined(h_data, num_records,
                                         r.run_gen_ms, rg_h2d, rg_d2h);
    r.num_runs = runs.size();
    printf("  %d runs in %.0f ms (%.2f GB/s effective)\n\n",
           r.num_runs, r.run_gen_ms, total_bytes/(r.run_gen_ms*1e6));

    // Free sort workspace but KEEP sort buffers — reuse as merge arena
    sort_ws.free();
    // d_buf[0..NBUFS-1] stay allocated — total ~16GB available for merge

    printf("== Phase 2: GPU Streaming Merge ==\n");
    double mg_h2d = 0, mg_d2h = 0;
    r.sorted_output = streaming_merge(h_data, num_records, runs,
                                       r.merge_ms, r.merge_passes, mg_h2d, mg_d2h,
                                       h_perm_prealloc);

    r.pcie_h2d_gb = (rg_h2d + mg_h2d) / 1e9;
    r.pcie_d2h_gb = (rg_d2h + mg_d2h) / 1e9;
    r.total_ms = r.run_gen_ms + r.merge_ms;

    printf("\n[ExternalSort] ═══════════════════════════════════════\n");
    printf("  DONE: %.0f ms (gen: %.0f + merge: %.0f)\n",
           r.total_ms, r.run_gen_ms, r.merge_ms);
    printf("  Throughput: %.2f GB/s\n", total_bytes/(r.total_ms*1e6));
    printf("  Runs: %d | Merge passes: %d\n", r.num_runs, r.merge_passes);
    printf("  PCIe: %.1f GB H2D + %.1f GB D2H = %.1f GB total (%.1fx amplification)\n",
           r.pcie_h2d_gb, r.pcie_d2h_gb,
           r.pcie_h2d_gb + r.pcie_d2h_gb,
           (r.pcie_h2d_gb + r.pcie_d2h_gb) / (total_bytes/1e9));
    printf("═══════════════════════════════════════════════════════\n");
    return r;
}

// ════════════════════════════════════════════════════════════════════

#ifdef EXTERNAL_SORT_MAIN

static void gen_data(uint8_t* d, uint64_t n) {
    srand(42);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* r = d + i*RECORD_SIZE;
        for (int b = 0; b < KEY_SIZE; b++) r[b] = (uint8_t)(rand() & 0xFF);
        memset(r+KEY_SIZE, 0, VALUE_SIZE);
        memcpy(r+KEY_SIZE, &i, sizeof(uint64_t));
    }
}

int main(int argc, char** argv) {
    double total_gb = 0.5;
    bool verify = true;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--total-gb") && i+1<argc) total_gb = atof(argv[++i]);
        else if (!strcmp(argv[i],"--no-verify")) verify = false;
        else { printf("Usage: %s [--total-gb N] [--no-verify]\n",argv[0]); return 0; }
    }

    uint64_t num_records = (uint64_t)(total_gb * 1e9) / RECORD_SIZE;
    uint64_t total_bytes = num_records * RECORD_SIZE;

    printf("════════════════════════════════════════════════════\n");
    printf("  GPU External Merge Sort — Streaming Benchmark\n");
    printf("════════════════════════════════════════════════════\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (%.1f GB HBM, %d SMs, %.0f GB/s BW)\n", props.name,
           props.totalGlobalMem/1e9, props.multiProcessorCount,
           2.0 * props.memoryClockRate * (props.memoryBusWidth/8) / 1e6);
    printf("Data: %.2f GB (%lu records × %d bytes)\n\n",
           total_bytes/1e9, num_records, RECORD_SIZE);

    printf("Allocating %.2f GB pinned host memory...\n", total_bytes/1e9);
    uint8_t* h_data;
    cudaError_t alloc_err = cudaMallocHost(&h_data, total_bytes);
    if (alloc_err != cudaSuccess) {
        printf("  cudaMallocHost failed, falling back to malloc\n");
        h_data = (uint8_t*)malloc(total_bytes);
    }
    if (!h_data) { fprintf(stderr,"allocation failed\n"); return 1; }
    // Request transparent huge pages for less TLB pressure during random gather
    madvise(h_data, total_bytes, MADV_HUGEPAGE);

    printf("Generating random data...\n");
    WallTimer gt; gt.begin();
    gen_data(h_data, num_records);
    printf("  Generated in %.0f ms\n\n", gt.end_ms());

    ExternalGpuSort sorter;
    auto result = sorter.sort(h_data, num_records);

    // sorted_output points to sorted data (either h_data for single-chunk, or malloc'd buffer)
    const uint8_t* sorted = result.sorted_output ? result.sorted_output : h_data;

    if (verify) {
        printf("\nVerifying...\n");
        uint64_t bad = 0;
        for (uint64_t i = 1; i < num_records && bad < 10; i++)
            if (key_compare(sorted+(i-1)*RECORD_SIZE, sorted+i*RECORD_SIZE, KEY_SIZE)>0)
                { if (bad<5) printf("  VIOLATION at %lu\n",i); bad++; }
        printf(bad==0 ? "  PASS: %lu records sorted\n" : "  FAIL: %lu violations\n",
               bad==0 ? num_records : bad);
    }
    if (result.sorted_output) free(result.sorted_output);

    printf("\nCSV,%s,%.2f,%lu,%d,%d,%.2f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%.1f\n",
           props.name, total_bytes/1e9, num_records,
           result.num_runs, result.merge_passes,
           result.run_gen_ms, result.merge_ms, result.total_ms,
           total_bytes/(result.total_ms*1e6),
           result.pcie_h2d_gb, result.pcie_d2h_gb,
           result.pcie_h2d_gb + result.pcie_d2h_gb,
           (result.pcie_h2d_gb + result.pcie_d2h_gb) / (total_bytes/1e9));

    if (alloc_err == cudaSuccess) cudaFreeHost(h_data);
    else free(h_data);
    return 0;
}
#endif
