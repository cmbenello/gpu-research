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
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
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

// Extract uint32 sort key (top 4 bytes, big-endian) from KEY_SIZE-stride buffer
// For random 10B keys, 4 bytes covers 2^32 values vs 600M records — ~0 ties
__global__ void extract_uint32_from_keys_kernel(
    const uint8_t* __restrict__ key_buffer,
    uint32_t* __restrict__ sort_keys,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* k = key_buffer + i * KEY_SIZE;
    uint32_t v = ((uint32_t)k[0] << 24) | ((uint32_t)k[1] << 16) |
                 ((uint32_t)k[2] << 8) | (uint32_t)k[3];
    sort_keys[i] = v;
}

// Gather uint64 values by permutation: out[i] = src[perm[i]]
__global__ void gather_uint64_kernel(
    const uint64_t* __restrict__ src,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ out,
    uint64_t n
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[perm[i]];
}

// Extract uint64 from a specific 8-byte chunk of the key, in permutation order.
// Reads key[perm[i]][byte_offset : byte_offset + chunk_bytes] as big-endian uint64.
__global__ void extract_uint64_chunk_kernel(
    const uint8_t* __restrict__ key_buffer,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes  // 1-8
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = key_buffer + (uint64_t)orig_idx * KEY_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    // Left-align: shift so MSB of chunk is in MSB of uint64
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}

// ── Compact key for TPC-H: strip constant bytes, keep only varying ones ───
// The 66B TPC-H key has ~40 constant bytes and only 26 varying bytes.
// Packing the varying bytes into a 32B compact key reduces LSD from 9 to 4 passes.
//
// Compile with -DUSE_COMPACT_KEY to enable. The byte mapping is:
//   compact[0]  = record[0]   returnflag
//   compact[1]  = record[1]   linestatus
//   compact[2]  = record[4]   shipdate[2]
//   compact[3]  = record[5]   shipdate[3]
//   compact[4]  = record[8]   commitdate[2]
//   compact[5]  = record[9]   commitdate[3]
//   compact[6]  = record[12]  receiptdate[2]
//   compact[7]  = record[13]  receiptdate[3]
//   compact[8]  = record[19]  extprice[5]
//   compact[9]  = record[20]  extprice[6]
//   compact[10] = record[21]  extprice[7]
//   compact[11] = record[29]  discount[7]
//   compact[12] = record[37]  tax[7]
//   compact[13] = record[44]  quantity[6]
//   compact[14] = record[45]  quantity[7]
//   compact[15] = record[51]  orderkey[5]
//   compact[16] = record[52]  orderkey[6]
//   compact[17] = record[53]  orderkey[7]
//   compact[18-21] = record[54-57]  partkey
//   compact[22] = record[59]  suppkey[1]
//   compact[23] = record[60]  suppkey[2]
//   compact[24] = record[61]  suppkey[3]
//   compact[25] = record[65]  linenumber[3]
//   compact[26-31] = 0 (padding to 32B)

#ifdef USE_COMPACT_KEY
static constexpr int COMPACT_KEY_SIZE = 32;
// Source byte offsets within the RECORD for each compact key byte
__constant__ int d_compact_map[26] = {
    0, 1,           // returnflag, linestatus
    4, 5,           // shipdate varying
    8, 9,           // commitdate varying
    12, 13,         // receiptdate varying
    19, 20, 21,     // extprice varying (3 bytes)
    29,             // discount varying
    37,             // tax varying
    44, 45,         // quantity varying
    51, 52, 53,     // orderkey varying
    54, 55, 56, 57, // partkey (all 4 bytes)
    59, 60, 61,     // suppkey varying
    65              // linenumber varying
};

__global__ void build_compact_keys_kernel(
    const uint8_t* __restrict__ records,
    uint8_t* __restrict__ compact_keys,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* rec = records + i * RECORD_SIZE;
    uint8_t* ck = compact_keys + i * COMPACT_KEY_SIZE;
    #pragma unroll
    for (int b = 0; b < 26; b++) ck[b] = rec[d_compact_map[b]];
    // Pad remaining bytes with zero
    ck[26] = 0; ck[27] = 0; ck[28] = 0; ck[29] = 0; ck[30] = 0; ck[31] = 0;
}

// Extract uint64 from compact key buffer at COMPACT_KEY_SIZE stride
__global__ void extract_uint64_from_compact_kernel(
    const uint8_t* __restrict__ compact_keys,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = compact_keys + (uint64_t)orig_idx * COMPACT_KEY_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}
#endif // USE_COMPACT_KEY

// ── OVC Architecture kernels ──────────────────────────────────────
// Compute OVCs for a sorted run: compare adjacent sorted records.
// OVC[0] = OVC_INITIAL, OVC[i] = ovc_compute_delta(record[i-1], record[i]).
__global__ void compute_run_ovcs_kernel(
    const uint8_t* __restrict__ sorted_records,
    uint32_t* __restrict__ ovcs,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    if (i == 0) {
        ovcs[i] = OVC_INITIAL;
    } else {
        const uint8_t* prev = sorted_records + (i - 1) * RECORD_SIZE;
        const uint8_t* curr = sorted_records + i * RECORD_SIZE;
        ovcs[i] = ovc_compute_delta(prev, curr, KEY_SIZE);
    }
}

// Extract 8-byte big-endian prefix from sorted records (for OVC tiebreaking)
__global__ void extract_prefix8_kernel(
    const uint8_t* __restrict__ sorted_records,
    uint64_t* __restrict__ prefixes,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* k = sorted_records + i * RECORD_SIZE;
    uint64_t v = 0;
    for (int b = 0; b < 8; b++) v = (v << 8) | k[b];
    prefixes[i] = v;
}

// Build global permutation: map sorted position to original h_data index.
// After sort_chunk_on_gpu, the chunk's records are sorted in d_buf[cur].
// local_perm[i] = the original index within the chunk of the i-th sorted record.
// global_perm[run_offset + i] = chunk_start + local_perm[i].
__global__ void build_global_perm_kernel(
    const uint32_t* __restrict__ local_perm,
    uint32_t* __restrict__ global_perm,
    uint64_t chunk_start,    // record offset of this chunk in h_data
    uint64_t run_offset,     // position in the global OVC/perm arrays
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    global_perm[run_offset + i] = (uint32_t)(chunk_start + local_perm[i]);
}

// Combined OVC + prefix extraction via permutation (for OVC merge architecture)
// For sorted position i: reads record[perm[i]], computes OVC vs record[perm[i-1]]
__global__ void extract_ovc_and_prefix_kernel(
    const uint8_t* __restrict__ records,
    const uint32_t* __restrict__ perm,
    uint32_t* __restrict__ ovcs,
    uint64_t* __restrict__ prefixes,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t idx = perm[i];
    const uint8_t* k = records + (uint64_t)idx * RECORD_SIZE;
    // Extract 8B big-endian prefix
    uint64_t v = 0;
    for (int b = 0; b < 8; b++) v = (v << 8) | k[b];
    prefixes[i] = v;
    // Compute OVC
    if (i == 0) {
        ovcs[i] = OVC_INITIAL;
    } else {
        uint32_t prev_idx = perm[i - 1];
        const uint8_t* prev_k = records + (uint64_t)prev_idx * RECORD_SIZE;
        ovcs[i] = ovc_compute_delta(prev_k, k, KEY_SIZE);
    }
}

// Extract uint64 from records at RECORD_SIZE stride (for in-chunk LSD sort)
__global__ void extract_uint64_from_records_kernel(
    const uint8_t* __restrict__ records,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = records + (uint64_t)orig_idx * RECORD_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}

// Extract uint64 sort key (bytes 0-7 big-endian) from KEY_SIZE-stride key buffer.
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

// Extract 16-bit tiebreaker (bytes 8-9 big-endian) using permutation to lookup original keys
__global__ void extract_tiebreaker_kernel(
    const uint8_t* __restrict__ key_buffer,
    const uint32_t* __restrict__ perm,
    uint16_t* __restrict__ tiebreakers,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = key_buffer + (uint64_t)orig_idx * KEY_SIZE;
    tiebreakers[i] = ((uint16_t)k[8] << 8) | k[9];
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

// Extract all KEY_SIZE bytes from records into packed key buffer (for LSD)
__global__ void extract_all_keys_kernel(
    const uint8_t* __restrict__ records,
    uint8_t* __restrict__ packed_keys,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* src = records + i * RECORD_SIZE;
    uint8_t* dst = packed_keys + i * KEY_SIZE;
    for (int b = 0; b < KEY_SIZE; b++) dst[b] = src[b];
}

// Extract uint64 from packed key buffer at KEY_SIZE stride (for LSD passes 2+)
__global__ void extract_uint64_from_packed_keys_kernel(
    const uint8_t* __restrict__ packed_keys,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = packed_keys + (uint64_t)orig_idx * KEY_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}

// Pre-allocated workspace for sort_chunk_on_gpu (allocated once, reused per chunk)
struct SortWorkspace {
    // CUB radix sort workspace
    uint64_t* d_keys = nullptr;
    uint64_t* d_keys_alt = nullptr;
    uint32_t* d_indices = nullptr;
    uint32_t* d_indices_alt = nullptr;
    uint8_t* d_compact = nullptr;   // Compact key buffer (USE_COMPACT_KEY)
    uint64_t capacity = 0;

    void allocate(uint64_t max_records) {
        if (capacity >= max_records) return;
        free();
        capacity = max_records;
        CUDA_CHECK(cudaMalloc(&d_keys, max_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_alt, max_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_indices, max_records * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_indices_alt, max_records * sizeof(uint32_t)));
#ifdef USE_COMPACT_KEY
        // Skip compact key buffer if OVC mode is active (saves 1.4GB GPU memory)
        cudaError_t ck_err = cudaMalloc(&d_compact, max_records * COMPACT_KEY_SIZE);
        if (ck_err != cudaSuccess) d_compact = nullptr; // Graceful fallback
#endif
    }
    void free() {
        if (d_keys) { cudaFree(d_keys); d_keys = nullptr; }
        if (d_keys_alt) { cudaFree(d_keys_alt); d_keys_alt = nullptr; }
        if (d_indices) { cudaFree(d_indices); d_indices = nullptr; }
        if (d_indices_alt) { cudaFree(d_indices_alt); d_indices_alt = nullptr; }
        if (d_compact) { cudaFree(d_compact); d_compact = nullptr; }
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

    // Persistent key buffer (for small-key single-pass path)
    uint8_t* d_key_buffer;
    uint64_t key_buffer_capacity;
    std::vector<uint64_t> run_key_offsets;

    // OVC merge buffers (for large-key path: persistent across run gen)
    uint32_t* d_ovc_buffer;        // 4B OVC per record
    uint64_t* d_prefix_buffer;     // 8B key prefix per record (tiebreaker)
    uint32_t* d_global_perm;       // global index per record
    uint64_t  ovc_buffer_records;  // capacity in records
    uint64_t  ovc_run_offset;     // current write position in OVC/perm/prefix buffers
    uint64_t  ovc_chunk_start;    // record offset of current chunk in h_data

    // Pre-allocated sort workspace (avoids cudaMalloc per chunk)
    SortWorkspace sort_ws;

public:
    struct TimingResult {
        double run_gen_ms, merge_ms, total_ms;
        int num_runs, merge_passes;
        double pcie_h2d_gb, pcie_d2h_gb;
        uint8_t* sorted_output;  // pointer to sorted data
        uint64_t sorted_output_size;  // size in bytes (for munmap)
        bool sorted_output_is_mmap;   // true = munmap, false = free
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
                              uint32_t* h_perm_prealloc, uint8_t* h_output_prealloc);

};

ExternalGpuSort::ExternalGpuSort() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Sort buffers get 65% of GPU memory (3 × ~5.5GB each)
    gpu_budget = (size_t)(free_mem * 0.65);
    key_buffer_capacity = 0;
    d_key_buffer = nullptr;
    d_ovc_buffer = nullptr;
    d_prefix_buffer = nullptr;
    d_global_perm = nullptr;
    ovc_buffer_records = 0;
    ovc_run_offset = 0;
    ovc_chunk_start = 0;
    buf_records = (gpu_budget / NBUFS) / RECORD_SIZE;
    buf_bytes = buf_records * RECORD_SIZE;

    printf("[ExternalSort] GPU: %.2f GB free, budget: %.2f GB\n",
           free_mem/1e9, gpu_budget/1e9);
    printf("[ExternalSort] Triple-buffer: %d × %.2f GB (%.0f M records each)\n",
           NBUFS, buf_bytes/1e9, buf_records/1e6);

    // Allocate host pinned buffers and streams/events (not GPU buffers yet — lazy alloc)
    for (int i = 0; i < NBUFS; i++) {
        d_buf[i] = nullptr;  // allocated lazily in sort()
        CUDA_CHECK(cudaMallocHost(&h_pin[i], buf_bytes));
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
    if (d_ovc_buffer) cudaFree(d_ovc_buffer);
    if (d_prefix_buffer) cudaFree(d_prefix_buffer);
    if (d_global_perm) cudaFree(d_global_perm);
    sort_ws.free();
}

// Sort a chunk on GPU using LSD radix sort on full KEY_SIZE-byte key + record reorder.
// For KEY_SIZE ≤ 8: single CUB pass on uint64.
// For KEY_SIZE > 8: ceil(KEY_SIZE/8) LSD passes, each sorting 8 bytes of key.
void ExternalGpuSort::sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_scratch,
                                         uint64_t n, cudaStream_t s) {
    int nthreads = 256;
    int nblks = (n + nthreads - 1) / nthreads;

    // Initialize identity permutation
    init_identity_kernel<<<nblks, nthreads, 0, s>>>(sort_ws.d_indices, n);

    uint32_t* perm_in = sort_ws.d_indices;
    uint32_t* perm_out = sort_ws.d_indices_alt;

#ifdef USE_COMPACT_KEY
    if (sort_ws.d_compact) {
    // Compact key path: build 32B keys from varying bytes, then 4 LSD passes
    build_compact_keys_kernel<<<nblks, nthreads, 0, s>>>(d_in, sort_ws.d_compact, n);

    int num_chunks = (COMPACT_KEY_SIZE + 7) / 8;  // 4 for 32B
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int byte_offset = chunk * 8;
        int chunk_bytes = std::min(8, COMPACT_KEY_SIZE - byte_offset);

        extract_uint64_from_compact_kernel<<<nblks, nthreads, 0, s>>>(
            sort_ws.d_compact, perm_in, sort_ws.d_keys, n, byte_offset, chunk_bytes);

        cub::DoubleBuffer<uint64_t> keys_buf(sort_ws.d_keys, sort_ws.d_keys_alt);
        cub::DoubleBuffer<uint32_t> idx_buf(perm_in, perm_out);
        size_t temp_bytes = buf_bytes;
        cub::DeviceRadixSort::SortPairs(d_scratch, temp_bytes,
            keys_buf, idx_buf, (int)n, 0, chunk_bytes * 8, s);
        perm_in = idx_buf.Current();
        perm_out = idx_buf.Alternate();
    }
    } else {
    // Compact buffer unavailable — fall through to standard LSD
#else
    {
#endif
    // Standard LSD path: ceil(KEY_SIZE/8) passes on full key
    int num_chunks = (KEY_SIZE + 7) / 8;
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int byte_offset = chunk * 8;
        int chunk_bytes = std::min(8, KEY_SIZE - byte_offset);

        extract_uint64_from_records_kernel<<<nblks, nthreads, 0, s>>>(
            d_in, perm_in, sort_ws.d_keys, n, byte_offset, chunk_bytes);

        cub::DoubleBuffer<uint64_t> keys_buf(sort_ws.d_keys, sort_ws.d_keys_alt);
        cub::DoubleBuffer<uint32_t> idx_buf(perm_in, perm_out);
        size_t temp_bytes = buf_bytes;
        cub::DeviceRadixSort::SortPairs(d_scratch, temp_bytes,
            keys_buf, idx_buf, (int)n, 0, chunk_bytes * 8, s);
        perm_in = idx_buf.Current();
        perm_out = idx_buf.Alternate();
    }
    } // end standard LSD / compact fallback

    // OVC mode: extract OVCs + prefixes + global perm instead of reordering records.
    // This avoids the D2H of sorted records — only OVCs + perm are downloaded.
    if (d_ovc_buffer && ovc_run_offset + n <= ovc_buffer_records) {
        // Compute OVCs by comparing adjacent records via permutation
        // OVC[0] = INITIAL, OVC[i] = delta(record[perm[i-1]], record[perm[i]])
        // We need a kernel that reads records via permutation, not from sorted output
        // (because we're NOT reordering records in OVC mode)

        // Extract OVCs and 8B prefixes from records via permutation
        extract_ovc_and_prefix_kernel<<<nblks, nthreads, 0, s>>>(
            d_in, perm_in, d_ovc_buffer + ovc_run_offset,
            d_prefix_buffer + ovc_run_offset, n);

        // Build global perm: global_perm[run_offset + i] = chunk_start + perm[i]
        build_global_perm_kernel<<<nblks, nthreads, 0, s>>>(
            perm_in, d_global_perm, ovc_chunk_start, ovc_run_offset, n);

        ovc_run_offset += n;
        return; // Don't reorder records — they stay unsorted in d_buf
    }

    // Standard path: reorder full records using the final sorted permutation
    reorder_records_kernel<<<nblks, nthreads, 0, s>>>(
        d_in, d_scratch, perm_in, n);

    // Copy sorted result back to d_in (async — no CPU sync needed)
    CUDA_CHECK(cudaMemcpyAsync(d_in, d_scratch, n * RECORD_SIZE,
                                cudaMemcpyDeviceToDevice, s));
    // Don't sync here — let the caller manage dependencies via events
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

    // Per-buffer D2H completion events: ensure H2D of buffer X doesn't start
    // before the previous D2H of buffer X completes (they're on different streams).
    cudaEvent_t d2h_done[2];
    CUDA_CHECK(cudaEventCreate(&d2h_done[0]));
    CUDA_CHECK(cudaEventCreate(&d2h_done[1]));

    for (int c = 0; c < total_chunks; c++) {
        uint64_t offset = (uint64_t)c * buf_records;
        uint64_t cur_n = std::min(buf_records, num_records - offset);
        uint64_t cur_bytes = cur_n * RECORD_SIZE;
        int cur = c % 2;
        int scratch = 2;

        // Wait for previous D2H of this buffer to complete before overwriting
        if (c >= 2) {
            CUDA_CHECK(cudaStreamWaitEvent(streams[0], d2h_done[cur], 0));
        }

        // Start H2D upload on stream[0]
        CUDA_CHECK(cudaMemcpyAsync(d_buf[cur], h_data + offset * RECORD_SIZE, cur_bytes,
                                    cudaMemcpyHostToDevice, streams[0]));
        h2d += cur_bytes;

        // Record event when H2D completes, make sort stream wait for it
        CUDA_CHECK(cudaEventRecord(events[0], streams[0]));
        CUDA_CHECK(cudaStreamWaitEvent(streams[1], events[0], 0));

        // Sort (GPU-side wait for H2D via event — no CPU blocking)
        // In OVC mode, sort_chunk_on_gpu will also extract OVCs + prefixes + global perm
        ovc_chunk_start = offset;  // record offset for global perm computation
        sort_chunk_on_gpu(d_buf[cur], d_buf[scratch], cur_n, streams[1]);

        // Extract keys to persistent GPU key buffer (for small-key single-pass path)
        uint64_t key_off = run_key_offsets.empty() ? 0 :
            run_key_offsets.back() + (runs.empty() ? 0 : runs.back().num_records * KEY_SIZE);
        if (d_key_buffer && (key_off + cur_n * KEY_SIZE) <= key_buffer_capacity) {
            int nthreads = 256;
            int nblocks_k = (cur_n + nthreads - 1) / nthreads;
            extract_keys_kernel<<<nblocks_k, nthreads, 0, streams[1]>>>(
                d_buf[cur], d_key_buffer + key_off, cur_n);
            run_key_offsets.push_back(key_off);
        }

        // Make D2H stream wait for sort+OVC extraction on stream[1] to finish
        CUDA_CHECK(cudaEventRecord(events[1], streams[1]));
        CUDA_CHECK(cudaStreamWaitEvent(streams[2], events[1], 0));

        if (d_ovc_buffer) {
            // OVC mode: skip D2H of sorted records — OVCs are already on GPU.
            // Just record the event so the next H2D can proceed.
            CUDA_CHECK(cudaEventRecord(d2h_done[cur], streams[1]));
            // No d2h traffic for sorted records
        } else {
            // Standard mode: download sorted records to h_data
            CUDA_CHECK(cudaMemcpyAsync(h_data + offset * RECORD_SIZE, d_buf[cur], cur_bytes,
                                        cudaMemcpyDeviceToHost, streams[2]));
            d2h += cur_bytes;
            CUDA_CHECK(cudaEventRecord(d2h_done[cur], streams[2]));
        }

        runs.push_back({offset * RECORD_SIZE, cur_n});
        printf("\r  Run %d/%d: %.1f MB sorted    ", c+1, total_chunks,
               cur_bytes/(1024.0*1024.0));
        fflush(stdout);
    }

    // Finalize: wait for last D2H to complete
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));
    CUDA_CHECK(cudaEventDestroy(d2h_done[0]));
    CUDA_CHECK(cudaEventDestroy(d2h_done[1]));
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

extern "C" void launch_merge_ovc(
    const uint32_t*, uint32_t*, const uint64_t*, uint64_t*,
    const uint32_t*, uint32_t*,
    const void*, int, int, cudaStream_t);

extern "C" void launch_merge_keys_only(
    const uint8_t*, uint8_t*, const uint32_t*, uint32_t*,
    const KeyMergePair*, int, int, cudaStream_t);

uint8_t* ExternalGpuSort::streaming_merge(
    uint8_t* h_data, uint64_t num_records,
    std::vector<RunInfo>& runs,
    double& ms, int& passes, double& h2d, double& d2h,
    uint32_t* h_perm_prealloc, uint8_t* h_output_prealloc
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
        // Split merge workspace across sort buffers + one extra alloc.
        // d_buf[0] (5.44GB): d_sort_keys (N×8B = 4.8GB)
        // d_buf[1] (5.44GB): d_sort_keys_alt (N×8B = 4.8GB)
        // d_buf[2] (5.44GB): d_perm_in (N×4B) + d_perm_out (N×4B) = 4.8GB
        // CUB temp: allocate separately (small, ~100MB)
        size_t keys_sz = num_records * sizeof(uint64_t);
        size_t perm_sz = num_records * sizeof(uint32_t);
        bool can_split = (d_buf[0] && d_buf[1] && d_buf[2] &&
                          keys_sz <= buf_bytes && 2 * perm_sz <= buf_bytes);
        if (can_split) {
            d_sort_keys = (uint64_t*)d_buf[0];
            d_sort_keys_alt = (uint64_t*)d_buf[1];
            d_perm_in = (uint32_t*)d_buf[2];
            d_perm_out = d_perm_in + num_records;
            // CUB temp fits after perm_out in d_buf[2] (0.6GB remaining)
            d_temp = (void*)(d_perm_out + num_records);
            cub_temp_bytes = buf_bytes - 2 * perm_sz;
            // ZERO cudaMalloc in merge phase!
        } else {
            // Fallback: single arena alloc
            cub::DoubleBuffer<uint64_t> dk(nullptr,nullptr);
            cub::DoubleBuffer<uint32_t> dp(nullptr,nullptr);
            cub_temp_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, dk, dp, (int)num_records, 0, 64, streams[0]);
            size_t arena_sz = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + cub_temp_bytes;
            for (int i = 0; i < NBUFS; i++) { if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; } }
            CUDA_CHECK(cudaMalloc(&d_merge_arena, arena_sz));
            d_sort_keys = (uint64_t*)d_merge_arena;
            d_sort_keys_alt = d_sort_keys + num_records;
            d_perm_in = (uint32_t*)(d_sort_keys_alt + num_records);
            d_perm_out = d_perm_in + num_records;
            d_temp = (void*)(d_perm_out + num_records);
        }
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
    uint32_t* h_perm = h_perm_prealloc;
    if (!h_perm) { CUDA_CHECK(cudaMallocHost(&h_perm, total_perm_bytes)); }

    // Start async perm download — CPU continues with alloc/free work in parallel
    CUDA_CHECK(cudaMemcpyAsync(h_perm, d_perm_in, total_perm_bytes,
                                cudaMemcpyDeviceToHost, streams[0]));
    d2h += total_perm_bytes;

    // While perm downloads: allocate gather output buffer + free GPU memory
    // These are CPU operations that run concurrently with the D2H DMA
    int num_threads = std::min(48, (int)std::thread::hardware_concurrency());
    printf("  CPU value gather (%.2f GB, %d threads)...\n", total_bytes/1e9, num_threads);

    // Use pre-allocated output buffer if available (allocated before run gen)
    uint8_t* h_output = h_output_prealloc;
    if (!h_output) {
        h_output = (uint8_t*)malloc(total_bytes);
        if (!h_output) { fprintf(stderr, "malloc failed for gather output\n"); ms = timer.end_ms(); return nullptr; }
        madvise(h_output, total_bytes, MADV_HUGEPAGE);
    }

    // Free GPU memory while perm downloads
    if (d_merge_arena) { cudaFree(d_merge_arena); d_merge_arena = nullptr; }
    for (int i = 0; i < NBUFS; i++) {
        if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; }
    }
    if (h_keys) { cudaFreeHost(h_keys); h_keys = nullptr; }

    // NOW wait for perm download to complete before starting gather
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    tpoint("perm downloaded + alloc + GPU freed (overlapped)");

    WallTimer gather_timer; gather_timer.begin();

    // Pre-compute run lookup table for O(1) run_id lookup instead of O(K) scan
    // run_global_base is sorted, so we can use it directly
    // But for speed, build a direct lookup: global_idx → (run_id, idx_in_run)
    // For K <= 64, binary search is fast enough

    // Pre-compute source pointers for all records in a block, then copy.
    // This separates the "find source" phase (random permutation reads) from
    // the "copy data" phase (random source reads + sequential writes).
    // Deeper prefetch pipeline hides more DRAM latency.
    auto gather_worker = [&](uint64_t start, uint64_t end) {
        constexpr int BLOCK = 256;  // balanced: enough prefetch depth without cache pressure
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

    // Lazy-allocate GPU buffers if needed
    if (!d_buf[0]) {
        for (int i = 0; i < NBUFS; i++)
            CUDA_CHECK(cudaMalloc(&d_buf[i], buf_bytes));
    }

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

    // ════════════════════════════════════════════════════════════════
    // SINGLE-PASS KEY-ONLY SORT
    //
    // Instead of: upload full records → GPU sort → download sorted records → merge
    // Do:         extract keys on CPU → upload 10% keys → GPU sort → download perm → CPU gather
    //
    // PCIe traffic: 8.4GB instead of 122.4GB for 60GB dataset (14.6× reduction!)
    // ════════════════════════════════════════════════════════════════

    uint64_t total_keys_bytes = num_records * KEY_SIZE;
    uint64_t total_perm_bytes = num_records * sizeof(uint32_t);

    // Pre-allocate host perm buffer (pinned for fast D2H)
    uint32_t* h_perm = nullptr;
    cudaMallocHost(&h_perm, total_perm_bytes);
    // Pre-allocate gather output buffer
    // Use mmap + MAP_POPULATE for pre-faulted pages (avoids page faults during gather)
    uint8_t* h_output = (uint8_t*)mmap(nullptr, total_bytes, PROT_READ|PROT_WRITE,
                                        MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    if (h_output == MAP_FAILED) h_output = nullptr;
    if (h_output) madvise(h_output, total_bytes, MADV_HUGEPAGE);
    bool h_output_is_mmap = (h_output != nullptr);
    if (!h_output) {
        h_output = (uint8_t*)malloc(total_bytes);
        if (h_output) madvise(h_output, total_bytes, MADV_HUGEPAGE);
    }

    WallTimer phase_timer;

    // ── Step 1+2: Upload keys to GPU ──
    // For small keys (KEY_SIZE ≤ 16): strided DMA extracts only keys
    // For large keys (KEY_SIZE > 16): contiguous upload of full records is faster
    //   because strided DMA has host-side read amplification
    printf("== Step 1+2: Key Upload ==\n");
    phase_timer.begin();

    // Free sort buffers and workspace to make room for key upload
    for (int i = 0; i < NBUFS; i++) { if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; } }
    sort_ws.free();
    if (d_key_buffer) { cudaFree(d_key_buffer); d_key_buffer = nullptr; }

    uint8_t* d_keys_10byte;
    uint8_t* h_keys = nullptr;

    // Variables used by both paths (prefix sort + LSD sort)
    uint32_t *d_perm_in = nullptr, *d_perm_out = nullptr;

    // Check if all keys fit in GPU memory (need keys + arena)
    size_t free_mem_now, dummy2;
    CUDA_CHECK(cudaMemGetInfo(&free_mem_now, &dummy2));
    size_t est_arena = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + 512*1024*1024;
    // PREFIX SORT experiment: disabled — fixup step (6-12s) makes it slower
    // than run-gen approach. The CPU fixup has to touch all data again,
    // negating the PCIe savings. Run-gen + K-way merge remains faster.
    // Keeping code for reference.
#if 0
    // For large keys: try PREFIX SORT (16B GPU sort + CPU fixup for ties).
    // Upload first 16 bytes via strided DMA, GPU sorts by 16B prefix (2 LSD passes),
    // CPU gathers into prefix-sorted order, then sorts within small contiguous groups.
    // PCIe = 16B×N up + 4B×N down vs 2×RECORD_SIZE×N for full-record round-trip.
    static constexpr int PREFIX_BYTES = 16;
    size_t prefix_total = num_records * PREFIX_BYTES;
    // Arena: 2×uint64 (sort keys) + 2×uint32 (perm) + CUB temp (~300MB for 600M records)
    size_t prefix_arena = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + 300*1024*1024;
    bool use_prefix_sort = (KEY_SIZE > 16 && prefix_total + prefix_arena < free_mem_now * 0.98);
    printf("  Prefix sort check: need %.1f GB, have %.1f GB (%.1f×0.95=%.1f) → %s\n",
           (prefix_total + prefix_arena)/1e9, free_mem_now/1e9,
           free_mem_now/1e9, free_mem_now*0.95/1e9,
           use_prefix_sort ? "YES" : "NO");

    if (use_prefix_sort) {
        printf("  Using PREFIX SORT: %dB prefix on GPU + CPU fixup for ties\n", PREFIX_BYTES);
        printf("  PCIe: %.1f GB H2D + %.1f GB D2H (vs %.1f GB full-record round-trip)\n",
               prefix_total/1e9, total_perm_bytes/1e9, 2*total_bytes/1e9);

        // Allocate prefix keys and sort workspace on GPU
        uint8_t* d_prefix_keys;
        CUDA_CHECK(cudaMalloc(&d_prefix_keys, prefix_total));
        CUDA_CHECK(cudaMalloc(&d_perm_in, num_records * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_perm_out, num_records * sizeof(uint32_t)));
        uint64_t* d_sort_keys, *d_sort_keys_alt;
        CUDA_CHECK(cudaMalloc(&d_sort_keys, num_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sort_keys_alt, num_records * sizeof(uint64_t)));

        // Step 1: Strided DMA — upload first PREFIX_BYTES of each record
        CUDA_CHECK(cudaMemcpy2D(d_prefix_keys, PREFIX_BYTES, h_data, RECORD_SIZE,
                                 PREFIX_BYTES, num_records, cudaMemcpyHostToDevice));
        double upload_ms = phase_timer.end_ms();
        r.pcie_h2d_gb = prefix_total / 1e9;
        printf("  Uploaded %.2f GB prefix keys in %.0f ms (%.2f GB/s effective)\n",
               prefix_total/1e9, upload_ms, total_bytes/(upload_ms*1e6));

        // Step 2: GPU LSD sort by 16-byte prefix (2 CUB passes)
        phase_timer.begin();
        int nthreads = 256;
        int nblks = (num_records + nthreads - 1) / nthreads;
        init_identity_kernel<<<nblks, nthreads>>>(d_perm_in, num_records);

        // CUB temp storage
        size_t cub_temp_bytes = 0;
        cub::DoubleBuffer<uint64_t> keys_buf(d_sort_keys, d_sort_keys_alt);
        cub::DoubleBuffer<uint32_t> idx_buf(d_perm_in, d_perm_out);

        cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
            keys_buf, idx_buf, (int)num_records, 0, 64);
        void* d_cub_temp;
        CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

        // LSD: sort by least significant 8 bytes first (bytes 8-15), then bytes 0-7
        uint32_t* perm_in = d_perm_in;
        uint32_t* perm_out = d_perm_out;

        for (int pass = 1; pass >= 0; pass--) {
            int byte_offset = pass * 8;
            // Extract 8 bytes from prefix buffer at PREFIX_BYTES stride
            thrust::counting_iterator<uint64_t> count_idx(0);
            thrust::device_ptr<uint64_t> sk_ptr(d_sort_keys);
            uint8_t* pk = d_prefix_keys;
            uint32_t* pi = perm_in;
            thrust::transform(count_idx, count_idx + num_records, sk_ptr,
                [pk, pi, byte_offset] __device__ (uint64_t i) {
                    uint32_t orig = pi[i];
                    const uint8_t* k = pk + (uint64_t)orig * PREFIX_BYTES + byte_offset;
                    uint64_t v = 0;
                    for (int b = 0; b < 8; b++) v = (v << 8) | k[b];
                    return v;
                });

            cub::DoubleBuffer<uint64_t> kb(d_sort_keys, d_sort_keys_alt);
            cub::DoubleBuffer<uint32_t> ib(perm_in, perm_out);
            size_t temp = cub_temp_bytes;
            cub::DeviceRadixSort::SortPairs(d_cub_temp, temp,
                kb, ib, (int)num_records, 0, 64);
            perm_in = ib.Current();
            perm_out = ib.Alternate();
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        double sort_ms = phase_timer.end_ms();
        printf("  GPU prefix sort (2 LSD passes): %.0f ms\n", sort_ms);

        // Step 3: Download sorted prefix keys + permutation
        phase_timer.begin();
        CUDA_CHECK(cudaMemcpy(h_perm, perm_in,
                               total_perm_bytes, cudaMemcpyDeviceToHost));
        // Download sorted prefix keys for group detection (need the 16B prefixes in sorted order)
        // Reconstruct by reading h_data[h_perm[i]]'s first 16 bytes on CPU (faster than GPU D2H)
        // Actually just download the sorted uint64 from the last CUB pass (bytes 0-7)
        uint8_t* h_sorted_prefixes = (uint8_t*)malloc(num_records * PREFIX_BYTES);
        // Read prefixes from h_data using permutation (sequential perm access, random h_data access)
        {
            int nt = std::max(1, (int)std::thread::hardware_concurrency());
            uint64_t per_t = (num_records + nt - 1) / nt;
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) {
                threads.emplace_back([&, t]() {
                    uint64_t lo = t * per_t, hi = std::min(lo + per_t, num_records);
                    for (uint64_t i = lo; i < hi; i++) {
                        memcpy(h_sorted_prefixes + i * PREFIX_BYTES,
                               h_data + (uint64_t)h_perm[i] * RECORD_SIZE, PREFIX_BYTES);
                    }
                });
            }
            for (auto& t : threads) t.join();
        }
        r.pcie_d2h_gb = total_perm_bytes / 1e9;
        double dl_ms = phase_timer.end_ms();
        printf("  Downloaded perm + extracted prefixes in %.0f ms\n", dl_ms);

        // Free GPU memory
        cudaFree(d_prefix_keys); cudaFree(d_perm_in); cudaFree(d_perm_out);
        cudaFree(d_sort_keys); cudaFree(d_sort_keys_alt); cudaFree(d_cub_temp);
        d_perm_in = d_perm_out = nullptr;

        // Step 4: CPU GATHER first (prefix-sorted order), THEN fixup on contiguous data
        // Gathering first puts each group's records contiguously in h_output,
        // so the fixup sort within each group has excellent L3 cache locality
        // (18.8MB per group fits in ~25MB L3).
        int hw_threads = std::max(1, (int)std::thread::hardware_concurrency());

        phase_timer.begin();
        printf("  CPU gather (prefix-sorted order)...\n");
        int PREFETCH_AHEAD = 256;
        uint64_t chunk_size = (num_records + hw_threads - 1) / hw_threads;
        std::vector<std::thread> threads;
        for (int t = 0; t < hw_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = t * chunk_size;
                uint64_t hi = std::min(lo + chunk_size, num_records);
                for (uint64_t i = lo; i < hi; i++) {
                    if (i + PREFETCH_AHEAD < hi)
                        __builtin_prefetch(h_data + (uint64_t)h_perm[i+PREFETCH_AHEAD] * RECORD_SIZE, 0);
                    memcpy(h_output + i * RECORD_SIZE,
                           h_data + (uint64_t)h_perm[i] * RECORD_SIZE, RECORD_SIZE);
                }
            });
        }
        for (auto& t : threads) t.join();
        double gather_ms = phase_timer.end_ms();
        printf("  Gathered %.2f GB in %.0f ms (%.2f GB/s)\n",
               total_bytes/1e9, gather_ms, total_bytes/(gather_ms*1e6));

        // Step 5: CPU fixup — sort within contiguous groups in h_output
        phase_timer.begin();
        printf("  CPU fixup: sorting within groups of equal 8B prefixes...\n");

        // Find group boundaries using sorted 16-byte prefixes
        std::vector<std::pair<uint64_t, uint64_t>> groups;
        uint64_t grp_start = 0;
        for (uint64_t i = 1; i <= num_records; i++) {
            if (i == num_records ||
                memcmp(h_sorted_prefixes + i * PREFIX_BYTES,
                       h_sorted_prefixes + grp_start * PREFIX_BYTES, PREFIX_BYTES) != 0) {
                groups.push_back({grp_start, i - grp_start});
                grp_start = i;
            }
        }
        printf("    %lu groups (avg %.0f records/group)\n",
               groups.size(), (double)num_records / groups.size());

        // Sort each group IN PLACE in h_output.
        // Groups avg 31 records → tiny, use simple insertion sort.
        uint64_t groups_per_thread = (groups.size() + hw_threads - 1) / hw_threads;
        threads.clear();
        for (int t = 0; t < hw_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t gs = t * groups_per_thread;
                uint64_t ge = std::min(gs + groups_per_thread, (uint64_t)groups.size());
                uint8_t tmp[RECORD_SIZE];
                for (uint64_t g = gs; g < ge; g++) {
                    auto [start, count] = groups[g];
                    if (count <= 1) continue;
                    // Insertion sort — optimal for small groups (avg 31)
                    uint8_t* base = h_output + start * RECORD_SIZE;
                    for (uint64_t i = 1; i < count; i++) {
                        uint8_t* cur = base + i * RECORD_SIZE;
                        if (key_compare(cur - RECORD_SIZE, cur, KEY_SIZE) <= 0) continue;
                        memcpy(tmp, cur, RECORD_SIZE);
                        uint64_t j = i;
                        while (j > 0 && key_compare(base + (j-1)*RECORD_SIZE, tmp, KEY_SIZE) > 0) {
                            memcpy(base + j*RECORD_SIZE, base + (j-1)*RECORD_SIZE, RECORD_SIZE);
                            j--;
                        }
                        memcpy(base + j*RECORD_SIZE, tmp, RECORD_SIZE);
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
        double fixup_ms = phase_timer.end_ms();
        printf("    Fixup: %.0f ms (%d threads)\n", fixup_ms, hw_threads);
        free(h_sorted_prefixes);

        r.run_gen_ms = upload_ms + sort_ms + dl_ms;
        r.merge_ms = fixup_ms + gather_ms;
        r.total_ms = r.run_gen_ms + r.merge_ms;
        r.num_runs = 1; r.merge_passes = 1;
        r.sorted_output = h_output;
        r.sorted_output_size = total_bytes;
        r.sorted_output_is_mmap = h_output_is_mmap;
        return r;
    }
#endif // PREFIX SORT disabled

    // Check: OVC buffers fit in GPU? Need: 4B OVC + 8B prefix + 4B perm = 16B per record
    size_t ovc_total = num_records * (sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t));
    bool use_ovc = (total_keys_bytes + est_arena > free_mem_now * 0.9) &&
                   (ovc_total + 2*buf_bytes + 512*1024*1024 < free_mem_now * 0.95);

    if (use_ovc) {
        printf("  Keys too large for single-pass (%.1f GB > %.1f GB)\n",
               total_keys_bytes/1e9, free_mem_now*0.9/1e9);
        printf("  Using OVC ARCHITECTURE: run-gen + GPU OVC merge + CPU gather\n");
        printf("  OVC buffers: %.1f GB (4B OVC + 8B prefix + 4B perm per record)\n", ovc_total/1e9);

        // Allocate OVC merge buffers first (persistent across run gen)
        CUDA_CHECK(cudaMalloc(&d_ovc_buffer, num_records * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_prefix_buffer, num_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_global_perm, num_records * sizeof(uint32_t)));
        ovc_buffer_records = num_records;
        ovc_run_offset = 0;

        // Re-allocate sort buffers with REDUCED size (OVC buffers take space)
        size_t ovc_used, ovc_dummy;
        CUDA_CHECK(cudaMemGetInfo(&ovc_used, &ovc_dummy));
        size_t ovc_budget = (size_t)(ovc_used * 0.90);
        uint64_t ovc_buf_records = (ovc_budget / NBUFS) / RECORD_SIZE;
        size_t ovc_buf_bytes = ovc_buf_records * RECORD_SIZE;
        printf("  Reduced triple-buffer: %d × %.2f GB (%.0f M records each)\n",
               NBUFS, ovc_buf_bytes/1e9, ovc_buf_records/1e6);

        for (int i = 0; i < NBUFS; i++) {
            CUDA_CHECK(cudaMalloc(&d_buf[i], ovc_buf_bytes));
        }
        // Update buf_records/buf_bytes for this sort run
        buf_records = ovc_buf_records;
        buf_bytes = ovc_buf_bytes;
        sort_ws.allocate(buf_records);

        // Run generation with per-chunk CUB sort
        printf("\n== Phase 1: Run Generation (chunked LSD sort) ==\n");
        double rg_h2d = 0, rg_d2h = 0;
        double rg_ms;
        auto runs = generate_runs_pipelined(h_data, num_records, rg_ms, rg_h2d, rg_d2h);
        r.run_gen_ms = rg_ms;
        r.num_runs = runs.size();
        r.pcie_h2d_gb = rg_h2d / 1e9;
        printf("  %d runs in %.0f ms (%.2f GB/s)\n\n", r.num_runs, rg_ms, total_bytes/(rg_ms*1e6));

        // Phase 2: GPU Prefix Merge — CUB radix sort on (8B prefix, global_index)
        // The 8B prefix is an ABSOLUTE value (not relative like OVC), so CUB
        // radix sort gives correct ordering. LSD: sort by global_index first
        // (preserves run-internal order for ties), then by prefix.
        // This replaces the broken OVC merge-path approach.
        sort_ws.free();
        for (int i = 0; i < NBUFS; i++) { if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; } }
        if (d_key_buffer) { cudaFree(d_key_buffer); d_key_buffer = nullptr; }

        printf("== Phase 2: GPU Prefix Merge (CUB LSD on 8B prefix + index) ==\n");
        WallTimer merge_timer; merge_timer.begin();

        // We have d_prefix_buffer (uint64, 8B per record) and d_global_perm (uint32).
        // LSD approach: 2 CUB passes
        //   Pass 1: sort by d_global_perm (ensures stable ordering within runs)
        //   Pass 2: sort by d_prefix_buffer (primary key)
        // After pass 2: records with equal prefixes preserve run-internal order.

        // Allocate alt buffers for CUB double-buffering
        uint64_t* d_pfx_alt;
        uint32_t* d_perm_alt;
        CUDA_CHECK(cudaMalloc(&d_pfx_alt, num_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_perm_alt, num_records * sizeof(uint32_t)));

        // Free OVC buffer (not needed for prefix merge)
        cudaFree(d_ovc_buffer); d_ovc_buffer = nullptr;

        // CUB temp storage
        size_t cub_temp_bytes = 0;
        {
            cub::DoubleBuffer<uint64_t> pfx_buf(d_prefix_buffer, d_pfx_alt);
            cub::DoubleBuffer<uint32_t> pm_buf(d_global_perm, d_perm_alt);
            cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
                pm_buf, pfx_buf, (int)num_records, 0, 32);
        }
        // Also check prefix sort temp
        size_t cub_temp_bytes2 = 0;
        {
            cub::DoubleBuffer<uint64_t> pfx_buf(d_prefix_buffer, d_pfx_alt);
            cub::DoubleBuffer<uint32_t> pm_buf(d_global_perm, d_perm_alt);
            cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes2,
                pfx_buf, pm_buf, (int)num_records, 0, 64);
        }
        size_t max_temp = std::max(cub_temp_bytes, cub_temp_bytes2);
        void* d_cub_temp;
        CUDA_CHECK(cudaMalloc(&d_cub_temp, max_temp));

        // Pass 1: Sort by global_perm (key=uint32 perm, value=uint64 prefix)
        // This establishes correct ordering within runs for equal prefixes.
        {
            cub::DoubleBuffer<uint32_t> key_buf(d_global_perm, d_perm_alt);
            cub::DoubleBuffer<uint64_t> val_buf(d_prefix_buffer, d_pfx_alt);
            size_t temp = max_temp;
            cub::DeviceRadixSort::SortPairs(d_cub_temp, temp,
                key_buf, val_buf, (int)num_records, 0, 32);
            // After: key_buf.Current() has sorted perm, val_buf.Current() has reordered prefixes
            d_global_perm = key_buf.Current();
            d_perm_alt = key_buf.Alternate();
            d_prefix_buffer = val_buf.Current();
            d_pfx_alt = val_buf.Alternate();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double pass1_ms = merge_timer.end_ms();
        printf("  Pass 1 (sort by index): %.0f ms\n", pass1_ms);

        merge_timer.begin();
        // Pass 2: Sort by prefix (key=uint64 prefix, value=uint32 perm)
        // CUB stable radix sort: equal prefixes maintain pass-1 order (by index).
        {
            cub::DoubleBuffer<uint64_t> key_buf(d_prefix_buffer, d_pfx_alt);
            cub::DoubleBuffer<uint32_t> val_buf(d_global_perm, d_perm_alt);
            size_t temp = max_temp;
            cub::DeviceRadixSort::SortPairs(d_cub_temp, temp,
                key_buf, val_buf, (int)num_records, 0, 64);
            d_prefix_buffer = key_buf.Current();
            d_global_perm = val_buf.Current();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double pass2_ms = merge_timer.end_ms();
        printf("  Pass 2 (sort by prefix): %.0f ms\n", pass2_ms);

        double merge_ms = pass1_ms + pass2_ms;
        printf("  GPU prefix merge: %.0f ms (2 LSD passes)\n", merge_ms);

        // Download final permutation
        phase_timer.begin();
        CUDA_CHECK(cudaMemcpy(h_perm, d_global_perm, total_perm_bytes, cudaMemcpyDeviceToHost));
        double dl_ms = phase_timer.end_ms();
        r.pcie_d2h_gb = total_perm_bytes / 1e9;
        printf("  Downloaded perm in %.0f ms\n", dl_ms);

        // Free GPU buffers
        cudaFree(d_prefix_buffer); d_prefix_buffer = nullptr;
        cudaFree(d_global_perm); d_global_perm = nullptr;
        cudaFree(d_pfx_alt); cudaFree(d_perm_alt); cudaFree(d_cub_temp);

        // CPU gather: apply permutation to original h_data
        phase_timer.begin();
        printf("== Phase 3: CPU Gather ==\n");
        int hw_threads = std::max(1, (int)std::thread::hardware_concurrency());
        int PREFETCH_AHEAD = 256;
        uint64_t chunk_per_t = (num_records + hw_threads - 1) / hw_threads;
        std::vector<std::thread> threads;
        for (int t = 0; t < hw_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = t * chunk_per_t;
                uint64_t hi = std::min(lo + chunk_per_t, num_records);
                for (uint64_t i = lo; i < hi; i++) {
                    if (i + PREFETCH_AHEAD < hi)
                        __builtin_prefetch(h_data + (uint64_t)h_perm[i+PREFETCH_AHEAD] * RECORD_SIZE, 0);
                    memcpy(h_output + i * RECORD_SIZE,
                           h_data + (uint64_t)h_perm[i] * RECORD_SIZE, RECORD_SIZE);
                }
            });
        }
        for (auto& t : threads) t.join();
        double gather_ms = phase_timer.end_ms();
        printf("  Gathered %.2f GB in %.0f ms (%.2f GB/s)\n",
               total_bytes/1e9, gather_ms, total_bytes/(gather_ms*1e6));

        r.merge_ms = merge_ms + dl_ms + gather_ms;
        r.merge_passes = pass;
        r.sorted_output = h_output;
        r.sorted_output_size = total_bytes;
        r.sorted_output_is_mmap = h_output_is_mmap;
        r.total_ms = r.run_gen_ms + r.merge_ms;
        printf("  Merged in %.0f ms\n", r.merge_ms);
        return r;
    }

    // Strided DMA: extract only KEY_SIZE bytes per record from host
    CUDA_CHECK(cudaMalloc(&d_keys_10byte, total_keys_bytes));
    CUDA_CHECK(cudaMemcpy2DAsync(
        d_keys_10byte, KEY_SIZE,
        h_data, RECORD_SIZE,
        KEY_SIZE, num_records,
        cudaMemcpyHostToDevice, streams[0]));
    r.pcie_h2d_gb = total_keys_bytes / 1e9;

    // Allocate arena for CUB sort: uint64 keys + uint32 perm + CUB temp
    size_t cub_temp_bytes = 0;
    {
        cub::DoubleBuffer<uint64_t> dk(nullptr,nullptr);
        cub::DoubleBuffer<uint32_t> dp(nullptr,nullptr);
        cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, dk, dp, (int)num_records, 0, 64);
    }
    size_t arena_sz = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + cub_temp_bytes;
    uint8_t* d_arena;
    CUDA_CHECK(cudaMalloc(&d_arena, arena_sz));

    uint64_t* d_sort_keys = (uint64_t*)d_arena;
    uint64_t* d_sort_keys_alt = d_sort_keys + num_records;
    d_perm_in = (uint32_t*)(d_sort_keys_alt + num_records);
    d_perm_out = d_perm_in + num_records;
    void* d_temp = (void*)(d_perm_out + num_records);

    double upload_ms = phase_timer.end_ms();
    printf("  Uploaded %.2f GB keys (strided, GPU-direct) in %.0f ms (%.2f GB/s effective)\n",
           total_keys_bytes/1e9, upload_ms, total_bytes/(upload_ms*1e6));

    // ── Step 3: GPU sort (extract uint64, init perm, CUB radix sort) ──
    printf("== Step 3: GPU CUB Radix Sort ==\n");
    phase_timer.begin();

    int nthreads = 256, nblks = (num_records + nthreads - 1) / nthreads;

    // ── LSD Multi-Pass Radix Sort for full KEY_SIZE correctness ──
    // Sort from least significant 8-byte chunk to most significant.
    // For KEY_SIZE=10: 2 passes (bytes 8-9, then bytes 0-7)
    // For KEY_SIZE=88: 11 passes (bytes 80-87, 72-79, ..., 0-7)
    // CUB radix sort is stable, so LSD ordering is correct.

    init_identity_kernel<<<nblks, nthreads, 0, streams[0]>>>(d_perm_in, num_records);

    int num_chunks = (KEY_SIZE + 7) / 8;  // ceil(KEY_SIZE / 8)
    printf("  LSD sort: %d passes for %d-byte key\n", num_chunks, KEY_SIZE);

    GpuTimer pass_timer;
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int byte_offset = chunk * 8;
        int chunk_bytes = std::min(8, KEY_SIZE - byte_offset);
        pass_timer.begin();

        // Extract uint64 from this chunk of the key (in permutation order)
        // Use a kernel that reads key[perm[i]][byte_offset:byte_offset+8]
        // For the FIRST pass (highest chunk idx), perm is identity → read directly
        if (chunk == num_chunks - 1 && chunk_bytes <= 2) {
            // Last chunk is ≤2 bytes — use uint16 sort (fewer radix passes)
            uint16_t* d_tie = reinterpret_cast<uint16_t*>(d_sort_keys);
            uint16_t* d_tie_alt = reinterpret_cast<uint16_t*>(d_sort_keys_alt);
            extract_tiebreaker_kernel<<<nblks, nthreads, 0, streams[0]>>>(
                d_keys_10byte, d_perm_in, d_tie, num_records);
            cub::DoubleBuffer<uint16_t> tie_buf(d_tie, d_tie_alt);
            cub::DoubleBuffer<uint32_t> perm_buf(d_perm_in, d_perm_out);
            size_t t = cub_temp_bytes;
            cub::DeviceRadixSort::SortPairs(d_temp, t,
                tie_buf, perm_buf, (int)num_records, 0, 16, streams[0]);
            d_perm_in = perm_buf.Current();
            d_perm_out = perm_buf.Alternate();
        } else {
            // Extract uint64 for this chunk, in current permutation order
            // Kernel: d_sort_keys[i] = big-endian uint64 from key[perm[i]][byte_offset:+8]
            extract_uint64_chunk_kernel<<<nblks, nthreads, 0, streams[0]>>>(
                d_keys_10byte, d_perm_in, d_sort_keys, num_records, byte_offset, chunk_bytes);

            cub::DoubleBuffer<uint64_t> keys_buf(d_sort_keys, d_sort_keys_alt);
            cub::DoubleBuffer<uint32_t> perm_buf(d_perm_in, d_perm_out);
            cub::DeviceRadixSort::SortPairs(d_temp, cub_temp_bytes,
                keys_buf, perm_buf, (int)num_records, 0, chunk_bytes * 8, streams[0]);
            d_perm_in = perm_buf.Current();
            d_perm_out = perm_buf.Alternate();
        }
        float pass_ms = pass_timer.end();
        printf("    Pass %d (bytes %d-%d): %.1f ms\n", num_chunks - chunk, byte_offset, byte_offset + chunk_bytes - 1, pass_ms);
    }
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    cudaFree(d_keys_10byte);

    double sort_ms = phase_timer.end_ms();
    printf("  Sorted %lu keys in %.0f ms\n", num_records, sort_ms);

    // ── Step 4: Download permutation ──
    printf("== Step 4: Download Permutation ==\n");
    phase_timer.begin();

    CUDA_CHECK(cudaMemcpy(h_perm, d_perm_in, total_perm_bytes, cudaMemcpyDeviceToHost));
    r.pcie_d2h_gb = total_perm_bytes / 1e9;

    cudaFree(d_arena);
    if (h_keys) cudaFreeHost(h_keys);

    double download_ms = phase_timer.end_ms();
    printf("  Downloaded %.2f GB perm in %.0f ms\n", total_perm_bytes/1e9, download_ms);

    // ── Step 5: CPU multi-threaded gather ──
    printf("== Step 5: CPU Gather ==\n");
    phase_timer.begin();

    {
        int num_threads = std::min(48, (int)std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        uint64_t chunk = (num_records + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; t++) {
            uint64_t start = (uint64_t)t * chunk;
            uint64_t end = std::min(start + chunk, num_records);
            if (start < end) {
                threads.emplace_back([=, &h_data, &h_output, &h_perm]() {
                    constexpr int BLOCK = 256;
                    const uint8_t* src_ptrs[BLOCK];
                    for (uint64_t base = start; base < end; base += BLOCK) {
                        int count = std::min((uint64_t)BLOCK, end - base);
                        for (int j = 0; j < count; j++) {
                            uint32_t idx = h_perm[base + j];
                            src_ptrs[j] = h_data + (uint64_t)idx * RECORD_SIZE;
                            __builtin_prefetch(src_ptrs[j], 0, 0);
                        }
                        for (int j = 0; j < count; j++) {
                            memcpy(h_output + (base + j) * RECORD_SIZE, src_ptrs[j], RECORD_SIZE);
                        }
                    }
                });
            }
        }
        for (auto& t : threads) t.join();
    }

    double gather_ms = phase_timer.end_ms();
    printf("  Gathered %.2f GB in %.0f ms (%.2f GB/s)\n",
           total_bytes/1e9, gather_ms, total_bytes/(gather_ms*1e6));

    cudaFreeHost(h_perm);

    r.run_gen_ms = upload_ms + sort_ms + download_ms;  // no separate extract step with cudaMemcpy2D
    r.merge_ms = gather_ms;
    r.merge_passes = 1;
    r.num_runs = 1;
    r.sorted_output = h_output;
    r.sorted_output_size = total_bytes;
    r.sorted_output_is_mmap = h_output_is_mmap;
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
    const char* input_file = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--total-gb") && i+1<argc) total_gb = atof(argv[++i]);
        else if (!strcmp(argv[i],"--input") && i+1<argc) input_file = argv[++i];
        else if (!strcmp(argv[i],"--no-verify")) verify = false;
        else { printf("Usage: %s [--total-gb N] [--input FILE] [--no-verify]\n",argv[0]); return 0; }
    }

    // Determine data size from file or --total-gb
    uint64_t num_records, total_bytes;
    if (input_file) {
        FILE* f = fopen(input_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
        fseek(f, 0, SEEK_END);
        total_bytes = ftell(f);
        fclose(f);
        num_records = total_bytes / RECORD_SIZE;
        total_bytes = num_records * RECORD_SIZE; // round down
    } else {
        num_records = (uint64_t)(total_gb * 1e9) / RECORD_SIZE;
        total_bytes = num_records * RECORD_SIZE;
    }

    printf("════════════════════════════════════════════════════\n");
    printf("  GPU External Merge Sort — Streaming Benchmark\n");
    printf("════════════════════════════════════════════════════\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (%.1f GB HBM, %d SMs, %.0f GB/s BW)\n", props.name,
           props.totalGlobalMem/1e9, props.multiProcessorCount,
           2.0 * props.memoryClockRate * (props.memoryBusWidth/8) / 1e6);
    printf("Data: %.2f GB (%lu records × %d bytes)%s\n\n",
           total_bytes/1e9, num_records, RECORD_SIZE,
           input_file ? " (from file)" : " (random)");

    printf("Allocating %.2f GB pinned host memory...\n", total_bytes/1e9);
    uint8_t* h_data;
    cudaError_t alloc_err = cudaMallocHost(&h_data, total_bytes);
    if (alloc_err != cudaSuccess) {
        printf("  cudaMallocHost failed, falling back to malloc\n");
        h_data = (uint8_t*)malloc(total_bytes);
    }
    if (!h_data) { fprintf(stderr,"allocation failed\n"); return 1; }
    madvise(h_data, total_bytes, MADV_HUGEPAGE);

    if (input_file) {
        printf("Loading from %s...\n", input_file);
        WallTimer gt; gt.begin();
        FILE* f = fopen(input_file, "rb");
        size_t read = fread(h_data, 1, total_bytes, f);
        fclose(f);
        printf("  Loaded %.2f GB in %.0f ms (%.2f GB/s)\n\n",
               read/1e9, gt.end_ms(), read/(gt.end_ms()*1e6));
    } else {
        printf("Generating random data...\n");
        WallTimer gt; gt.begin();
        gen_data(h_data, num_records);
        printf("  Generated in %.0f ms\n\n", gt.end_ms());
    }

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
    if (result.sorted_output) {
        if (result.sorted_output_is_mmap) munmap(result.sorted_output, result.sorted_output_size);
        else free(result.sorted_output);
    }

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
