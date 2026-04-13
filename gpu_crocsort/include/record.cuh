#pragma once
#include <cstdint>
#include <cstdio>

// Record format: configurable via compile flags
// GenSort: KEY_SIZE=10, VALUE_SIZE=90, RECORD_SIZE=100
// TPC-H lineitem: KEY_SIZE=88, VALUE_SIZE=32, RECORD_SIZE=120
#ifndef KEY_SIZE
#define KEY_SIZE 10
#endif
#ifndef VALUE_SIZE
#define VALUE_SIZE 90
#endif
static constexpr int RECORD_SIZE = KEY_SIZE + VALUE_SIZE;

// Run generation config
static constexpr int RECORDS_PER_BLOCK = 512;    // Records sorted per thread block
static constexpr int BLOCK_THREADS = 256;         // Threads per block for run generation
static constexpr int SPARSE_INDEX_STRIDE = 64;    // Sample every 64th record

// ── Merge config (merge-path based 2-way merge) ───────────────────
//
// Uses merge path algorithm (Green et al., 2012): every thread does work.
//
// PARALLELISM:
//   - 256 threads per block, each merges 8 records = 2048 records/block
//   - Blocks per pass = total_records / 2048
//     e.g., 10M records → 4,883 blocks, ALL threads active
//   - Passes = ceil(log2(num_runs))
//     e.g., 19,531 runs → 15 passes
//
// COMPARISON:
//   Old loser tree: 1 active thread per block × 128 blocks = 128 threads
//   New merge path: 256 threads per block × 4883 blocks = 1.25M threads
//   That's ~10,000x more parallelism in the merge phase.
//
// WHY 2-WAY INSTEAD OF K-WAY:
//   K-way merge with a tournament tree is inherently sequential (1 thread).
//   2-way merge path is embarrassingly parallel (all threads).
//   Even though 2-way needs more passes (log2 vs logK), each pass is
//   so much faster that total time is lower. The GPU's strength is
//   massive parallelism, not single-thread performance.
//
static constexpr int MERGE_ITEMS_PER_THREAD_CFG = 8;   // Tunable: 4-16
static constexpr int MERGE_BLOCK_THREADS_CFG = 256;     // Full warp occupancy

// Sort key: we use the first 8 bytes as a uint64_t for radix sort,
// then the remaining 2 bytes as a tiebreaker
struct __align__(4) SortKey {
    uint64_t hi;   // Bytes 0-7 of key (big-endian for correct ordering)
    uint16_t lo;   // Bytes 8-9 of key

    __host__ __device__ bool operator<(const SortKey& other) const {
        if (hi != other.hi) return hi < other.hi;
        return lo < other.lo;
    }
    __host__ __device__ bool operator>(const SortKey& other) const {
        return other < *this;
    }
    __host__ __device__ bool operator==(const SortKey& other) const {
        return hi == other.hi && lo == other.lo;
    }
    __host__ __device__ bool operator<=(const SortKey& other) const {
        return !(other < *this);
    }
};

// Extract SortKey from raw key bytes (big-endian for correct lexicographic order)
__host__ __device__ inline SortKey make_sort_key(const uint8_t* key) {
    SortKey sk;
    sk.hi = 0;
    for (int i = 0; i < 8; i++) {
        sk.hi = (sk.hi << 8) | key[i];
    }
    sk.lo = ((uint16_t)key[8] << 8) | key[9];
    return sk;
}

// Write SortKey back to raw bytes
__host__ __device__ inline void write_sort_key(uint8_t* dst, SortKey sk) {
    for (int i = 7; i >= 0; i--) {
        dst[i] = (uint8_t)(sk.hi & 0xFF);
        sk.hi >>= 8;
    }
    dst[8] = (uint8_t)(sk.lo >> 8);
    dst[9] = (uint8_t)(sk.lo & 0xFF);
}

// Sparse index entry: sampled during run generation
struct SparseEntry {
    uint64_t byte_offset;     // Byte offset within the run
    uint8_t  key[KEY_SIZE];   // Full key at this sample point
};

// Run descriptor: metadata about one sorted run
struct RunDescriptor {
    uint64_t offset;          // Byte offset in the runs buffer
    uint64_t num_records;     // Number of records in this run
    uint64_t num_bytes;       // Total bytes (num_records * RECORD_SIZE)
    int      sparse_offset;   // Index into sparse index array
    int      sparse_count;    // Number of sparse entries for this run
};

// Merge descriptors are defined in merge.cu and host_sort.cu
// (PairDesc2Way for 2-way, KWayPartition for K-way merge tree)

// Timer utility
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }
    float end(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
