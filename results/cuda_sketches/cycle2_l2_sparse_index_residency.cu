// CrocSort GPU Transfer: 5% L2 Residency Control for Sparse Indexes
// Maps from: src/sort/core/sparse_index.rs:SparseIndex (page-backed, 4KB pages)
//            src/sort/core/engine.rs:INDEX_BUDGET_PCT = 5 (line 50)
//            src/sort/core/sparse_index.rs:SparseIndexPagePool (page recycling)
// Concept: Reserve 5% of GPU L2 cache for sparse index pages using
//          CUDA L2 persistence API, ensuring merge boundary lookups hit L2.

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

// Matches CrocSort's SPARSE_INDEX_PAGE_SIZE = 4 * 1024 (sparse_index.rs:10)
static constexpr int SPARSE_INDEX_PAGE_SIZE = 4096;

// Matches CrocSort's ENTRY_HEADER_SIZE = 10 (sparse_index.rs:21)
// Layout: [file_offset: u64 (8B)] [key_len: u16 (2B)] [key: variable]
static constexpr int ENTRY_HEADER_SIZE = 10;

// Matches CrocSort's INDEX_BUDGET_PCT = 5 (engine.rs:50)
static constexpr float INDEX_BUDGET_PCT = 0.05f;

// Sparse index entry — matches SparseIndex::push layout (sparse_index.rs:197)
struct __align__(8) SparseIndexEntry {
    uint64_t file_offset;   // Byte offset in run file
    uint16_t key_len;       // Key length
    // Followed by key bytes (variable length)
};

// Handle encoding — matches encode_handle/decode_handle (sparse_index.rs:93)
// Upper 20 bits: page_idx, Lower 12 bits: byte_offset
__device__ __forceinline__ uint32_t encode_handle(uint32_t page_idx, uint32_t byte_offset) {
    return (page_idx << 12) | byte_offset;
}

__device__ __forceinline__ void decode_handle(uint32_t handle, uint32_t& page_idx, uint32_t& byte_offset) {
    page_idx = handle >> 12;
    byte_offset = handle & 0xFFF;
}

// GPU sparse index structure — maps from SparseIndex struct (sparse_index.rs:112)
struct GpuSparseIndex {
    uint32_t* pos;          // Handle array (page_idx << 12 | byte_offset)
    uint8_t* pages;         // Concatenated 4KB pages
    int num_entries;
    int num_pages;
    int run_id;
};

// Multi-index structure — maps from MultiSparseIndexes (run_format.rs)
struct GpuMultiSparseIndexes {
    GpuSparseIndex* indexes;
    int num_indexes;
    int total_entries;
    uint64_t total_bytes;
};

// Configure L2 persistence for sparse index pages
// This ensures merge boundary lookups hit L2 instead of going to HBM
void configure_l2_persistence_for_sparse_indexes(
    GpuMultiSparseIndexes* d_indexes,
    int num_indexes,
    size_t l2_cache_size
) {
    // Calculate total sparse index bytes
    size_t total_index_bytes = 0;
    // TODO: sum up page counts from host-side metadata

    // Reserve INDEX_BUDGET_PCT of L2 for sparse indexes
    size_t l2_budget = static_cast<size_t>(l2_cache_size * INDEX_BUDGET_PCT);

    // Set L2 persistence window
    // Maps CrocSort's SparseIndexPagePool concept to GPU L2 persistence
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = nullptr; // Set per-index below
    stream_attr.accessPolicyWindow.num_bytes = l2_budget;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;    // Try to keep all of it in L2
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    // Apply persistence to the sparse index page memory
    // The pages pointer covers all sparse index data
    // TODO: Set base_ptr to actual device memory for sparse index pages
    // stream_attr.accessPolicyWindow.base_ptr = d_sparse_pages;
    // stream_attr.accessPolicyWindow.num_bytes = min(total_index_bytes, l2_budget);

    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);

    // All merge boundary search kernels should use this stream
    // to benefit from L2 persistence
    cudaStreamDestroy(stream);
}

// Binary search within a sparse index — maps to boundary search in
// run_format.rs: select_boundary_by_size / select_boundary_by_count
__device__ int sparse_index_upper_bound(
    const GpuSparseIndex* idx,
    const uint8_t* key,
    int key_len,
    int lo,
    int hi
) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;

        // Decode handle to find entry location
        uint32_t page_idx, byte_offset;
        decode_handle(idx->pos[mid], page_idx, byte_offset);

        // Read entry from page — this read should hit L2 due to persistence
        const uint8_t* page = idx->pages + page_idx * SPARSE_INDEX_PAGE_SIZE;
        const SparseIndexEntry* entry = reinterpret_cast<const SparseIndexEntry*>(
            page + byte_offset);
        const uint8_t* entry_key = page + byte_offset + ENTRY_HEADER_SIZE;
        int entry_key_len = entry->key_len;

        // Compare keys (lexicographic)
        int cmp_len = min(key_len, entry_key_len);
        int cmp = 0;
        for (int i = 0; i < cmp_len; i++) {
            if (key[i] != entry_key[i]) {
                cmp = (key[i] < entry_key[i]) ? -1 : 1;
                break;
            }
        }
        if (cmp == 0) {
            cmp = (key_len < entry_key_len) ? -1 : (key_len > entry_key_len) ? 1 : 0;
        }

        if (cmp >= 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Double binary search for byte-balanced boundary selection
// Maps from: run_format.rs select_boundary_by_size
// This is the core of CrocSort's merge partitioning
__global__ void select_merge_boundaries_kernel(
    const GpuMultiSparseIndexes* __restrict__ multi_idx,
    uint64_t target_bytes,          // Target byte volume for this partition
    uint32_t* __restrict__ result_handle,  // Output: boundary handle
    uint8_t* __restrict__ result_key,      // Output: boundary key
    int* __restrict__ result_key_len
) {
    // Each thread block computes one boundary
    const int tid = threadIdx.x;
    const int num_indexes = multi_idx->num_indexes;

    // Shared memory for per-index lo/hi search windows
    extern __shared__ int smem[];
    int* lo = smem;                         // [num_indexes]
    int* hi = smem + num_indexes;           // [num_indexes]

    // Initialize search windows
    if (tid < num_indexes) {
        lo[tid] = 0;
        hi[tid] = multi_idx->indexes[tid].num_entries;
    }
    __syncthreads();

    // Outer loop: double binary search
    // Maps from run_format.rs boundary search logic
    const int MAX_ITERATIONS = 40; // log2(max_entries) * num_indexes

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Find largest active range (picks pivot source)
        // Maps: "largest byte range" heuristic from merge_partitioning doc
        __shared__ int pivot_idx;
        __shared__ int pivot_mid;
        __shared__ bool made_progress;

        if (tid == 0) {
            int64_t max_range = -1;
            pivot_idx = 0;
            for (int k = 0; k < num_indexes; k++) {
                // TODO: Use bytes_before for size mode instead of entry count
                int64_t range = hi[k] - lo[k];
                if (range > max_range) {
                    max_range = range;
                    pivot_idx = k;
                }
            }
            pivot_mid = lo[pivot_idx] + (hi[pivot_idx] - lo[pivot_idx]) / 2;
            made_progress = false;
        }
        __syncthreads();

        if (hi[pivot_idx] - lo[pivot_idx] <= 1) break;

        // Read pivot key from the selected index
        // This sparse index read should hit L2 (persistence)
        __shared__ uint8_t pivot_key[256]; // TODO: dynamic key length
        __shared__ int pivot_key_len;

        if (tid == 0) {
            uint32_t page_idx, byte_offset;
            decode_handle(multi_idx->indexes[pivot_idx].pos[pivot_mid], page_idx, byte_offset);
            const uint8_t* page = multi_idx->indexes[pivot_idx].pages +
                                  page_idx * SPARSE_INDEX_PAGE_SIZE;
            const SparseIndexEntry* entry = reinterpret_cast<const SparseIndexEntry*>(
                page + byte_offset);
            pivot_key_len = min((int)entry->key_len, 255);
            memcpy(pivot_key, page + byte_offset + ENTRY_HEADER_SIZE, pivot_key_len);
        }
        __syncthreads();

        // Each thread does upper_bound for one index (parallel across indexes)
        if (tid < num_indexes) {
            int new_pos = sparse_index_upper_bound(
                &multi_idx->indexes[tid],
                pivot_key, pivot_key_len,
                lo[tid], hi[tid]);

            // Accumulate bytes objective
            // TODO: compute bytes_before(new_pos) for size-balanced mode
        }
        __syncthreads();

        // Update lo/hi based on objective vs target
        // TODO: compute total objective, update lo/hi, track progress
        // Maps: run_format.rs update rule: if objective < target: lo = pos; else: hi = pos;

        if (tid == 0 && !made_progress) break;
        __syncthreads();
    }

    // Finalization: gather candidates from lo[k], sort, pick first >= target
    // Maps: run_format.rs candidate finalization
    if (tid == 0) {
        // TODO: Collect candidates, sort by (key, run_id, offset), select boundary
        *result_key_len = 0; // placeholder
    }
}

// Host-side launch:
// size_t l2_size;
// cudaDeviceGetAttribute((int*)&l2_size, cudaDevAttrL2CacheSize, 0);
// configure_l2_persistence_for_sparse_indexes(d_indexes, num_indexes, l2_size);
// int smem = num_indexes * 2 * sizeof(int);
// select_merge_boundaries_kernel<<<num_partitions, max(num_indexes, 32), smem, stream>>>(
//     d_multi_idx, target_bytes, d_result_handle, d_result_key, d_result_key_len);
