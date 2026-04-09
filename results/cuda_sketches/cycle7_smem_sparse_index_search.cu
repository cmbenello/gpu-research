// CrocSort GPU Transfer: Shared-Memory Sparse Index for Boundary Search
// Maps from: src/sort/core/sparse_index.rs:SparseIndex (4KB pages, handle encoding)
//            docs/merge_partitioning_with_sparse_indexes.md (double binary search)
// GPU primitive: Shared memory for sub-microsecond boundary lookups

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

static constexpr int SPARSE_INDEX_PAGE_SIZE = 4096;
static constexpr int ENTRY_HEADER_SIZE = 10; // 8B offset + 2B key_len

// Sparse index entry in shared memory
// Matches SparseIndex page layout (sparse_index.rs:197)
struct SmemIndexEntry {
    uint64_t file_offset;   // Byte offset in run file
    uint16_t key_len;       // Key length
    // key bytes follow immediately (variable length)
};

// Compact index entry for shared memory (fixed-size for coalesced access)
// Stores only the minimum needed for boundary search
struct __align__(16) CompactIndexEntry {
    uint64_t file_offset;       // For byte-balanced partitioning
    uint32_t key_prefix;        // First 4 bytes of key (for coarse search)
    uint16_t key_len;           // Full key length
    uint16_t run_id;            // Which run this belongs to
};

// ── Cooperative sparse index loading ───────────────────────────────
// Load sparse index pages from HBM to shared memory using CUB BlockLoad
// Maps from: SparseIndex::key() and file_offset() accessors

__device__ void load_sparse_index_to_smem(
    const uint8_t* __restrict__ d_pages,     // HBM: concatenated 4KB pages
    const uint32_t* __restrict__ d_handles,  // HBM: handle array (page_idx << 12 | offset)
    int num_entries,
    CompactIndexEntry* __restrict__ s_entries,  // SMEM: compact entries
    int max_smem_entries
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    int entries_to_load = min(num_entries, max_smem_entries);

    for (int i = tid; i < entries_to_load; i += block_size) {
        uint32_t handle = d_handles[i];
        uint32_t page_idx = handle >> 12;
        uint32_t byte_offset = handle & 0xFFF;

        // Decode entry from page — matches sparse_index.rs:168-172
        const uint8_t* page = d_pages + page_idx * SPARSE_INDEX_PAGE_SIZE;
        const uint8_t* entry_ptr = page + byte_offset;

        // Read file_offset (8B LE)
        uint64_t file_offset;
        memcpy(&file_offset, entry_ptr, 8);

        // Read key_len (2B LE)
        uint16_t key_len;
        memcpy(&key_len, entry_ptr + 8, 2);

        // Read key prefix (first 4 bytes)
        uint32_t key_prefix = 0;
        const uint8_t* key_ptr = entry_ptr + ENTRY_HEADER_SIZE;
        int copy_len = min((int)key_len, 4);
        for (int j = 0; j < copy_len; j++) {
            key_prefix |= ((uint32_t)key_ptr[j]) << (24 - 8*j);
        }

        s_entries[i].file_offset = file_offset;
        s_entries[i].key_prefix = key_prefix;
        s_entries[i].key_len = key_len;
        s_entries[i].run_id = 0; // Set by caller
    }
    __syncthreads();
}

// ── Shared-memory binary search ────────────────────────────────────
// Upper bound search within shared memory entries
// Maps from: sparse_index_upper_bound concept in run_format.rs

__device__ int smem_upper_bound(
    const CompactIndexEntry* __restrict__ s_entries,
    int lo, int hi,
    uint32_t search_key_prefix,    // First 4 bytes of search key
    const uint8_t* full_key,       // Full key for tie-breaking (in HBM)
    int full_key_len
) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;

        // Fast path: compare 4-byte prefix (single instruction)
        if (s_entries[mid].key_prefix < search_key_prefix) {
            lo = mid + 1;
        } else if (s_entries[mid].key_prefix > search_key_prefix) {
            hi = mid;
        } else {
            // Prefix tie: need full key comparison
            // TODO: fetch full key from HBM via cp.async for exact comparison
            // For now, use prefix + file_offset as approximation
            lo = mid + 1; // Conservative: treat ties as "less than"
        }
    }
    return lo;
}

// ── Double binary search for byte-balanced boundaries ──────────────
// Maps from: docs/merge_partitioning_with_sparse_indexes.md
// Outer loop: pick pivot from largest active range
// Inner loop: binary search all indexes in parallel

__global__ void byte_balanced_boundary_search(
    // Sparse index data (already loaded to shared memory in production)
    const uint8_t** __restrict__ d_pages_per_run,
    const uint32_t** __restrict__ d_handles_per_run,
    const int* __restrict__ entries_per_run,
    const uint64_t* __restrict__ total_bytes_per_run,
    const int num_runs,
    // Boundary computation
    const uint64_t target_bytes,          // Target byte volume for this boundary
    // Output
    uint32_t* __restrict__ out_boundary_key_prefix,
    uint64_t* __restrict__ out_boundary_file_offset,
    int* __restrict__ out_boundary_run_id
) {
    extern __shared__ uint8_t smem_raw[];

    const int tid = threadIdx.x;

    // Shared memory: per-run lo/hi arrays + compact index entries
    int* lo = reinterpret_cast<int*>(smem_raw);
    int* hi = lo + num_runs;
    uint64_t* base_bytes = reinterpret_cast<uint64_t*>(hi + num_runs);
    // Remaining smem for compact entries (loaded on demand)

    // Initialize search windows
    // Maps from: run_format.rs boundary search initialization
    if (tid < num_runs) {
        lo[tid] = 0;
        hi[tid] = entries_per_run[tid];
    }
    __syncthreads();

    // Compute base_bytes prefix sum (for global byte offset)
    // Maps from: MultiSparseIndexes base_bytes computation
    if (tid == 0) {
        uint64_t cumulative = 0;
        for (int k = 0; k < num_runs; k++) {
            base_bytes[k] = cumulative;
            cumulative += total_bytes_per_run[k];
        }
    }
    __syncthreads();

    // ── Outer loop: double binary search ──
    const int MAX_ITER = 50;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find largest active range (by bytes)
        // Maps from: "largest byte range" pivot selection
        __shared__ int pivot_run;
        __shared__ int pivot_mid;

        if (tid == 0) {
            int64_t max_byte_range = -1;
            pivot_run = 0;
            for (int k = 0; k < num_runs; k++) {
                // Approximate byte range from entry count * avg entry size
                int64_t entry_range = hi[k] - lo[k];
                if (entry_range <= 0) continue;
                // TODO: use actual bytes_before for size mode
                int64_t byte_range = entry_range * (total_bytes_per_run[k] /
                    max(entries_per_run[k], 1));
                if (byte_range > max_byte_range) {
                    max_byte_range = byte_range;
                    pivot_run = k;
                }
            }
            pivot_mid = lo[pivot_run] + (hi[pivot_run] - lo[pivot_run]) / 2;
        }
        __syncthreads();

        if (hi[pivot_run] - lo[pivot_run] <= 1) break;

        // Read pivot key from selected run's sparse index
        // This access should be in L2 if persistence is configured (cycle 2)
        __shared__ uint32_t pivot_key_prefix;
        if (tid == 0) {
            uint32_t handle = d_handles_per_run[pivot_run][pivot_mid];
            uint32_t page_idx = handle >> 12;
            uint32_t byte_offset = handle & 0xFFF;
            const uint8_t* page = d_pages_per_run[pivot_run] +
                                  page_idx * SPARSE_INDEX_PAGE_SIZE;
            const uint8_t* key_ptr = page + byte_offset + ENTRY_HEADER_SIZE;
            pivot_key_prefix = 0;
            for (int j = 0; j < min(4, (int)(page[byte_offset + 8] | (page[byte_offset + 9] << 8))); j++) {
                pivot_key_prefix |= ((uint32_t)key_ptr[j]) << (24 - 8*j);
            }
        }
        __syncthreads();

        // ── Inner loop: parallel binary search across all runs ──
        // Each thread handles one run (tid < num_runs)
        __shared__ int pos[256]; // Max 256 runs
        __shared__ bool made_progress;

        if (tid == 0) made_progress = false;
        __syncthreads();

        if (tid < num_runs) {
            // Binary search for pivot in this run's index
            // Maps from: upper_bound in boundary search inner loop
            int my_lo = lo[tid];
            int my_hi = hi[tid];
            while (my_lo < my_hi) {
                int mid = my_lo + (my_hi - my_lo) / 2;
                // Read key prefix from this run's sparse index
                uint32_t handle = d_handles_per_run[tid][mid];
                uint32_t page_idx = handle >> 12;
                uint32_t byte_off = handle & 0xFFF;
                const uint8_t* page = d_pages_per_run[tid] +
                                      page_idx * SPARSE_INDEX_PAGE_SIZE;
                uint32_t entry_key_prefix = 0;
                const uint8_t* kp = page + byte_off + ENTRY_HEADER_SIZE;
                for (int j = 0; j < 4; j++) {
                    entry_key_prefix |= ((uint32_t)kp[j]) << (24 - 8*j);
                }

                if (entry_key_prefix <= pivot_key_prefix) {
                    my_lo = mid + 1;
                } else {
                    my_hi = mid;
                }
            }
            pos[tid] = my_lo;
        }
        __syncthreads();

        // Compute objective: sum of bytes across all runs
        // Maps from: size mode objective = sum(bytes_before(index[k], pos[k]))
        if (tid == 0) {
            uint64_t total_obj = 0;
            for (int k = 0; k < num_runs; k++) {
                // Approximate bytes_before using proportional estimation
                if (entries_per_run[k] > 0) {
                    total_obj += base_bytes[k] +
                        (uint64_t)pos[k] * total_bytes_per_run[k] / entries_per_run[k];
                }
            }

            // Update lo/hi based on objective vs target
            // Maps from: run_format.rs update rule
            bool any_changed = false;
            if (total_obj < target_bytes) {
                for (int k = 0; k < num_runs; k++) {
                    if (pos[k] > lo[k]) { lo[k] = pos[k]; any_changed = true; }
                }
            } else {
                for (int k = 0; k < num_runs; k++) {
                    if (pos[k] < hi[k]) { hi[k] = pos[k]; any_changed = true; }
                }
            }
            made_progress = any_changed;
        }
        __syncthreads();

        // Stall detection — matches doc: "if !made_progress: break"
        if (!made_progress) break;
    }

    // ── Finalization: pick boundary from candidates ──
    // Maps from: run_format.rs candidate finalization
    // Gather lo[k] entries, sort by (key, run_id, offset), pick first >= target
    if (tid == 0) {
        // TODO: Implement full candidate expansion for stall case
        // For now: pick the entry at lo[pivot_run]
        *out_boundary_run_id = 0; // placeholder
        *out_boundary_key_prefix = 0;
        *out_boundary_file_offset = 0;
    }
}

// Host-side launch:
// int smem = num_runs * (sizeof(int)*2 + sizeof(uint64_t)) + 256*sizeof(int);
// byte_balanced_boundary_search<<<num_boundaries, max(num_runs, 32), smem>>>(
//     d_pages_per_run, d_handles_per_run, entries_per_run,
//     total_bytes_per_run, num_runs, target_bytes,
//     d_boundary_key, d_boundary_offset, d_boundary_run_id);
