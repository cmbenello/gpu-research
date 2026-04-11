#include "record.cuh"
#include "ovc.cuh"

// ============================================================================
// Phase 1: Run Generation — GPU block-sort with OVC delta computation
//
// Each thread block loads a chunk of records into shared memory,
// sorts them by key, computes OVC deltas, and writes a sorted run.
//
// Maps from: CrocSort run generation (engine.rs:execute_run_generation)
// but uses block-sort instead of replacement selection for GPU efficiency.
// ============================================================================

// ── Bitonic sort network for SortKey ────────────────────────────────
// Standard bitonic sort: correct for any power-of-2 padded array.
// Threads cooperate to sort RECORDS_PER_BLOCK items in shared memory.

__device__ void bitonic_sort_shared(
    SortKey* __restrict__ s_keys,
    int* __restrict__ s_indices,
    int n
) {
    int tid = threadIdx.x;

    // Bitonic sort: O(n * log^2(n)) comparators
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int i = tid; i < n; i += blockDim.x) {
                int partner = i ^ j;

                // Standard bitonic: compare (i, partner) with direction based on k
                if (partner > i && partner < n) {
                    bool ascending = ((i & k) == 0);
                    bool need_swap = ascending
                        ? (s_keys[i] > s_keys[partner])
                        : (s_keys[i] < s_keys[partner]);
                    if (need_swap) {
                        SortKey tk = s_keys[i];
                        s_keys[i] = s_keys[partner];
                        s_keys[partner] = tk;
                        int ti = s_indices[i];
                        s_indices[i] = s_indices[partner];
                        s_indices[partner] = ti;
                    }
                }
            }
        }
    }
    __syncthreads();
}

// ── Run generation kernel ──────────────────────────────────────────

__global__ void run_generation_kernel(
    const uint8_t* __restrict__ input,       // Raw input records (RECORD_SIZE each)
    uint8_t* __restrict__ output,            // Sorted output with OVC prepended
    uint32_t* __restrict__ ovc_array,        // OVC values per record (separate array)
    uint64_t total_records,
    // Sparse index output
    SparseEntry* __restrict__ sparse_entries,
    int* __restrict__ sparse_counts          // Per-block count
) {
    // Shared memory for sorting (must be power-of-2 for bitonic sort)
    // RECORDS_PER_BLOCK should be power of 2 (512)
    __shared__ SortKey s_keys[RECORDS_PER_BLOCK];
    __shared__ int s_indices[RECORDS_PER_BLOCK];

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    uint64_t block_start = (uint64_t)bid * RECORDS_PER_BLOCK;
    int block_records = min((uint64_t)RECORDS_PER_BLOCK, total_records - block_start);
    if (block_records <= 0) return;

    // ── Step 1: Load keys into shared memory ──
    for (int i = tid; i < block_records; i += blockDim.x) {
        uint64_t record_idx = block_start + i;
        const uint8_t* rec = input + record_idx * RECORD_SIZE;

        s_keys[i] = make_sort_key(rec);
        s_indices[i] = i;
    }

    // Pad unused slots with max values
    for (int i = block_records + tid; i < RECORDS_PER_BLOCK; i += blockDim.x) {
        s_keys[i].hi = UINT64_MAX;
        s_keys[i].lo = UINT16_MAX;
        s_indices[i] = i;
    }
    __syncthreads();

    // ── Step 2: Sort keys in shared memory ──
    // Must pass power-of-2 size (padded with max values above)
    bitonic_sort_shared(s_keys, s_indices, RECORDS_PER_BLOCK);

    // ── Step 3: Write sorted records to output ──
    for (int i = tid; i < block_records; i += blockDim.x) {
        uint64_t out_record_idx = block_start + i;
        int orig_idx = s_indices[i];

        // Source record in input
        const uint8_t* src_record = input + (block_start + orig_idx) * RECORD_SIZE;

        // Destination in output (sorted position)
        uint8_t* dst_record = output + out_record_idx * RECORD_SIZE;

        // Copy full record (key + value) to output in sorted order
        for (int b = 0; b < RECORD_SIZE; b++) {
            dst_record[b] = src_record[b];
        }
    }
    __syncthreads();

    // ── Step 4: Compute OVC deltas from sorted output ──
    // OVC compares consecutive keys in sorted order.
    // Thread i computes OVC for record i by reading sorted keys from output.
    // (Reading from global memory is fine here — keys are in L2 cache from the write above)

    int sparse_base = bid * ((RECORDS_PER_BLOCK + SPARSE_INDEX_STRIDE - 1) / SPARSE_INDEX_STRIDE);

    for (int i = tid; i < block_records; i += blockDim.x) {
        uint64_t out_record_idx = block_start + i;
        const uint8_t* curr_key = output + out_record_idx * RECORD_SIZE;

        uint32_t ovc;
        if (i == 0) {
            ovc = OVC_INITIAL;
        } else {
            const uint8_t* prev_key = output + (out_record_idx - 1) * RECORD_SIZE;
            ovc = ovc_compute_delta(prev_key, curr_key, KEY_SIZE);
        }
        ovc_array[out_record_idx] = ovc;

        // Sparse index sample
        if (i % SPARSE_INDEX_STRIDE == 0) {
            int sparse_idx = sparse_base + (i / SPARSE_INDEX_STRIDE);
            sparse_entries[sparse_idx].byte_offset = (uint64_t)i * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++) {
                sparse_entries[sparse_idx].key[b] = curr_key[b];
            }
        }
    }

    __syncthreads();
    if (tid == 0) {
        sparse_counts[bid] = (block_records + SPARSE_INDEX_STRIDE - 1) / SPARSE_INDEX_STRIDE;
    }
}

// ── Host interface ─────────────────────────────────────────────────

extern "C" void launch_run_generation(
    const uint8_t* d_input,
    uint8_t* d_output,
    uint32_t* d_ovc,
    uint64_t total_records,
    SparseEntry* d_sparse,
    int* d_sparse_counts,
    int num_blocks,
    cudaStream_t stream
) {
    run_generation_kernel<<<num_blocks, BLOCK_THREADS, 0, stream>>>(
        d_input, d_output, d_ovc, total_records,
        d_sparse, d_sparse_counts
    );
}
