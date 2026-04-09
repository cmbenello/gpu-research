// CrocSort GPU Transfer: SM-Partitioned T_gen/T_merge Decoupling
// Maps from: SorterCore::run_generation_internal (engine.rs:292)
//            and SorterCore::multi_merge_internal (engine.rs:309)
// CrocSort Eq.5: T_gen * T_merge <= E * rho^2 * M^2 / (D * P)
// GPU mapping: M = shared_mem_per_SM, D = total_HBM_data,
//              P = HBM_cache_line (128B), T_gen = runs_per_SM, T_merge = fanin

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

// Configuration derived from CrocSort analytical model
struct GpuSortConfig {
    int sm_count_run_gen;   // SMs allocated to run generation
    int sm_count_merge;     // SMs allocated to merge phase
    int smem_per_sm;        // Shared memory budget per SM (bytes)
    int merge_fanin;        // Merge fan-in per SM
    int hbm_data_bytes;     // Total data size in HBM
    float rho;              // Memory utilization factor (CrocSort: 0.8)
    float E;                // Expansion factor (replacement selection: 2.0)
};

// Loser tree node in shared memory — maps from LoserTree<T> in tree_of_losers.rs
struct __align__(8) SmemLoserNode {
    uint32_t key_prefix;    // First 4 bytes of key for fast comparison
    uint16_t source_idx;    // Which input stream this came from
    uint16_t flags;         // Sentinel flags (maps to SentinelValue trait)
};

// Per-SM run generation kernel
// Each SM independently generates sorted runs using replacement selection
// Maps from: ReplacementSelectionMM in tol_mm.rs
__global__ void run_generation_kernel(
    const uint8_t* __restrict__ input_data,
    uint64_t* __restrict__ run_offsets,     // Output: byte offsets of run boundaries
    uint8_t* __restrict__ output_runs,      // Output: sorted run data in HBM
    const int num_records,
    const int record_size,
    const int smem_tree_capacity            // Capacity of loser tree in shared memory
) {
    // Shared memory layout:
    // [0..tree_bytes): Loser tree nodes
    // [tree_bytes..smem_limit): Record buffer (replacement selection workspace)
    extern __shared__ uint8_t smem[];

    cg::grid_group grid = cg::this_grid();
    const int sm_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    SmemLoserNode* tree = reinterpret_cast<SmemLoserNode*>(smem);
    const int tree_bytes = smem_tree_capacity * sizeof(SmemLoserNode);
    uint8_t* record_buffer = smem + tree_bytes;

    // Phase 1: Each SM claims a chunk of input data (round-robin by SM)
    // Maps CrocSort's create_parallel_scanners (lib.rs:314)
    const int records_per_sm = (num_records + gridDim.x - 1) / gridDim.x;
    const int my_start = sm_id * records_per_sm;
    const int my_end = min(my_start + records_per_sm, num_records);

    // Phase 2: Initial fill of loser tree (replacement selection)
    // Maps: ReplacementSelectionMM::insert_initial + build() (tol_mm.rs:162-173)
    int tree_size = 0;
    int buffer_offset = 0;
    const int max_buffer_records = (smem_tree_capacity * 2); // rho * smem budget

    for (int i = my_start + tid; i < my_start + max_buffer_records && i < my_end; i += block_size) {
        // Load record key prefix into shared memory
        int local_idx = i - my_start;
        if (local_idx < smem_tree_capacity) {
            uint32_t key_prefix = *reinterpret_cast<const uint32_t*>(
                input_data + i * record_size);
            tree[local_idx].key_prefix = key_prefix;
            tree[local_idx].source_idx = local_idx;
            tree[local_idx].flags = 0; // NormalValue
        }
    }
    __syncthreads();

    // Phase 3: Build loser tree tournament
    // Maps: LoserTree::reset_from_iter (tree_of_losers.rs:46)
    // TODO: Parallel tree construction — each thread handles one internal node
    if (tid == 0) {
        // Sequential build for correctness — parallelize later
        for (int i = smem_tree_capacity - 1; i >= 1; i--) {
            int left = 2 * i;
            int right = 2 * i + 1;
            if (left < smem_tree_capacity && right < smem_tree_capacity) {
                if (tree[left].key_prefix <= tree[right].key_prefix) {
                    tree[i] = tree[right];  // Loser stays, winner bubbles up
                } else {
                    SmemLoserNode tmp = tree[i];
                    tree[i] = tree[left];
                    // Winner (right) continues up
                }
            }
        }
    }
    __syncthreads();

    // Phase 4: Replacement selection main loop
    // Maps: ReplacementSelectionMM::absorb_record_with (tol_mm.rs:190)
    // Each thread cooperatively processes the output stream
    int run_count = 0;
    int output_offset = sm_id * records_per_sm * record_size;

    for (int input_idx = my_start + max_buffer_records; input_idx < my_end; input_idx++) {
        // Thread 0 extracts winner and replays tournament
        if (tid == 0) {
            // Extract winner (tree[0])
            SmemLoserNode winner = tree[0];

            // Write winner to HBM output
            // TODO: Use cp.async for async HBM writes
            *reinterpret_cast<uint32_t*>(output_runs + output_offset) = winner.key_prefix;
            output_offset += record_size;

            // Load new record from HBM input
            uint32_t new_key = *reinterpret_cast<const uint32_t*>(
                input_data + input_idx * record_size);

            // Check if new record belongs to current run
            // Maps: should_defer_to_next_run in heap.rs:123
            if (new_key < winner.key_prefix) {
                // Defer to next run — mark with LateFence
                tree[0].flags = 4; // LateFence
                // TODO: Track next_run_buffer count
                run_count++;
            }

            // Replay tournament (push new value)
            // Maps: LoserTree::push (tree_of_losers.rs:98)
            tree[0].key_prefix = new_key;
            int slot = (smem_tree_capacity + winner.source_idx) / 2;
            while (slot >= 1) {
                if (tree[0].key_prefix > tree[slot].key_prefix) {
                    SmemLoserNode tmp = tree[0];
                    tree[0] = tree[slot];
                    tree[slot] = tmp;
                }
                slot /= 2;
            }
        }
        __syncthreads();
    }

    // Record run boundary offsets
    if (tid == 0) {
        run_offsets[sm_id] = run_count;
    }
}

// Host-side launch sketch:
// GpuSortConfig config = compute_gpu_config(data_size, device_props);
// int smem_bytes = config.smem_per_sm;
// int tree_cap = smem_bytes / (2 * sizeof(SmemLoserNode));
// run_generation_kernel<<<config.sm_count_run_gen, 256, smem_bytes>>>(
//     d_input, d_run_offsets, d_output_runs,
//     num_records, record_size, tree_cap);
// cudaDeviceSynchronize();
// // Then launch merge kernel with config.sm_count_merge SMs
