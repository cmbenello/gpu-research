// CrocSort GPU Transfer: SM Cluster Replacement Selection on H100
// Maps from: src/replacement_selection/tol_mm_ovc.rs:ReplacementSelectionOVCMM
//            src/replacement_selection/memory.rs:MemoryManager (workspace management)
// GPU primitive: H100 SM clusters, distributed shared memory, cp.async.bulk

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

// ── H100 SM Cluster Configuration ──────────────────────────────────
// H100 clusters: 4 SMs share distributed shared memory
// Total smem per cluster: 4 * 228KB = 912KB
// Maps from: MemoryManager::with_limit (memory.rs:122)
// CrocSort's workspace_size determines run length via T*
// With 912KB: T* = 0.7 * 912KB * 1.41 / sqrt(80GB * 128B) = ~0.28

static constexpr int CLUSTER_SIZE = 4;           // SMs per cluster
static constexpr int SMEM_PER_SM = 228 * 1024;   // 228KB per SM
static constexpr int TOTAL_CLUSTER_SMEM = CLUSTER_SIZE * SMEM_PER_SM;  // 912KB

// Layout within cluster shared memory:
// SM0: Loser tree (OVC) + control state    (~32KB)
// SM0: Record buffer partition 0            (~196KB)
// SM1: Record buffer partition 1            (~228KB)
// SM2: Record buffer partition 2            (~228KB)
// SM3: Record buffer partition 3            (~228KB)
// Total record buffer: ~880KB

static constexpr int TREE_REGION_SIZE = 32 * 1024;
static constexpr int RECORD_BUF_SM0 = SMEM_PER_SM - TREE_REGION_SIZE;
static constexpr int RECORD_BUF_OTHER = SMEM_PER_SM;
static constexpr int TOTAL_RECORD_BUF = RECORD_BUF_SM0 + 3 * RECORD_BUF_OTHER;

// OVC Loser Tree node for replacement selection
// Maps from: OVCKeyValueMM (tol_mm_ovc.rs:19)
struct __align__(8) OVCTreeNodeCluster {
    uint32_t ovc;           // OVCU32 packed value
    uint16_t sm_id;         // Which SM's memory holds the record (0-3)
    uint16_t buf_offset_hi; // High 16 bits of buffer offset
    uint32_t buf_offset_lo; // Low 32 bits of buffer offset
    // Total: 12 bytes per node
};

// ── Cluster replacement selection kernel ───────────────────────────
// Launched with cudaLaunchKernelEx with cluster size = 4

__global__ void __cluster_dims__(4, 1, 1)
cluster_replacement_selection(
    const uint8_t* __restrict__ input_data,
    uint64_t input_size,
    uint8_t* __restrict__ output_runs,
    uint64_t* __restrict__ run_boundaries,
    int* __restrict__ num_runs_out,
    int record_size,
    int tree_capacity                        // Max records in loser tree
) {
    extern __shared__ uint8_t local_smem[];

    // Get cluster info
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int sm_rank = cluster.block_rank(); // 0-3 within cluster
    const int tid = threadIdx.x;

    // ── Distributed shared memory access ──
    // SM0 accesses its own smem directly
    // SM0 accesses SM1-3 smem via distributed shared memory (dsmem)
    // Cross-SM access: ~5 cycle latency vs ~1 cycle for local

    // Pointer to this SM's shared memory
    uint8_t* my_smem = local_smem;

    // Get pointers to other SMs' shared memory via cluster
    // cuda::dsmem::get_cluster_shared_memory(rank)
    // TODO: use actual CUDA dsmem API when available

    // SM0: manages the loser tree
    // Maps from: ReplacementSelectionOVCMM (tol_mm_ovc.rs:149)
    OVCTreeNodeCluster* tree = nullptr;
    int* next_run_count = nullptr;
    int* current_output_offset = nullptr;

    if (sm_rank == 0) {
        tree = reinterpret_cast<OVCTreeNodeCluster*>(my_smem);
        next_run_count = reinterpret_cast<int*>(my_smem + tree_capacity * sizeof(OVCTreeNodeCluster));
        current_output_offset = next_run_count + 1;
    }

    // Record buffer regions
    // SM0: records at offset TREE_REGION_SIZE
    // SM1-3: records at offset 0
    uint8_t* my_record_buf = nullptr;
    int my_buf_size = 0;

    if (sm_rank == 0) {
        my_record_buf = my_smem + TREE_REGION_SIZE;
        my_buf_size = RECORD_BUF_SM0;
    } else {
        my_record_buf = my_smem;
        my_buf_size = RECORD_BUF_OTHER;
    }

    // ── Phase 1: Initial fill ──
    // Maps from: ReplacementSelectionOVCMM::insert_initial + build()
    // All SMs cooperatively load records into their local buffers

    __shared__ int local_record_count;
    __shared__ int local_buf_used;

    if (tid == 0) {
        local_record_count = 0;
        local_buf_used = 0;
    }
    __syncthreads();

    // Calculate input range for this cluster
    uint64_t cluster_id = blockIdx.x / CLUSTER_SIZE; // Cluster index
    uint64_t records_per_cluster = (input_size / record_size + gridDim.x / CLUSTER_SIZE - 1) /
                                   (gridDim.x / CLUSTER_SIZE);
    uint64_t my_start = cluster_id * records_per_cluster * record_size;
    uint64_t my_end = min(my_start + records_per_cluster * record_size, input_size);

    // Each SM loads records into its local buffer
    int max_records_per_sm = my_buf_size / record_size;
    uint64_t sm_input_start = my_start + sm_rank * (records_per_cluster / CLUSTER_SIZE) * record_size;

    for (int i = tid; i < max_records_per_sm; i += blockDim.x) {
        uint64_t record_offset = sm_input_start + i * record_size;
        if (record_offset + record_size <= my_end) {
            // Load record into local smem buffer
            memcpy(my_record_buf + i * record_size,
                   input_data + record_offset,
                   record_size);
            atomicAdd(&local_record_count, 1);
        }
    }
    __syncthreads();

    // Cluster-wide barrier: all SMs have loaded their records
    cluster.sync();

    // ── Phase 2: Build loser tree on SM0 ──
    // Maps from: ReplacementSelectionOVCMM::build (tol_mm_ovc.rs:176)
    // SM0 reads key prefixes from all SMs' buffers and builds the tree

    if (sm_rank == 0 && tid == 0) {
        // For each record across all 4 SMs, compute OVC and insert into tree
        // Access SM1-3 records via distributed shared memory

        // TODO: Use dsmem API to read from other SMs' shared memory
        // For now, conceptual:
        // for each SM s in cluster:
        //   for each record r in SM[s].buffer:
        //     uint32_t key_prefix = read_from_dsmem(s, r.offset);
        //     ovc = compute_ovc(prev_key, key_prefix);
        //     tree[tree_size++] = {ovc, s, r.offset};
        //
        // build_tournament(tree, tree_size);
    }

    cluster.sync();

    // ── Phase 3: Replacement selection main loop ──
    // Maps from: ReplacementSelectionOVCMM::absorb_record_with (tol_mm_ovc.rs:196)
    // SM0 manages the tree; other SMs handle I/O

    int run_count = 0;
    uint64_t output_offset = 0;

    // SM0: tree management
    // SM1: input record loading (cp.async from HBM)
    // SM2: output record writing (to HBM)
    // SM3: sparse index construction (cycle 7)

    // Main loop: process remaining input records
    uint64_t remaining_start = my_start + CLUSTER_SIZE * max_records_per_sm * record_size;

    while (remaining_start < my_end) {
        if (sm_rank == 0 && tid == 0) {
            // Extract winner from tree
            OVCTreeNodeCluster winner = tree[0];

            // Read winner's record from its SM's buffer via dsmem
            // uint8_t* winner_record = dsmem_ptr(winner.sm_id) + winner.buf_offset;

            // Write winner to HBM output
            // memcpy(output_runs + output_offset, winner_record, record_size);
            output_offset += record_size;

            // Check if new record belongs to current run
            // Maps from: tol_mm_ovc.rs absorb logic
            // If new_key < winner_key: defer to next run (run_count++)

            // Replay tournament with new record
            // tree[0] = new_record;
            // replay_tournament(tree, tree_capacity);
        }

        cluster.sync(); // Sync across cluster after each record
        remaining_start += record_size;
    }

    // ── Phase 4: Drain remaining records ──
    if (sm_rank == 0 && tid == 0) {
        // Drain tree
        // Maps from: ReplacementSelectionOVCMM::drain_with

        *num_runs_out = run_count;
    }
}

// Host-side launch:
// cudaLaunchConfig_t config;
// config.gridDim = num_clusters * CLUSTER_SIZE;
// config.blockDim = 256;
// config.dynamicSmemBytes = SMEM_PER_SM;
//
// cudaLaunchAttribute attrs[1];
// attrs[0].id = cudaLaunchAttributeClusterDimension;
// attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
// config.numAttrs = 1;
// config.attrs = attrs;
//
// cudaLaunchKernelEx(&config, cluster_replacement_selection,
//     d_input, input_size, d_output, d_run_boundaries,
//     d_num_runs, record_size, tree_capacity);
//
// printf("Cluster SMEM: %d KB, expected E=%.1f, expected T*=%.2f\n",
//        TOTAL_CLUSTER_SMEM/1024, 2.0,
//        0.7 * TOTAL_CLUSTER_SMEM * 1.41 / sqrt(80e9 * 128.0));
