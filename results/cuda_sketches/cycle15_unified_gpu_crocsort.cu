// CrocSort GPU Transfer: Unified End-to-End GPU CrocSort Architecture
// Maps from: src/sort/ovc/sorter.rs:ExternalSorterWithOVC — full pipeline
//            Pipeline: run-gen -> sparse index -> byte-balanced partition -> OVC merge
// Combines: cycles 1,2,4,5,7,10,12,14 into one system

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

// ══════════════════════════════════════════════════════════════════════
// GPU CrocSort Configuration (derived from analytical model)
// Maps from: SorterCore (engine.rs:218) + Eq.5 feasibility constraint
// ══════════════════════════════════════════════════════════════════════

struct GpuCrocSortConfig {
    // Hardware parameters
    int num_sms;                // Total SMs on device
    int smem_per_sm;            // Shared memory per SM (bytes)
    int l2_cache_size;          // L2 cache size (bytes)
    uint64_t hbm_size;          // Total HBM (bytes)
    float hbm_bandwidth;        // HBM bandwidth (bytes/sec)

    // Derived from Eq.5: T_gen * T_merge <= E * rho^2 * M^2 / (D * P)
    int sms_for_run_gen;        // SMs allocated to run generation
    int sms_for_merge;          // SMs allocated to merge
    int merge_fanin;            // Merge fan-in per SM
    int merge_threads;          // Parallel merge partitions

    // CrocSort parameters
    float rho;                  // Memory utilization factor (0.7 for GPU)
    float E;                    // Expansion factor (2.0 for random)
    int sparse_index_stride;    // Sampling stride for sparse index
    float index_budget_pct;     // L2 budget for sparse indexes (5%)

    // Compute optimal SM split from Eq.5
    void compute_optimal_split(uint64_t data_size) {
        // T* = rho * M * sqrt(E) / sqrt(D * P)
        // where M = smem_per_sm, D = data_size, P = 128 (cache line)
        double T_star_per_sm = rho * smem_per_sm * sqrt(E) / sqrt((double)data_size * 128.0);

        // We need T_gen = num_sms_runGen * T_star_per_sm to be sufficient
        // CrocSort's T_gen > T_merge optimal -> allocate more SMs to run-gen
        sms_for_run_gen = (int)(num_sms * 0.6);  // 60% for run-gen (compute-heavy)
        sms_for_merge = num_sms - sms_for_run_gen; // 40% for merge (bandwidth-heavy)

        // Merge fanin: limited by L2 cache (each source needs a read buffer)
        int read_buf_size = 64 * 1024; // 64KB per source buffer
        merge_fanin = l2_cache_size / (sms_for_merge * read_buf_size);
        merge_fanin = max(2, min(merge_fanin, 256));

        merge_threads = sms_for_merge;
        sparse_index_stride = 1000;
        index_budget_pct = 0.05f;
    }
};

// ══════════════════════════════════════════════════════════════════════
// Phase 1: Run Generation with OVC + Sparse Index
// Combines: cycle 10 (smem allocator) + cycle 14 (SM cluster)
// Maps from: ExternalSorterWithOVC::run_generation (sorter.rs)
// ══════════════════════════════════════════════════════════════════════

struct RunGenOutput {
    uint8_t* run_data;          // Sorted run data in HBM
    uint64_t run_size;          // Bytes written
    uint32_t* sparse_pos;       // Sparse index handles
    uint8_t* sparse_pages;      // Sparse index pages
    int sparse_entry_count;
    int num_runs;                // Run count from this SM
};

__global__ void gpu_crocsort_run_generation(
    const uint8_t* __restrict__ input_data,
    uint64_t input_size,
    RunGenOutput* __restrict__ outputs,  // Per-SM output
    int record_size,
    int tree_capacity,
    int sparse_stride
) {
    extern __shared__ uint8_t smem[];
    const int sm_id = blockIdx.x;
    const int tid = threadIdx.x;

    // Shared memory layout (cycle 10 allocator):
    // [0..32KB): OVC loser tree
    // [32KB..228KB): Record buffer (managed by best-fit allocator)

    // Phase 1a: Load initial records, build OVC loser tree
    // Uses: ReplacementSelectionOVCMM pattern from tol_mm_ovc.rs

    // Phase 1b: Replacement selection main loop
    // For each input record:
    //   - Compare OVC (register operation, cycle 4)
    //   - Emit winner to HBM output
    //   - Sample sparse index entry every sparse_stride records (cycle 7)

    // Phase 1c: Compute OVC deltas for output stream
    // Uses: warp-level parallel derive_ovc_from (cycle 4 idea 5)

    // TODO: Full implementation combining cycle 10 + 14 kernels
}

// ══════════════════════════════════════════════════════════════════════
// Phase 2: Byte-Balanced Boundary Selection
// Combines: cycle 2 (L2 persistence) + cycle 7 (smem boundary search)
// Maps from: select_boundary_by_size in run_format.rs
// ══════════════════════════════════════════════════════════════════════

__global__ void gpu_crocsort_boundary_selection(
    // Sparse indexes from all runs (pinned in L2 via cycle 2)
    const uint8_t** __restrict__ sparse_pages_per_run,
    const uint32_t** __restrict__ sparse_handles_per_run,
    const int* __restrict__ entries_per_run,
    const uint64_t* __restrict__ bytes_per_run,
    int num_runs,
    int num_partitions,
    // Output: boundaries for each partition
    uint8_t** __restrict__ boundary_keys,
    uint64_t* __restrict__ boundary_offsets,
    int* __restrict__ boundary_run_ids
) {
    // Each thread block computes one boundary
    // Uses: byte_balanced_boundary_search from cycle 7

    // Double binary search:
    // 1. Pick pivot from largest active range (by bytes)
    // 2. Parallel binary search across all runs (one thread per run)
    // 3. Compute byte objective
    // 4. Update lo/hi
    // 5. Stall detection + candidate expansion (from CrocSort doc)

    // TODO: Call cycle 7 kernel
}

// ══════════════════════════════════════════════════════════════════════
// Phase 3: OVC-Enhanced Zero-Copy Merge
// Combines: cycle 4 (register OVC) + cycle 5 (selective fetch) + cycle 12 (DMA)
// Maps from: ZeroCopyMergeWithOVC::merge_into (merge.rs:163)
// ══════════════════════════════════════════════════════════════════════

__global__ void gpu_crocsort_ovc_merge(
    // Input runs (OVC-encoded)
    const uint8_t** __restrict__ run_bases,
    const uint64_t* __restrict__ partition_starts,
    const uint64_t* __restrict__ partition_ends,
    // Output
    uint8_t* __restrict__ output,
    uint64_t* __restrict__ output_size,
    // Value transfer queue (for DMA engine, cycle 12)
    // ValueTransfer* __restrict__ value_queue,
    // Config
    int K,           // Merge fan-in
    int key_size,
    int value_size
) {
    const int partition_id = blockIdx.x;
    const int tid = threadIdx.x;

    // Register-resident OVC loser tree (cycle 4)
    uint32_t ovc[16];
    uint16_t src[16];

    // Statistics (matching CrocSort's SortStats)
    uint64_t records_merged = 0;
    uint64_t ovc_comparisons = 0;    // Resolved by OVC alone
    uint64_t full_comparisons = 0;   // Required full key comparison
    uint64_t duplicate_shortcuts = 0; // OVC duplicate fast path

    // Initialize tree from partition boundaries
    // TODO: init from boundary keys

    uint64_t out_off = 0;

    // ── Main merge loop ──
    while (ovc[0] < 0x80000000) { // while not LateFence (flag < 4)
        int winner = src[0];

        // Check for duplicate fast path (cycle 4 idea 2)
        // Maps from: merge.rs:42-47 duplicate OVC handling
        // if next_ovc.is_duplicate_value(): replace_top_ovc, skip tree push

        // OVC comparison (cycle 4): single 32-bit register compare
        // Only on tie: selective key fetch from HBM (cycle 5)
        // Value streaming: queue for DMA engine (cycle 12)

        records_merged++;

        // Advance winner source, push to tree
        // All operations are register-only when OVC is decisive

        // TODO: full implementation combining cycles 4, 5, 12
        break; // placeholder
    }

    // Report statistics (matching CrocSort SortStats)
    if (tid == 0) {
        // Maps from: lib.rs SortStats display
        // printf("Partition %d: %lu records, OVC hit rate: %.1f%%, "
        //        "duplicates: %lu\n",
        //        partition_id, records_merged,
        //        100.0 * ovc_comparisons / max(ovc_comparisons + full_comparisons, 1ul),
        //        duplicate_shortcuts);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Host-side orchestration
// Maps from: SorterCore::sort() in engine.rs
// ══════════════════════════════════════════════════════════════════════

// void gpu_crocsort_sort(
//     uint8_t* d_input,
//     uint64_t input_size,
//     uint8_t* d_output,
//     int record_size,
//     int key_size
// ) {
//     // ── Configure from analytical model (Eq.5) ──
//     GpuCrocSortConfig config;
//     cudaDeviceProp props;
//     cudaGetDeviceProperties(&props, 0);
//     config.num_sms = props.multiProcessorCount;
//     config.smem_per_sm = props.sharedMemPerMultiprocessor;
//     config.l2_cache_size = props.l2CacheSize;
//     config.hbm_size = props.totalGlobalMem;
//     config.rho = 0.7f;
//     config.E = 2.0f;
//     config.compute_optimal_split(input_size);
//
//     printf("GPU CrocSort Config: %d SMs for run-gen, %d for merge, "
//            "fanin=%d, T*=%.3f\n",
//            config.sms_for_run_gen, config.sms_for_merge,
//            config.merge_fanin,
//            config.rho * config.smem_per_sm * sqrtf(config.E) /
//            sqrtf((float)input_size * 128.0f));
//
//     // ── Phase 1: Run Generation ──
//     RunGenOutput* d_run_outputs;
//     cudaMalloc(&d_run_outputs, config.sms_for_run_gen * sizeof(RunGenOutput));
//     int tree_cap = config.smem_per_sm / (2 * 12); // 12 bytes per tree node
//
//     gpu_crocsort_run_generation<<<config.sms_for_run_gen, 256,
//                                   config.smem_per_sm>>>(
//         d_input, input_size, d_run_outputs, record_size, tree_cap,
//         config.sparse_index_stride);
//
//     // ── Configure L2 persistence for sparse indexes (cycle 2) ──
//     // size_t l2_budget = config.l2_cache_size * config.index_budget_pct;
//     // configure_l2_persistence(sparse_index_memory, l2_budget);
//
//     // ── Phase 2: Byte-Balanced Boundary Selection ──
//     gpu_crocsort_boundary_selection<<<config.merge_threads, 256>>>(
//         /* sparse index data */, config.merge_threads,
//         /* boundary outputs */);
//
//     // ── Phase 3: OVC Merge ──
//     gpu_crocsort_ovc_merge<<<config.merge_threads, 32>>>(
//         /* run data */, /* boundaries */, d_output, /* sizes */,
//         config.merge_fanin, key_size, record_size - key_size);
//
//     // ── Report results (matching CrocSort SortStats format) ──
//     // printf("GPU CrocSort complete. Runs: %d, Merge levels: %d, "
//     //        "Read amplification: %.2fx\n", ...);
// }
