// CrocSort GPU Transfer: Multi-GPU NVLink Sort with M/D Transfer Property
// Maps from: src/sort/core/engine.rs:SorterCore (config space)
//            Cloud paper: M/D ratio governs performance surface; top configs transfer across scales
// GPU primitive: NCCL AllToAll, NVLink inter-GPU communication

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdint>
#include <vector>

// ── Multi-GPU CrocSort Architecture ────────────────────────────────
//
// CrocSort transfer property: optimal config found at scale S transfers
// to scale N*S within 5% performance. On GPU:
//   - 1 GPU: M = SMEM (228KB), D = HBM (80GB), M/D = 0.0003%
//   - N GPUs: M = N*SMEM, D = N*HBM, M/D = 0.0003% (INVARIANT!)
//
// This means optimal per-GPU config (block size, smem allocation,
// merge fanin) found on 1 GPU transfers directly to N GPUs.
//
// Architecture:
// Phase 1: Per-GPU run generation (independent, no communication)
// Phase 2: Global sample + boundary computation (small AllReduce)
// Phase 3: Data redistribution (AllToAll via NVLink)
// Phase 4: Per-GPU merge of redistributed data (independent)

struct MultiGpuSortConfig {
    int num_gpus;
    int smem_per_sm;          // Per-GPU shared memory budget
    int merge_fanin;          // Per-GPU merge fan-in
    int run_gen_sms;          // SMs for run generation per GPU
    int merge_sms;            // SMs for merge per GPU
    float imbalance_factor;   // CrocSort's partition imbalance factor
};

// Validate transfer property: config found on 1 GPU should work for N GPUs
bool validate_transfer_property(
    const MultiGpuSortConfig& config_1gpu,
    const MultiGpuSortConfig& config_ngpu
) {
    // M/D ratio should be invariant
    float md_ratio_1 = (float)config_1gpu.smem_per_sm /
                        (80.0f * 1024 * 1024 * 1024); // 80GB
    float md_ratio_n = (float)config_ngpu.smem_per_sm /
                        (80.0f * 1024 * 1024 * 1024); // Same per GPU
    return fabsf(md_ratio_1 - md_ratio_n) < 0.001f;
}

// ── Phase 1: Per-GPU Run Generation ────────────────────────────────
// Each GPU independently runs replacement selection on its data shard
// Maps from: execute_run_generation in engine.rs

__global__ void per_gpu_run_generation(
    const uint8_t* __restrict__ local_data,
    uint64_t local_data_size,
    uint8_t* __restrict__ local_runs,
    uint64_t* __restrict__ run_offsets,
    int* __restrict__ num_runs,
    int smem_budget
) {
    extern __shared__ uint8_t smem[];
    // Run replacement selection kernel (cycle 1/10 implementations)
    // Each GPU processes 1/N of the total data
    // TODO: call replacement_selection_with_smem_alloc (cycle 10)
}

// ── Phase 2: Global Sample Exchange ────────────────────────────────
// Exchange sparse index samples across GPUs to compute global boundaries
// Maps from: MultiSparseIndexes + boundary search

struct GpuSampleExchange {
    // Each GPU contributes its sparse index samples
    // Exchanged via NCCL AllGather (small: ~1MB per GPU)

    static void exchange_samples(
        ncclComm_t comm,
        cudaStream_t stream,
        const uint8_t* local_samples,     // This GPU's sparse index entries
        int local_sample_count,
        uint8_t* global_samples,           // All GPUs' samples concatenated
        int* global_sample_counts,
        int num_gpus
    ) {
        // Gather sample counts
        ncclAllGather(
            &local_sample_count,
            global_sample_counts,
            1,
            ncclInt,
            comm,
            stream
        );

        // Gather actual samples
        // Each sample is (key_prefix: 4B, file_offset: 8B, gpu_id: 2B) = 14B
        int max_samples_per_gpu = 100000;  // Budget per GPU
        ncclAllGather(
            local_samples,
            global_samples,
            max_samples_per_gpu * 14,  // bytes per GPU
            ncclChar,
            comm,
            stream
        );
    }
};

// ── Phase 3: Data Redistribution via NVLink ────────────────────────
// Send each record to the GPU responsible for its key range
// Maps from: CrocSort's merge partitioning, but across GPUs

__global__ void compute_send_destinations(
    const uint8_t* __restrict__ local_run_data,
    int num_records,
    int record_size,
    const uint32_t* __restrict__ boundary_key_prefixes,  // N-1 boundaries
    int num_gpus,
    int* __restrict__ send_counts,     // [num_gpus] counts per destination
    int* __restrict__ send_offsets     // Per-record destination GPU
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_records) return;

    // Read key prefix for this record
    uint32_t key_prefix = *reinterpret_cast<const uint32_t*>(
        local_run_data + tid * record_size);

    // Binary search for destination GPU
    // Maps from: byte-balanced boundary search (cycle 7)
    int dest_gpu = 0;
    int lo = 0, hi = num_gpus - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (key_prefix >= boundary_key_prefixes[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    dest_gpu = lo;

    send_offsets[tid] = dest_gpu;
    atomicAdd(&send_counts[dest_gpu], 1);
}

// Host-side AllToAll redistribution
void redistribute_data(
    ncclComm_t comm,
    cudaStream_t stream,
    uint8_t** gpu_send_buffers,    // Per-destination send buffer on each GPU
    uint8_t** gpu_recv_buffers,    // Per-source receive buffer on each GPU
    int* send_sizes,               // Bytes to send to each GPU
    int* recv_sizes,               // Bytes to receive from each GPU
    int num_gpus
) {
    // Exchange sizes first
    ncclGroupStart();
    for (int i = 0; i < num_gpus; i++) {
        ncclSend(gpu_send_buffers[i], send_sizes[i], ncclChar, i, comm, stream);
        ncclRecv(gpu_recv_buffers[i], recv_sizes[i], ncclChar, i, comm, stream);
    }
    ncclGroupEnd();
}

// ── Phase 4: Per-GPU Final Merge ───────────────────────────────────
// Each GPU merges its received data (from all GPUs) into final sorted output
// Maps from: multi_merge_with_hooks in engine.rs

__global__ void per_gpu_final_merge(
    const uint8_t** __restrict__ received_runs,   // Received from N GPUs
    const uint64_t* __restrict__ run_sizes,
    uint8_t* __restrict__ sorted_output,
    int num_runs,                                  // = num_gpus (one run per source GPU)
    int record_size
) {
    // Use OVC merge kernel from cycle 4/5
    // Each GPU's received data is already sorted (came from sorted runs)
    // So this is a K-way merge where K = num_gpus
    // TODO: call ovc_merge_kernel or ovc_selective_fetch_merge
}

// ── Host orchestration ─────────────────────────────────────────────

// void multi_gpu_crocsort(
//     int num_gpus,
//     uint8_t** d_data,          // Per-GPU data pointers
//     uint64_t* data_sizes,      // Per-GPU data sizes
//     MultiGpuSortConfig config
// ) {
//     // Validate transfer property
//     assert(validate_transfer_property(config, config));
//
//     ncclComm_t comm;
//     ncclCommInitAll(&comm, num_gpus, ...);
//
//     // Phase 1: Per-GPU run generation (embarrassingly parallel)
//     for (int g = 0; g < num_gpus; g++) {
//         cudaSetDevice(g);
//         per_gpu_run_generation<<<grid, block, smem, streams[g]>>>(
//             d_data[g], data_sizes[g], d_runs[g], ...);
//     }
//
//     // Phase 2: Exchange sparse index samples
//     GpuSampleExchange::exchange_samples(comm, stream, ...);
//
//     // Compute global boundaries on GPU 0
//     // byte_balanced_boundary_search<<<...>>>(global_samples, ...);
//
//     // Phase 3: AllToAll redistribution via NVLink
//     redistribute_data(comm, stream, send_bufs, recv_bufs, ...);
//
//     // Phase 4: Per-GPU final merge
//     for (int g = 0; g < num_gpus; g++) {
//         cudaSetDevice(g);
//         per_gpu_final_merge<<<grid, block, smem, streams[g]>>>(
//             d_received[g], ...);
//     }
//
//     ncclCommDestroy(comm);
// }
