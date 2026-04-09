// CrocSort GPU Transfer: Zero-Copy Merge with DMA Value Streaming
// Maps from: src/sort/ovc/merge.rs:ZeroCopyMergeWithOVC (line 79)
//            Keys in loser tree, values flow source->output via copy_value_to
// GPU primitive: Concurrent SM compute + copy engine DMA for value transfer

#include <cuda_runtime.h>
#include <cstdint>

// ── Zero-copy merge architecture ───────────────────────────────────
// CrocSort's ZeroCopyMergeWithOVC (merge.rs:79-210):
//   - Keys live in the loser tree (shared memory / registers)
//   - Values are NEVER buffered
//   - Values flow directly: source AlignedReader -> output AlignedWriter
//   - copy_value_to(writer, len) does the direct transfer
//
// GPU mapping:
//   - Keys: in registers (OVC) or shared memory (full key on tie-break)
//   - Values: transferred HBM-to-HBM via copy engine (separate from SM)
//   - SM handles key comparison, copy engine handles value movement
//   - Two CUDA streams: compute_stream (SM) and copy_stream (DMA)

// Value transfer descriptor: batched for efficiency
struct ValueTransfer {
    uint64_t src_offset;    // Source HBM address offset
    uint64_t dst_offset;    // Destination HBM address offset
    uint32_t length;        // Value length in bytes
    uint32_t padding;       // Alignment padding
};

// Merge state per thread block (one partition)
struct MergePartitionState {
    // OVC loser tree (register-resident for K<=16)
    uint32_t ovc_tree[16];
    uint16_t src_tree[16];

    // Per-source stream positions
    uint64_t src_key_offset[16];   // Current key read position per source
    uint64_t src_val_offset[16];   // Current value read position per source
    uint64_t src_end[16];          // End of each source's range

    // Output position
    uint64_t out_key_offset;       // Where to write next key
    uint64_t out_val_offset;       // Where to write next value (may differ due to batching)

    // Value transfer batch
    ValueTransfer batch[64];       // Batch up to 64 value transfers
    int batch_count;
};

// ── Key-only merge kernel (runs on SMs) ────────────────────────────
// This kernel ONLY processes keys. Values are handled by DMA.
// Maps from: ZeroCopyMergeWithOVC::merge_into key path

__global__ void key_merge_kernel(
    // Input sources (key data only — values at separate offsets)
    const uint8_t** __restrict__ src_key_bases,
    const uint8_t** __restrict__ src_val_bases,
    const uint64_t* __restrict__ partition_key_starts,
    const uint64_t* __restrict__ partition_key_ends,
    // Output
    uint8_t* __restrict__ out_keys,
    // Value transfer queue (consumed by DMA kernel/stream)
    ValueTransfer* __restrict__ value_queue,
    uint32_t* __restrict__ queue_head,  // Atomic producer index
    // Config
    const int K,
    const int key_stride,     // Fixed key record stride
    const int val_stride      // Fixed value record stride
) {
    const int partition_id = blockIdx.x;
    const int tid = threadIdx.x;

    // Shared memory for OVC tie-break key comparison
    extern __shared__ uint8_t smem[];

    // Register-resident OVC tree
    uint32_t ovc[16];
    uint16_t src[16];
    uint64_t src_off[16];
    uint64_t src_end_local[16];
    uint64_t val_off[16];

    // Initialize
    for (int i = 0; i < K; i++) {
        src_off[i] = partition_key_starts[partition_id * K + i];
        src_end_local[i] = partition_key_ends[partition_id * K + i];
        val_off[i] = partition_key_starts[partition_id * K + i]; // TODO: separate val offsets

        if (src_off[i] < src_end_local[i]) {
            ovc[i] = *reinterpret_cast<const uint32_t*>(
                src_key_bases[i] + src_off[i]);
            src[i] = i;
        } else {
            ovc[i] = 0xFFFFFFFF; // LateFence
            src[i] = i;
        }
    }

    // TODO: build tournament

    uint64_t out_key_off = 0;
    int batch_count = 0;
    const int BATCH_SIZE = 32; // Batch 32 value transfers before flushing

    // ── Merge loop: keys only ──
    while (ovc[0] != 0xFFFFFFFF) { // while not LateFence
        int winner_src = src[0];

        // 1. Write winner's OVC + key to output key buffer
        // Maps from: output.append_header_and_key(ovc, key, value_len)
        // in merge.rs:173
        if (tid == 0) {
            *reinterpret_cast<uint32_t*>(out_keys + out_key_off) = ovc[0];
            out_key_off += 4;

            // Copy key data (fixed stride for simplicity)
            memcpy(out_keys + out_key_off,
                   src_key_bases[winner_src] + src_off[winner_src] + 4,
                   key_stride - 4);
            out_key_off += key_stride - 4;
        }

        // 2. Queue value transfer for DMA (NOT copied by SM!)
        // Maps from: source.copy_value_to(output.writer(), value_len)
        // in merge.rs:176
        // Instead of SM doing the copy, we queue it for the DMA engine
        if (tid == 0) {
            // Add to local batch
            // In production: use shared memory batch buffer
            uint32_t slot = atomicAdd(queue_head, 1);
            value_queue[slot].src_offset = (uint64_t)(src_val_bases[winner_src] + val_off[winner_src]);
            value_queue[slot].dst_offset = 0; // TODO: compute output value offset
            value_queue[slot].length = val_stride;
            value_queue[slot].padding = 0;

            val_off[winner_src] += val_stride;
        }

        // 3. Advance: read next OVC from winner's source
        src_off[winner_src] += key_stride;

        if (src_off[winner_src] < src_end_local[winner_src]) {
            uint32_t next_ovc = *reinterpret_cast<const uint32_t*>(
                src_key_bases[winner_src] + src_off[winner_src]);

            // Push to OVC tree (register operations only — no HBM access!)
            uint32_t cand_ovc = next_ovc;
            uint16_t cand_src = (uint16_t)winner_src;
            int slot = (K + winner_src) / 2;

            #pragma unroll
            for (int l = 0; l < 4; l++) {
                if (slot >= 1 && slot < K) {
                    if (cand_ovc > ovc[slot]) {
                        uint32_t t = cand_ovc; cand_ovc = ovc[slot]; ovc[slot] = t;
                        uint16_t s = cand_src; cand_src = src[slot]; src[slot] = s;
                    }
                    slot /= 2;
                }
            }
            ovc[0] = cand_ovc;
            src[0] = cand_src;
        } else {
            // Source exhausted
            uint32_t cand_ovc = 0xFFFFFFFF;
            uint16_t cand_src = (uint16_t)winner_src;
            int slot = (K + winner_src) / 2;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                if (slot >= 1 && slot < K) {
                    if (cand_ovc > ovc[slot]) {
                        uint32_t t = cand_ovc; cand_ovc = ovc[slot]; ovc[slot] = t;
                        uint16_t s = cand_src; cand_src = src[slot]; src[slot] = s;
                    }
                    slot /= 2;
                }
            }
            ovc[0] = cand_ovc;
            src[0] = cand_src;
        }
    }
}

// ── Host-side: Launch with concurrent compute + copy streams ──────

// cudaStream_t compute_stream, copy_stream;
// cudaStreamCreate(&compute_stream);
// cudaStreamCreate(&copy_stream);
//
// // Launch key merge on compute stream (uses SMs)
// key_merge_kernel<<<num_partitions, 32, smem_bytes, compute_stream>>>(
//     d_src_key_bases, d_src_val_bases,
//     d_partition_key_starts, d_partition_key_ends,
//     d_out_keys, d_value_queue, d_queue_head,
//     K, key_stride, val_stride);
//
// // Process value transfers on copy stream (uses DMA engine)
// // Option A: Host-driven batched cudaMemcpyAsync
// while (queue not empty) {
//     // Read batch from value_queue
//     // Issue cudaMemcpyAsync per transfer on copy_stream
//     cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, copy_stream);
// }
//
// // Option B: Device-side copy kernel on copy_stream
// // value_copy_kernel<<<grid, block, 0, copy_stream>>>(d_value_queue, ...);
//
// cudaStreamSynchronize(compute_stream);
// cudaStreamSynchronize(copy_stream);
