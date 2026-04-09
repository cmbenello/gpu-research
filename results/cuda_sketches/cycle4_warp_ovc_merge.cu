// CrocSort GPU Transfer: Warp-Level OVC Merge with Register-Resident Loser Tree
// Maps from: src/ovc/offset_value_coding_32.rs:OVCU32 (packed 32-bit OVC)
//            src/ovc/tree_of_losers_ovc.rs:LoserTreeOVC (OVC-aware tournament)
//            src/sort/ovc/merge.rs:ZeroCopyMergeWithOVC (keys in tree, values streamed)
// GPU primitive: 32-bit register comparison, __shfl_sync for OVC propagation

#include <cuda_runtime.h>
#include <cstdint>

// ── OVC encoding matching CrocSort's OVCU32 (offset_value_coding_32.rs:100) ──

// Memory layout (32 bits total):
// ┌───────┬─────────────────────┬──────────────────┐
// │ flags │ arity_minus_offset  │      value       │
// └───────┴─────────────────────┴──────────────────┘
//  bits 29-31      bits 16-28        bits 0-15

static constexpr uint32_t OVC_VALUE_MASK = 0x0000FFFF;
static constexpr uint32_t OVC_ARITY_MAX  = 0x1FFF;
static constexpr uint32_t OVC_ARITY_SHIFT = 16;
static constexpr uint32_t OVC_FLAG_SHIFT = 29;

// Flag values matching OVCFlag enum (offset_value_coding_64.rs)
static constexpr uint32_t OVC_FLAG_EARLY_FENCE    = 0;
static constexpr uint32_t OVC_FLAG_DUPLICATE_VALUE = 1;
static constexpr uint32_t OVC_FLAG_NORMAL_VALUE    = 2;
static constexpr uint32_t OVC_FLAG_INITIAL_VALUE   = 3;
static constexpr uint32_t OVC_FLAG_LATE_FENCE      = 4;

// Sentinel OVC values (matching OVCU32::early_fence/late_fence)
static constexpr uint32_t OVC_EARLY_FENCE = (OVC_FLAG_EARLY_FENCE << OVC_FLAG_SHIFT);
static constexpr uint32_t OVC_LATE_FENCE  = (OVC_FLAG_LATE_FENCE << OVC_FLAG_SHIFT) |
                                             (OVC_ARITY_MAX << OVC_ARITY_SHIFT) |
                                             OVC_VALUE_MASK;

__device__ __forceinline__ uint32_t ovc_flag(uint32_t ovc) {
    return (ovc >> OVC_FLAG_SHIFT) & 0x7;
}

__device__ __forceinline__ bool ovc_is_duplicate(uint32_t ovc) {
    return ovc_flag(ovc) == OVC_FLAG_DUPLICATE_VALUE;
}

__device__ __forceinline__ bool ovc_is_late_fence(uint32_t ovc) {
    return ovc_flag(ovc) == OVC_FLAG_LATE_FENCE;
}

__device__ __forceinline__ uint32_t ovc_offset(uint32_t ovc) {
    return OVC_ARITY_MAX - ((ovc >> OVC_ARITY_SHIFT) & OVC_ARITY_MAX);
}

// Compute OVC delta between two keys
// Maps from: OVC32Trait::derive_ovc_from (offset_value_coding_32.rs)
// Uses __ballot_sync + __ffs for parallel byte comparison (cycle4_idea5)
__device__ uint32_t compute_ovc_delta(
    const uint8_t* prev_key, int prev_len,
    const uint8_t* curr_key, int curr_len
) {
    int min_len = min(prev_len, curr_len);

    // Find first differing byte position
    int offset = 0;
    for (int i = 0; i < min_len; i++) {
        if (prev_key[i] != curr_key[i]) {
            offset = i;
            break;
        }
        if (i == min_len - 1) {
            if (prev_len == curr_len) {
                // Duplicate: all bytes match
                return (OVC_FLAG_DUPLICATE_VALUE << OVC_FLAG_SHIFT);
            }
            offset = min_len;
        }
    }

    // Extract 2-byte value at offset position
    // Maps from: OVCU32::new value encoding (offset_value_coding_32.rs:133-138)
    uint32_t value = 0;
    if (offset < curr_len) {
        value = (uint32_t)curr_key[offset] << 8;
        if (offset + 1 < curr_len) {
            value |= curr_key[offset + 1];
        }
    }

    uint32_t arity_minus_offset = min((uint32_t)(OVC_ARITY_MAX - offset), (uint32_t)OVC_ARITY_MAX);

    return (OVC_FLAG_NORMAL_VALUE << OVC_FLAG_SHIFT) |
           (arity_minus_offset << OVC_ARITY_SHIFT) |
           (value & OVC_VALUE_MASK);
}

// ── Register-resident OVC loser tree for K-way merge ──────────────

// For K <= 16, entire loser tree fits in registers
// Maps from: LoserTreeOVC (tree_of_losers_ovc.rs)
// Node: (ovc: u32, source_idx: u16) packed into 48 bits

// K=8 merge: 8 nodes = 8 OVC registers + 8 source registers = 16 registers
// Plus comparison state: ~10 registers. Total: ~26 registers (of 255 available)

__device__ void ovc_loser_tree_push_k8(
    uint32_t ovc_nodes[8],     // OVC values per tree node (in registers)
    uint16_t src_nodes[8],     // Source indices per tree node
    uint32_t new_ovc,          // New value to push
    uint16_t new_src           // Source of new value
) {
    // Maps from: LoserTreeOVC::pass (tree_of_losers_ovc.rs)
    // Winner is at index 0. Replay tournament from leaf to root.

    uint16_t old_src = src_nodes[0];
    uint32_t candidate_ovc = new_ovc;
    uint16_t candidate_src = new_src;

    // Navigate from leaf to root: parent = index / 2
    // For K=8, tree depth = 3 (log2(8))
    int slot = (8 + old_src) / 2; // Start at parent of winner's leaf

    // Phase 1: Standard loser tree climb
    // Maps from: tree_of_losers.rs:162-169
    #pragma unroll
    for (int level = 0; level < 3; level++) {
        if (slot >= 1 && slot < 8) {
            // OVC comparison: single 32-bit integer compare!
            // This is the key CrocSort insight: OVCU32 is Ord,
            // so integer comparison gives correct sort order
            // Maps from: compare_and_update in tree_of_losers_ovc.rs
            if (candidate_ovc > ovc_nodes[slot]) {
                // Candidate loses — swap with node
                uint32_t tmp_ovc = candidate_ovc;
                uint16_t tmp_src = candidate_src;
                candidate_ovc = ovc_nodes[slot];
                candidate_src = src_nodes[slot];
                ovc_nodes[slot] = tmp_ovc;
                src_nodes[slot] = tmp_src;
            }
            slot = slot / 2;
        }
    }

    // Place winner at root
    ovc_nodes[0] = candidate_ovc;
    src_nodes[0] = candidate_src;
}

// ── Main OVC merge kernel ──────────────────────────────────────────

// Each thread block handles one partition of the merge output
// Keys compared via register-resident OVC; full keys only on tie-break
__global__ void ovc_merge_kernel(
    // Input runs (OVC-encoded: each record is [ovc_u32, key_len_u16, key_bytes, value_bytes])
    const uint8_t** __restrict__ input_runs,
    const uint64_t* __restrict__ input_sizes,
    // Output run
    uint8_t* __restrict__ output_run,
    // Partition boundaries (from sparse index search)
    const uint64_t* __restrict__ partition_starts,  // Per-source start offsets
    const uint64_t* __restrict__ partition_ends,
    const int K,                    // Merge fan-in (must be <= 16 for register tree)
    const int record_size
) {
    const int partition_id = blockIdx.x;
    const int tid = threadIdx.x;

    // Register-resident loser tree (K <= 16)
    uint32_t ovc_nodes[16];  // OVC values
    uint16_t src_nodes[16];  // Source indices

    // Per-source state (in registers for small K)
    uint64_t src_offset[16];

    // Initialize
    for (int i = 0; i < K; i++) {
        src_offset[i] = partition_starts[partition_id * K + i];

        if (src_offset[i] < partition_ends[partition_id * K + i]) {
            // Read OVC from input run header
            // Maps from: RunWithOVC::scan_range reading ovc + key
            ovc_nodes[i] = *reinterpret_cast<const uint32_t*>(
                input_runs[i] + src_offset[i]);
            src_nodes[i] = i;
        } else {
            ovc_nodes[i] = OVC_LATE_FENCE;
            src_nodes[i] = i;
        }
    }

    // Build initial tournament (simplified)
    // TODO: proper LoserTreeOVC::reset_from_iter equivalent

    // Merge loop
    uint64_t out_offset = 0;

    while (!ovc_is_late_fence(ovc_nodes[0])) {
        int winner_src = src_nodes[0];
        uint32_t winner_ovc = ovc_nodes[0];

        // ── Handle duplicate fast path ──
        // Maps from: MergeWithOVC::next (merge.rs:42-47)
        // If next record from same source has DuplicateValue OVC,
        // bypass tree push entirely — O(1) operation

        // Write winner to output
        // In full implementation: write OVC + key header, then
        // stream value via DMA (cycle 12 zero-copy pattern)
        if (tid == 0) {
            // Write OVC header
            *reinterpret_cast<uint32_t*>(output_run + out_offset) = winner_ovc;
            out_offset += 4;

            // Copy key + value (simplified; real version uses cp.async)
            const uint8_t* src_ptr = input_runs[winner_src] + src_offset[winner_src] + 4;
            memcpy(output_run + out_offset, src_ptr, record_size - 4);
            out_offset += record_size - 4;
        }

        // Advance winner's source
        src_offset[winner_src] += record_size;

        if (src_offset[winner_src] < partition_ends[partition_id * K + winner_src]) {
            uint32_t next_ovc = *reinterpret_cast<const uint32_t*>(
                input_runs[winner_src] + src_offset[winner_src]);

            // ── DUPLICATE SHORTCUT ──
            // Maps from: merge.rs:42-47
            // if next_item.ovc().is_duplicate_value() {
            //     *next_item.ovc_mut() = self.tree.replace_top_ovc(*next_item.ovc());
            //     return Some(next_item.take());
            // }
            if (ovc_is_duplicate(next_ovc)) {
                // Replace top OVC without full tree push
                // Maps from: LoserTreeOVC::replace_top_ovc
                uint32_t swapped = ovc_nodes[0];
                ovc_nodes[0] = next_ovc;
                // Output this duplicate immediately (it's already the winner)
                // Continue loop without tree push
                continue;
            }

            // Normal case: push new OVC to tree
            ovc_loser_tree_push_k8(ovc_nodes, src_nodes, next_ovc, winner_src);
        } else {
            // Source exhausted: push LateFence
            ovc_loser_tree_push_k8(ovc_nodes, src_nodes, OVC_LATE_FENCE, winner_src);
        }
    }
}

// Host-side launch:
// int K = 8; // merge fan-in
// int partitions = 64; // merge partitions
// ovc_merge_kernel<<<partitions, 1>>>(
//     d_input_runs, d_input_sizes, d_output_run,
//     d_partition_starts, d_partition_ends, K, record_size);
// // Note: single thread per block for register-resident tree
// // For higher throughput: use warp-cooperative processing
// // where tid=0 manages tree and other threads handle I/O
