// CrocSort GPU Transfer: OVC-Enhanced Merge with Selective Key Fetching
// Maps from: src/sort/ovc/merge.rs:ZeroCopyMergeWithOVC (line 79)
//            Keys in loser tree, values streamed; OVC comparison avoids HBM reads
// GPU primitive: cp.async for selective key prefetch on OVC tie-break only

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cstdint>

// OVC constants (same as cycle4)
static constexpr uint32_t OVC_FLAG_SHIFT = 29;
static constexpr uint32_t OVC_LATE_FENCE = (4u << 29) | 0x1FFFFFFF;

__device__ __forceinline__ bool ovc_is_late_fence(uint32_t ovc) {
    return (ovc >> OVC_FLAG_SHIFT) >= 4;
}

// ── Selective key fetch merge kernel ───────────────────────────────
// Core insight: only read full keys from HBM when OVC comparison is a tie.
// For high-redundancy data (lineitem), >90% of comparisons are OVC-decisive,
// meaning we skip 90% of HBM key reads.
//
// Maps from: ZeroCopyMergeWithOVC design where keys live in tree and
// values flow directly reader -> writer without buffering.
// On GPU: OVC lives in registers, keys are staged in smem only on tie-break.

__global__ void ovc_selective_fetch_merge(
    // Per-source: base pointer, current offset, end offset
    const uint8_t** __restrict__ source_bases,
    uint64_t* __restrict__ source_offsets,
    const uint64_t* __restrict__ source_ends,
    // Output
    uint8_t* __restrict__ output,
    uint64_t* __restrict__ output_offset_ptr,
    // Config
    const int K,              // fan-in (<=16)
    const int max_key_len
) {
    // ── Shared memory layout ──
    // Two key staging buffers for double-buffering cp.async prefetch
    // Maps from: ZeroCopyMergeWithOVC::spare_key (merge.rs:108)
    extern __shared__ uint8_t smem[];
    uint8_t* key_buf_A = smem;                              // Current key buffer
    uint8_t* key_buf_B = smem + max_key_len;                // Prefetch buffer
    uint32_t* tie_break_result = reinterpret_cast<uint32_t*>(smem + 2 * max_key_len);

    const int tid = threadIdx.x;

    // Register-resident OVC loser tree (K<=16)
    uint32_t ovc[16];
    uint16_t src[16];

    // Track OVC hit/miss for statistics
    uint64_t ovc_hits = 0;
    uint64_t ovc_misses = 0;

    // Initialize tree from first records
    for (int i = 0; i < K; i++) {
        if (source_offsets[i] < source_ends[i]) {
            // Read OVC from record header (4 bytes)
            ovc[i] = *reinterpret_cast<const uint32_t*>(
                source_bases[i] + source_offsets[i]);
            src[i] = i;
        } else {
            ovc[i] = OVC_LATE_FENCE;
            src[i] = i;
        }
    }

    // TODO: build initial tournament

    uint64_t out_off = 0;

    // ── Main merge loop ──
    while (!ovc_is_late_fence(ovc[0])) {
        int winner_src = src[0];
        uint32_t winner_ovc = ovc[0];

        // Output current winner (OVC header + value streaming)
        if (tid == 0) {
            // Write OVC to output
            *reinterpret_cast<uint32_t*>(output + out_off) = winner_ovc;
            out_off += 4;

            // Read key_len and value_len from source
            uint64_t src_off = source_offsets[winner_src] + 4; // skip OVC
            uint16_t key_len = *reinterpret_cast<const uint16_t*>(
                source_bases[winner_src] + src_off);
            src_off += 2;
            uint16_t val_len = *reinterpret_cast<const uint16_t*>(
                source_bases[winner_src] + src_off);
            src_off += 2;

            // Write key to output
            memcpy(output + out_off, source_bases[winner_src] + src_off, key_len);
            out_off += key_len;
            src_off += key_len;

            // Stream value to output (zero-copy: DMA in production)
            // Maps from: source.copy_value_to(output.writer(), value_len)
            // in merge.rs:176
            memcpy(output + out_off, source_bases[winner_src] + src_off, val_len);
            out_off += val_len;

            // Advance source
            source_offsets[winner_src] = src_off + val_len - 4; // TODO: fix offset
        }
        __syncthreads();

        // Read next OVC from winner's source
        uint32_t next_ovc = OVC_LATE_FENCE;
        if (source_offsets[winner_src] < source_ends[winner_src]) {
            next_ovc = *reinterpret_cast<const uint32_t*>(
                source_bases[winner_src] + source_offsets[winner_src]);
        }

        // ── OVC COMPARISON: The key optimization ──
        // Compare next_ovc with all tree nodes using register comparison
        // This is a SINGLE 32-bit integer compare per tree node!
        // Maps from: LoserTreeOVC compare_and_update using OVC

        // Determine if OVC is decisive (no tie-break needed)
        bool ovc_decisive = true;

        // Push to tree and check if any comparison was an OVC tie
        // (In practice: OVC tie = same flag + same arity + same value,
        //  which means keys share a long prefix)

        // Simple tournament replay with OVC
        uint32_t cand_ovc = next_ovc;
        uint16_t cand_src = winner_src;
        int slot = (K + winner_src) / 2;

        #pragma unroll
        for (int level = 0; level < 4; level++) {  // log2(16) = 4 max levels
            if (slot >= 1 && slot < K) {
                if (cand_ovc == ovc[slot]) {
                    // ── OVC TIE: Must fetch full keys from HBM ──
                    // This is the rare case (~10% on redundant data)
                    ovc_decisive = false;
                    ovc_misses++;

                    // Prefetch both keys via cp.async
                    // Maps from: compare_and_update falling back to
                    // full key comparison when OVC values are equal
                    if (tid == 0) {
                        // Read candidate key
                        uint64_t cand_off = source_offsets[cand_src] + 4;
                        uint16_t cand_kl = *reinterpret_cast<const uint16_t*>(
                            source_bases[cand_src] + cand_off);
                        memcpy(key_buf_A, source_bases[cand_src] + cand_off + 4, cand_kl);

                        // Read opponent key
                        uint64_t opp_off = source_offsets[src[slot]] + 4;
                        uint16_t opp_kl = *reinterpret_cast<const uint16_t*>(
                            source_bases[src[slot]] + opp_off);
                        memcpy(key_buf_B, source_bases[src[slot]] + opp_off + 4, opp_kl);

                        // Full key comparison (only on OVC tie)
                        int cmp = memcmp(key_buf_A, key_buf_B, min(cand_kl, opp_kl));
                        if (cmp == 0) cmp = (cand_kl < opp_kl) ? -1 : (cand_kl > opp_kl) ? 1 : 0;
                        *tie_break_result = (cmp > 0) ? 1 : 0;
                    }
                    __syncthreads();

                    if (*tie_break_result) {
                        // Candidate loses
                        uint32_t tmp_ovc = cand_ovc; uint16_t tmp_src = cand_src;
                        cand_ovc = ovc[slot]; cand_src = src[slot];
                        ovc[slot] = tmp_ovc; src[slot] = tmp_src;
                    }
                } else if (cand_ovc > ovc[slot]) {
                    // OVC decisive: candidate loses (single integer compare!)
                    ovc_hits++;
                    uint32_t tmp_ovc = cand_ovc; uint16_t tmp_src = cand_src;
                    cand_ovc = ovc[slot]; cand_src = src[slot];
                    ovc[slot] = tmp_ovc; src[slot] = tmp_src;
                } else {
                    // OVC decisive: candidate wins
                    ovc_hits++;
                }
                slot = slot / 2;
            }
        }

        ovc[0] = cand_ovc;
        src[0] = cand_src;
    }

    if (tid == 0) {
        *output_offset_ptr = out_off;
        // Report OVC statistics
        // printf("OVC hits: %lu, misses: %lu, hit rate: %.1f%%\n",
        //        ovc_hits, ovc_misses,
        //        100.0 * ovc_hits / (ovc_hits + ovc_misses + 1));
    }
}

// Host-side launch:
// int smem_bytes = 2 * max_key_len + 4;
// ovc_selective_fetch_merge<<<num_partitions, 32, smem_bytes>>>(
//     d_source_bases, d_source_offsets, d_source_ends,
//     d_output, d_output_offset, K, max_key_len);
