#pragma once
#include <cstdint>
#include "record.cuh"

// ============================================================================
// OVC (Offset-Value Coding) — GPU port of CrocSort's OVCU32
// Source: crocsort_repo/src/ovc/offset_value_coding_32.rs
//
// Bit layout (32 bits):
// [bits 29-31: flag (3)]  [bits 16-28: arity_minus_offset (13)]  [bits 0-15: value (16)]
//
// The encoding is designed so that natural uint32_t ordering matches sort order:
//   EarlyFence < DuplicateValue < NormalValue < InitialValue < LateFence
// Within NormalValue: higher arity_minus_offset = smaller offset = earlier in key = "larger" OVC
// ============================================================================

// Flag values (3 bits, stored in bits 29-31)
static constexpr uint32_t OVC_FLAG_EARLY_FENCE    = 0;  // Sentinel: less than everything
static constexpr uint32_t OVC_FLAG_DUPLICATE       = 1;  // Key identical to predecessor
static constexpr uint32_t OVC_FLAG_NORMAL          = 2;  // Normal: offset + 2-byte value
static constexpr uint32_t OVC_FLAG_INITIAL         = 3;  // First record in a run
static constexpr uint32_t OVC_FLAG_LATE_FENCE      = 4;  // Sentinel: greater than everything

// Bit field constants
static constexpr uint32_t OVC_VALUE_MASK    = 0x0000FFFF;  // Bits 0-15
static constexpr uint32_t OVC_ARITY_MAX     = 0x1FFF;      // 13 bits = 8191
static constexpr uint32_t OVC_ARITY_SHIFT   = 16;
static constexpr uint32_t OVC_FLAG_SHIFT    = 29;
static constexpr uint32_t OVC_FLAG_MASK     = 0x7;

// Pre-computed sentinel values
static constexpr uint32_t OVC_EARLY_FENCE = (OVC_FLAG_EARLY_FENCE << OVC_FLAG_SHIFT);
static constexpr uint32_t OVC_LATE_FENCE  = (OVC_FLAG_LATE_FENCE << OVC_FLAG_SHIFT)
                                          | (OVC_ARITY_MAX << OVC_ARITY_SHIFT)
                                          | OVC_VALUE_MASK;
static constexpr uint32_t OVC_INITIAL     = (OVC_FLAG_INITIAL << OVC_FLAG_SHIFT)
                                          | (OVC_ARITY_MAX << OVC_ARITY_SHIFT)
                                          | OVC_VALUE_MASK;
static constexpr uint32_t OVC_DUPLICATE   = (OVC_FLAG_DUPLICATE << OVC_FLAG_SHIFT);

// ── Encode/decode functions ────────────────────────────────────────

// Encode an OVC from components
// Maps from: OVCU32::new (offset_value_coding_32.rs:133)
__host__ __device__ inline uint32_t ovc_encode(uint32_t flag, int offset, uint16_t value) {
    uint32_t arity_minus_offset = OVC_ARITY_MAX - (uint32_t)offset;
    if (arity_minus_offset > OVC_ARITY_MAX) arity_minus_offset = 0; // Underflow guard
    return (flag << OVC_FLAG_SHIFT)
         | (arity_minus_offset << OVC_ARITY_SHIFT)
         | ((uint32_t)value & OVC_VALUE_MASK);
}

// Extract flag from OVC
__host__ __device__ inline uint32_t ovc_flag(uint32_t ovc) {
    return (ovc >> OVC_FLAG_SHIFT) & OVC_FLAG_MASK;
}

// Extract offset from OVC (inverted arity encoding)
__host__ __device__ inline int ovc_offset(uint32_t ovc) {
    uint32_t arity = (ovc >> OVC_ARITY_SHIFT) & OVC_ARITY_MAX;
    return (int)(OVC_ARITY_MAX - arity);
}

// Extract 2-byte value from OVC
__host__ __device__ inline uint16_t ovc_value(uint32_t ovc) {
    return (uint16_t)(ovc & OVC_VALUE_MASK);
}

// ── Flag checks ────────────────────────────────────────────────────

__host__ __device__ inline bool ovc_is_early_fence(uint32_t ovc) {
    return ovc_flag(ovc) == OVC_FLAG_EARLY_FENCE;
}

__host__ __device__ inline bool ovc_is_duplicate(uint32_t ovc) {
    return ovc_flag(ovc) == OVC_FLAG_DUPLICATE;
}

__host__ __device__ inline bool ovc_is_normal(uint32_t ovc) {
    return ovc_flag(ovc) == OVC_FLAG_NORMAL;
}

__host__ __device__ inline bool ovc_is_initial(uint32_t ovc) {
    return ovc_flag(ovc) == OVC_FLAG_INITIAL;
}

__host__ __device__ inline bool ovc_is_late_fence(uint32_t ovc) {
    return ovc_flag(ovc) == OVC_FLAG_LATE_FENCE;
}

__host__ __device__ inline bool ovc_is_sentinel(uint32_t ovc) {
    uint32_t f = ovc_flag(ovc);
    return f == OVC_FLAG_EARLY_FENCE || f == OVC_FLAG_LATE_FENCE;
}

// ── OVC delta computation ──────────────────────────────────────────
// Maps from: OVC32Trait::derive_ovc_from (offset_value_coding_32.rs)
// Compares two keys byte-by-byte, finds first differing position,
// encodes the 2-byte value at that position (aligned to 2-byte chunks).

__host__ __device__ inline uint32_t ovc_compute_delta(
    const uint8_t* prev_key,
    const uint8_t* curr_key,
    int key_len
) {
    // Find first differing byte
    int first_diff = key_len; // If all bytes match -> duplicate
    for (int i = 0; i < key_len; i++) {
        if (prev_key[i] != curr_key[i]) {
            first_diff = i;
            break;
        }
    }

    // All bytes match -> duplicate
    if (first_diff >= key_len) {
        return OVC_DUPLICATE;
    }

    // Align offset to 2-byte chunk boundary (OVC32_CHUNK_SIZE = 2)
    int offset = (first_diff / 2) * 2;

    // Extract 2-byte value at the aligned offset
    uint16_t value = 0;
    if (offset < key_len) {
        value = (uint16_t)curr_key[offset] << 8;
        if (offset + 1 < key_len) {
            value |= (uint16_t)curr_key[offset + 1];
        }
    }

    return ovc_encode(OVC_FLAG_NORMAL, offset, value);
}

// ── Full key comparison (used on OVC tie-break) ────────────────────
// Maps from: compare_and_update falling back to byte comparison
// Returns: <0 if a<b, 0 if a==b, >0 if a>b

__host__ __device__ inline int key_compare(
    const uint8_t* key_a,
    const uint8_t* key_b,
    int key_len
) {
    for (int i = 0; i < key_len; i++) {
        if (key_a[i] != key_b[i]) {
            return (int)key_a[i] - (int)key_b[i];
        }
    }
    return 0;
}

// Compare two keys starting from a given offset (after OVC resolves prefix)
// Maps from: compare_and_update resuming from offset + OVC32_CHUNK_SIZE
__host__ __device__ inline int key_compare_from(
    const uint8_t* key_a,
    const uint8_t* key_b,
    int key_len,
    int start_offset
) {
    for (int i = start_offset; i < key_len; i++) {
        if (key_a[i] != key_b[i]) {
            return (int)key_a[i] - (int)key_b[i];
        }
    }
    return 0;
}
