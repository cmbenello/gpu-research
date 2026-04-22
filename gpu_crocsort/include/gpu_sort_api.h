// GPU Sort C API — thin wrapper for DuckDB integration
// Sorts byte-comparable key blobs using GPU radix sort + CPU fixup.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Sort fixed-size byte-comparable keys and produce a sorted permutation.
//
// Parameters:
//   keys          - contiguous buffer of key rows, each `key_stride` bytes
//   key_size      - number of prefix bytes to sort on (comparison_size)
//   key_stride    - total bytes per key row (entry_size, >= key_size)
//   num_records   - number of records
//   perm_out      - output: sorted permutation array (caller-allocated, num_records entries)
//                   perm_out[i] = original row index of i-th sorted record
//
// Returns 0 on success, non-zero on error.
int gpu_sort_keys(
    const uint8_t* keys,
    uint32_t key_size,
    uint32_t key_stride,
    uint64_t num_records,
    uint32_t* perm_out
);

// Sort keys AND gather payload rows on GPU.
// Avoids CPU-side payload reorder by doing the gather on GPU.
//
// Parameters:
//   keys           - contiguous key rows (key_stride bytes each)
//   key_size       - comparison bytes per key
//   key_stride     - total bytes per key row
//   payload        - contiguous payload rows (payload_stride bytes each)
//   payload_stride - bytes per payload row
//   num_records    - number of records
//   sorted_payload - output: reordered payload (caller-allocated, num_records * payload_stride)
//
// Returns 0 on success, non-zero on error.
int gpu_sort_and_gather(
    const uint8_t* keys,
    uint32_t key_size,
    uint32_t key_stride,
    const uint8_t* payload,
    uint32_t payload_stride,
    uint64_t num_records,
    uint8_t* sorted_payload
);

// Pin host memory for fast DMA transfers. Call before gpu_sort_keys.
// Returns 0 on success. The caller must call gpu_unpin_memory to unpin.
int gpu_pin_memory(void* ptr, size_t size);
int gpu_unpin_memory(void* ptr);

// Allocate pinned (page-locked) host memory for maximum DMA bandwidth.
void* gpu_alloc_pinned(size_t size);
void gpu_free_pinned(void* ptr);

// Cached pinned host alloc — grows but never shrinks.
void* gpu_pinned_cache_alloc(size_t bytes);

// Query GPU availability. Returns 1 if a CUDA GPU is available, 0 otherwise.
int gpu_sort_available(void);

// Query GPU memory. Returns 0 on success, fills free_bytes and total_bytes.
int gpu_query_memory(size_t* free_bytes, size_t* total_bytes);

// Get timing from last sort call (milliseconds).
typedef struct {
    double total_ms;
    double upload_ms;
    double gpu_sort_ms;
    double download_ms;
    double gather_ms;     // GPU-side gather time (for sort_and_gather)
    double fixup_ms;
    uint64_t num_fixup_groups;
    uint64_t num_fixup_records;
} GpuSortTiming;

void gpu_sort_get_timing(GpuSortTiming* out);

// ── Low-level device memory API ─────────────────────────────────
// For zero-copy integration: upload blocks directly to GPU, sort on device,
// download results directly to output buffer. Avoids host-side concat.

void* gpu_device_alloc(size_t bytes);
void  gpu_device_free(void* d_ptr);
// Cached device alloc: slot 0-3, grows but never shrinks (avoids cudaMalloc overhead)
void* gpu_cache_alloc(int slot, size_t bytes);
void  gpu_cache_free_all(void);
void  gpu_memcpy_h2d(void* d_dst, const void* h_src, size_t bytes);
void  gpu_memcpy_d2h(void* h_dst, const void* d_src, size_t bytes);

// Sort + gather on data already resident on GPU device memory.
// d_keys, d_payload are device pointers (populated by caller via gpu_memcpy_h2d).
// d_sorted_payload is a device pointer for output (caller downloads via gpu_memcpy_d2h).
// Returns 0 on success.
int gpu_sort_and_gather_device(
    uint8_t* d_keys, uint32_t key_size, uint32_t key_stride,
    uint8_t* d_payload, uint32_t payload_stride,
    uint64_t num_records,
    uint8_t* d_sorted_payload
);

// GPU-side gather only: reorder payload using permutation.
int gpu_gather_device(
    const uint8_t* d_payload, uint32_t payload_stride,
    const uint32_t* d_perm, uint64_t num_records,
    uint8_t* d_sorted_payload
);

// Sort keys on GPU device memory and return permutation only (no payload gather).
// d_perm_out is a device pointer (caller-allocated, num_records * sizeof(uint32_t)).
// Use this for columnar mode: upload only keys, get permutation, CPU-side gather.
// Returns 0 on success.
int gpu_sort_permutation_device(
    uint8_t* d_keys, uint32_t key_size, uint32_t key_stride,
    uint64_t num_records,
    uint32_t* d_perm_out
);

// ── Stream API for async transfers ──────────────────────────────
void* gpu_stream_create(void);
void  gpu_stream_sync(void* stream);
void  gpu_stream_destroy(void* stream);
void  gpu_memcpy_h2d_async(void* d_dst, const void* h_src, size_t bytes, void* stream);
void  gpu_memcpy_d2h_async(void* h_dst, const void* d_src, size_t bytes, void* stream);

// ── Staged upload: concat scattered blocks through pinned staging buffer ──
// Uses a persistent pinned (cudaMallocHost) staging buffer for true DMA.
// Collects blocks into staging, then single DMA to GPU = max PCIe throughput.
void gpu_upload_staged(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks);

// ── Pipelined upload: double-buffered pinned staging with DMA overlap ──
// Overlaps CPU memcpy (block → pinned) with GPU DMA (pinned → device).
// Uses two small pinned staging buffers + two CUDA streams.
void gpu_upload_pipelined(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks);

// ── Direct pinned upload: pin→DMA→unpin (no staging buffer copy) ──
void gpu_upload_direct_pinned(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks);

// ── Chunked pipelined upload: fixed 128MB pinned staging, double-buffered ──
// Avoids per-block cudaHostRegister overhead by using pre-allocated pinned memory.
void gpu_upload_chunked(void* d_dst, const void** src_ptrs, const size_t* src_sizes, uint32_t num_blocks);


// ── Pipelined download: double-buffered pinned staging for DMA overlap ──
void gpu_download_pipelined(void* h_dst, const void* d_src, size_t bytes, size_t chunk_size);

// ── Direct pinned download: pin→DMA→unpin (no staging copy) ──
void gpu_download_direct_pinned(void* h_dst, const void* d_src, size_t bytes);

#ifdef __cplusplus
}
#endif
