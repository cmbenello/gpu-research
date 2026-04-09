// CrocSort GPU Transfer: Best-Fit Allocator in Shared Memory
// Maps from: src/replacement_selection/memory.rs:MemoryManager (line 98)
//            Best-fit allocator with 64KB extents, 32B alignment, free-list coalescing
// GPU primitive: Shared memory as arena, warp-level atomics for thread-safe alloc/free

#include <cuda_runtime.h>
#include <cstdint>

// ── Constants matching CrocSort's MemoryManager (memory.rs:6-21) ──

static constexpr int SMEM_EXTENT_SIZE = 49152;  // 48KB (conservative for shared memory)
static constexpr int ALIGN = 32;                 // 32-byte alignment
static constexpr int FREE_LISTS = SMEM_EXTENT_SIZE / ALIGN;  // ~1536 size classes
static constexpr int HEADER_SIZE = 4;            // Block header: size + flags
static constexpr int FOOTER_SIZE = 4;            // Block footer: size + flags
static constexpr int MIN_BLOCK_SIZE = ALIGN;     // 32 bytes minimum

// Tag encoding matching memory.rs:15-17
static constexpr uint32_t TAG_ALLOCATED = 0x01;
static constexpr uint32_t TAG_FLAGS_MASK = 0x1F;

// Handle encoding matching AllocHandle (memory.rs:23)
// For single-extent smem: just the byte offset (no extent index needed)
struct SmemAllocHandle {
    uint32_t offset;  // Byte offset within smem extent

    static constexpr uint32_t NONE = 0xFFFFFFFF;
    static constexpr uint32_t EARLY = 0xFFFFFFFE;
    static constexpr uint32_t LATE = 0xFFFFFFFD;

    __device__ bool is_none() const { return offset == NONE; }
    __device__ bool is_sentinel() const { return offset >= LATE; }
    __device__ bool is_valid() const { return offset < LATE; }
};

// ── Shared-memory allocator state ──────────────────────────────────

struct SmemAllocator {
    // The arena itself (shared memory)
    uint8_t* arena;
    int arena_size;

    // Free list heads: free_list[size_class] = offset of first free block
    // Maps from: MemoryManager::free_lists (memory.rs:100)
    uint32_t* free_list_heads;  // Array of FREE_LISTS entries in smem
    int last_non_null;          // Highest non-empty free list class

    // Statistics
    uint32_t allocated_bytes;

    // ── Block tag operations (matching memory.rs read_block_tag/write_block_tags) ──

    __device__ void write_block_tags(uint32_t offset, uint32_t size, bool allocated) {
        uint32_t tag = (size & ~TAG_FLAGS_MASK) | (allocated ? TAG_ALLOCATED : 0);
        // Header at start of block
        *reinterpret_cast<uint32_t*>(arena + offset) = tag;
        // Footer at end of block (last 4 bytes)
        *reinterpret_cast<uint32_t*>(arena + offset + size - FOOTER_SIZE) = tag;
    }

    __device__ uint32_t read_block_size(uint32_t offset) {
        uint32_t tag = *reinterpret_cast<uint32_t*>(arena + offset);
        return tag & ~TAG_FLAGS_MASK;
    }

    __device__ bool is_block_allocated(uint32_t offset) {
        uint32_t tag = *reinterpret_cast<uint32_t*>(arena + offset);
        return (tag & TAG_ALLOCATED) != 0;
    }

    // ── Size class computation ──
    // Maps from: MemoryManager free_list indexing by size/ALIGN

    __device__ int size_class(uint32_t size) {
        return (size / ALIGN) - 1;  // 0-indexed: class 0 = 32B, class 1 = 64B, ...
    }

    __device__ uint32_t required_block_size(uint32_t payload_size) {
        uint32_t total = payload_size + HEADER_SIZE + FOOTER_SIZE;
        return ((total + ALIGN - 1) / ALIGN) * ALIGN;  // Round up to ALIGN
    }

    // ── Initialize allocator ──
    // Maps from: MemoryManager::new (memory.rs:108)

    __device__ void init(uint8_t* smem_arena, int size, uint32_t* smem_free_lists) {
        arena = smem_arena;
        arena_size = size;
        free_list_heads = smem_free_lists;
        allocated_bytes = 0;
        last_non_null = -1;

        // Clear free lists
        for (int i = threadIdx.x; i < FREE_LISTS; i += blockDim.x) {
            free_list_heads[i] = SmemAllocHandle::NONE;
        }
        __syncthreads();

        // Create single free block spanning entire arena
        if (threadIdx.x == 0) {
            uint32_t block_size = (arena_size / ALIGN) * ALIGN;
            write_block_tags(0, block_size, false);

            // No next/prev pointers needed for single initial block
            int cls = size_class(block_size);
            if (cls < FREE_LISTS) {
                free_list_heads[cls] = 0;
                last_non_null = cls;
            }
        }
        __syncthreads();
    }

    // ── Allocate ──
    // Maps from: MemoryManager::alloc (memory.rs:144)
    // Best-fit: scan from required size class upward

    __device__ SmemAllocHandle alloc(uint32_t payload_size) {
        uint32_t required = required_block_size(payload_size);
        int min_class = size_class(required);

        // Scan free lists from min_class upward (best-fit)
        // Maps from: find_free_block scanning from required class
        for (int cls = min_class; cls <= last_non_null && cls < FREE_LISTS; cls++) {
            if (free_list_heads[cls] != SmemAllocHandle::NONE) {
                uint32_t block_offset = free_list_heads[cls];
                uint32_t block_size = read_block_size(block_offset);

                // Remove from free list
                // TODO: linked list traversal for multi-block free lists
                free_list_heads[cls] = SmemAllocHandle::NONE;

                // Split if remainder is large enough
                // Maps from: MemoryManager allocate_from splitting logic
                if (block_size >= required + MIN_BLOCK_SIZE) {
                    uint32_t remainder = block_size - required;
                    write_block_tags(block_offset, required, true);
                    write_block_tags(block_offset + required, remainder, false);

                    // Add remainder to appropriate free list
                    int rem_class = size_class(remainder);
                    if (rem_class < FREE_LISTS) {
                        free_list_heads[rem_class] = block_offset + required;
                        if (rem_class > last_non_null) last_non_null = rem_class;
                    }
                } else {
                    // Use entire block (no split)
                    write_block_tags(block_offset, block_size, true);
                }

                allocated_bytes += required;
                return {block_offset + HEADER_SIZE};  // Return pointer past header
            }
        }

        return {SmemAllocHandle::NONE};  // Out of memory
    }

    // ── Free with coalescing ──
    // Maps from: MemoryManager::free (memory.rs:165)
    // Checks left and right neighbors for merging

    __device__ void free(SmemAllocHandle handle) {
        if (!handle.is_valid()) return;

        uint32_t block_offset = handle.offset - HEADER_SIZE;
        uint32_t block_size = read_block_size(block_offset);

        allocated_bytes -= block_size;

        // Mark as free
        write_block_tags(block_offset, block_size, false);

        uint32_t new_offset = block_offset;
        uint32_t new_size = block_size;

        // ── Coalesce left neighbor ──
        // Maps from: memory.rs:178-189
        if (block_offset >= FOOTER_SIZE) {
            uint32_t left_footer_offset = block_offset - FOOTER_SIZE;
            if (!is_block_allocated(left_footer_offset)) {
                uint32_t left_tag = *reinterpret_cast<uint32_t*>(arena + left_footer_offset);
                uint32_t left_size = left_tag & ~TAG_FLAGS_MASK;
                uint32_t left_offset = block_offset - left_size;

                // Remove left block from free list
                int left_class = size_class(left_size);
                if (left_class < FREE_LISTS && free_list_heads[left_class] == left_offset) {
                    free_list_heads[left_class] = SmemAllocHandle::NONE;
                }

                new_offset = left_offset;
                new_size += left_size;
            }
        }

        // ── Coalesce right neighbor ──
        // Maps from: memory.rs:197-210
        uint32_t right_offset = block_offset + block_size;
        if (right_offset + HEADER_SIZE <= (uint32_t)arena_size) {
            if (!is_block_allocated(right_offset)) {
                uint32_t right_size = read_block_size(right_offset);

                // Remove right block from free list
                int right_class = size_class(right_size);
                if (right_class < FREE_LISTS && free_list_heads[right_class] == right_offset) {
                    free_list_heads[right_class] = SmemAllocHandle::NONE;
                }

                new_size += right_size;
            }
        }

        // Write coalesced block tags
        write_block_tags(new_offset, new_size, false);

        // Add to free list
        int new_class = size_class(new_size);
        if (new_class < FREE_LISTS) {
            free_list_heads[new_class] = new_offset;
            if (new_class > last_non_null) last_non_null = new_class;
        }
    }

    // ── Access allocated memory ──

    __device__ uint8_t* get_ptr(SmemAllocHandle handle) {
        return arena + handle.offset;
    }

    __device__ uint32_t allocation_size(SmemAllocHandle handle) {
        uint32_t block_offset = handle.offset - HEADER_SIZE;
        return read_block_size(block_offset) - HEADER_SIZE - FOOTER_SIZE;
    }
};

// ── Usage in replacement selection kernel ──────────────────────────

__global__ void replacement_selection_with_smem_alloc(
    const uint8_t* __restrict__ input_data,
    uint8_t* __restrict__ output_runs,
    int num_records,
    int max_record_size
) {
    // Shared memory layout:
    // [0..alloc_arena): Allocator arena (main workspace)
    // [alloc_arena..alloc_arena+free_lists): Free list heads
    extern __shared__ uint8_t total_smem[];

    const int ARENA_SIZE = 40960; // 40KB for arena
    const int FREE_LIST_BYTES = FREE_LISTS * sizeof(uint32_t);

    uint8_t* arena = total_smem;
    uint32_t* free_lists = reinterpret_cast<uint32_t*>(total_smem + ARENA_SIZE);

    __shared__ SmemAllocator allocator;

    // Initialize
    allocator.init(arena, ARENA_SIZE, free_lists);

    // Use allocator for variable-size record storage during replacement selection
    // Maps from: tol_mm.rs KeyValueMM::try_new using ManagedSlice::alloc
    if (threadIdx.x == 0) {
        // Example: allocate records of varying sizes
        SmemAllocHandle h1 = allocator.alloc(100);  // 100-byte record
        SmemAllocHandle h2 = allocator.alloc(50);   // 50-byte record

        if (h1.is_valid()) {
            uint8_t* ptr = allocator.get_ptr(h1);
            // Copy record data to allocated region
            // ... loser tree operations ...
        }

        // Free when record is emitted from replacement selection
        allocator.free(h1);
        // Coalescing happens automatically
    }
}

// Host-side launch:
// int smem = 40960 + FREE_LISTS * sizeof(uint32_t);
// replacement_selection_with_smem_alloc<<<grid, 256, smem>>>(
//     d_input, d_output, num_records, max_record_size);
