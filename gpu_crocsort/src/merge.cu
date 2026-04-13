#include "record.cuh"
#include "ovc.cuh"
#include <cstdint>

// ============================================================================
// Merge Phase: Partition-Parallel K-Way Merge
//
// STRATEGY 1 — "merge_path_2way": 2-way merge path, all threads active,
//   but needs log2(N) passes. Good compute, bad total bandwidth.
//
// STRATEGY 2 — "kway_partition": K-way merge with partition parallelism.
//   Only log_K(N) passes. Each partition = 1 block with cooperative I/O.
//   Thread 0 runs loser tree, other threads prefetch + write.
//   Thousands of partitions → thousands of blocks → enough parallelism.
//
// STRATEGY 3 — "smem_merge_tree" (NOVEL): K-way merge decomposed as
//   log2(K) levels of 2-way merge-path IN shared memory. Every thread
//   is active at every level. Achieves BOTH high fanin AND full parallelism.
//   Only the final output goes to HBM. This is the hybrid approach.
//
// All three are implemented below. Host chooses based on benchmarking.
// ============================================================================

// Pair descriptor for 2-way merge
struct PairDesc2Way {
    uint64_t a_byte_offset;
    int      a_count;
    uint64_t b_byte_offset;
    int      b_count;
    uint64_t out_byte_offset;
    int      first_block;
};

// ────────────────────────────────────────────────────────────────────
// STRATEGY 1: 2-way merge path (existing, kept as baseline)
// ────────────────────────────────────────────────────────────────────

static constexpr int MP2_ITEMS_PER_THREAD = 8;
static constexpr int MP2_BLOCK_THREADS = 256;

__device__ int merge_path_search(
    const uint8_t* __restrict__ A, int a_len,
    const uint8_t* __restrict__ B, int b_len,
    int diag
) {
    int lo = max(0, diag - b_len);
    int hi = min(diag, a_len);
    while (lo < hi) {
        int a_mid = (lo + hi) >> 1;
        int b_mid = diag - 1 - a_mid;
        bool a_greater;
        if (a_mid >= a_len) a_greater = true;
        else if (b_mid < 0 || b_mid >= b_len) a_greater = false;
        else a_greater = (key_compare(A + (uint64_t)a_mid * RECORD_SIZE,
                                       B + (uint64_t)b_mid * RECORD_SIZE, KEY_SIZE) > 0);
        if (a_greater) hi = a_mid;
        else lo = a_mid + 1;
    }
    return lo;
}

__global__ void merge_2way_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const PairDesc2Way* __restrict__ pairs,
    int num_pairs
) {
    int pair_id = 0;
    { int lo = 0, hi = num_pairs;
      while (lo < hi) { int m = (lo+hi)>>1;
        if (pairs[m].first_block <= (int)blockIdx.x) { pair_id = m; lo = m+1; } else hi = m; } }

    const PairDesc2Way& p = pairs[pair_id];
    const uint8_t* A = input + p.a_byte_offset;
    const uint8_t* B = input + p.b_byte_offset;
    int total = p.a_count + p.b_count;
    int block_in_pair = blockIdx.x - p.first_block;
    int items_per_block = MP2_ITEMS_PER_THREAD * MP2_BLOCK_THREADS;
    int t_start = block_in_pair * items_per_block + threadIdx.x * MP2_ITEMS_PER_THREAD;
    int t_end = min(t_start + MP2_ITEMS_PER_THREAD, min((block_in_pair+1)*items_per_block, total));
    if (t_start >= total) return;

    int ai = merge_path_search(A, p.a_count, B, p.b_count, t_start);
    int bi = t_start - ai;
    uint8_t* out_base = output + p.out_byte_offset;

    for (int i = t_start; i < t_end; i++) {
        bool take_a = (ai < p.a_count) && (bi >= p.b_count ||
            key_compare(A + (uint64_t)ai*RECORD_SIZE, B + (uint64_t)bi*RECORD_SIZE, KEY_SIZE) <= 0);
        const uint8_t* src = take_a ? A + (uint64_t)ai*RECORD_SIZE : B + (uint64_t)bi*RECORD_SIZE;
        if (take_a) ai++; else bi++;
        uint8_t* dst = out_base + (uint64_t)i * RECORD_SIZE;
        for (int b = 0; b < RECORD_SIZE; b += 4) {
            *reinterpret_cast<uint32_t*>(dst + b) =
                *reinterpret_cast<const uint32_t*>(src + b);
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// STRATEGY 3: Shared-Memory K-Way Merge Tree (NOVEL)
//
// Decomposes K-way merge as log2(K) levels of 2-way merge-path,
// all happening in shared memory. Every thread participates at
// every level. Only the final merged output touches HBM.
//
// For K=8: 3 levels.
//   Level 0: 4 independent 2-way merges (64 threads each)
//   Level 1: 2 independent 2-way merges (128 threads each)
//   Level 2: 1 final 2-way merge (256 threads)
// ────────────────────────────────────────────────────────────────────

static constexpr int SMMT_K = 8;               // Fan-in (must be power of 2)
static constexpr int SMMT_LEVELS = 3;           // log2(K)
static constexpr int SMMT_BLOCK_THREADS = 256;

// Partition descriptor for K-way merge tree
struct KWayPartition {
    int      src_rec_start[SMMT_K]; // Start record index in each source run
    int      src_rec_count[SMMT_K]; // Number of records from each source
    uint64_t src_byte_off[SMMT_K];  // Byte offset of each source run in input
    uint64_t out_byte_offset;       // Output byte offset
    int      total_records;         // Total records in this partition
};

// Merge-path search in shared memory (keys are at stride RECORD_SIZE)
__device__ int smem_merge_path(
    const uint8_t* A, int a_len,
    const uint8_t* B, int b_len,
    int diag, int stride
) {
    int lo = max(0, diag - b_len);
    int hi = min(diag, a_len);
    while (lo < hi) {
        int am = (lo + hi) >> 1;
        int bm = diag - 1 - am;
        bool a_gt;
        if (am >= a_len) a_gt = true;
        else if (bm < 0 || bm >= b_len) a_gt = false;
        else a_gt = (key_compare(A + am * stride, B + bm * stride, KEY_SIZE) > 0);
        if (a_gt) hi = am;
        else lo = am + 1;
    }
    return lo;
}

// 2-way merge-path in shared memory, using a subset of threads
__device__ void smem_merge_2way(
    const uint8_t* __restrict__ A, int a_len,
    const uint8_t* __restrict__ B, int b_len,
    uint8_t* __restrict__ out,
    int my_tid,       // Thread index within this merge (0..threads_for_this_merge-1)
    int num_threads   // Total threads assigned to this merge
) {
    int total = a_len + b_len;
    int items_per_thread = (total + num_threads - 1) / num_threads;
    int t_start = my_tid * items_per_thread;
    int t_end = min(t_start + items_per_thread, total);
    if (t_start >= total) return;

    int ai = smem_merge_path(A, a_len, B, b_len, t_start, RECORD_SIZE);
    int bi = t_start - ai;

    for (int i = t_start; i < t_end; i++) {
        bool take_a = (ai < a_len) && (bi >= b_len ||
            key_compare(A + ai * RECORD_SIZE, B + bi * RECORD_SIZE, KEY_SIZE) <= 0);
        const uint8_t* src = take_a ? A + ai * RECORD_SIZE : B + bi * RECORD_SIZE;
        if (take_a) ai++; else bi++;

        uint8_t* dst = out + i * RECORD_SIZE;
        // Copy record in shared memory (16-byte chunks where possible)
        for (int b = 0; b < RECORD_SIZE; b += 4) {
            *reinterpret_cast<uint32_t*>(dst + b) =
                *reinterpret_cast<const uint32_t*>(src + b);
        }
    }
}

__global__ void __launch_bounds__(SMMT_BLOCK_THREADS)
smem_kway_merge_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const KWayPartition* __restrict__ partitions
) {
    int pid = blockIdx.x;
    const KWayPartition& part = partitions[pid];
    int total = part.total_records;
    if (total == 0) return;

    // Dynamic shared memory: 2 buffers for ping-pong merge
    // Each buffer holds `total` records of RECORD_SIZE bytes
    extern __shared__ uint8_t smem[];
    uint8_t* buf_a = smem;
    uint8_t* buf_b = smem + total * RECORD_SIZE;

    // ── Step 1: Cooperative load from K sources into buf_a ──
    // Records laid out as: [src0_records][src1_records]...[srcK_records]
    // We track where each source's records start in the buffer
    __shared__ int seq_start[SMMT_K + 1]; // Prefix sum of source counts
    if (threadIdx.x == 0) {
        seq_start[0] = 0;
        for (int k = 0; k < SMMT_K; k++) {
            seq_start[k + 1] = seq_start[k] + part.src_rec_count[k];
        }
    }
    __syncthreads();

    // Load: each thread handles a strided portion of records
    for (int k = 0; k < SMMT_K; k++) {
        int count = part.src_rec_count[k];
        uint64_t src_base = part.src_byte_off[k] + (uint64_t)part.src_rec_start[k] * RECORD_SIZE;
        int dst_start = seq_start[k];
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const uint8_t* src = input + src_base + (uint64_t)i * RECORD_SIZE;
            uint8_t* dst = buf_a + (dst_start + i) * RECORD_SIZE;
            // 100-byte copy
            for (int b = 0; b < RECORD_SIZE; b += 4) {
                *reinterpret_cast<uint32_t*>(dst + b) =
                    *reinterpret_cast<const uint32_t*>(src + b);
            }
        }
    }
    __syncthreads();

    // ── Step 2: Merge tree — log2(K) levels of 2-way merge path ──
    //
    // Level 0 (K=8): 4 merges, 64 threads each
    //   Merge (seq0, seq1) → out0, (seq2, seq3) → out1, ...
    // Level 1: 2 merges, 128 threads each
    //   Merge (out0, out1) → out2, (out3, out4) → out5
    // Level 2: 1 merge, 256 threads
    //   Merge (out2, out5) → final output

    uint8_t* src_buf = buf_a;
    uint8_t* dst_buf = buf_b;

    // We maintain a "sequence table": for each sequence, (start_record, count)
    // Initially K sequences, halved each level
    __shared__ int s_seq_start_arr[SMMT_K]; // Start record index of each sequence
    __shared__ int s_seq_count_arr[SMMT_K]; // Record count of each sequence

    // Initialize sequence table from source counts
    if (threadIdx.x < SMMT_K) {
        s_seq_start_arr[threadIdx.x] = seq_start[threadIdx.x];
        s_seq_count_arr[threadIdx.x] = part.src_rec_count[threadIdx.x];
    }
    __syncthreads();

    int num_seqs = SMMT_K;

    for (int level = 0; level < SMMT_LEVELS; level++) {
        int num_merges = num_seqs / 2;
        int threads_per_merge = SMMT_BLOCK_THREADS / num_merges;

        int my_merge = threadIdx.x / threads_per_merge;
        int my_tid_in_merge = threadIdx.x % threads_per_merge;

        if (my_merge < num_merges) {
            int seq_a = 2 * my_merge;
            int seq_b = 2 * my_merge + 1;

            int a_start = s_seq_start_arr[seq_a];
            int a_count = s_seq_count_arr[seq_a];
            int b_start = s_seq_start_arr[seq_b];
            int b_count = s_seq_count_arr[seq_b];

            const uint8_t* A = src_buf + a_start * RECORD_SIZE;
            const uint8_t* B = src_buf + b_start * RECORD_SIZE;

            // Output starts at the position of seq_a (merged pair replaces both)
            int out_start = a_start; // Dense packing: merged result goes to a_start
            uint8_t* out = dst_buf + out_start * RECORD_SIZE;

            smem_merge_2way(A, a_count, B, b_count, out,
                           my_tid_in_merge, threads_per_merge);
        }
        __syncthreads();

        // Update sequence table for next level
        if (threadIdx.x < num_merges) {
            int seq_a = 2 * threadIdx.x;
            int seq_b = 2 * threadIdx.x + 1;
            // Merged sequence: starts at seq_a's old start, count = sum
            s_seq_start_arr[threadIdx.x] = s_seq_start_arr[seq_a];
            s_seq_count_arr[threadIdx.x] = s_seq_count_arr[seq_a] + s_seq_count_arr[seq_b];
        }
        __syncthreads();

        num_seqs = num_merges;

        // Swap buffers
        uint8_t* tmp = src_buf;
        src_buf = dst_buf;
        dst_buf = tmp;
    }

    // ── Step 3: Write merged result from shared memory to HBM ──
    // After SMMT_LEVELS iterations, result is in src_buf (which may be buf_a or buf_b)
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const uint8_t* src = src_buf + i * RECORD_SIZE;
        uint8_t* dst = output + part.out_byte_offset + (uint64_t)i * RECORD_SIZE;
        for (int b = 0; b < RECORD_SIZE; b += 4) {
            *reinterpret_cast<uint32_t*>(dst + b) =
                *reinterpret_cast<const uint32_t*>(src + b);
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// KEY-ONLY MERGE: merges keys (KEY_SIZE stride), outputs permutation
// Used by external sort to avoid sending full records over PCIe.
// ────────────────────────────────────────────────────────────────────

__device__ int merge_path_search_keys(
    const uint8_t* __restrict__ A, int a_len,
    const uint8_t* __restrict__ B, int b_len,
    int diag
) {
    int lo = max(0, diag - b_len);
    int hi = min(diag, a_len);
    while (lo < hi) {
        int a_mid = (lo + hi) >> 1;
        int b_mid = diag - 1 - a_mid;
        bool a_greater;
        if (a_mid >= a_len) a_greater = true;
        else if (b_mid < 0 || b_mid >= b_len) a_greater = false;
        else a_greater = (key_compare(A + (uint64_t)a_mid * KEY_SIZE,
                                       B + (uint64_t)b_mid * KEY_SIZE, KEY_SIZE) > 0);
        if (a_greater) hi = a_mid;
        else lo = a_mid + 1;
    }
    return lo;
}

// Descriptor for key-only merge pair
struct KeyMergePair {
    uint64_t a_key_offset;    // byte offset of A keys in d_keys_in
    int      a_count;
    uint64_t b_key_offset;    // byte offset of B keys in d_keys_in
    int      b_count;
    uint64_t out_key_offset;  // byte offset of output keys
    uint64_t out_perm_offset; // element offset of output permutation
    uint64_t a_perm_offset;   // element offset of A's permutation in d_perm_in
    uint64_t b_perm_offset;   // element offset of B's permutation in d_perm_in
    int      first_block;
};

__global__ void merge_keys_only_kernel(
    const uint8_t* __restrict__ keys_in,
    uint8_t* __restrict__ keys_out,
    const uint32_t* __restrict__ perm_in,  // input permutation (carried forward)
    uint32_t* __restrict__ perm_out,       // output permutation
    const KeyMergePair* __restrict__ pairs,
    int num_pairs
) {
    int pair_id = 0;
    { int lo = 0, hi = num_pairs;
      while (lo < hi) { int m = (lo+hi)>>1;
        if (pairs[m].first_block <= (int)blockIdx.x) { pair_id = m; lo = m+1; } else hi = m; } }

    const KeyMergePair& p = pairs[pair_id];
    const uint8_t* A = keys_in + p.a_key_offset;
    const uint8_t* B = keys_in + p.b_key_offset;
    const uint32_t* perm_A = perm_in + p.a_perm_offset;
    const uint32_t* perm_B = perm_in + p.b_perm_offset;
    int total = p.a_count + p.b_count;
    int block_in_pair = blockIdx.x - p.first_block;
    int items_per_block = MP2_ITEMS_PER_THREAD * MP2_BLOCK_THREADS;
    int t_start = block_in_pair * items_per_block + threadIdx.x * MP2_ITEMS_PER_THREAD;
    int t_end = min(t_start + MP2_ITEMS_PER_THREAD, min((block_in_pair+1)*items_per_block, total));
    if (t_start >= total) return;

    int ai = merge_path_search_keys(A, p.a_count, B, p.b_count, t_start);
    int bi = t_start - ai;

    uint8_t* out_keys = keys_out + p.out_key_offset;
    uint32_t* out_perm = perm_out + p.out_perm_offset;

    for (int i = t_start; i < t_end; i++) {
        bool take_a = (ai < p.a_count) && (bi >= p.b_count ||
            key_compare(A + (uint64_t)ai * KEY_SIZE, B + (uint64_t)bi * KEY_SIZE, KEY_SIZE) <= 0);

        const uint8_t* src = take_a ? A + (uint64_t)ai * KEY_SIZE : B + (uint64_t)bi * KEY_SIZE;
        // Carry forward the original global index from the input permutation
        uint32_t global_idx = take_a ? perm_A[ai] : perm_B[bi];
        if (take_a) ai++; else bi++;

        // Write key (10 bytes, byte-by-byte to avoid alignment issues)
        uint8_t* dk = out_keys + (uint64_t)i * KEY_SIZE;
        for (int b = 0; b < KEY_SIZE; b++) dk[b] = src[b];

        // Write permutation index (original global record index)
        out_perm[i] = global_idx;
    }
}

// ────────────────────────────────────────────────────────────────────
// OVC MERGE: merges using 4-byte OVC comparisons (16× less traffic than key merge)
// On OVC tie: compares 8-byte key prefix. On double tie: compare full keys via perm.
// ────────────────────────────────────────────────────────────────────

struct OvcMergePair {
    uint64_t a_ovc_offset;    // element offset of A's OVCs
    int      a_count;
    uint64_t b_ovc_offset;    // element offset of B's OVCs
    int      b_count;
    uint64_t out_ovc_offset;  // element offset of output OVCs
    uint64_t out_perm_offset; // element offset of output perm
    uint64_t a_perm_offset;
    uint64_t b_perm_offset;
    uint64_t a_prefix_offset; // element offset of A's 8B prefixes
    uint64_t b_prefix_offset;
    uint64_t out_prefix_offset;
    int      first_block;
};

// OVC merge-path binary search: compare by OVC, then prefix
__device__ int merge_path_search_ovc(
    const uint32_t* __restrict__ A_ovcs, const uint64_t* __restrict__ A_pfx, int a_len,
    const uint32_t* __restrict__ B_ovcs, const uint64_t* __restrict__ B_pfx, int b_len,
    int diag
) {
    int lo = max(0, diag - b_len);
    int hi = min(diag, a_len);
    while (lo < hi) {
        int a_mid = (lo + hi) >> 1;
        int b_mid = diag - 1 - a_mid;
        bool a_greater;
        if (a_mid >= a_len) a_greater = true;
        else if (b_mid < 0 || b_mid >= b_len) a_greater = false;
        else {
            uint32_t oa = A_ovcs[a_mid], ob = B_ovcs[b_mid];
            if (oa != ob) a_greater = (oa > ob);
            else {
                uint64_t pa = A_pfx[a_mid], pb = B_pfx[b_mid];
                a_greater = (pa > pb);
                // On double tie (very rare): treat as equal → take A (stable)
            }
        }
        if (a_greater) hi = a_mid;
        else lo = a_mid + 1;
    }
    return lo;
}

__global__ void merge_ovc_kernel(
    const uint32_t* __restrict__ ovcs_in,
    uint32_t* __restrict__ ovcs_out,
    const uint64_t* __restrict__ pfx_in,
    uint64_t* __restrict__ pfx_out,
    const uint32_t* __restrict__ perm_in,
    uint32_t* __restrict__ perm_out,
    const OvcMergePair* __restrict__ pairs,
    int num_pairs
) {
    int pair_id = 0;
    { int lo = 0, hi = num_pairs;
      while (lo < hi) { int m = (lo+hi)>>1;
        if (pairs[m].first_block <= (int)blockIdx.x) { pair_id = m; lo = m+1; } else hi = m; } }

    const OvcMergePair& p = pairs[pair_id];
    const uint32_t* A_ovc = ovcs_in + p.a_ovc_offset;
    const uint32_t* B_ovc = ovcs_in + p.b_ovc_offset;
    const uint64_t* A_pfx = pfx_in + p.a_prefix_offset;
    const uint64_t* B_pfx = pfx_in + p.b_prefix_offset;
    const uint32_t* A_perm = perm_in + p.a_perm_offset;
    const uint32_t* B_perm = perm_in + p.b_perm_offset;
    int total = p.a_count + p.b_count;
    int block_in_pair = blockIdx.x - p.first_block;
    int items_per_block = MP2_ITEMS_PER_THREAD * MP2_BLOCK_THREADS;
    int t_start = block_in_pair * items_per_block + threadIdx.x * MP2_ITEMS_PER_THREAD;
    int t_end = min(t_start + MP2_ITEMS_PER_THREAD, min((block_in_pair+1)*items_per_block, total));
    if (t_start >= total) return;

    int ai = merge_path_search_ovc(A_ovc, A_pfx, p.a_count, B_ovc, B_pfx, p.b_count, t_start);
    int bi = t_start - ai;

    uint32_t* out_ovc = ovcs_out + p.out_ovc_offset;
    uint64_t* out_pfx = pfx_out + p.out_prefix_offset;
    uint32_t* out_perm = perm_out + p.out_perm_offset;

    for (int i = t_start; i < t_end; i++) {
        bool take_a = (ai < p.a_count) && (bi >= p.b_count || ({
            uint32_t oa = A_ovc[ai], ob = B_ovc[bi];
            (oa < ob) || (oa == ob && A_pfx[ai] <= B_pfx[bi]);
        }));

        out_ovc[i]  = take_a ? A_ovc[ai]  : B_ovc[bi];
        out_pfx[i]  = take_a ? A_pfx[ai]  : B_pfx[bi];
        out_perm[i] = take_a ? A_perm[ai] : B_perm[bi];
        if (take_a) ai++; else bi++;
    }
}

extern "C" void launch_merge_ovc(
    const uint32_t* ovcs_in, uint32_t* ovcs_out,
    const uint64_t* pfx_in, uint64_t* pfx_out,
    const uint32_t* perm_in, uint32_t* perm_out,
    const OvcMergePair* d_pairs, int num_pairs, int total_blocks,
    cudaStream_t stream
) {
    if (total_blocks > 0) {
        merge_ovc_kernel<<<total_blocks, MP2_BLOCK_THREADS, 0, stream>>>(
            ovcs_in, ovcs_out, pfx_in, pfx_out, perm_in, perm_out, d_pairs, num_pairs);
    }
}

// ── Host interfaces ────────────────────────────────────────────────

extern "C" void launch_merge_keys_only(
    const uint8_t* d_keys_in, uint8_t* d_keys_out,
    const uint32_t* d_perm_in, uint32_t* d_perm_out,
    const KeyMergePair* d_pairs, int num_pairs, int total_blocks,
    cudaStream_t stream
) {
    if (total_blocks > 0) {
        merge_keys_only_kernel<<<total_blocks, MP2_BLOCK_THREADS, 0, stream>>>(
            d_keys_in, d_keys_out, d_perm_in, d_perm_out, d_pairs, num_pairs);
    }
}

extern "C" void launch_merge_2way(
    const uint8_t* d_input, uint8_t* d_output,
    const PairDesc2Way* d_pairs, int num_pairs, int total_blocks,
    cudaStream_t stream
) {
    if (total_blocks > 0) {
        merge_2way_kernel<<<total_blocks, MP2_BLOCK_THREADS, 0, stream>>>(
            d_input, d_output, d_pairs, num_pairs);
    }
}

extern "C" void launch_merge_kway(
    const uint8_t* d_input, uint8_t* d_output,
    const KWayPartition* d_partitions, int num_partitions,
    int max_records_per_partition,
    cudaStream_t stream
) {
    if (num_partitions > 0) {
        // Dynamic shared memory: 2 × max_records × RECORD_SIZE
        int smem_bytes = 2 * max_records_per_partition * RECORD_SIZE;
        // Opt-in for extended shared memory (>48KB on sm_75+)
        cudaFuncSetAttribute(
            (const void*)smem_kway_merge_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        smem_kway_merge_kernel<<<num_partitions, SMMT_BLOCK_THREADS, smem_bytes, stream>>>(
            d_input, d_output, d_partitions);
    }
}
