// ============================================================================
// KV-Separation Merge Benchmark
//
// Measures the bandwidth savings of separating keys from values during merge.
// Standard merge reads 100B per comparison (wastes 90B of value data).
// KV-separate merge reads 10B keys for comparison, then scatters 90B values.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 -I../include kv_separate_bench.cu \
//        -o kv_separate_bench && ./kv_separate_bench [num_records]
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "../include/record.cuh"

// ── Merge-path binary search on full 100B records ────────────────────

__device__ int merge_path_standard(
    const uint8_t* A, int a_len,
    const uint8_t* B, int b_len,
    int diag)
{
    int lo = max(0, diag - b_len);
    int hi = min(diag, a_len);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int bmid = diag - 1 - mid;
        if (bmid >= 0 && bmid < b_len) {
            SortKey ka = make_sort_key(A + (uint64_t)mid * RECORD_SIZE);
            SortKey kb = make_sort_key(B + (uint64_t)bmid * RECORD_SIZE);
            if (ka > kb) hi = mid;
            else         lo = mid + 1;
        } else lo = mid + 1;
    }
    return lo;
}

// ── Standard merge kernel: reads full 100B records for comparison ────

__global__ void merge_standard(
    const uint8_t* __restrict__ A, int a_len,
    const uint8_t* __restrict__ B, int b_len,
    uint8_t* __restrict__ out,
    int items_per_thread)
{
    int total = a_len + b_len;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t_start = tid * items_per_thread;
    int t_end = min(t_start + items_per_thread, total);
    if (t_start >= total) return;

    int ai = merge_path_standard(A, a_len, B, b_len, t_start);
    int bi = t_start - ai;

    for (int i = t_start; i < t_end; i++) {
        bool take_a;
        if      (ai >= a_len)  take_a = false;
        else if (bi >= b_len)  take_a = true;
        else {
            SortKey ka = make_sort_key(A + (uint64_t)ai * RECORD_SIZE);
            SortKey kb = make_sort_key(B + (uint64_t)bi * RECORD_SIZE);
            take_a = (ka <= kb);
        }
        const uint8_t* src = take_a
            ? A + (uint64_t)ai * RECORD_SIZE
            : B + (uint64_t)bi * RECORD_SIZE;
        uint8_t* dst = out + (uint64_t)i * RECORD_SIZE;
        // Copy full 100B record
        for (int b = 0; b < RECORD_SIZE; b += 4)
            *(uint32_t*)(dst + b) = *(const uint32_t*)(src + b);
        if (take_a) ai++; else bi++;
    }
}

// ── Merge-path binary search on KEY_SIZE-stride key arrays ───────────

__device__ int merge_path_keys(
    const uint8_t* A, int a_len,
    const uint8_t* B, int b_len,
    int diag)
{
    int lo = max(0, diag - b_len);
    int hi = min(diag, a_len);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int bmid = diag - 1 - mid;
        if (bmid >= 0 && bmid < b_len) {
            SortKey ka = make_sort_key(A + (uint64_t)mid * KEY_SIZE);
            SortKey kb = make_sort_key(B + (uint64_t)bmid * KEY_SIZE);
            if (ka > kb) hi = mid;
            else         lo = mid + 1;
        } else lo = mid + 1;
    }
    return lo;
}

// ── Phase 1: merge keys only, emit permutation (source_indices) ──────

__global__ void merge_keys_only(
    const uint8_t* __restrict__ keys_a, int a_len,
    const uint8_t* __restrict__ keys_b, int b_len,
    uint8_t* __restrict__ merged_keys,
    int* __restrict__ source_indices,   // <a_len → index in A, >=a_len → offset into B
    int items_per_thread)
{
    int total = a_len + b_len;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t_start = tid * items_per_thread;
    int t_end = min(t_start + items_per_thread, total);
    if (t_start >= total) return;

    int ai = merge_path_keys(keys_a, a_len, keys_b, b_len, t_start);
    int bi = t_start - ai;

    for (int i = t_start; i < t_end; i++) {
        bool take_a;
        if      (ai >= a_len)  take_a = false;
        else if (bi >= b_len)  take_a = true;
        else {
            SortKey ka = make_sort_key(keys_a + (uint64_t)ai * KEY_SIZE);
            SortKey kb = make_sort_key(keys_b + (uint64_t)bi * KEY_SIZE);
            take_a = (ka <= kb);
        }
        const uint8_t* src = take_a
            ? keys_a + (uint64_t)ai * KEY_SIZE
            : keys_b + (uint64_t)bi * KEY_SIZE;
        uint8_t* dst = merged_keys + (uint64_t)i * KEY_SIZE;
        // Copy 10B key (two 4B + one 2B)
        *(uint32_t*)(dst + 0) = *(const uint32_t*)(src + 0);
        *(uint32_t*)(dst + 4) = *(const uint32_t*)(src + 4);
        *(uint16_t*)(dst + 8) = *(const uint16_t*)(src + 8);
        // Record source: encode as index into combined [A|B] space
        source_indices[i] = take_a ? ai : (a_len + bi);
        if (take_a) ai++; else bi++;
    }
}

// ── Phase 2: scatter values by source_indices ────────────────────────

__global__ void scatter_values(
    const uint8_t* __restrict__ vals_a,
    const uint8_t* __restrict__ vals_b,
    const int* __restrict__ source_indices,
    uint8_t* __restrict__ merged_vals,
    int a_len, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int src_idx = source_indices[i];
    const uint8_t* src = (src_idx < a_len)
        ? vals_a + (uint64_t)src_idx * VALUE_SIZE
        : vals_b + (uint64_t)(src_idx - a_len) * VALUE_SIZE;
    uint8_t* dst = merged_vals + (uint64_t)i * VALUE_SIZE;
    // Copy 90B value (22 x 4B + 2B padding — just do 92B aligned reads are fine,
    // but safer to do exactly VALUE_SIZE)
    for (int b = 0; b < VALUE_SIZE - 3; b += 4)
        *(uint32_t*)(dst + b) = *(const uint32_t*)(src + b);
    // Last 2 bytes (90 = 22*4 + 2)
    *(uint16_t*)(dst + 88) = *(const uint16_t*)(src + 88);
}

// ── Host helpers ─────────────────────────────────────────────────────

static void fill_sorted_run(uint8_t* h_data, int n, int stride, int offset) {
    for (int i = 0; i < n; i++) {
        uint8_t* rec = h_data + (uint64_t)i * RECORD_SIZE;
        uint64_t val = (uint64_t)(i * stride + offset);
        // Write big-endian key
        for (int b = 0; b < 8; b++)
            rec[7 - b] = (uint8_t)((val >> (b * 8)) & 0xFF);
        rec[8] = rec[9] = 0;
        // Fill value with pattern
        uint32_t tag = (uint32_t)i;
        memcpy(rec + KEY_SIZE, &tag, sizeof(tag));
    }
}

// ── Main benchmark ───────────────────────────────────────────────────

int main(int argc, char** argv) {
    int num_records = 10000000;
    if (argc > 1) num_records = atoi(argv[1]);
    int half = num_records / 2;
    int total = half * 2; // ensure even

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("================================================================\n");
    printf("  KV-Separation Merge Benchmark\n");
    printf("================================================================\n");
    printf("GPU: %s\n", props.name);
    printf("Records: %d (each %dB = %dB key + %dB value)\n",
           total, RECORD_SIZE, KEY_SIZE, VALUE_SIZE);
    printf("Two pre-sorted runs of %d records each, merged into %d.\n\n", half, total);

    // Allocate and fill host data: two interleaved sorted runs
    uint64_t run_bytes = (uint64_t)half * RECORD_SIZE;
    uint8_t* h_a = (uint8_t*)malloc(run_bytes);
    uint8_t* h_b = (uint8_t*)malloc(run_bytes);
    fill_sorted_run(h_a, half, 2, 0);  // even keys
    fill_sorted_run(h_b, half, 2, 1);  // odd keys

    // ── Standard layout: upload as interleaved [key|val] records ──────
    uint8_t *d_std_a, *d_std_b, *d_std_out;
    CUDA_CHECK(cudaMalloc(&d_std_a,   run_bytes));
    CUDA_CHECK(cudaMalloc(&d_std_b,   run_bytes));
    CUDA_CHECK(cudaMalloc(&d_std_out, (uint64_t)total * RECORD_SIZE));
    CUDA_CHECK(cudaMemcpy(d_std_a, h_a, run_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std_b, h_b, run_bytes, cudaMemcpyHostToDevice));

    // ── Separated layout: split into key and value arrays ────────────
    uint64_t keys_bytes = (uint64_t)half * KEY_SIZE;
    uint64_t vals_bytes = (uint64_t)half * VALUE_SIZE;
    uint8_t *d_keys_a, *d_keys_b, *d_keys_out;
    uint8_t *d_vals_a, *d_vals_b, *d_vals_out;
    int *d_indices;

    CUDA_CHECK(cudaMalloc(&d_keys_a,   keys_bytes));
    CUDA_CHECK(cudaMalloc(&d_keys_b,   keys_bytes));
    CUDA_CHECK(cudaMalloc(&d_keys_out, (uint64_t)total * KEY_SIZE));
    CUDA_CHECK(cudaMalloc(&d_vals_a,   vals_bytes));
    CUDA_CHECK(cudaMalloc(&d_vals_b,   vals_bytes));
    CUDA_CHECK(cudaMalloc(&d_vals_out, (uint64_t)total * VALUE_SIZE));
    CUDA_CHECK(cudaMalloc(&d_indices,  (uint64_t)total * sizeof(int)));

    // Extract keys and values from interleaved host data
    uint8_t* h_keys_a = (uint8_t*)malloc(keys_bytes);
    uint8_t* h_vals_a = (uint8_t*)malloc(vals_bytes);
    uint8_t* h_keys_b = (uint8_t*)malloc(keys_bytes);
    uint8_t* h_vals_b = (uint8_t*)malloc(vals_bytes);
    for (int i = 0; i < half; i++) {
        memcpy(h_keys_a + (uint64_t)i * KEY_SIZE,   h_a + (uint64_t)i * RECORD_SIZE, KEY_SIZE);
        memcpy(h_vals_a + (uint64_t)i * VALUE_SIZE,  h_a + (uint64_t)i * RECORD_SIZE + KEY_SIZE, VALUE_SIZE);
        memcpy(h_keys_b + (uint64_t)i * KEY_SIZE,   h_b + (uint64_t)i * RECORD_SIZE, KEY_SIZE);
        memcpy(h_vals_b + (uint64_t)i * VALUE_SIZE,  h_b + (uint64_t)i * RECORD_SIZE + KEY_SIZE, VALUE_SIZE);
    }
    CUDA_CHECK(cudaMemcpy(d_keys_a, h_keys_a, keys_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keys_b, h_keys_b, keys_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals_a, h_vals_a, vals_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals_b, h_vals_b, vals_bytes, cudaMemcpyHostToDevice));

    // ── Launch config ────────────────────────────────────────────────
    const int IPT = MERGE_ITEMS_PER_THREAD_CFG;  // items per thread
    const int BLK = MERGE_BLOCK_THREADS_CFG;      // threads per block
    int total_threads = (total + IPT - 1) / IPT;
    int grid = (total_threads + BLK - 1) / BLK;
    int scatter_grid = (total + BLK - 1) / BLK;

    GpuTimer timer;
    const int TRIALS = 10;

    // ── Warmup ───────────────────────────────────────────────────────
    merge_standard<<<grid, BLK>>>(d_std_a, half, d_std_b, half, d_std_out, IPT);
    merge_keys_only<<<grid, BLK>>>(d_keys_a, half, d_keys_b, half,
                                    d_keys_out, d_indices, IPT);
    scatter_values<<<scatter_grid, BLK>>>(d_vals_a, d_vals_b, d_indices,
                                           d_vals_out, half, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Benchmark standard merge ─────────────────────────────────────
    float std_ms_total = 0;
    for (int t = 0; t < TRIALS; t++) {
        timer.begin();
        merge_standard<<<grid, BLK>>>(d_std_a, half, d_std_b, half, d_std_out, IPT);
        std_ms_total += timer.end();
    }
    float std_ms = std_ms_total / TRIALS;
    double std_read_gb  = (2.0 * run_bytes) / 1e9;  // read both runs
    double std_write_gb = ((uint64_t)total * RECORD_SIZE) / 1e9;
    double std_total_gb = std_read_gb + std_write_gb;
    double std_bw = std_total_gb / (std_ms / 1000.0);

    // ── Benchmark KV-separate merge ──────────────────────────────────
    float kv_key_ms_total = 0, kv_scatter_ms_total = 0;
    for (int t = 0; t < TRIALS; t++) {
        timer.begin();
        merge_keys_only<<<grid, BLK>>>(d_keys_a, half, d_keys_b, half,
                                        d_keys_out, d_indices, IPT);
        kv_key_ms_total += timer.end();

        timer.begin();
        scatter_values<<<scatter_grid, BLK>>>(d_vals_a, d_vals_b, d_indices,
                                               d_vals_out, half, total);
        kv_scatter_ms_total += timer.end();
    }
    float kv_key_ms = kv_key_ms_total / TRIALS;
    float kv_scat_ms = kv_scatter_ms_total / TRIALS;
    float kv_total_ms = kv_key_ms + kv_scat_ms;

    double kv_key_read_gb   = (2.0 * keys_bytes) / 1e9;
    double kv_key_write_gb  = ((uint64_t)total * KEY_SIZE) / 1e9;
    double kv_idx_write_gb  = ((uint64_t)total * sizeof(int)) / 1e9;
    double kv_scat_read_gb  = (2.0 * vals_bytes + (uint64_t)total * sizeof(int)) / 1e9;
    double kv_scat_write_gb = ((uint64_t)total * VALUE_SIZE) / 1e9;
    double kv_merge_traffic = kv_key_read_gb + kv_key_write_gb + kv_idx_write_gb;
    double kv_scat_traffic  = kv_scat_read_gb + kv_scat_write_gb;
    double kv_total_traffic = kv_merge_traffic + kv_scat_traffic;

    // ── Results ──────────────────────────────────────────────────────
    printf("── Standard Merge (100B records) ──\n");
    printf("  Time:      %.2f ms\n", std_ms);
    printf("  Traffic:   %.3f GB (read %.3f + write %.3f)\n",
           std_total_gb, std_read_gb, std_write_gb);
    printf("  Bandwidth: %.1f GB/s\n\n", std_bw);

    printf("── KV-Separate Merge ──\n");
    printf("  Phase 1 (key merge):    %.2f ms  traffic %.3f GB  BW %.1f GB/s\n",
           kv_key_ms, kv_merge_traffic, kv_merge_traffic / (kv_key_ms / 1000.0));
    printf("  Phase 2 (value scatter): %.2f ms  traffic %.3f GB  BW %.1f GB/s\n",
           kv_scat_ms, kv_scat_traffic, kv_scat_traffic / (kv_scat_ms / 1000.0));
    printf("  Total:     %.2f ms  traffic %.3f GB\n", kv_total_ms, kv_total_traffic);
    printf("  Bandwidth: %.1f GB/s (effective)\n\n", kv_total_traffic / (kv_total_ms / 1000.0));

    printf("── Comparison ──\n");
    printf("  Standard:    %.2f ms\n", std_ms);
    printf("  KV-separate: %.2f ms (keys %.2f + scatter %.2f)\n",
           kv_total_ms, kv_key_ms, kv_scat_ms);
    printf("  Speedup:     %.2fx\n", std_ms / kv_total_ms);
    printf("  Key merge is %.1fx faster than standard merge alone\n",
           std_ms / kv_key_ms);
    printf("  Scatter overhead: %.2f ms (%.0f%% of key merge time)\n\n",
           kv_scat_ms, 100.0 * kv_scat_ms / kv_key_ms);

    printf("── Insight ──\n");
    printf("  Standard merge comparison reads %dB per element (wastes %dB value data)\n",
           RECORD_SIZE, VALUE_SIZE);
    printf("  KV-separate comparison reads %dB per element (%dx less)\n",
           KEY_SIZE, RECORD_SIZE / KEY_SIZE);
    printf("  Trade-off: extra scatter pass reads values + indices\n");
    printf("  Net win depends on whether merge is bandwidth-bound (likely yes)\n");

    // Cleanup
    free(h_a); free(h_b);
    free(h_keys_a); free(h_vals_a); free(h_keys_b); free(h_vals_b);
    cudaFree(d_std_a); cudaFree(d_std_b); cudaFree(d_std_out);
    cudaFree(d_keys_a); cudaFree(d_keys_b); cudaFree(d_keys_out);
    cudaFree(d_vals_a); cudaFree(d_vals_b); cudaFree(d_vals_out);
    cudaFree(d_indices);
    return 0;
}
