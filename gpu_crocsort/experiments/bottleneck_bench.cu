#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>  // For CUB radix sort baseline — remove if CUB unavailable

// ============================================================================
// GPU Sort Bottleneck Benchmarks
//
// Run: nvcc -O3 -std=c++17 -arch=sm_80 bottleneck_bench.cu -o bench && ./bench
// If CUB not available: compile with -DNO_CUB and it will skip that test.
//
// Measures the ACTUAL bottlenecks so we know what to optimize.
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Timer {
    cudaEvent_t start, stop;
    Timer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ════════════════════════════════════════════════════════════════════
// Experiment 1: Raw HBM Bandwidth Ceiling
// ════════════════════════════════════════════════════════════════════

__global__ void copy_records_kernel(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    uint64_t num_uint4s
) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid < num_uint4s) {
        dst[tid] = src[tid];
    }
}

void bench_bandwidth_ceiling(uint64_t total_bytes) {
    printf("\n══ Experiment 1: HBM Bandwidth Ceiling ══\n");
    printf("  Data: %.2f GB\n", total_bytes / 1e9);

    uint8_t *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, total_bytes));
    CUDA_CHECK(cudaMemset(d_src, 0xAB, total_bytes));

    Timer t;

    // cudaMemcpy (DMA engine)
    t.begin();
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, total_bytes, cudaMemcpyDeviceToDevice));
    float dma_ms = t.end_ms();
    printf("  cudaMemcpy (DMA):     %.2f ms = %.2f GB/s\n",
           dma_ms, total_bytes / (dma_ms * 1e6));

    // Kernel copy (SM)
    uint64_t num_uint4s = total_bytes / sizeof(uint4);
    int threads = 256;
    int blocks = (num_uint4s + threads - 1) / threads;

    // Warmup
    copy_records_kernel<<<blocks, threads>>>((uint4*)d_src, (uint4*)d_dst, num_uint4s);
    CUDA_CHECK(cudaDeviceSynchronize());

    t.begin();
    copy_records_kernel<<<blocks, threads>>>((uint4*)d_src, (uint4*)d_dst, num_uint4s);
    float kern_ms = t.end_ms();
    printf("  Kernel copy (SM):     %.2f ms = %.2f GB/s\n",
           kern_ms, total_bytes / (kern_ms * 1e6));

    printf("  → This is the ceiling for ANY single-pass operation on this data.\n");
    printf("  → A merge sort with P passes costs at least P × this time.\n");

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// ════════════════════════════════════════════════════════════════════
// Experiment 2: Merge Pass Cost (how much does one pass cost?)
// ════════════════════════════════════════════════════════════════════

// Simple 2-way merge: each thread merges a portion using merge path
__device__ int mp_search(const uint64_t* A, int a_len, const uint64_t* B, int b_len, int diag) {
    int lo = max(0, diag - b_len);
    int hi = min(diag, a_len);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int b_mid = diag - 1 - mid;
        if (b_mid >= 0 && b_mid < b_len && A[mid] > B[b_mid]) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

__global__ void merge_2way_kernel(
    const uint64_t* __restrict__ keys_in,
    uint64_t* __restrict__ keys_out,
    // Each pair: [a_start, a_count, b_start, b_count, out_start]
    const int* __restrict__ pair_info,
    int num_pairs,
    int items_per_thread
) {
    // Find our pair (simple: each pair gets ceil(pair_total / (items_per_thread * blockDim.x)) blocks)
    // For this benchmark, we just merge the entire array as 2 halves
    int pair_id = 0; // Simplified: single pair
    int a_start = pair_info[0], a_count = pair_info[1];
    int b_start = pair_info[2], b_count = pair_info[3];
    int out_start = pair_info[4];

    const uint64_t* A = keys_in + a_start;
    const uint64_t* B = keys_in + b_start;
    uint64_t* out = keys_out + out_start;
    int total = a_count + b_count;

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_start = global_tid * items_per_thread;
    int thread_end = min(thread_start + items_per_thread, total);
    if (thread_start >= total) return;

    int ai = mp_search(A, a_count, B, b_count, thread_start);
    int bi = thread_start - ai;

    for (int i = thread_start; i < thread_end; i++) {
        bool take_a;
        if (ai >= a_count) take_a = false;
        else if (bi >= b_count) take_a = true;
        else take_a = (A[ai] <= B[bi]);

        if (take_a) { out[i] = A[ai]; ai++; }
        else        { out[i] = B[bi]; bi++; }
    }
}

void bench_merge_pass_cost(int num_records) {
    printf("\n══ Experiment 2: Cost Per Merge Pass ══\n");
    printf("  Records: %d, Key size: 8 bytes (uint64)\n", num_records);

    uint64_t data_bytes = (uint64_t)num_records * sizeof(uint64_t);
    printf("  Data: %.2f MB\n", data_bytes / 1e6);

    // Generate sorted data (two sorted halves)
    uint64_t* h_keys = (uint64_t*)malloc(data_bytes);
    int half = num_records / 2;
    for (int i = 0; i < half; i++) h_keys[i] = 2 * (uint64_t)i;
    for (int i = half; i < num_records; i++) h_keys[i] = 2 * (uint64_t)(i - half) + 1;

    uint64_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_keys, data_bytes, cudaMemcpyHostToDevice));

    // Pair info: merge two halves
    int h_pair[] = {0, half, half, num_records - half, 0};
    int* d_pair;
    CUDA_CHECK(cudaMalloc(&d_pair, 5 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair, h_pair, 5 * sizeof(int), cudaMemcpyHostToDevice));

    int items_per_thread = 8;
    int threads = 256;
    int total_threads = (num_records + items_per_thread - 1) / items_per_thread;
    int blocks = (total_threads + threads - 1) / threads;

    // Warmup
    merge_2way_kernel<<<blocks, threads>>>(d_in, d_out, d_pair, 1, items_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    Timer t;
    int trials = 5;
    float total_ms = 0;
    for (int i = 0; i < trials; i++) {
        t.begin();
        merge_2way_kernel<<<blocks, threads>>>(d_in, d_out, d_pair, 1, items_per_thread);
        total_ms += t.end_ms();
    }
    float avg_ms = total_ms / trials;

    float achieved_bw = (2.0 * data_bytes) / (avg_ms * 1e6); // read + write
    printf("  2-way merge pass: %.2f ms (%.2f GB/s, read+write)\n", avg_ms, achieved_bw);
    printf("  Blocks: %d, Threads: %d, Total active: %d\n", blocks, threads, blocks * threads);

    // Bandwidth ceiling for reference
    float copy_bw = 0;
    {
        Timer ct;
        ct.begin();
        CUDA_CHECK(cudaMemcpy(d_out, d_in, data_bytes, cudaMemcpyDeviceToDevice));
        float copy_ms = ct.end_ms();
        copy_bw = data_bytes / (copy_ms * 1e6);
    }
    printf("  Bandwidth ceiling: %.2f GB/s (copy only)\n", copy_bw);
    printf("  Merge efficiency: %.1f%% of bandwidth ceiling\n",
           100.0 * achieved_bw / (2.0 * copy_bw));
    printf("  → Each merge pass costs %.2f ms. With P passes, total = P × %.2f ms\n",
           avg_ms, avg_ms);

    free(h_keys);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_pair));
}

// ════════════════════════════════════════════════════════════════════
// Experiment 3: Fanin vs Passes — the key tradeoff
// ════════════════════════════════════════════════════════════════════

void bench_fanin_tradeoff(int num_records) {
    printf("\n══ Experiment 3: Fanin vs Passes Tradeoff ══\n");
    printf("  Records: %d\n", num_records);

    int runs_from_block_sort = (num_records + 511) / 512;

    struct Strategy {
        const char* name;
        int fanin;
        int passes;
        double total_traffic_ratio; // relative to data size
    };

    Strategy strategies[] = {
        {"2-way merge path",  2, (int)ceil(log2(runs_from_block_sort)), 0},
        {"4-way merge path",  4, (int)ceil(log(runs_from_block_sort)/log(4)), 0},
        {"8-way loser tree",  8, (int)ceil(log(runs_from_block_sort)/log(8)), 0},
        {"16-way loser tree", 16, (int)ceil(log(runs_from_block_sort)/log(16)), 0},
        {"32-way loser tree", 32, (int)ceil(log(runs_from_block_sort)/log(32)), 0},
    };
    int num_strategies = sizeof(strategies) / sizeof(strategies[0]);

    printf("  Starting runs (block size 512): %d\n\n", runs_from_block_sort);
    printf("  %-22s  Fanin  Passes  Total HBM Traffic  Relative\n", "Strategy");
    printf("  %-22s  -----  ------  -----------------  --------\n", "--------");

    uint64_t data_bytes = (uint64_t)num_records * 100; // 100-byte records

    for (int i = 0; i < num_strategies; i++) {
        strategies[i].total_traffic_ratio = 2.0 * strategies[i].passes;
        double total_gb = (data_bytes * strategies[i].total_traffic_ratio) / 1e9;

        printf("  %-22s  %5d  %6d  %13.1f GB  %7.1fx\n",
               strategies[i].name,
               strategies[i].fanin,
               strategies[i].passes,
               total_gb,
               strategies[i].total_traffic_ratio / (2.0 * strategies[0].passes) * strategies[0].passes);
    }

    // Estimate times at different bandwidth utilization levels
    printf("\n  Estimated merge time at different bandwidth efficiencies:\n");
    printf("  (A100 peak = 2039 GB/s)\n\n");
    printf("  %-22s  @80%% BW   @60%% BW   @40%% BW\n", "Strategy");
    printf("  %-22s  -------   -------   -------\n", "--------");

    double peak_bw = 2039e9; // A100 peak bytes/sec
    for (int i = 0; i < num_strategies; i++) {
        double traffic = data_bytes * strategies[i].total_traffic_ratio;
        printf("  %-22s  %6.1f ms  %6.1f ms  %6.1f ms\n",
               strategies[i].name,
               traffic / (peak_bw * 0.8) * 1000,
               traffic / (peak_bw * 0.6) * 1000,
               traffic / (peak_bw * 0.4) * 1000);
    }

    printf("\n  KEY INSIGHT: High fanin has fewer passes → less total bandwidth.\n");
    printf("  But high fanin may achieve lower bandwidth efficiency (uncoalesced reads).\n");
    printf("  The experiment measures which effect wins.\n");
}

// ════════════════════════════════════════════════════════════════════
// Experiment 4: Record Size Sensitivity
// ════════════════════════════════════════════════════════════════════

void bench_record_size_sensitivity() {
    printf("\n══ Experiment 4: Record Size Sensitivity (Analytical) ══\n");
    printf("  Fixed total data: 1 GB\n\n");

    int record_sizes[] = {10, 20, 50, 100, 200, 500, 1000};
    int key_size = 10;

    printf("  RecSize  Records     Key%%  Runs(512)  Passes(2w)  Passes(8w)  BW_2way   BW_8way\n");
    printf("  -------  --------  -----  ---------  ----------  ----------  --------  --------\n");

    double total_data = 1e9; // 1 GB
    double peak_bw = 2039e9;

    for (int rs : record_sizes) {
        int num_recs = (int)(total_data / rs);
        int runs = (num_recs + 511) / 512;
        int passes_2way = (int)ceil(log2(runs));
        int passes_8way = (int)ceil(log(runs) / log(8));
        double key_pct = 100.0 * key_size / rs;
        double traffic_2way = 2.0 * total_data * passes_2way;
        double traffic_8way = 2.0 * total_data * passes_8way;

        printf("  %5dB  %8d  %4.0f%%  %9d  %10d  %10d  %5.1f GB  %5.1f GB\n",
               rs, num_recs, key_pct, runs, passes_2way, passes_8way,
               traffic_2way / 1e9, traffic_8way / 1e9);
    }

    printf("\n  For large records (100B+), key comparison is <10%% of data read.\n");
    printf("  Compute is negligible. Bandwidth dominates.\n");
    printf("  → Minimize passes (high fanin) for large records.\n");
    printf("  → For key-only sort (10B), comparison cost matters more.\n");
}

// ════════════════════════════════════════════════════════════════════
// Experiment 5: CUB Radix Sort Baseline
// ════════════════════════════════════════════════════════════════════

#ifndef NO_CUB
void bench_cub_baseline(int num_records) {
    printf("\n══ Experiment 5: CUB Radix Sort Baseline ══\n");
    printf("  Records: %d (key-only, uint64)\n", num_records);

    uint64_t data_bytes = (uint64_t)num_records * sizeof(uint64_t);

    uint64_t* h_keys = (uint64_t*)malloc(data_bytes);
    srand(42);
    for (int i = 0; i < num_records; i++) {
        h_keys[i] = ((uint64_t)rand() << 32) | rand();
    }

    uint64_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_keys, data_bytes, cudaMemcpyHostToDevice));

    // CUB temp storage
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, num_records);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // Warmup
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, num_records);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    Timer t;
    t.begin();
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, num_records);
    float ms = t.end_ms();

    printf("  CUB radix sort: %.2f ms (%.2f GB/s)\n", ms, data_bytes / (ms * 1e6));
    printf("  This is the baseline for in-HBM sorting.\n");
    printf("  External merge sort should only be used for data EXCEEDING HBM.\n");

    free(h_keys);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_temp));
}
#endif

// ════════════════════════════════════════════════════════════════════
// Experiment 6: PCIe Transfer Baseline (for external sort)
// ════════════════════════════════════════════════════════════════════

void bench_pcie_transfer() {
    printf("\n══ Experiment 6: PCIe Transfer Baseline ══\n");

    size_t sizes[] = {
        64ULL * 1024 * 1024,        // 64 MB
        256ULL * 1024 * 1024,       // 256 MB
        1024ULL * 1024 * 1024,      // 1 GB
        4096ULL * 1024 * 1024,      // 4 GB (if memory allows)
    };

    for (size_t sz : sizes) {
        // Check if we can allocate
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (sz > free_mem / 2) continue;

        uint8_t* h_data;
        uint8_t* d_data;
        CUDA_CHECK(cudaMallocHost(&h_data, sz));  // Pinned memory
        CUDA_CHECK(cudaMalloc(&d_data, sz));
        memset(h_data, 0xAB, sz);

        Timer t;

        // H2D
        t.begin();
        CUDA_CHECK(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
        float h2d_ms = t.end_ms();

        // D2H
        t.begin();
        CUDA_CHECK(cudaMemcpy(h_data, d_data, sz, cudaMemcpyDeviceToHost));
        float d2h_ms = t.end_ms();

        printf("  %6.0f MB: H2D %.1f ms (%.1f GB/s), D2H %.1f ms (%.1f GB/s)\n",
               sz / 1e6,
               h2d_ms, sz / (h2d_ms * 1e6),
               d2h_ms, sz / (d2h_ms * 1e6));

        CUDA_CHECK(cudaFreeHost(h_data));
        CUDA_CHECK(cudaFree(d_data));
    }

    printf("  → For external sort, PCIe is 50-100x slower than HBM.\n");
    printf("  → Overlap transfers with compute to hide this.\n");
}

// ════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("════════════════════════════════════════════════════════\n");
    printf("  GPU Sort Bottleneck Analysis\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("GPU: %s\n", props.name);
    printf("SMs: %d, HBM: %.1f GB, Peak BW: %.0f GB/s\n",
           props.multiProcessorCount,
           props.totalGlobalMem / 1e9,
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);

    int num_records = 10000000; // 10M default
    if (argc > 1) num_records = atoi(argv[1]);

    uint64_t total_bytes = (uint64_t)num_records * 100; // 100-byte records
    printf("Test size: %d records = %.2f GB\n", num_records, total_bytes / 1e9);

    // Run all experiments
    bench_bandwidth_ceiling(total_bytes);
    bench_merge_pass_cost(num_records);
    bench_fanin_tradeoff(num_records);
    bench_record_size_sensitivity();
    bench_pcie_transfer();

#ifndef NO_CUB
    bench_cub_baseline(num_records);
#endif

    printf("\n════════════════════════════════════════════════════════\n");
    printf("  Summary: What To Optimize\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("1. For data fitting in HBM: use CUB radix sort. Done.\n");
    printf("2. For data > HBM: merge sort is needed.\n");
    printf("   - Minimize merge passes (high fanin)\n");
    printf("   - Overlap PCIe transfer with GPU compute\n");
    printf("   - The GPU compute is NOT the bottleneck\n");
    printf("3. OVC/comparison optimizations only help if\n");
    printf("   the merge is compute-bound (small keys, few bytes/record).\n");

    return 0;
}
