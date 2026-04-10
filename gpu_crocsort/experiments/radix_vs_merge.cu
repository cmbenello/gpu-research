// ============================================================================
// Radix Sort vs Merge Sort Crossover Analysis
//
// KEY QUESTION: At what data size does our merge sort beat CUB radix sort?
// CUB radix sort is ~1.5 TB/s but only works for data fitting in HBM.
// Our merge sort works for any size but has multi-pass overhead.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_80 experiments/radix_vs_merge.cu -o radix_vs_merge
// Run:   ./radix_vs_merge
//
// Requires CUB (ships with CUDA toolkit 11+)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

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
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
    }
};

// ── CUB radix sort on key-value pairs ──────────────────────────────
// Sort by 10-byte key. We pack key into (uint64_t, uint16_t) pair.
// CUB sorts the uint64_t (first 8 bytes), ties broken by uint16_t.

float bench_cub_radix(int num_records, int trials = 3) {
    // Allocate keys (packed as uint64_t for radix sort) + values
    uint64_t* d_keys_in;
    uint64_t* d_keys_out;
    uint32_t* d_vals_in;  // Record indices as "values"
    uint32_t* d_vals_out;

    CUDA_CHECK(cudaMalloc(&d_keys_in, num_records * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, num_records * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_in, num_records * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_out, num_records * sizeof(uint32_t)));

    // Generate random keys on host, upload
    uint64_t* h_keys = (uint64_t*)malloc(num_records * sizeof(uint64_t));
    srand(42);
    for (int i = 0; i < num_records; i++) {
        h_keys[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
    }
    CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys, num_records * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // CUB temp storage
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys_out,
                                     d_vals_in, d_vals_out, num_records);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // Warmup
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys_out,
                                     d_vals_in, d_vals_out, num_records);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    Timer t;
    float best = 1e9;
    for (int i = 0; i < trials; i++) {
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys, num_records * sizeof(uint64_t), cudaMemcpyHostToDevice));
        t.begin();
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys_out,
                                         d_vals_in, d_vals_out, num_records);
        float ms = t.end_ms();
        best = fminf(best, ms);
    }

    free(h_keys);
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_vals_in));
    CUDA_CHECK(cudaFree(d_vals_out));
    CUDA_CHECK(cudaFree(d_temp));
    return best;
}

// ── Analytical model for merge sort time ───────────────────────────

struct MergeSortModel {
    double bandwidth_gbps;   // Achieved merge bandwidth (GB/s)
    double run_gen_gbps;     // Run generation throughput (GB/s)
    int records_per_run;     // Block sort size
    int merge_fanin;         // K-way merge fan-in

    double predict_ms(int num_records, int record_size) const {
        double data_gb = (double)num_records * record_size / 1e9;
        int num_runs = (num_records + records_per_run - 1) / records_per_run;
        int passes = (int)ceil(log((double)num_runs) / log((double)merge_fanin));

        double run_gen_ms = data_gb / (run_gen_gbps / 1000.0);
        double merge_ms = (2.0 * data_gb * passes) / (bandwidth_gbps / 1000.0);
        return run_gen_ms + merge_ms;
    }
};

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("═══════════════════════════════════════════════════════\n");
    printf("  Radix Sort vs Merge Sort Crossover Analysis\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("GPU: %s\n", props.name);
    printf("HBM: %.1f GB, Peak BW: %.0f GB/s\n\n",
           props.totalGlobalMem / 1e9,
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);

    // ── Part 1: CUB radix sort scaling ──
    printf("Part 1: CUB Radix Sort (key-only, uint64)\n");
    printf("  %-12s  %-10s  %-12s\n", "Records", "Time (ms)", "Throughput");
    printf("  %-12s  %-10s  %-12s\n", "-------", "---------", "----------");

    int sizes[] = {100000, 500000, 1000000, 5000000, 10000000, 50000000};
    for (int n : sizes) {
        size_t needed = n * sizeof(uint64_t) * 4; // keys_in + keys_out + vals_in + vals_out
        if (needed > props.totalGlobalMem / 2) continue;

        float ms = bench_cub_radix(n);
        double gb = n * sizeof(uint64_t) / 1e9;
        printf("  %10d  %8.2f ms  %8.2f GB/s\n", n, ms, gb / (ms / 1000.0));
    }

    // ── Part 2: Merge sort prediction ──
    printf("\nPart 2: Merge Sort Prediction (100-byte records)\n");
    printf("  Assuming: 80 GB/s run generation, 200 GB/s merge bandwidth\n");
    printf("  (Conservative estimates — run benchmarks to get real numbers)\n\n");

    MergeSortModel model_2way  = {200.0, 80.0, 512, 2};
    MergeSortModel model_8way  = {150.0, 80.0, 512, 8};
    MergeSortModel model_16way = {120.0, 80.0, 512, 16};

    printf("  %-12s  %-8s  %-10s  %-10s  %-10s\n",
           "Records", "Data", "2-way", "8-way", "16-way");
    printf("  %-12s  %-8s  %-10s  %-10s  %-10s\n",
           "-------", "----", "-----", "-----", "------");

    int record_size = 100;
    int msizes[] = {100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000};
    for (int n : msizes) {
        double data_mb = (double)n * record_size / 1e6;
        printf("  %10d  %5.0f MB  %7.1f ms  %7.1f ms  %7.1f ms\n",
               n, data_mb,
               model_2way.predict_ms(n, record_size),
               model_8way.predict_ms(n, record_size),
               model_16way.predict_ms(n, record_size));
    }

    // ── Part 3: Crossover analysis ──
    printf("\nPart 3: When to use what\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │ Data ≤ HBM, key-only:  Use CUB radix sort             │\n");
    printf("  │ Data ≤ HBM, key+value: Use CUB sort + scatter         │\n");
    printf("  │ Data > HBM:            Use GPU merge sort (this proj)  │\n");
    printf("  │                                                         │\n");
    printf("  │ Within merge sort:                                      │\n");
    printf("  │   Few large runs:  K-way merge tree (fewer passes)     │\n");
    printf("  │   Many small runs: 2-way merge path (more parallelism) │\n");
    printf("  │   Best: high-fanin merge tree for all cases            │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n");

    // ── Part 4: Memory requirements ──
    printf("\nPart 4: Memory Requirements\n");
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU free memory: %.2f GB\n", free_mem / 1e9);
    printf("  Sort needs ~3x data (input + 2 merge buffers)\n");

    double max_data_gb = free_mem / 3.0 / 1e9;
    int max_records_100b = (int)(max_data_gb * 1e9 / 100);
    printf("  Max in-HBM sort (100B records): %.2f GB = %d M records\n",
           max_data_gb, max_records_100b / 1000000);
    printf("  Larger data → use external sort (src/external_sort.cu)\n");

    return 0;
}
