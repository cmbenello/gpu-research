/**
 * C2: Compressibility-vs-speedup sweep on synthetic data.
 *
 * Generate 100M-row synthetic uint32 datasets with controlled cardinality
 * (which controls compressibility). Run CUB radix sort with full 32 bits
 * vs. truncated bits (FOR equivalent). Plot speedup vs. compression ratio.
 *
 * Build: nvcc -O3 -std=c++17 -arch=sm_75 -o c2_sweep scripts/c2_synthetic_sweep.cu
 * Run:   ./c2_sweep
 */
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cub/cub.cuh>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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

float bench_sort(uint32_t* d_keys_src, size_t N, int sort_bits, int warmup=1, int runs=5) {
    uint32_t* d_keys, *d_keys_alt, *d_vals, *d_vals_alt;
    CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_alt, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vals, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_alt, N * sizeof(uint32_t)));

    size_t temp_bytes = 0;
    cub::DoubleBuffer<uint32_t> kb(d_keys, d_keys_alt);
    cub::DoubleBuffer<uint32_t> vb(d_vals, d_vals_alt);
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, kb, vb, (int)N, 0, sort_bits);
    void* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemcpy(d_keys, d_keys_src, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        kb = cub::DoubleBuffer<uint32_t>(d_keys, d_keys_alt);
        vb = cub::DoubleBuffer<uint32_t>(d_vals, d_vals_alt);
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, kb, vb, (int)N, 0, sort_bits);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Timer t;
    float total_ms = 0;
    for (int r = 0; r < runs; r++) {
        CUDA_CHECK(cudaMemcpy(d_keys, d_keys_src, N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        kb = cub::DoubleBuffer<uint32_t>(d_keys, d_keys_alt);
        vb = cub::DoubleBuffer<uint32_t>(d_vals, d_vals_alt);
        t.begin();
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, kb, vb, (int)N, 0, sort_bits);
        total_ms += t.end_ms();
    }

    cudaFree(d_keys); cudaFree(d_keys_alt);
    cudaFree(d_vals); cudaFree(d_vals_alt);
    cudaFree(d_temp);

    return total_ms / runs;
}

int main() {
    size_t N = 100000000;  // 100M
    printf("C2: Compressibility-vs-speedup sweep (N=%zuM)\n\n", N/1000000);

    FILE* csv = fopen("results/overnight/c2_synthetic_sweep.csv", "w");
    fprintf(csv, "cardinality,effective_bits,compression_ratio,sort_32b_ms,sort_for_ms,speedup\n");

    // Different cardinalities = different compressibility
    // cardinality 256 = 8 bits, 65536 = 16 bits, etc.
    size_t cardinalities[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 16777216, 4294967295UL};

    uint32_t* h_data = new uint32_t[N];
    uint32_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint32_t)));

    for (size_t card : cardinalities) {
        // Generate data
        std::mt19937 rng(42);
        uint32_t max_val = (card > 0 && card <= 4294967295UL) ? (uint32_t)(card - 1) : 0xFFFFFFFF;
        std::uniform_int_distribution<uint32_t> dist(0, max_val);
        for (size_t i = 0; i < N; i++) h_data[i] = dist(rng);

        CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Compute effective bits
        int eff_bits = 1;
        uint64_t v = card;
        while (v > (1ULL << eff_bits)) eff_bits++;
        if (eff_bits > 32) eff_bits = 32;

        float compression = 32.0f / eff_bits;

        // Baseline: sort all 32 bits
        float base_ms = bench_sort(d_data, N, 32);

        // FOR: sort only effective bits
        float for_ms = bench_sort(d_data, N, eff_bits);

        float speedup = base_ms / for_ms;

        printf("  card=%10zu eff_bits=%2d ratio=%.1fx: 32b=%.1fms for=%.1fms speedup=%.2fx\n",
               card, eff_bits, compression, base_ms, for_ms, speedup);
        fprintf(csv, "%zu,%d,%.2f,%.2f,%.2f,%.2f\n",
                card, eff_bits, compression, base_ms, for_ms, speedup);
    }

    fclose(csv);
    cudaFree(d_data);
    delete[] h_data;

    printf("\nWrote results/overnight/c2_synthetic_sweep.csv\n");
    return 0;
}
