/*
 * GPU Sort Comparison Benchmark
 * Compares: CUB DeviceRadixSort, Thrust sort_by_key, CPU std::sort/stable_sort
 *
 * Build: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_75 comparison_bench.cu -o comparison_bench
 * Run:   ./comparison_bench
 */

#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include <string>
#include <thread>

// ── Helpers ────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct BenchResult {
    std::string name;
    size_t num_records;
    double median_ms;
    double throughput_gb_s;   // data throughput (keys+values)
    double throughput_mrec_s; // million records/sec
};

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0;
    return v[n/2];
}

// ── GPU Memory Query ───────────────────────────────────────────────

size_t gpu_free_bytes() {
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// ── Data Generation ────────────────────────────────────────────────

void generate_data(std::vector<uint64_t>& keys, std::vector<uint32_t>& vals, size_t n) {
    std::mt19937_64 rng(42);
    keys.resize(n);
    vals.resize(n);
    for (size_t i = 0; i < n; i++) {
        keys[i] = rng();
        vals[i] = static_cast<uint32_t>(i);
    }
}

// ── CUB DeviceRadixSort Benchmark ──────────────────────────────────

BenchResult bench_cub(size_t n, int warmup_runs = 1, int timed_runs = 7) {
    // Allocate device memory
    uint64_t *d_keys_in, *d_keys_out;
    uint32_t *d_vals_in, *d_vals_out;
    CUDA_CHECK(cudaMalloc(&d_keys_in,  n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_in,  n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vals_out, n * sizeof(uint32_t)));

    // Generate and upload data
    std::vector<uint64_t> h_keys;
    std::vector<uint32_t> h_vals;
    generate_data(h_keys, h_vals, n);

    // Determine temp storage
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_keys_in, d_keys_out, d_vals_in, d_vals_out, n);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;

    for (int r = 0; r < warmup_runs + timed_runs; r++) {
        // Re-upload data each run
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals_in, h_vals.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        if (r >= warmup_runs) {
            times.push_back(ms);
        }
    }

    double med = median(times);
    double data_bytes = n * (sizeof(uint64_t) + sizeof(uint32_t));
    double gb_s = (data_bytes / 1e9) / (med / 1000.0);
    double mrec_s = (n / 1e6) / (med / 1000.0);

    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_vals_in));
    CUDA_CHECK(cudaFree(d_vals_out));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return {"CUB RadixSort", n, med, gb_s, mrec_s};
}

// ── Thrust sort_by_key Benchmark ───────────────────────────────────

BenchResult bench_thrust(size_t n, int warmup_runs = 1, int timed_runs = 7) {
    std::vector<uint64_t> h_keys;
    std::vector<uint32_t> h_vals;
    generate_data(h_keys, h_vals, n);

    uint64_t *d_keys;
    uint32_t *d_vals;
    CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_vals, n * sizeof(uint32_t)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;

    for (int r = 0; r < warmup_runs + timed_runs; r++) {
        CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        thrust::sort_by_key(thrust::device, d_keys, d_keys + n, d_vals);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        if (r >= warmup_runs) {
            times.push_back(ms);
        }
    }

    double med = median(times);
    double data_bytes = n * (sizeof(uint64_t) + sizeof(uint32_t));
    double gb_s = (data_bytes / 1e9) / (med / 1000.0);
    double mrec_s = (n / 1e6) / (med / 1000.0);

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return {"Thrust sort_by_key", n, med, gb_s, mrec_s};
}

// ── CUB Keys-Only Benchmark ───────────────────────────────────────

BenchResult bench_cub_keys_only(size_t n, int warmup_runs = 1, int timed_runs = 7) {
    uint64_t *d_keys_in, *d_keys_out;
    CUDA_CHECK(cudaMalloc(&d_keys_in,  n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint64_t)));

    std::vector<uint64_t> h_keys;
    std::vector<uint32_t> dummy;
    generate_data(h_keys, dummy, n);

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_in, d_keys_out, n);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;

    for (int r = 0; r < warmup_runs + timed_runs; r++) {
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_in, d_keys_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        if (r >= warmup_runs) {
            times.push_back(ms);
        }
    }

    double med = median(times);
    double data_bytes = n * sizeof(uint64_t);
    double gb_s = (data_bytes / 1e9) / (med / 1000.0);
    double mrec_s = (n / 1e6) / (med / 1000.0);

    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return {"CUB RadixSort (keys-only)", n, med, gb_s, mrec_s};
}

// ── CPU std::sort Benchmark ────────────────────────────────────────

struct KVPair {
    uint64_t key;
    uint32_t val;
    bool operator<(const KVPair& o) const { return key < o.key; }
};

BenchResult bench_cpu_sort(size_t n, int timed_runs = 5) {
    std::vector<uint64_t> h_keys;
    std::vector<uint32_t> h_vals;
    generate_data(h_keys, h_vals, n);

    // Pack into struct for std::sort
    std::vector<KVPair> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = {h_keys[i], h_vals[i]};
    }

    std::vector<KVPair> working(n);
    std::vector<double> times;

    for (int r = 0; r < timed_runs; r++) {
        std::copy(data.begin(), data.end(), working.begin());

        auto t0 = std::chrono::high_resolution_clock::now();
        std::sort(working.begin(), working.end());
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    double med = median(times);
    double data_bytes = n * (sizeof(uint64_t) + sizeof(uint32_t));
    double gb_s = (data_bytes / 1e9) / (med / 1000.0);
    double mrec_s = (n / 1e6) / (med / 1000.0);

    return {"CPU std::sort (1-thread)", n, med, gb_s, mrec_s};
}

// Simple parallel merge sort using std::threads
void parallel_merge_sort(KVPair* data, KVPair* temp, size_t n, int threads) {
    if (threads <= 1 || n < 10000) {
        std::sort(data, data + n);
        return;
    }
    size_t mid = n / 2;
    std::thread t([&]{ parallel_merge_sort(data, temp, mid, threads / 2); });
    parallel_merge_sort(data + mid, temp + mid, n - mid, threads / 2);
    t.join();
    std::merge(data, data + mid, data + mid, data + n, temp);
    std::copy(temp, temp + n, data);
}

BenchResult bench_cpu_parallel_sort(size_t n, int timed_runs = 5) {
    std::vector<uint64_t> h_keys;
    std::vector<uint32_t> h_vals;
    generate_data(h_keys, h_vals, n);

    std::vector<KVPair> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = {h_keys[i], h_vals[i]};
    }

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8;

    std::vector<KVPair> working(n);
    std::vector<KVPair> temp(n);
    std::vector<double> times;

    for (int r = 0; r < timed_runs; r++) {
        std::copy(data.begin(), data.end(), working.begin());

        auto t0 = std::chrono::high_resolution_clock::now();
        parallel_merge_sort(working.data(), temp.data(), n, num_threads);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    double med = median(times);
    double data_bytes = n * (sizeof(uint64_t) + sizeof(uint32_t));
    double gb_s = (data_bytes / 1e9) / (med / 1000.0);
    double mrec_s = (n / 1e6) / (med / 1000.0);

    char name[64];
    snprintf(name, sizeof(name), "CPU parallel sort (%d-thr)", num_threads);
    return {name, n, med, gb_s, mrec_s};
}

// ── Main ───────────────────────────────────────────────────────────

void print_result(const BenchResult& r) {
    printf("| %-26s | %10zu | %10.2f ms | %8.2f GB/s | %8.1f Mrec/s |\n",
           r.name.c_str(), r.num_records, r.median_ms, r.throughput_gb_s, r.throughput_mrec_s);
}

int main(int argc, char** argv) {
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, %zu MB, SM %d.%d\n", prop.name,
           prop.totalGlobalMem / (1024*1024), prop.major, prop.minor);
    printf("Free GPU memory: %zu MB\n\n", gpu_free_bytes() / (1024*1024));

    // Record size: 8 bytes key + 4 bytes value = 12 bytes per record
    // CUB needs 2x keys + 2x vals + temp ≈ 24 bytes/record + overhead
    // With 24GB, max ~800M records for CUB (conservative: use 280M)

    std::vector<size_t> gpu_sizes = {
        1'000'000,      // 1M
        10'000'000,     // 10M
        60'000'000,     // 60M
        100'000'000,    // 100M
        200'000'000,    // 200M
        280'000'000,    // 280M — near GPU limit for key+val pairs
    };

    // CPU sizes are smaller (much slower)
    std::vector<size_t> cpu_sizes = {
        1'000'000,      // 1M
        10'000'000,     // 10M
        60'000'000,     // 60M
        100'000'000,    // 100M
    };

    bool run_cpu = true;
    bool run_large_cpu = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-cpu") run_cpu = false;
        if (std::string(argv[i]) == "--large-cpu") run_large_cpu = true;
    }

    if (run_large_cpu) {
        cpu_sizes.push_back(200'000'000);
        cpu_sizes.push_back(280'000'000);
    }

    printf("=== GPU Sort Benchmarks ===\n");
    printf("| %-26s | %10s | %13s | %12s | %15s |\n",
           "Method", "Records", "Median Time", "Throughput", "Records/sec");
    printf("|%s|%s|%s|%s|%s|\n",
           "----------------------------", "------------", "---------------",
           "--------------", "-----------------");

    // CUB keys-only
    printf("\n--- CUB RadixSort (keys-only, uint64) ---\n");
    for (size_t n : gpu_sizes) {
        auto r = bench_cub_keys_only(n);
        print_result(r);
    }

    // CUB key+value
    printf("\n--- CUB RadixSort (uint64 key + uint32 value) ---\n");
    for (size_t n : gpu_sizes) {
        auto r = bench_cub(n);
        print_result(r);
    }

    // Thrust
    printf("\n--- Thrust sort_by_key (uint64 key + uint32 value) ---\n");
    for (size_t n : gpu_sizes) {
        auto r = bench_thrust(n);
        print_result(r);
    }

    // CPU benchmarks
    if (run_cpu) {
        printf("\n--- CPU std::sort (single-threaded, uint64 key + uint32 value) ---\n");
        for (size_t n : cpu_sizes) {
            auto r = bench_cpu_sort(n);
            print_result(r);
        }

        printf("\n--- CPU std::sort (parallel, uint64 key + uint32 value) ---\n");
        for (size_t n : cpu_sizes) {
            auto r = bench_cpu_parallel_sort(n);
            print_result(r);
        }
    }

    printf("\nDone.\n");
    return 0;
}
