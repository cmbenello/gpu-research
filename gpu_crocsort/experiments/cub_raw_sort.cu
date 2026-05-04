// 4.4 — Raw CUB DeviceRadixSort on synthetic GPU data.
//
// Sorts uint64 keys + uint32 values pair on H100, no host I/O. This
// is the GPU sort *ceiling* — what gpu_crocsort would hit if it had
// zero CPU/PCIe overhead.
//
// Build: nvcc -O3 -arch=sm_90 -I/usr/local/cuda/include \
//          experiments/cub_raw_sort.cu -o experiments/cub_raw_sort
// Run:   ./experiments/cub_raw_sort
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CHK(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA %d: %s\n", err__, cudaGetErrorString(err__)); std::exit(1); } } while(0)

int main(int argc, char** argv) {
    int dev = 0;
    CHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHK(cudaGetDeviceProperties(&p, dev));
    printf("GPU: %s, %.1f GB HBM\n", p.name, p.totalGlobalMem / 1e9);

    // For comparable scale, use record counts matching SF50/100/300.
    // Each "record" is uint64 key + uint32 value = 12 B = ~1/10 the
    // 120 B real-record size.
    std::vector<uint64_t> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) sizes.push_back(std::strtoull(argv[i], nullptr, 10));
    } else {
        sizes = {300000000ULL, 600000000ULL, 1800000000ULL};  // SF50, SF100, SF300 record counts
    }

    printf("scale,records,bytes_keys,total_ms,gb_per_s_keys,gb_per_s_record_equiv\n");

    for (uint64_t N : sizes) {
        size_t key_bytes  = N * sizeof(uint64_t);
        size_t val_bytes  = N * sizeof(uint32_t);
        size_t rec_equiv  = N * 120;  // pretend each is 120 B for fair comparison

        // Allocate keys + values on device
        uint64_t* d_keys_in;
        uint64_t* d_keys_out;
        uint32_t* d_vals_in;
        uint32_t* d_vals_out;
        CHK(cudaMalloc(&d_keys_in,  key_bytes));
        CHK(cudaMalloc(&d_keys_out, key_bytes));
        CHK(cudaMalloc(&d_vals_in,  val_bytes));
        CHK(cudaMalloc(&d_vals_out, val_bytes));

        // Random init
        std::vector<uint64_t> h_keys(N);
        std::mt19937_64 rng(42);
        for (uint64_t i = 0; i < N; i++) h_keys[i] = rng();
        CHK(cudaMemcpy(d_keys_in, h_keys.data(), key_bytes, cudaMemcpyHostToDevice));
        CHK(cudaMemset(d_vals_in, 0xab, val_bytes));

        // CUB temp storage
        size_t cub_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, cub_bytes, d_keys_in, d_keys_out,
                                          d_vals_in, d_vals_out, (int)N);
        void* d_cub_temp = nullptr;
        CHK(cudaMalloc(&d_cub_temp, cub_bytes));

        // Warmup
        cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_bytes, d_keys_in, d_keys_out,
                                          d_vals_in, d_vals_out, (int)N);
        CHK(cudaDeviceSynchronize());

        // Bench: 5 iterations
        cudaEvent_t evs, eve;
        CHK(cudaEventCreate(&evs));
        CHK(cudaEventCreate(&eve));
        const int ITERS = 5;
        CHK(cudaEventRecord(evs));
        for (int i = 0; i < ITERS; i++) {
            cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_bytes, d_keys_in, d_keys_out,
                                              d_vals_in, d_vals_out, (int)N);
        }
        CHK(cudaEventRecord(eve));
        CHK(cudaEventSynchronize(eve));
        float ms = 0;
        CHK(cudaEventElapsedTime(&ms, evs, eve));
        ms /= ITERS;  // per-iter ms

        double gb_per_s_keys     = (key_bytes / 1e9) / (ms / 1e3);
        double gb_per_s_rec      = (rec_equiv / 1e9) / (ms / 1e3);
        printf("SF%llu,%llu,%llu,%.2f,%.2f,%.2f\n",
               (unsigned long long)(N / 6000000),
               (unsigned long long)N,
               (unsigned long long)key_bytes,
               ms, gb_per_s_keys, gb_per_s_rec);
        fflush(stdout);

        CHK(cudaFree(d_keys_in));
        CHK(cudaFree(d_keys_out));
        CHK(cudaFree(d_vals_in));
        CHK(cudaFree(d_vals_out));
        CHK(cudaFree(d_cub_temp));
        CHK(cudaEventDestroy(evs));
        CHK(cudaEventDestroy(eve));
    }
    return 0;
}
