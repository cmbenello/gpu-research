// 6.2 — HBM3 bandwidth saturation test on H100 NVL.
//
// Pure memcpy + reduction kernels to measure peak HBM throughput.
// Sets the upper bound on what sort throughput can ever be on H100.
//
// Build: nvcc -O3 -arch=sm_90 experiments/hbm_saturation.cu -o hbm_saturation
// Run:   ./hbm_saturation
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %d: %s\n", e, cudaGetErrorString(e)); std::exit(1); } } while(0)

// Simple read-write kernel: out[i] = in[i] * 2 + 1
__global__ void memcpy_modify_kernel(const uint64_t* __restrict__ in,
                                     uint64_t* __restrict__ out,
                                     uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i] * 2 + 1;  // simple compute to prevent dead-code elimination
}

// Pure memcpy kernel (1 read + 1 write per uint64)
__global__ void pure_memcpy_kernel(const uint64_t* __restrict__ in,
                                   uint64_t* __restrict__ out,
                                   uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i];
}

// Streaming reduction (read-only)
__global__ void streaming_sum_kernel(const uint64_t* __restrict__ in,
                                     uint64_t* __restrict__ partial,
                                     uint64_t n) {
    extern __shared__ uint64_t smem[];
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v = (i < n) ? in[i] : 0;
    smem[threadIdx.x] = v;
    __syncthreads();
    // Block reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = smem[0];
}

int main() {
    int dev = 0;
    CHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHK(cudaGetDeviceProperties(&p, dev));
    printf("GPU: %s, %.1f GB HBM\n", p.name, p.totalGlobalMem / 1e9);

    // Allocate big buffers (~30 GB each so 60 GB total — leaves room)
    size_t bytes = 30ULL * 1024 * 1024 * 1024;  // 30 GB
    uint64_t n = bytes / sizeof(uint64_t);
    uint64_t* d_in;
    uint64_t* d_out;
    CHK(cudaMalloc(&d_in, bytes));
    CHK(cudaMalloc(&d_out, bytes));
    CHK(cudaMemset(d_in, 0xab, bytes));

    int nthreads = 256;
    uint64_t nblks = (n + nthreads - 1) / nthreads;
    if (nblks > (1ULL << 31)) nblks = 1ULL << 31;

    auto bench = [&](const char* name, void (*kernel)(const uint64_t*, uint64_t*, uint64_t),
                     int reads, int writes) {
        // Warmup
        kernel<<<nblks, nthreads>>>(d_in, d_out, n);
        CHK(cudaDeviceSynchronize());

        cudaEvent_t evs, eve;
        CHK(cudaEventCreate(&evs));
        CHK(cudaEventCreate(&eve));
        const int ITERS = 5;
        CHK(cudaEventRecord(evs));
        for (int i = 0; i < ITERS; i++) {
            kernel<<<nblks, nthreads>>>(d_in, d_out, n);
        }
        CHK(cudaEventRecord(eve));
        CHK(cudaEventSynchronize(eve));
        float ms = 0;
        CHK(cudaEventElapsedTime(&ms, evs, eve));
        // Total bytes moved per iteration: reads + writes
        double total_bytes_per_iter = (double)bytes * (reads + writes);
        double gb_per_s = (total_bytes_per_iter * ITERS / 1e9) / (ms / 1e3);
        printf("  %-20s %d-read %d-write : %.0f ms / %d iters → %.0f GB/s\n",
               name, reads, writes, ms, ITERS, gb_per_s);
        cudaEventDestroy(evs);
        cudaEventDestroy(eve);
    };

    printf("\n%.1f GB working set, %lu uint64 elements\n\n", bytes/1e9, n);

    bench("pure_memcpy",   pure_memcpy_kernel,   1, 1);
    bench("modify_2x_p1",  memcpy_modify_kernel, 1, 1);

    // Reduction (streaming read only) — needs partial output
    uint64_t* d_partial;
    CHK(cudaMalloc(&d_partial, nblks * sizeof(uint64_t)));
    {
        cudaEvent_t evs, eve;
        cudaEventCreate(&evs); cudaEventCreate(&eve);
        cudaEventRecord(evs);
        const int ITERS = 5;
        for (int i = 0; i < ITERS; i++) {
            streaming_sum_kernel<<<nblks, nthreads, nthreads*sizeof(uint64_t)>>>(d_in, d_partial, n);
        }
        cudaEventRecord(eve);
        cudaEventSynchronize(eve);
        float ms = 0;
        cudaEventElapsedTime(&ms, evs, eve);
        double gb_per_s = ((double)bytes * ITERS / 1e9) / (ms / 1e3);
        printf("  %-20s %d-read %d-write : %.0f ms / %d iters → %.0f GB/s\n",
               "streaming_sum", 1, 0, ms, ITERS, gb_per_s);
        cudaEventDestroy(evs); cudaEventDestroy(eve);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_partial);
    return 0;
}
