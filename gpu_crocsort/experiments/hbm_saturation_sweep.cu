// 6.2.1 — HBM saturation sweep across working-set sizes and 4 GPUs.
//
// Measures effective HBM3 bandwidth across:
//   - working set: 1, 4, 16, 32, 64 GB
//   - device: GPU 0..N-1 (whatever CUDA_VISIBLE_DEVICES exposes)
//
// Build: nvcc -O3 -arch=sm_90 experiments/hbm_saturation_sweep.cu -o experiments/hbm_saturation_sweep
// Run:   ./experiments/hbm_saturation_sweep
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA %d: %s\n", err__, cudaGetErrorString(err__)); std::exit(1); } } while(0)

__global__ void rw_kernel(const uint64_t* __restrict__ in,
                          uint64_t* __restrict__ out,
                          uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i] * 2 + 1;
}

__global__ void read_kernel(const uint64_t* __restrict__ in,
                            uint64_t* __restrict__ acc_out,
                            uint64_t n) {
    extern __shared__ uint64_t smem[];
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t v = (i < n) ? in[i] : 0;
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) acc_out[blockIdx.x] = smem[0];
}

double bench_rw(uint64_t* d_in, uint64_t* d_out, uint64_t n, size_t bytes) {
    int nthreads = 256;
    uint64_t nblks = (n + nthreads - 1) / nthreads;
    if (nblks > (1ULL << 31)) nblks = 1ULL << 31;
    rw_kernel<<<nblks, nthreads>>>(d_in, d_out, n);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CHK(cudaEventCreate(&s));
    CHK(cudaEventCreate(&e));
    const int ITERS = 5;
    CHK(cudaEventRecord(s));
    for (int i = 0; i < ITERS; i++) rw_kernel<<<nblks, nthreads>>>(d_in, d_out, n);
    CHK(cudaEventRecord(e));
    CHK(cudaEventSynchronize(e));
    float ms = 0;
    CHK(cudaEventElapsedTime(&ms, s, e));
    CHK(cudaEventDestroy(s));
    CHK(cudaEventDestroy(e));
    return ((double)bytes * 2.0 * ITERS / 1e9) / (ms / 1e3);
}

double bench_read(uint64_t* d_in, uint64_t* d_partial, uint64_t n, size_t bytes) {
    int nthreads = 256;
    uint64_t nblks = (n + nthreads - 1) / nthreads;
    if (nblks > (1ULL << 31)) nblks = 1ULL << 31;
    read_kernel<<<nblks, nthreads, nthreads*sizeof(uint64_t)>>>(d_in, d_partial, n);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CHK(cudaEventCreate(&s));
    CHK(cudaEventCreate(&e));
    const int ITERS = 5;
    CHK(cudaEventRecord(s));
    for (int i = 0; i < ITERS; i++)
        read_kernel<<<nblks, nthreads, nthreads*sizeof(uint64_t)>>>(d_in, d_partial, n);
    CHK(cudaEventRecord(e));
    CHK(cudaEventSynchronize(e));
    float ms = 0;
    CHK(cudaEventElapsedTime(&ms, s, e));
    CHK(cudaEventDestroy(s));
    CHK(cudaEventDestroy(e));
    return ((double)bytes * ITERS / 1e9) / (ms / 1e3);
}

int main() {
    int ndev = 0;
    CHK(cudaGetDeviceCount(&ndev));
    printf("# 6.2.1 HBM saturation sweep across %d GPU(s) and 5 working-set sizes\n", ndev);
    printf("device,size_gb,read_gbs,rw_gbs\n");

    // Cap at 32 GB so two buffers fit in 94 GB H100 HBM with margin.
    std::vector<size_t> sizes_gb = {1, 4, 16, 32, 40};
    for (int dev = 0; dev < ndev; dev++) {
        CHK(cudaSetDevice(dev));
        cudaDeviceProp p;
        CHK(cudaGetDeviceProperties(&p, dev));
        for (size_t gb : sizes_gb) {
            size_t bytes = gb * 1024ULL * 1024ULL * 1024ULL;
            uint64_t n = bytes / sizeof(uint64_t);
            uint64_t* d_in;
            uint64_t* d_out;
            uint64_t* d_partial;
            CHK(cudaMalloc(&d_in, bytes));
            CHK(cudaMalloc(&d_out, bytes));
            int nthreads = 256;
            uint64_t nblks = (n + nthreads - 1) / nthreads;
            if (nblks > (1ULL << 31)) nblks = 1ULL << 31;
            CHK(cudaMalloc(&d_partial, nblks * sizeof(uint64_t)));
            CHK(cudaMemset(d_in, 0xab, bytes));

            double rd = bench_read(d_in, d_partial, n, bytes);
            double rw = bench_rw(d_in, d_out, n, bytes);

            printf("%d,%zu,%.0f,%.0f\n", dev, gb, rd, rw);
            fflush(stdout);

            CHK(cudaFree(d_in));
            CHK(cudaFree(d_out));
            CHK(cudaFree(d_partial));
        }
    }
    return 0;
}
