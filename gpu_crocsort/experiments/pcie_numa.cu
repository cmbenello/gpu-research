// 6.3.1 — PCIe HtoD/DtoH transfer rate with vs without numactl wrap.
//
// Measures whether co-locating the host pinned buffer with the GPU's
// NUMA-affined CPU root port matters for PCIe5 transfer bandwidth.
//
// Build: nvcc -O3 -arch=sm_90 experiments/pcie_numa.cu -o experiments/pcie_numa
// Run (compare):
//   ./experiments/pcie_numa
//   numactl --cpunodebind=0 --membind=0 ./experiments/pcie_numa
//   numactl --cpunodebind=1 --membind=1 ./experiments/pcie_numa
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA %d: %s\n", err__, cudaGetErrorString(err__)); std::exit(1); } } while(0)

int main(int argc, char**argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);
    CHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHK(cudaGetDeviceProperties(&p, dev));
    fprintf(stderr, "GPU: %s on dev %d\n", p.name, dev);

    const size_t bytes = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    void* h = nullptr;
    CHK(cudaMallocHost(&h, bytes));
    void* d = nullptr;
    CHK(cudaMalloc(&d, bytes));

    // Warmup
    CHK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    CHK(cudaDeviceSynchronize());

    cudaEvent_t evs, eve;
    CHK(cudaEventCreate(&evs));
    CHK(cudaEventCreate(&eve));
    const int ITERS = 5;

    // HtoD
    CHK(cudaEventRecord(evs));
    for (int i = 0; i < ITERS; i++)
        CHK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CHK(cudaEventRecord(eve));
    CHK(cudaEventSynchronize(eve));
    float ms_h2d;
    CHK(cudaEventElapsedTime(&ms_h2d, evs, eve));

    // DtoH
    CHK(cudaEventRecord(evs));
    for (int i = 0; i < ITERS; i++)
        CHK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    CHK(cudaEventRecord(eve));
    CHK(cudaEventSynchronize(eve));
    float ms_d2h;
    CHK(cudaEventElapsedTime(&ms_d2h, evs, eve));

    double h2d_gbs = ((double)bytes * ITERS / 1e9) / (ms_h2d / 1e3);
    double d2h_gbs = ((double)bytes * ITERS / 1e9) / (ms_d2h / 1e3);
    printf("dev=%d, h2d=%.2f GB/s, d2h=%.2f GB/s, bytes=%zu, iters=%d\n",
           dev, h2d_gbs, d2h_gbs, bytes, ITERS);

    CHK(cudaFreeHost(h));
    CHK(cudaFree(d));
    return 0;
}
