// 6.2.2 — Multi-stream HBM saturation
//
// Single-stream HBM peak measured in 6.2 was 2191 GB/s read+write.
// The H100 NVL's 5 HBM3 stacks each have their own controller; with
// multiple concurrent kernels working different memory regions, can
// we push closer to the spec 3.35 TB/s?
//
// Build: nvcc -O3 -arch=sm_90 experiments/hbm_multistream.cu -o experiments/hbm_multistream
// Run:   ./experiments/hbm_multistream
#include <cstdio>
#include <cstdlib>
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

int main() {
    int dev = 0;
    CHK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHK(cudaGetDeviceProperties(&p, dev));
    printf("GPU: %s, %.1f GB HBM\n", p.name, p.totalGlobalMem / 1e9);

    // Per-stream working set: 4 GB.  With NSTREAMS=8 → 32 GB IN + 32 GB OUT.
    const size_t per_stream_bytes = 4ULL * 1024 * 1024 * 1024;
    const uint64_t per_stream_n = per_stream_bytes / sizeof(uint64_t);

    printf("policy,nstreams,per_stream_gb,total_gb,iters,wall_ms,gbs_rw\n");
    int nthreads = 256;

    for (int NSTREAMS : {1, 2, 4, 8, 16}) {
        size_t total_bytes = NSTREAMS * per_stream_bytes;
        if (total_bytes > 80ULL * 1024 * 1024 * 1024) {
            // Don't blow the 94 GB HBM budget
            continue;
        }
        std::vector<cudaStream_t> streams(NSTREAMS);
        std::vector<uint64_t*> d_in(NSTREAMS);
        std::vector<uint64_t*> d_out(NSTREAMS);
        for (int s = 0; s < NSTREAMS; s++) {
            CHK(cudaStreamCreate(&streams[s]));
            CHK(cudaMalloc(&d_in[s], per_stream_bytes));
            CHK(cudaMalloc(&d_out[s], per_stream_bytes));
            CHK(cudaMemset(d_in[s], 0xab, per_stream_bytes));
        }

        uint64_t nblks = (per_stream_n + nthreads - 1) / nthreads;
        if (nblks > (1ULL << 31)) nblks = 1ULL << 31;

        // Warmup
        for (int s = 0; s < NSTREAMS; s++) {
            rw_kernel<<<nblks, nthreads, 0, streams[s]>>>(d_in[s], d_out[s], per_stream_n);
        }
        CHK(cudaDeviceSynchronize());

        const int ITERS = 5;
        cudaEvent_t evs, eve;
        CHK(cudaEventCreate(&evs));
        CHK(cudaEventCreate(&eve));
        CHK(cudaEventRecord(evs));
        for (int it = 0; it < ITERS; it++) {
            for (int s = 0; s < NSTREAMS; s++) {
                rw_kernel<<<nblks, nthreads, 0, streams[s]>>>(d_in[s], d_out[s], per_stream_n);
            }
        }
        CHK(cudaEventRecord(eve));
        CHK(cudaEventSynchronize(eve));
        float ms = 0;
        CHK(cudaEventElapsedTime(&ms, evs, eve));

        // total bus bytes per iter = NSTREAMS × per_stream_bytes × 2 (rw)
        double total_bus_per_iter = (double)total_bytes * 2.0;
        double gbs = (total_bus_per_iter * ITERS / 1e9) / (ms / 1e3);
        printf("multistream,%d,%.1f,%.1f,%d,%.0f,%.0f\n",
               NSTREAMS,
               per_stream_bytes / 1e9,
               total_bytes / 1e9,
               ITERS, ms, gbs);
        fflush(stdout);

        CHK(cudaEventDestroy(evs));
        CHK(cudaEventDestroy(eve));
        for (int s = 0; s < NSTREAMS; s++) {
            CHK(cudaFree(d_in[s]));
            CHK(cudaFree(d_out[s]));
            CHK(cudaStreamDestroy(streams[s]));
        }
    }
    return 0;
}
