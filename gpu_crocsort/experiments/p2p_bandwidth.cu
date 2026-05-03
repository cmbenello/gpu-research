// 15.2 — pair-wise NVLink p2p bandwidth on the 4× H100 NVL box.
//
// Measures cudaMemcpyPeer throughput between every GPU pair. Compares
// against the topology baseline from `nvidia-smi nvlink --status`
// (NV6 = 6 links × 26.562 GB/s/link/direction ≈ 159 GB/s per direction
// per pair). Reports a 4×4 GB/s matrix.
//
// Build:
//   nvcc -O3 -arch=sm_90 experiments/p2p_bandwidth.cu -o p2p_bandwidth
// Run:
//   ./p2p_bandwidth                   # default 1 GiB transfer, 5 iters
//   ./p2p_bandwidth --gb 4 --iters 10
//
// Output: human-readable matrix + a CSV line for the autoresearch loop.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %d at %s:%d: %s\n", e, __FILE__, __LINE__, \
            cudaGetErrorString(e)); std::exit(1); } } while(0)

int main(int argc, char** argv) {
    double gb_transfer = 1.0;
    int n_iters = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gb") && i+1 < argc) gb_transfer = atof(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) n_iters = atoi(argv[++i]);
    }
    size_t bytes = (size_t)(gb_transfer * 1e9);

    int n_gpus = 0;
    CHK(cudaGetDeviceCount(&n_gpus));
    printf("Detected %d GPUs. Transfer size: %.2f GB, %d iters per pair.\n\n",
           n_gpus, gb_transfer, n_iters);

    // Print per-GPU info
    for (int g = 0; g < n_gpus; g++) {
        cudaDeviceProp p;
        CHK(cudaGetDeviceProperties(&p, g));
        size_t free_mem = 0, total_mem = 0;
        CHK(cudaSetDevice(g));
        CHK(cudaMemGetInfo(&free_mem, &total_mem));
        printf("  GPU %d: %s, %.1f GB free of %.1f GB\n",
               g, p.name, free_mem/1e9, total_mem/1e9);
    }
    putchar('\n');

    // Enable peer access between every pair.
    for (int a = 0; a < n_gpus; a++) {
        CHK(cudaSetDevice(a));
        for (int b = 0; b < n_gpus; b++) {
            if (a == b) continue;
            int can_peer = 0;
            CHK(cudaDeviceCanAccessPeer(&can_peer, a, b));
            if (can_peer) {
                cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    printf("  enablePeerAccess(%d, %d) failed: %s\n",
                           a, b, cudaGetErrorString(e));
                }
            } else {
                printf("  GPU %d cannot access GPU %d as peer\n", a, b);
            }
        }
    }

    // Allocate one buffer per GPU.
    std::vector<void*> bufs(n_gpus, nullptr);
    for (int g = 0; g < n_gpus; g++) {
        CHK(cudaSetDevice(g));
        CHK(cudaMalloc(&bufs[g], bytes));
        CHK(cudaMemset(bufs[g], 0xab, bytes));
    }
    CHK(cudaDeviceSynchronize());

    // Measure pair-wise bandwidth. For each (src, dst) pair, time `n_iters`
    // cudaMemcpyPeer copies on the SRC device's default stream.
    std::vector<std::vector<double>> mat(n_gpus, std::vector<double>(n_gpus, 0.0));
    for (int src = 0; src < n_gpus; src++) {
        for (int dst = 0; dst < n_gpus; dst++) {
            if (src == dst) {
                // Intra-GPU memcpy as a reference (HBM bandwidth)
                CHK(cudaSetDevice(src));
                cudaEvent_t start, stop;
                CHK(cudaEventCreate(&start));
                CHK(cudaEventCreate(&stop));
                CHK(cudaEventRecord(start));
                for (int it = 0; it < n_iters; it++) {
                    CHK(cudaMemcpyAsync(bufs[src], bufs[src], bytes,
                                        cudaMemcpyDeviceToDevice));
                }
                CHK(cudaEventRecord(stop));
                CHK(cudaEventSynchronize(stop));
                float ms = 0;
                CHK(cudaEventElapsedTime(&ms, start, stop));
                mat[src][dst] = (double)n_iters * bytes / 1e9 / (ms / 1e3);
                CHK(cudaEventDestroy(start));
                CHK(cudaEventDestroy(stop));
                continue;
            }
            CHK(cudaSetDevice(src));
            cudaEvent_t start, stop;
            CHK(cudaEventCreate(&start));
            CHK(cudaEventCreate(&stop));

            // Warm-up
            CHK(cudaMemcpyPeerAsync(bufs[dst], dst, bufs[src], src, bytes));
            CHK(cudaDeviceSynchronize());

            CHK(cudaEventRecord(start));
            for (int it = 0; it < n_iters; it++) {
                CHK(cudaMemcpyPeerAsync(bufs[dst], dst, bufs[src], src, bytes));
            }
            CHK(cudaEventRecord(stop));
            CHK(cudaEventSynchronize(stop));
            float ms = 0;
            CHK(cudaEventElapsedTime(&ms, start, stop));
            mat[src][dst] = (double)n_iters * bytes / 1e9 / (ms / 1e3);
            CHK(cudaEventDestroy(start));
            CHK(cudaEventDestroy(stop));
        }
    }

    // Pretty-print matrix
    printf("Unidirectional GB/s matrix (rows = src, cols = dst):\n\n");
    printf("       ");
    for (int d = 0; d < n_gpus; d++) printf("  GPU%d ", d);
    putchar('\n');
    for (int s = 0; s < n_gpus; s++) {
        printf("  GPU%d ", s);
        for (int d = 0; d < n_gpus; d++) {
            printf("  %5.1f", mat[s][d]);
        }
        putchar('\n');
    }
    putchar('\n');

    // CSV summary: one line, max-off-diagonal + min-off-diagonal + intra
    double max_off = 0, min_off = 1e9, intra = 0;
    int n_intra = 0;
    for (int s = 0; s < n_gpus; s++) for (int d = 0; d < n_gpus; d++) {
        if (s == d) { intra += mat[s][d]; n_intra++; }
        else { if (mat[s][d] > max_off) max_off = mat[s][d];
               if (mat[s][d] < min_off) min_off = mat[s][d]; }
    }
    intra /= n_intra;
    printf("CSV,p2p_bandwidth,n_gpus=%d,gb_per_transfer=%.2f,intra_GB/s=%.1f,p2p_min_GB/s=%.1f,p2p_max_GB/s=%.1f\n",
           n_gpus, gb_transfer, intra, min_off, max_off);

    // Cleanup
    for (int g = 0; g < n_gpus; g++) {
        CHK(cudaSetDevice(g));
        cudaFree(bufs[g]);
    }
    return 0;
}
