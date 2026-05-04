// 19.5 — cudaMemcpy from pinned vs unpinned host memory.
//
// Tests whether the 1.2 GB/s pin cost is worth the higher per-transfer
// bandwidth. If unpinned cudaMemcpy is fast enough, we can skip the
// pin and save 4-7 min per SF1500-bucket.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CHK(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA %d: %s\n", err__, cudaGetErrorString(err__)); std::exit(1); } } while(0)

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        return 1;
    }
    CHK(cudaSetDevice(0));
    const char* path = argv[1];
    struct stat st;
    if (stat(path, &st) != 0) { perror(path); return 1; }
    size_t sz = st.st_size;
    const size_t MAX_SZ = 4ULL * 1024 * 1024 * 1024;  // 4 GB max for the test
    if (sz > MAX_SZ) sz = MAX_SZ;
    double size_gb = sz / 1e9;
    printf("Testing %.2f GB from %s\n", size_gb, path);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 1; }
    void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    // Touch to bring into page cache (eliminate disk I/O from the comparison)
    volatile char x = 0;
    for (size_t i = 0; i < sz; i += 4096) x = ((char*)m)[i];
    (void)x;

    void* d;
    CHK(cudaMalloc(&d, sz));

    // Test 1: cudaMemcpy from non-pinned mmap (no cudaHostRegister)
    {
        // Warmup
        CHK(cudaMemcpy(d, m, sz, cudaMemcpyHostToDevice));
        double t0 = now_ms();
        const int ITERS = 3;
        for (int i = 0; i < ITERS; i++)
            CHK(cudaMemcpy(d, m, sz, cudaMemcpyHostToDevice));
        CHK(cudaDeviceSynchronize());
        double dt = now_ms() - t0;
        double gbs = (size_gb * ITERS) / (dt / 1e3);
        printf("UNPINNED memcpy: %.0f ms (avg %d iters) → %.2f GB/s\n",
               dt / ITERS, ITERS, gbs);
    }

    // Test 2: Pin first, then memcpy
    {
        double pin_start = now_ms();
        CHK(cudaHostRegister(m, sz, cudaHostRegisterReadOnly | cudaHostRegisterMapped));
        double pin_ms = now_ms() - pin_start;

        // Warmup
        CHK(cudaMemcpy(d, m, sz, cudaMemcpyHostToDevice));
        double t0 = now_ms();
        const int ITERS = 3;
        for (int i = 0; i < ITERS; i++)
            CHK(cudaMemcpy(d, m, sz, cudaMemcpyHostToDevice));
        CHK(cudaDeviceSynchronize());
        double dt = now_ms() - t0;
        double gbs = (size_gb * ITERS) / (dt / 1e3);
        printf("PINNED memcpy:   %.0f ms (avg %d iters) → %.2f GB/s [pin took %.0f ms = %.2f GB/s]\n",
               dt / ITERS, ITERS, gbs, pin_ms, size_gb / (pin_ms / 1e3));

        // Compute amortized: pin once + N transfers, what's net?
        double net1_ms = pin_ms + (dt / ITERS);
        double net1_gbs = size_gb / (net1_ms / 1e3);
        printf("  net (pin+1 transfer):  %.0f ms → %.2f GB/s\n", net1_ms, net1_gbs);

        cudaHostUnregister(m);
    }

    cudaFree(d);
    munmap(m, sz);
    close(fd);
    return 0;
}
