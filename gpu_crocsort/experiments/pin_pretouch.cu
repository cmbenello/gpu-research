// 19.11 — cudaHostRegister cost on PRE-TOUCHED memory.
//
// Hypothesis: 19.4's 1.2 GB/s pin rate includes NVMe page-fault cost.
// Pre-touch all pages first (so they're in OS page cache), then pin.
// Should expose the pure cudaHostRegister cost.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
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
    if (argc < 2) { fprintf(stderr, "Usage: %s <file>\n", argv[0]); return 1; }
    CHK(cudaSetDevice(0));
    const char* path = argv[1];
    struct stat st;
    if (stat(path, &st) != 0) { perror(path); return 1; }
    size_t sz = st.st_size;
    double size_gb = sz / 1e9;
    printf("file,size_gb,pretouch_ms,pretouch_gbs,pin_ms,pin_gbs\n");

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 1; }
    void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    // Step 1: pre-touch by reading every page
    double t0 = now_ms();
    volatile uint64_t sum = 0;
    const uint8_t* p = (const uint8_t*)m;
    for (size_t i = 0; i < sz; i += 4096) sum += p[i];
    (void)sum;
    double pretouch_ms = now_ms() - t0;
    double pretouch_gbs = size_gb / (pretouch_ms / 1e3);

    // Step 2: pin (should be much faster now)
    double t1 = now_ms();
    cudaError_t err = cudaHostRegister(m, sz, cudaHostRegisterReadOnly | cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostRegister failed: %s\n", cudaGetErrorString(err));
        munmap(m, sz);
        close(fd);
        return 1;
    }
    double pin_ms = now_ms() - t1;
    double pin_gbs = size_gb / (pin_ms / 1e3);

    printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f\n",
           path, size_gb, pretouch_ms, pretouch_gbs, pin_ms, pin_gbs);

    cudaHostUnregister(m);
    munmap(m, sz);
    close(fd);
    return 0;
}
