// cuFile chunked read test: read N×CHUNK_SIZE bytes from input.bin
// in CHUNK_SIZE chunks, reusing one GPU buffer. Tests sustained throughput.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cufile.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/mnt/data/lineitem_sf300.bin";
    uint64_t total_bytes = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : (50ULL << 30);
    uint64_t chunk_bytes = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : (4ULL << 30);

    printf("Chunked cuFile: %.2f GB total in %.2f GB chunks from %s\n",
           total_bytes/1e9, chunk_bytes/1e9, path);

    cuFileDriverOpen();

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    CUfileDescr_t descr = {};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t fh;
    CUfileError_t err = cuFileHandleRegister(&fh, &descr);
    if (err.err != 0) { fprintf(stderr, "cuFileHandleRegister err=%d\n", err.err); return 1; }

    uint8_t* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, chunk_bytes));
    err = cuFileBufRegister(d_buf, chunk_bytes, 0);
    if (err.err != 0) { fprintf(stderr, "BufRegister err=%d\n", err.err); return 1; }

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t total_read = 0;
    while (total_read < total_bytes) {
        uint64_t want = std::min(chunk_bytes, total_bytes - total_read);
        ssize_t n = cuFileRead(fh, d_buf, want, total_read, 0);
        if (n != (ssize_t)want) {
            fprintf(stderr, "Short read %zd vs %lu at offset %lu\n", n, want, total_read);
            break;
        }
        total_read += n;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Read %lu bytes in %.0f ms (%.2f GB/s sustained)\n",
           total_read, ms, total_read / (ms * 1e6));

    cuFileBufDeregister(d_buf);
    cudaFree(d_buf);
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileDriverClose();
    return 0;
}
