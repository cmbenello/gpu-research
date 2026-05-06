// cuFile smoke test: read 1 GB from input.bin directly into GPU memory.
// If this works, GDS plumbing is functional and we can build a streaming
// partition_sort that reads via cuFile.
//
// Build: nvcc -O3 -arch=sm_90 experiments/cufile_smoke.cu -lcufile -o experiments/cufile_smoke
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
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

#define CU_CHECK_ERR(s) do { CUfileError_t err = (s); \
    if (err.err != 0) { fprintf(stderr, "cuFile error %d at %s:%d\n", err.err, __FILE__, __LINE__); exit(1); } } while(0)

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : "/mnt/data/lineitem_sf50.bin";
    uint64_t bytes = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : (1ULL << 30);

    printf("cuFile smoke test: reading %.2f GB from %s into GPU memory\n",
           bytes/1e9, input_path);

    // Init cuFile
    auto t_init0 = std::chrono::high_resolution_clock::now();
    CU_CHECK_ERR(cuFileDriverOpen());
    auto t_init1 = std::chrono::high_resolution_clock::now();
    printf("cuFileDriverOpen: %.0f ms\n",
           std::chrono::duration<double, std::milli>(t_init1 - t_init0).count());

    // Open input file with O_DIRECT
    int fd = open(input_path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        // Fall back without O_DIRECT
        fd = open(input_path, O_RDONLY);
        printf("WARNING: O_DIRECT failed, using regular open\n");
    }
    if (fd < 0) { perror("open"); return 1; }

    CUfileDescr_t descr = {};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t fh;
    CUfileError_t reg_err = cuFileHandleRegister(&fh, &descr);
    if (reg_err.err != 0) {
        fprintf(stderr, "cuFileHandleRegister failed (err=%d). Common cause: filesystem doesn't support GDS or O_DIRECT path failed.\n", reg_err.err);
        return 2;
    }

    // Allocate GPU buffer + register
    uint8_t* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));
    CU_CHECK_ERR(cuFileBufRegister(d_buf, bytes, 0));

    // Read via cuFile
    auto t_read0 = std::chrono::high_resolution_clock::now();
    ssize_t n = cuFileRead(fh, d_buf, bytes, 0, 0);
    auto t_read1 = std::chrono::high_resolution_clock::now();
    double read_ms = std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();

    if (n != (ssize_t)bytes) {
        fprintf(stderr, "cuFileRead returned %zd, expected %lu\n", n, bytes);
        return 3;
    }

    printf("cuFileRead: %lu bytes in %.0f ms (%.2f GB/s)\n",
           n, read_ms, bytes / (read_ms * 1e6));

    // Verify by copying first 16 bytes back to host
    uint8_t check[16];
    CUDA_CHECK(cudaMemcpy(check, d_buf, 16, cudaMemcpyDeviceToHost));
    printf("First 16 bytes: ");
    for (int i = 0; i < 16; i++) printf("%02x", check[i]);
    printf("\n");

    cuFileBufDeregister(d_buf);
    cudaFree(d_buf);
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileDriverClose();

    printf("cuFile smoke test PASSED (%.2f GB/s)\n", bytes / (read_ms * 1e6));
    return 0;
}
