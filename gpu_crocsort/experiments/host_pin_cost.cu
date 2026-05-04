// 19.4 — cudaHostRegister cost vs file size.
//
// Measures the cold-pin overhead that dominates SF1000+ wall time.
// Pins existing files via mmap+cudaHostRegister and measures the
// wall time to do so.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

#define CHK(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA %d: %s\n", err__, cudaGetErrorString(err__)); std::exit(1); } } while(0)

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file> [<file2> ...]\n", argv[0]);
        return 1;
    }
    CHK(cudaSetDevice(0));
    printf("file,size_gb,mmap_ms,register_ms,unregister_ms,munmap_ms,total_ms,gb_per_s\n");
    for (int i = 1; i < argc; i++) {
        const char* path = argv[i];
        struct stat st;
        if (stat(path, &st) != 0) { perror(path); continue; }
        size_t sz = st.st_size;
        double size_gb = sz / 1e9;

        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror(path); continue; }

        double t0 = now_ms();
        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { perror("mmap"); close(fd); continue; }
        double t_mmap = now_ms();

        // Register pages with CUDA — this is the slow step
        // Allow flag override via env var
        unsigned int flags = cudaHostRegisterDefault;
        const char* flag_env = getenv("PIN_FLAGS");
        if (flag_env) {
            if (!strcmp(flag_env, "default"))   flags = cudaHostRegisterDefault;
            else if (!strcmp(flag_env, "ro"))   flags = cudaHostRegisterReadOnly;
            else if (!strcmp(flag_env, "mapped")) flags = cudaHostRegisterMapped;
            else if (!strcmp(flag_env, "ro_mapped")) flags = cudaHostRegisterReadOnly | cudaHostRegisterMapped;
            else if (!strcmp(flag_env, "portable")) flags = cudaHostRegisterPortable;
        }
        cudaError_t err = cudaHostRegister(m, sz, flags);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: cudaHostRegister failed: %s\n", path, cudaGetErrorString(err));
            munmap(m, sz);
            close(fd);
            continue;
        }
        double t_reg = now_ms();

        cudaHostUnregister(m);
        double t_unreg = now_ms();
        munmap(m, sz);
        double t_munmap = now_ms();
        close(fd);

        double mmap_ms     = t_mmap - t0;
        double register_ms = t_reg - t_mmap;
        double unreg_ms    = t_unreg - t_reg;
        double munmap_ms   = t_munmap - t_unreg;
        double total_ms    = t_munmap - t0;
        double gbs         = size_gb / (total_ms / 1e3);
        printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f\n",
               path, size_gb, mmap_ms, register_ms, unreg_ms, munmap_ms, total_ms, gbs);
        fflush(stdout);
    }
    return 0;
}
