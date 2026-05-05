// 19.21 — Sort a single compact bucket file (40-byte records).
// Reads (32-byte key + 8-byte offset) records, sorts by key, writes
// 8-byte sorted offsets to output file. Output indexes the original input.bin.
//
// Build: nvcc -O3 -arch=sm_90 -std=c++17 experiments/sort_compact_bucket.cu -o experiments/sort_compact_bucket
//        (skip Makefile — standalone tool)
// Run:   ./sort_compact_bucket BUCKET.bin OUT.bin
//
// At SF1500 K=16, each bucket is ~22.8 GB (570M × 40 B). Output is ~4.6 GB
// (570M × 8 B). Per-bucket wall projected ~10-12 sec on H100.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

static constexpr int COMPACT_KEY_SIZE = 32;
static constexpr int COMPACT_REC_SIZE = COMPACT_KEY_SIZE + 8;

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

// Extract 8-byte chunk of compact key for one LSD pass: out[i] = compact[perm[i]] bytes [byte_off..byte_off+7]
__global__ void extract_chunk(
    const uint8_t* __restrict__ compact_recs,  // n × 40 bytes
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ out,
    uint64_t n,
    int byte_off,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t idx = perm[i];
    const uint8_t* k = compact_recs + (uint64_t)idx * COMPACT_REC_SIZE + byte_off;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    out[i] = v;
}

__global__ void init_identity(uint32_t* perm, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) perm[i] = (uint32_t)i;
}

// Look up sorted offsets via final perm: out_offsets[i] = compact_recs[perm[i]].offset (last 8 bytes)
__global__ void gather_offsets(
    const uint8_t* __restrict__ compact_recs,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ out_offsets,
    uint64_t n
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t idx = perm[i];
    out_offsets[i] = *(const uint64_t*)(compact_recs + (uint64_t)idx * COMPACT_REC_SIZE + COMPACT_KEY_SIZE);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s BUCKET.bin OUT.bin\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];

    auto t0 = std::chrono::high_resolution_clock::now();

    struct stat st;
    if (stat(in_path, &st) != 0) { perror("stat"); return 1; }
    uint64_t total_bytes = st.st_size;
    uint64_t n = total_bytes / COMPACT_REC_SIZE;
    if (total_bytes != n * COMPACT_REC_SIZE) {
        fprintf(stderr, "ERROR: %s size %lu not a multiple of %d\n", in_path, total_bytes, COMPACT_REC_SIZE);
        return 1;
    }
    printf("Input: %lu records (%.2f GB compact)\n", n, total_bytes/1e9);

    // Allocate pinned host buffer + load
    auto t_load_start = std::chrono::high_resolution_clock::now();
    uint8_t* h_compact;
    CUDA_CHECK(cudaMallocHost(&h_compact, total_bytes));
    {
        FILE* f = fopen(in_path, "rb");
        if (!f) { perror("fopen"); return 1; }
        size_t r = fread(h_compact, 1, total_bytes, f);
        fclose(f);
        if (r != total_bytes) {
            fprintf(stderr, "short read %zu vs %lu\n", r, total_bytes);
            return 1;
        }
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    printf("Load: %.0f ms (%.2f GB/s)\n", load_ms, total_bytes/(load_ms*1e6));

    // Upload to GPU
    auto t_h2d_start = std::chrono::high_resolution_clock::now();
    uint8_t* d_compact;
    CUDA_CHECK(cudaMalloc(&d_compact, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_compact, h_compact, total_bytes, cudaMemcpyHostToDevice));
    auto t_h2d_end = std::chrono::high_resolution_clock::now();
    double h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();
    printf("H2D: %.0f ms (%.2f GB/s)\n", h2d_ms, total_bytes/(h2d_ms*1e6));

    // Sort: 4 LSD passes of 8 bytes each on the 32-byte compact key
    auto t_sort_start = std::chrono::high_resolution_clock::now();

    // Allocate perm + scratch
    uint32_t *d_perm_a, *d_perm_b;
    CUDA_CHECK(cudaMalloc(&d_perm_a, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_perm_b, n * sizeof(uint32_t)));
    init_identity<<<(n+255)/256, 256>>>(d_perm_a, n);

    uint64_t *d_keys_a, *d_keys_b;
    CUDA_CHECK(cudaMalloc(&d_keys_a, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_b, n * sizeof(uint64_t)));

    // Determine CUB temp storage size (using radix sort pairs)
    size_t cub_temp_bytes = 0;
    cub::DoubleBuffer<uint64_t> kb(d_keys_a, d_keys_b);
    cub::DoubleBuffer<uint32_t> vb(d_perm_a, d_perm_b);
    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, kb, vb, (int)n, 0, 64);
    void* d_cub_temp;
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
    printf("CUB temp: %.2f GB\n", cub_temp_bytes/1e9);

    // 4 LSD passes from least-significant chunk (bytes 24-31) to most-significant (bytes 0-7)
    for (int chunk = 3; chunk >= 0; chunk--) {
        int byte_off = chunk * 8;
        extract_chunk<<<(n+255)/256, 256>>>(d_compact, vb.Current(), kb.Current(), n, byte_off, 8);
        cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_temp_bytes, kb, vb, (int)n, 0, 64);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_sort_end = std::chrono::high_resolution_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();
    printf("Sort: %.0f ms (4 LSD passes)\n", sort_ms);

    // Gather offsets in sorted order
    auto t_gather_start = std::chrono::high_resolution_clock::now();
    uint64_t* d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, n * sizeof(uint64_t)));
    gather_offsets<<<(n+255)/256, 256>>>(d_compact, vb.Current(), d_offsets, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download
    uint64_t* h_offsets;
    CUDA_CHECK(cudaMallocHost(&h_offsets, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(h_offsets, d_offsets, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    auto t_gather_end = std::chrono::high_resolution_clock::now();
    double gather_ms = std::chrono::duration<double, std::milli>(t_gather_end - t_gather_start).count();
    printf("Gather + D2H: %.0f ms\n", gather_ms);

    // Write to output
    auto t_write_start = std::chrono::high_resolution_clock::now();
    {
        FILE* f = fopen(out_path, "wb");
        if (!f) { perror("fopen out"); return 1; }
        size_t w = fwrite(h_offsets, sizeof(uint64_t), n, f);
        fclose(f);
        if (w != n) {
            fprintf(stderr, "short write %zu vs %lu\n", w, n);
            return 1;
        }
    }
    auto t_write_end = std::chrono::high_resolution_clock::now();
    double write_ms = std::chrono::duration<double, std::milli>(t_write_end - t_write_start).count();
    printf("Write: %.0f ms (%.2f GB written)\n", write_ms, n*8/1e9);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total: %.0f ms\n", total_ms);
    printf("CSV,sort_compact_bucket,records=%lu,total_ms=%.0f,load_ms=%.0f,h2d_ms=%.0f,sort_ms=%.0f,gather_ms=%.0f,write_ms=%.0f\n",
           n, total_ms, load_ms, h2d_ms, sort_ms, gather_ms, write_ms);

    cudaFreeHost(h_compact);
    cudaFreeHost(h_offsets);
    cudaFree(d_compact);
    cudaFree(d_perm_a);
    cudaFree(d_perm_b);
    cudaFree(d_keys_a);
    cudaFree(d_keys_b);
    cudaFree(d_offsets);
    cudaFree(d_cub_temp);

    return 0;
}
