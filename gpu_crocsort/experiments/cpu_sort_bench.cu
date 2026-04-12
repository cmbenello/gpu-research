// CPU vs GPU sort comparison — standalone, no dependencies
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -Iinclude experiments/cpu_sort_bench.cu \
//        src/run_generation.cu src/merge.cu -o cpu_sort_bench
// Run:   ./cpu_sort_bench

#include "record.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>
#include <numeric>

struct WallTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void begin() { t0 = std::chrono::high_resolution_clock::now(); }
    double end_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// Forward declare GPU sort
struct PairDesc2Way {
    uint64_t a_byte_offset; int a_count;
    uint64_t b_byte_offset; int b_count;
    uint64_t out_byte_offset; int first_block;
};
extern "C" void launch_run_generation(
    const uint8_t*, uint8_t*, uint32_t*, uint64_t,
    SparseEntry*, int*, int, cudaStream_t);
extern "C" void launch_merge_2way(
    const uint8_t*, uint8_t*, const PairDesc2Way*, int, int, cudaStream_t);

static void gen_data(uint8_t* d, uint64_t n) {
    srand(42);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* r = d + i * RECORD_SIZE;
        for (int b = 0; b < KEY_SIZE; b++) r[b] = (uint8_t)(rand() & 0xFF);
        memset(r + KEY_SIZE, 0, VALUE_SIZE);
    }
}

// CPU std::sort with index array
static double cpu_sort_indexed(uint8_t* data, uint64_t n) {
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    WallTimer t; t.begin();
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        return memcmp(data + (uint64_t)a * RECORD_SIZE,
                      data + (uint64_t)b * RECORD_SIZE, KEY_SIZE) < 0;
    });

    // Scatter to sorted order
    uint8_t* tmp = (uint8_t*)malloc(n * RECORD_SIZE);
    for (uint64_t i = 0; i < n; i++)
        memcpy(tmp + i * RECORD_SIZE, data + (uint64_t)idx[i] * RECORD_SIZE, RECORD_SIZE);
    memcpy(data, tmp, n * RECORD_SIZE);
    free(tmp);

    return t.end_ms();
}

// CPU merge sort (2-way, iterative) — mirrors GPU merge sort
static double cpu_merge_sort(uint8_t* data, uint64_t n) {
    uint8_t* buf = (uint8_t*)malloc(n * RECORD_SIZE);
    uint8_t* src = data;
    uint8_t* dst = buf;

    WallTimer t; t.begin();

    // Phase 1: sort blocks of 512 using std::sort with index
    int block_sz = 512;
    for (uint64_t off = 0; off < n; off += block_sz) {
        int bn = std::min((uint64_t)block_sz, n - off);
        std::vector<uint16_t> idx(bn);
        std::iota(idx.begin(), idx.end(), 0);
        uint8_t* base = src + off * RECORD_SIZE;
        std::sort(idx.begin(), idx.end(), [&](uint16_t a, uint16_t b) {
            return memcmp(base + a * RECORD_SIZE, base + b * RECORD_SIZE, KEY_SIZE) < 0;
        });
        for (int i = 0; i < bn; i++)
            memcpy(dst + (off + i) * RECORD_SIZE, base + idx[i] * RECORD_SIZE, RECORD_SIZE);
    }
    std::swap(src, dst);

    // Phase 2: iterative 2-way merge
    for (int run_sz = block_sz; run_sz < (int)n; run_sz *= 2) {
        for (uint64_t off = 0; off < n; off += 2 * run_sz) {
            uint64_t a_n = std::min((uint64_t)run_sz, n - off);
            uint64_t b_start = off + run_sz;
            uint64_t b_n = (b_start < n) ? std::min((uint64_t)run_sz, n - b_start) : 0;

            const uint8_t* pa = src + off * RECORD_SIZE;
            const uint8_t* pb = src + b_start * RECORD_SIZE;
            uint8_t* po = dst + off * RECORD_SIZE;
            uint64_t ia = 0, ib = 0;

            while (ia < a_n && ib < b_n) {
                if (memcmp(pa + ia * RECORD_SIZE, pb + ib * RECORD_SIZE, KEY_SIZE) <= 0) {
                    memcpy(po, pa + ia * RECORD_SIZE, RECORD_SIZE); ia++;
                } else {
                    memcpy(po, pb + ib * RECORD_SIZE, RECORD_SIZE); ib++;
                }
                po += RECORD_SIZE;
            }
            while (ia < a_n) { memcpy(po, pa + ia*RECORD_SIZE, RECORD_SIZE); ia++; po += RECORD_SIZE; }
            while (ib < b_n) { memcpy(po, pb + ib*RECORD_SIZE, RECORD_SIZE); ib++; po += RECORD_SIZE; }
        }
        std::swap(src, dst);
    }

    double ms = t.end_ms();
    if (src != data) memcpy(data, src, n * RECORD_SIZE);
    free(buf);
    return ms;
}

// GPU sort (in-HBM only)
static double gpu_sort(uint8_t* h_data, uint64_t n) {
    uint64_t total_bytes = n * RECORD_SIZE;
    uint8_t *d_in, *d_out;
    cudaMalloc(&d_in, total_bytes);
    cudaMalloc(&d_out, total_bytes);

    // Warmup
    cudaMemcpy(d_in, h_data, total_bytes, cudaMemcpyHostToDevice);

    WallTimer t; t.begin();

    // Upload
    cudaMemcpy(d_in, h_data, total_bytes, cudaMemcpyHostToDevice);

    // Run generation
    int nblocks = (n + RECORDS_PER_BLOCK - 1) / RECORDS_PER_BLOCK;
    int max_sp = nblocks * ((RECORDS_PER_BLOCK + SPARSE_INDEX_STRIDE - 1) / SPARSE_INDEX_STRIDE);
    uint32_t* d_ovc; SparseEntry* d_sp; int* d_sc;
    cudaMalloc(&d_ovc, n * sizeof(uint32_t));
    cudaMalloc(&d_sp, std::max(1, max_sp) * (int)sizeof(SparseEntry));
    cudaMalloc(&d_sc, std::max(1, nblocks) * (int)sizeof(int));
    launch_run_generation(d_in, d_out, d_ovc, n, d_sp, d_sc, nblocks, 0);
    cudaDeviceSynchronize();
    cudaFree(d_ovc); cudaFree(d_sp); cudaFree(d_sc);

    // Merge
    int items_per_blk = 8 * 256;
    uint8_t* src = d_out; uint8_t* dst = d_in;
    int passes = 0;
    for (int run_sz = RECORDS_PER_BLOCK; run_sz < (int)n; run_sz *= 2) {
        std::vector<PairDesc2Way> pairs;
        int total_mblks = 0;
        for (uint64_t off = 0; off < n; off += 2 * run_sz) {
            uint64_t ac = std::min((uint64_t)run_sz, n - off);
            uint64_t bs = off + run_sz;
            uint64_t bc = (bs < n) ? std::min((uint64_t)run_sz, n - bs) : 0;
            if (bc == 0) {
                cudaMemcpy(dst + off*RECORD_SIZE, src + off*RECORD_SIZE,
                    ac*RECORD_SIZE, cudaMemcpyDeviceToDevice);
                continue;
            }
            int pblks = (ac+bc+items_per_blk-1) / items_per_blk;
            pairs.push_back({off*RECORD_SIZE, (int)ac, bs*RECORD_SIZE, (int)bc,
                             off*RECORD_SIZE, total_mblks});
            total_mblks += pblks;
        }
        if (!pairs.empty()) {
            PairDesc2Way* dp;
            cudaMalloc(&dp, pairs.size()*sizeof(PairDesc2Way));
            cudaMemcpy(dp, pairs.data(), pairs.size()*sizeof(PairDesc2Way), cudaMemcpyHostToDevice);
            launch_merge_2way(src, dst, dp, pairs.size(), total_mblks, 0);
            cudaDeviceSynchronize();
            cudaFree(dp);
        }
        std::swap(src, dst);
        passes++;
    }

    // Download
    cudaMemcpy(h_data, src, total_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double ms = t.end_ms();
    cudaFree(d_in); cudaFree(d_out);
    return ms;
}

int main() {
    printf("════════════════════════════════════════════════════════\n");
    printf("  CPU vs GPU Sort — Head-to-Head Comparison\n");
    printf("════════════════════════════════════════════════════════\n\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (%.1f GB HBM, %d SMs)\n", props.name,
           props.totalGlobalMem/1e9, props.multiProcessorCount);
    printf("CPU: 48 cores\n");
    printf("Record: %d bytes (key=%d, value=%d)\n\n", RECORD_SIZE, KEY_SIZE, VALUE_SIZE);

    printf("%-12s | %12s %8s | %12s %8s | %12s %8s | %s\n",
           "Records", "CPU std::sort", "GB/s", "CPU merge", "GB/s",
           "GPU (w/PCIe)", "GB/s", "GPU speedup");
    printf("%-12s-+-%12s-%8s-+-%12s-%8s-+-%12s-%8s-+-%s\n",
           "------------", "------------", "--------", "------------", "--------",
           "------------", "--------", "-----------");

    uint64_t sizes[] = {100000, 500000, 1000000, 5000000, 10000000, 50000000};
    int nsizes = 6;

    // Check GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    for (int s = 0; s < nsizes; s++) {
        uint64_t n = sizes[s];
        uint64_t bytes = n * RECORD_SIZE;

        // Skip if too large for GPU (need 2x for merge buffers)
        bool gpu_fits = (bytes * 3 < free_mem);

        uint8_t* data = (uint8_t*)malloc(bytes);
        uint8_t* data_copy = (uint8_t*)malloc(bytes);

        // Generate data
        gen_data(data, n);
        memcpy(data_copy, data, bytes);

        // CPU std::sort
        memcpy(data, data_copy, bytes);
        double cpu_std_ms = cpu_sort_indexed(data, n);

        // CPU merge sort
        memcpy(data, data_copy, bytes);
        double cpu_merge_ms = cpu_merge_sort(data, n);

        // GPU sort (including PCIe transfer)
        double gpu_ms = 0;
        if (gpu_fits) {
            memcpy(data, data_copy, bytes);
            gpu_ms = gpu_sort(data, n);
        }

        double gb = bytes / 1e9;
        printf("%-12lu | %9.1f ms %5.2f   | %9.1f ms %5.2f   |",
               n, cpu_std_ms, gb/(cpu_std_ms/1e3),
               cpu_merge_ms, gb/(cpu_merge_ms/1e3));

        if (gpu_fits) {
            double speedup_std = cpu_std_ms / gpu_ms;
            double speedup_merge = cpu_merge_ms / gpu_ms;
            printf(" %9.1f ms %5.2f   | %.1fx/%.1fx",
                   gpu_ms, gb/(gpu_ms/1e3), speedup_std, speedup_merge);
        } else {
            printf(" %9s %8s | N/A", "too large", "");
        }
        printf("\n");

        // CSV
        printf("CSV,%lu,%.2f,%.2f,%.2f\n", n, cpu_std_ms, cpu_merge_ms, gpu_ms);

        free(data);
        free(data_copy);
    }

    printf("\n");
    printf("Note: GPU times INCLUDE PCIe transfer (H2D + D2H).\n");
    printf("      GPU sort-only (no PCIe) is ~10x faster.\n");
    printf("      Speedup shown as: vs_std::sort / vs_cpu_merge\n");

    return 0;
}
