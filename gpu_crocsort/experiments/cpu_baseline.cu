// Build: nvcc -O3 -std=c++17 -arch=sm_80 -Iinclude experiments/cpu_baseline.cu src/run_generation.cu src/merge.cu src/host_sort.cu -o cpu_vs_gpu
// Run: ./cpu_vs_gpu --num-records 1000000

#include "record.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <chrono>

#if __has_include(<execution>)
#include <execution>
#define HAS_PARALLEL_STL 1
#else
#define HAS_PARALLEL_STL 0
#endif

// GPU sort entry point (defined in host_sort.cu)
extern void gpu_crocsort_in_hbm(uint8_t* d_data, uint64_t num_records, bool verify);

// ── Random data generation (matches main.cu: seed 42, 100-byte records) ──

static void generate_random_records(uint8_t* h_data, uint64_t num_records, unsigned seed) {
    srand(seed);
    for (uint64_t i = 0; i < num_records; i++) {
        uint8_t* rec = h_data + i * RECORD_SIZE;
        for (int b = 0; b < KEY_SIZE; b++)
            rec[b] = (uint8_t)(rand() & 0xFF);
        memset(rec + KEY_SIZE, 0, VALUE_SIZE);
        uint64_t idx = i;
        memcpy(rec + KEY_SIZE, &idx, sizeof(uint64_t));
    }
}

// ── CPU sort: index-based std::sort with memcmp key comparison ───────

struct KeyCmp {
    const uint8_t* data;
    KeyCmp(const uint8_t* d) : data(d) {}
    bool operator()(uint32_t a, uint32_t b) const {
        return memcmp(data + (uint64_t)a * RECORD_SIZE,
                      data + (uint64_t)b * RECORD_SIZE, KEY_SIZE) < 0;
    }
};

static void scatter_by_index(const uint8_t* src, uint8_t* dst,
                             const std::vector<uint32_t>& idx, uint64_t n) {
    for (uint64_t i = 0; i < n; i++)
        memcpy(dst + i * RECORD_SIZE, src + (uint64_t)idx[i] * RECORD_SIZE, RECORD_SIZE);
}

// ── CPU 2-way merge sort (simulates CrocSort's merge strategy) ──────

static void merge_pass(const uint8_t* src, uint8_t* dst,
                       uint64_t n, uint64_t run_len) {
    for (uint64_t base = 0; base < n; base += 2 * run_len) {
        uint64_t mid = std::min(base + run_len, n);
        uint64_t end = std::min(base + 2 * run_len, n);
        uint64_t i = base, j = mid, k = base;
        while (i < mid && j < end) {
            if (memcmp(src + i * RECORD_SIZE, src + j * RECORD_SIZE, KEY_SIZE) <= 0)
                memcpy(dst + k++ * RECORD_SIZE, src + i++ * RECORD_SIZE, RECORD_SIZE);
            else
                memcpy(dst + k++ * RECORD_SIZE, src + j++ * RECORD_SIZE, RECORD_SIZE);
        }
        while (i < mid)
            memcpy(dst + k++ * RECORD_SIZE, src + i++ * RECORD_SIZE, RECORD_SIZE);
        while (j < end)
            memcpy(dst + k++ * RECORD_SIZE, src + j++ * RECORD_SIZE, RECORD_SIZE);
    }
}

static double cpu_merge_sort(uint8_t* data, uint64_t n) {
    uint64_t bytes = n * RECORD_SIZE;
    uint8_t* buf = (uint8_t*)malloc(bytes);
    uint8_t* src = data;
    uint8_t* dst = buf;

    // Sort initial runs of RECORDS_PER_BLOCK using std::sort on indices
    std::vector<uint32_t> idx(RECORDS_PER_BLOCK);
    for (uint64_t base = 0; base < n; base += RECORDS_PER_BLOCK) {
        uint64_t count = std::min((uint64_t)RECORDS_PER_BLOCK, n - base);
        std::iota(idx.begin(), idx.begin() + count, 0u);
        KeyCmp cmp(src + base * RECORD_SIZE);
        std::sort(idx.begin(), idx.begin() + count, cmp);
        // Scatter in-place via temp
        for (uint64_t i = 0; i < count; i++)
            memcpy(dst + (base + i) * RECORD_SIZE,
                   src + (base + idx[i]) * RECORD_SIZE, RECORD_SIZE);
    }
    std::swap(src, dst);

    // 2-way merge passes
    for (uint64_t run_len = RECORDS_PER_BLOCK; run_len < n; run_len *= 2) {
        merge_pass(src, dst, n, run_len);
        std::swap(src, dst);
    }

    // Ensure result ends up in data
    if (src != data) memcpy(data, src, bytes);
    free(buf);
    return 0; // timing done externally
}

// ── Timing helper ────────────────────────────────────────────────────

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        steady_clock::now().time_since_epoch()).count();
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    uint64_t num_records = 1000000;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--num-records") == 0 && i + 1 < argc)
            num_records = strtoull(argv[++i], nullptr, 10);
    }

    uint64_t total_bytes = num_records * RECORD_SIZE;
    printf("CPU vs GPU Sort Benchmark\n");
    printf("  Records: %lu  (%lu bytes each, %.2f MB total)\n",
           num_records, (unsigned long)RECORD_SIZE, total_bytes / (1024.0 * 1024.0));
    printf("  Key size: %d bytes, seed: 42\n\n", KEY_SIZE);

    // ── Generate canonical data ──
    uint8_t* original = (uint8_t*)malloc(total_bytes);
    generate_random_records(original, num_records, 42);

    uint8_t* work = (uint8_t*)malloc(total_bytes);
    std::vector<uint32_t> indices(num_records);

    // ════════════════════════════════════════
    // 1. CPU std::sort (index-based)
    // ════════════════════════════════════════
    memcpy(work, original, total_bytes);
    std::iota(indices.begin(), indices.end(), 0u);
    KeyCmp cmp(work);

    double t0 = now_ms();
    std::sort(indices.begin(), indices.end(), cmp);
    double cpu_sort_ms = now_ms() - t0;

    // Include scatter time (records must be reordered)
    uint8_t* sorted_buf = (uint8_t*)malloc(total_bytes);
    t0 = now_ms();
    scatter_by_index(work, sorted_buf, indices, num_records);
    cpu_sort_ms += now_ms() - t0;
    free(sorted_buf);

    double cpu_sort_gbs = total_bytes / (cpu_sort_ms * 1e6);
    printf("CPU std::sort:    %8.2f ms  (%.2f GB/s)\n", cpu_sort_ms, cpu_sort_gbs);

    // ════════════════════════════════════════
    // 2. CPU parallel sort
    // ════════════════════════════════════════
    double par_sort_ms, par_sort_gbs;
#if HAS_PARALLEL_STL
    memcpy(work, original, total_bytes);
    std::iota(indices.begin(), indices.end(), 0u);
    KeyCmp cmp2(work);

    t0 = now_ms();
    std::sort(std::execution::par_unseq, indices.begin(), indices.end(), cmp2);
    par_sort_ms = now_ms() - t0;

    sorted_buf = (uint8_t*)malloc(total_bytes);
    t0 = now_ms();
    scatter_by_index(work, sorted_buf, indices, num_records);
    par_sort_ms += now_ms() - t0;
    free(sorted_buf);

    par_sort_gbs = total_bytes / (par_sort_ms * 1e6);
    printf("CPU parallel:     %8.2f ms  (%.2f GB/s)\n", par_sort_ms, par_sort_gbs);
#else
    par_sort_ms = cpu_sort_ms;
    par_sort_gbs = cpu_sort_gbs;
    printf("CPU parallel:     (unavailable, <execution> not found)\n");
#endif

    // ════════════════════════════════════════
    // 3. CPU 2-way merge sort
    // ════════════════════════════════════════
    memcpy(work, original, total_bytes);

    t0 = now_ms();
    cpu_merge_sort(work, num_records);
    double merge_sort_ms = now_ms() - t0;
    double merge_sort_gbs = total_bytes / (merge_sort_ms * 1e6);
    printf("CPU 2-way merge:  %8.2f ms  (%.2f GB/s)\n", merge_sort_ms, merge_sort_gbs);

    // ════════════════════════════════════════
    // 4. GPU CrocSort
    // ════════════════════════════════════════
    uint8_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, original, total_bytes, cudaMemcpyHostToDevice));

    // Warm up
    gpu_crocsort_in_hbm(d_data, num_records, false);

    // Timed run
    CUDA_CHECK(cudaMemcpy(d_data, original, total_bytes, cudaMemcpyHostToDevice));
    GpuTimer gpu_timer;
    gpu_timer.begin();
    gpu_crocsort_in_hbm(d_data, num_records, false);
    float gpu_ms = gpu_timer.end();
    double gpu_gbs = total_bytes / (gpu_ms * 1e6);

    CUDA_CHECK(cudaFree(d_data));

    // ════════════════════════════════════════
    // Comparison table
    // ════════════════════════════════════════
    printf("\n========================================\n");
    printf("Results (%lu records, %.2f MB)\n", num_records, total_bytes / (1024.0 * 1024.0));
    printf("========================================\n");
    printf("CPU std::sort:    %8.2f ms  (%.2f GB/s)\n", cpu_sort_ms, cpu_sort_gbs);
#if HAS_PARALLEL_STL
    printf("CPU parallel:     %8.2f ms  (%.2f GB/s)\n", par_sort_ms, par_sort_gbs);
#else
    printf("CPU parallel:     N/A\n");
#endif
    printf("CPU 2-way merge:  %8.2f ms  (%.2f GB/s)\n", merge_sort_ms, merge_sort_gbs);
    printf("GPU CrocSort:     %8.2f ms  (%.2f GB/s)\n", gpu_ms, gpu_gbs);
    printf("----------------------------------------\n");
    printf("GPU speedup:      %.1fx over std::sort", cpu_sort_ms / gpu_ms);
#if HAS_PARALLEL_STL
    printf(", %.1fx over parallel", par_sort_ms / gpu_ms);
#endif
    printf(", %.1fx over 2-way merge\n", merge_sort_ms / gpu_ms);
    printf("========================================\n");

    free(original);
    free(work);
    return 0;
}
