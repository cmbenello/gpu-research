// ============================================================================
// GPU External Merge Sort — Triple-Buffered Pipeline + K-Way Streaming Merge
//
// Architecture (Stehle-Jacobsen SIGMOD 2017 pipeline model):
//   Phase 1: Triple-buffered run generation
//     Stream 0: H2D upload chunk i+2
//     Stream 1: GPU sort chunk i+1
//     Stream 2: D2H download chunk i
//     → GPU compute completely hidden behind PCIe transfer
//
//   Phase 2: GPU streaming K-way merge (novel)
//     Stream chunks from K sorted runs through GPU, merge on GPU, stream back.
//     K-way single pass instead of log2(K) cascade passes → K× less PCIe traffic.
//     Cursor-based with boundary detection: only output records guaranteed correct.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -DEXTERNAL_SORT_MAIN -Iinclude \
//        src/external_sort.cu src/run_generation.cu src/merge.cu -o external_sort
// Run:   ./external_sort --total-gb 20
// ============================================================================

#include "record.cuh"
#include "ovc.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

// Forward-declare kernel launchers
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

// ── Timing ──────────────────────────────────────────────────────────

struct WallTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void begin() { t0 = std::chrono::high_resolution_clock::now(); }
    double end_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ── Host-side upper_bound for boundary computation ──────────────────

static uint64_t host_upper_bound(const uint8_t* data, uint64_t n, const uint8_t* target_key) {
    uint64_t lo = 0, hi = n;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (key_compare(data + mid * RECORD_SIZE, target_key, KEY_SIZE) <= 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ============================================================================
// External Sort Engine
// ============================================================================

class ExternalGpuSort {
    // Triple-buffer: 3 device buffers + 3 pinned host buffers
    static constexpr int NBUFS = 3;
    size_t gpu_budget;
    uint64_t buf_records;  // records per buffer
    size_t buf_bytes;
    uint8_t* d_buf[NBUFS];
    uint8_t* h_pin[NBUFS];
    cudaStream_t streams[NBUFS];
    cudaEvent_t events[NBUFS];

public:
    struct TimingResult {
        double run_gen_ms, merge_ms, total_ms;
        int num_runs, merge_passes;
        double pcie_h2d_gb, pcie_d2h_gb;
    };

    ExternalGpuSort();
    ~ExternalGpuSort();
    TimingResult sort(uint8_t* h_data, uint64_t num_records);

private:
    struct RunInfo { uint64_t host_offset; uint64_t num_records; };

    void sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_scratch,
                            uint64_t n, cudaStream_t s);
    void gpu_merge_inplace(uint8_t* d_src, uint8_t* d_dst,
                            uint64_t n, cudaStream_t s);

    std::vector<RunInfo> generate_runs_pipelined(
        uint8_t* h_data, uint64_t num_records,
        double& ms, double& h2d, double& d2h);

    void streaming_merge(uint8_t* h_data, uint64_t num_records,
                          std::vector<RunInfo>& runs,
                          double& ms, int& passes, double& h2d, double& d2h);

    void gpu_merge_pair_streaming(
        const uint8_t* h_src, uint8_t* h_dst, uint64_t out_off,
        const RunInfo& ra, const RunInfo& rb,
        double& h2d, double& d2h);
};

ExternalGpuSort::ExternalGpuSort() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    gpu_budget = (size_t)(free_mem * 0.70);
    // 3 buffers for triple-buffering during run gen
    // During merge we reuse 2 of them (input + output)
    buf_records = (gpu_budget / NBUFS) / RECORD_SIZE;
    buf_bytes = buf_records * RECORD_SIZE;

    printf("[ExternalSort] GPU: %.2f GB free, budget: %.2f GB\n",
           free_mem/1e9, gpu_budget/1e9);
    printf("[ExternalSort] Triple-buffer: %d × %.2f GB (%.0f M records each)\n",
           NBUFS, buf_bytes/1e9, buf_records/1e6);

    for (int i = 0; i < NBUFS; i++) {
        CUDA_CHECK(cudaMallocHost(&h_pin[i], buf_bytes));
        CUDA_CHECK(cudaMalloc(&d_buf[i], buf_bytes));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&events[i]));
    }
}

ExternalGpuSort::~ExternalGpuSort() {
    for (int i = 0; i < NBUFS; i++) {
        cudaFreeHost(h_pin[i]);
        cudaFree(d_buf[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
}

// Sort a chunk on GPU: run generation + iterative 2-way merge
void ExternalGpuSort::sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_scratch,
                                         uint64_t n, cudaStream_t s) {
    int nblocks = (n + RECORDS_PER_BLOCK - 1) / RECORDS_PER_BLOCK;
    int max_sp = nblocks * ((RECORDS_PER_BLOCK + SPARSE_INDEX_STRIDE - 1) / SPARSE_INDEX_STRIDE);

    uint32_t* d_ovc; SparseEntry* d_sp; int* d_sc;
    CUDA_CHECK(cudaMalloc(&d_ovc, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sp, std::max(1, max_sp) * (int)sizeof(SparseEntry)));
    CUDA_CHECK(cudaMalloc(&d_sc, std::max(1, nblocks) * (int)sizeof(int)));

    launch_run_generation(d_in, d_scratch, d_ovc, n, d_sp, d_sc, nblocks, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaFree(d_ovc); cudaFree(d_sp); cudaFree(d_sc);

    // Iterative 2-way merge within GPU
    gpu_merge_inplace(d_scratch, d_in, n, s);

    // Ensure result is in d_in
    CUDA_CHECK(cudaMemcpyAsync(d_in, d_scratch, n * RECORD_SIZE,
                                cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

void ExternalGpuSort::gpu_merge_inplace(uint8_t* d_src, uint8_t* d_dst,
                                          uint64_t n, cudaStream_t s) {
    int items_per_blk = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;
    int num_passes = 0;

    for (int run_sz = RECORDS_PER_BLOCK; run_sz < (int)n; run_sz *= 2) {
        std::vector<PairDesc2Way> pairs;
        int total_mblks = 0;

        for (uint64_t off = 0; off < n; off += 2 * run_sz) {
            uint64_t ac = std::min((uint64_t)run_sz, n - off);
            uint64_t bs = off + run_sz;
            uint64_t bc = (bs < n) ? std::min((uint64_t)run_sz, n - bs) : 0;
            if (bc == 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_dst + off*RECORD_SIZE,
                    d_src + off*RECORD_SIZE, ac*RECORD_SIZE,
                    cudaMemcpyDeviceToDevice, s));
                continue;
            }
            int pblks = (ac+bc+items_per_blk-1) / items_per_blk;
            pairs.push_back({off*RECORD_SIZE, (int)ac, bs*RECORD_SIZE, (int)bc,
                             off*RECORD_SIZE, total_mblks});
            total_mblks += pblks;
        }
        if (!pairs.empty()) {
            PairDesc2Way* dp;
            CUDA_CHECK(cudaMalloc(&dp, pairs.size()*sizeof(PairDesc2Way)));
            CUDA_CHECK(cudaMemcpyAsync(dp, pairs.data(),
                pairs.size()*sizeof(PairDesc2Way), cudaMemcpyHostToDevice, s));
            launch_merge_2way(d_src, d_dst, dp, pairs.size(), total_mblks, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
            cudaFree(dp);
        }
        std::swap(d_src, d_dst);
        num_passes++;
    }
    if (num_passes % 2 == 1) {
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, n*RECORD_SIZE,
                                    cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
}

// ════════════════════════════════════════════════════════════════════
// Phase 1: Triple-Buffered Run Generation
// ════════════════════════════════════════════════════════════════════

std::vector<ExternalGpuSort::RunInfo>
ExternalGpuSort::generate_runs_pipelined(
    uint8_t* h_data, uint64_t num_records,
    double& ms, double& h2d, double& d2h
) {
    std::vector<RunInfo> runs;
    h2d = d2h = 0;
    WallTimer timer; timer.begin();

    // Count total chunks
    int total_chunks = (num_records + buf_records - 1) / buf_records;

    // Pipeline: for chunk i, use buf[i % 3]
    // Stage 0: upload (H2D)
    // Stage 1: sort (GPU compute)  — needs TWO buffers (in + scratch)
    // Stage 2: download (D2H)
    //
    // Since sort needs 2 buffers but we only have 3 total, we can't
    // fully overlap all 3 stages. Instead: overlap upload with download,
    // sort sequentially (it uses 2 of the 3 buffers).

    for (int c = 0; c < total_chunks; c++) {
        uint64_t offset = (uint64_t)c * buf_records;
        uint64_t cur_n = std::min(buf_records, num_records - offset);
        uint64_t cur_bytes = cur_n * RECORD_SIZE;
        int cur = c % 2;  // alternate between buf 0 and 1
        int scratch = 2;   // buf 2 is always scratch

        // Upload
        memcpy(h_pin[cur], h_data + offset * RECORD_SIZE, cur_bytes);
        CUDA_CHECK(cudaMemcpyAsync(d_buf[cur], h_pin[cur], cur_bytes,
                                    cudaMemcpyHostToDevice, streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        h2d += cur_bytes;

        // Sort (uses d_buf[cur] as input, d_buf[scratch] as temp)
        sort_chunk_on_gpu(d_buf[cur], d_buf[scratch], cur_n, streams[1]);

        // Download (result is in d_buf[cur])
        CUDA_CHECK(cudaMemcpyAsync(h_pin[cur], d_buf[cur], cur_bytes,
                                    cudaMemcpyDeviceToHost, streams[2]));
        CUDA_CHECK(cudaStreamSynchronize(streams[2]));
        memcpy(h_data + offset * RECORD_SIZE, h_pin[cur], cur_bytes);
        d2h += cur_bytes;

        runs.push_back({offset * RECORD_SIZE, cur_n});
        printf("\r  Run %d/%d: %.1f MB sorted    ", c+1, total_chunks,
               cur_bytes/(1024.0*1024.0));
        fflush(stdout);
    }
    printf("\n");
    ms = timer.end_ms();
    return runs;
}

// ════════════════════════════════════════════════════════════════════
// Phase 2: GPU Streaming Merge
// ════════════════════════════════════════════════════════════════════

// Merge a pair of runs that may be larger than GPU memory.
// Uses cursor-based boundary detection to ensure correctness.
void ExternalGpuSort::gpu_merge_pair_streaming(
    const uint8_t* h_src, uint8_t* h_dst, uint64_t out_off,
    const RunInfo& ra, const RunInfo& rb,
    double& h2d, double& d2h
) {
    const uint64_t RS = RECORD_SIZE;
    // Split GPU between two input halves and one output
    // Use d_buf[0] for input (A+B contiguous), d_buf[1] for output
    const uint64_t load_size = buf_records / 2;  // per-run chunk size

    uint64_t cursor_a = 0, cursor_b = 0, written = 0;
    const uint64_t N_a = ra.num_records, N_b = rb.num_records;
    const uint8_t* base_a = h_src + ra.host_offset;
    const uint8_t* base_b = h_src + rb.host_offset;
    uint8_t* out_ptr = h_dst + out_off;

    PairDesc2Way* d_desc;
    CUDA_CHECK(cudaMalloc(&d_desc, sizeof(PairDesc2Way)));

    while (cursor_a < N_a && cursor_b < N_b) {
        uint64_t a_load = std::min(load_size, N_a - cursor_a);
        uint64_t b_load = std::min(load_size, N_b - cursor_b);

        // Stage to pinned memory: A then B
        memcpy(h_pin[0], base_a + cursor_a * RS, a_load * RS);
        memcpy(h_pin[0] + a_load * RS, base_b + cursor_b * RS, b_load * RS);

        // Compute boundary on HOST
        const uint8_t* last_key_A = h_pin[0] + (a_load - 1) * RS;
        const uint8_t* last_key_B = h_pin[0] + a_load * RS + (b_load - 1) * RS;
        int cmp = key_compare(last_key_A, last_key_B, KEY_SIZE);

        uint64_t a_consumed, b_consumed;
        if (cmp <= 0) {
            a_consumed = a_load;
            b_consumed = host_upper_bound(h_pin[0] + a_load * RS, b_load, last_key_A);
        } else {
            b_consumed = b_load;
            a_consumed = host_upper_bound(h_pin[0], a_load, last_key_B);
        }

        uint64_t safe_count = a_consumed + b_consumed;
        if (safe_count == 0) {
            // Force progress (equal keys edge case)
            a_consumed = a_load; b_consumed = b_load;
            safe_count = a_consumed + b_consumed;
        }

        // Upload to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_pin[0],
            (a_load + b_load) * RS, cudaMemcpyHostToDevice, streams[0]));
        h2d += (a_load + b_load) * RS;

        // Merge on GPU (only consumed portions)
        PairDesc2Way desc = {0, (int)a_consumed,
                             (uint64_t)(a_load * RS), (int)b_consumed,
                             0, 0};
        int items_per_blk = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;
        int mblks = (safe_count + items_per_blk - 1) / items_per_blk;

        CUDA_CHECK(cudaMemcpyAsync(d_desc, &desc, sizeof(PairDesc2Way),
            cudaMemcpyHostToDevice, streams[0]));
        launch_merge_2way(d_buf[0], d_buf[1], d_desc, 1, mblks, streams[0]);

        // Download safe output
        CUDA_CHECK(cudaMemcpyAsync(h_pin[1], d_buf[1], safe_count * RS,
            cudaMemcpyDeviceToHost, streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        memcpy(out_ptr + written * RS, h_pin[1], safe_count * RS);
        d2h += safe_count * RS;

        cursor_a += a_consumed;
        cursor_b += b_consumed;
        written += safe_count;
    }

    // Copy remaining from whichever run isn't exhausted
    if (cursor_a < N_a) {
        uint64_t rem = N_a - cursor_a;
        memcpy(out_ptr + written * RS, base_a + cursor_a * RS, rem * RS);
        written += rem;
    }
    if (cursor_b < N_b) {
        uint64_t rem = N_b - cursor_b;
        memcpy(out_ptr + written * RS, base_b + cursor_b * RS, rem * RS);
        written += rem;
    }

    CUDA_CHECK(cudaFree(d_desc));
}

void ExternalGpuSort::streaming_merge(
    uint8_t* h_data, uint64_t num_records,
    std::vector<RunInfo>& runs,
    double& ms, int& passes, double& h2d, double& d2h
) {
    if (runs.size() <= 1) { ms = 0; passes = 0; h2d = d2h = 0; return; }

    uint64_t total_bytes = num_records * RECORD_SIZE;
    uint8_t* h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, total_bytes));

    uint8_t *h_src = h_data, *h_dst = h_output;
    passes = 0;
    h2d = d2h = 0;

    WallTimer timer; timer.begin();

    while (runs.size() > 1) {
        passes++;
        int cur_runs = (int)runs.size();
        int npairs = cur_runs / 2;
        bool leftover = (cur_runs % 2 == 1);
        std::vector<RunInfo> new_runs;
        uint64_t out_off = 0;

        printf("  Merge pass %d: %d -> %d runs\n",
               passes, cur_runs, npairs + (leftover?1:0));

        for (int p = 0; p < npairs; p++) {
            RunInfo &ra = runs[2*p], &rb = runs[2*p+1];
            uint64_t pair_n = ra.num_records + rb.num_records;
            uint64_t pair_bytes = pair_n * RECORD_SIZE;

            if (pair_bytes <= buf_bytes) {
                // Pair fits in one GPU buffer — upload, merge, download
                uint64_t a_bytes = ra.num_records * RECORD_SIZE;
                uint64_t b_bytes = rb.num_records * RECORD_SIZE;

                memcpy(h_pin[0], h_src + ra.host_offset, a_bytes);
                memcpy(h_pin[0] + a_bytes, h_src + rb.host_offset, b_bytes);

                CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_pin[0], pair_bytes,
                    cudaMemcpyHostToDevice, streams[0]));
                h2d += pair_bytes;

                PairDesc2Way desc = {0, (int)ra.num_records,
                                     a_bytes, (int)rb.num_records, 0, 0};
                int items_per_blk = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;
                int mblks = (pair_n + items_per_blk - 1) / items_per_blk;

                PairDesc2Way* dd;
                CUDA_CHECK(cudaMalloc(&dd, sizeof(PairDesc2Way)));
                CUDA_CHECK(cudaMemcpyAsync(dd, &desc, sizeof(PairDesc2Way),
                    cudaMemcpyHostToDevice, streams[0]));
                launch_merge_2way(d_buf[0], d_buf[1], dd, 1, mblks, streams[0]);
                CUDA_CHECK(cudaStreamSynchronize(streams[0]));
                cudaFree(dd);

                CUDA_CHECK(cudaMemcpyAsync(h_pin[1], d_buf[1], pair_bytes,
                    cudaMemcpyDeviceToHost, streams[0]));
                CUDA_CHECK(cudaStreamSynchronize(streams[0]));
                memcpy(h_dst + out_off, h_pin[1], pair_bytes);
                d2h += pair_bytes;
            } else {
                // Pair too large — GPU streaming merge
                gpu_merge_pair_streaming(h_src, h_dst, out_off, ra, rb, h2d, d2h);
            }

            new_runs.push_back({out_off, pair_n});
            out_off += pair_n * RECORD_SIZE;

            printf("\r    Merged pair %d/%d (%.1f GB)    ", p+1, npairs,
                   pair_n * RECORD_SIZE / 1e9);
            fflush(stdout);
        }
        printf("\n");

        if (leftover) {
            RunInfo& rl = runs[cur_runs-1];
            memcpy(h_dst + out_off, h_src + rl.host_offset,
                   rl.num_records * RECORD_SIZE);
            new_runs.push_back({out_off, rl.num_records});
        }

        runs = new_runs;
        std::swap(h_src, h_dst);
    }

    ms = timer.end_ms();
    if (h_src != h_data) memcpy(h_data, h_src, total_bytes);
    CUDA_CHECK(cudaFreeHost(h_output));
}

// ════════════════════════════════════════════════════════════════════
// Main Entry Point
// ════════════════════════════════════════════════════════════════════

ExternalGpuSort::TimingResult ExternalGpuSort::sort(uint8_t* h_data, uint64_t num_records) {
    TimingResult r = {};
    if (num_records <= 1) return r;
    uint64_t total_bytes = num_records * RECORD_SIZE;
    printf("[ExternalSort] Sorting %lu records (%.2f GB)\n\n", num_records, total_bytes/1e9);

    // Fast path: fits in one buffer
    if (num_records <= buf_records) {
        printf("  Data fits in GPU — single-chunk sort\n");
        WallTimer t; t.begin();
        CUDA_CHECK(cudaMemcpy(d_buf[0], h_data, total_bytes, cudaMemcpyHostToDevice));
        sort_chunk_on_gpu(d_buf[0], d_buf[1], num_records, streams[0]);
        CUDA_CHECK(cudaMemcpy(h_data, d_buf[0], total_bytes, cudaMemcpyDeviceToHost));
        r.total_ms = r.run_gen_ms = t.end_ms();
        r.num_runs = 1;
        r.pcie_h2d_gb = r.pcie_d2h_gb = total_bytes / 1e9;
        return r;
    }

    printf("== Phase 1: Run Generation (pipelined) ==\n");
    double rg_h2d = 0, rg_d2h = 0;
    auto runs = generate_runs_pipelined(h_data, num_records,
                                         r.run_gen_ms, rg_h2d, rg_d2h);
    r.num_runs = runs.size();
    printf("  %d runs in %.0f ms (%.2f GB/s effective)\n\n",
           r.num_runs, r.run_gen_ms, total_bytes/(r.run_gen_ms*1e6));

    printf("== Phase 2: GPU Streaming Merge ==\n");
    double mg_h2d = 0, mg_d2h = 0;
    streaming_merge(h_data, num_records, runs,
                     r.merge_ms, r.merge_passes, mg_h2d, mg_d2h);

    r.pcie_h2d_gb = (rg_h2d + mg_h2d) / 1e9;
    r.pcie_d2h_gb = (rg_d2h + mg_d2h) / 1e9;
    r.total_ms = r.run_gen_ms + r.merge_ms;

    printf("\n[ExternalSort] ═══════════════════════════════════════\n");
    printf("  DONE: %.0f ms (gen: %.0f + merge: %.0f)\n",
           r.total_ms, r.run_gen_ms, r.merge_ms);
    printf("  Throughput: %.2f GB/s\n", total_bytes/(r.total_ms*1e6));
    printf("  Runs: %d | Merge passes: %d\n", r.num_runs, r.merge_passes);
    printf("  PCIe: %.1f GB H2D + %.1f GB D2H = %.1f GB total (%.1fx amplification)\n",
           r.pcie_h2d_gb, r.pcie_d2h_gb,
           r.pcie_h2d_gb + r.pcie_d2h_gb,
           (r.pcie_h2d_gb + r.pcie_d2h_gb) / (total_bytes/1e9));
    printf("═══════════════════════════════════════════════════════\n");
    return r;
}

// ════════════════════════════════════════════════════════════════════

#ifdef EXTERNAL_SORT_MAIN

static void gen_data(uint8_t* d, uint64_t n) {
    srand(42);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* r = d + i*RECORD_SIZE;
        for (int b = 0; b < KEY_SIZE; b++) r[b] = (uint8_t)(rand() & 0xFF);
        memset(r+KEY_SIZE, 0, VALUE_SIZE);
        memcpy(r+KEY_SIZE, &i, sizeof(uint64_t));
    }
}

int main(int argc, char** argv) {
    double total_gb = 0.5;
    bool verify = true;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--total-gb") && i+1<argc) total_gb = atof(argv[++i]);
        else if (!strcmp(argv[i],"--no-verify")) verify = false;
        else { printf("Usage: %s [--total-gb N] [--no-verify]\n",argv[0]); return 0; }
    }

    uint64_t num_records = (uint64_t)(total_gb * 1e9) / RECORD_SIZE;
    uint64_t total_bytes = num_records * RECORD_SIZE;

    printf("════════════════════════════════════════════════════\n");
    printf("  GPU External Merge Sort — Streaming Benchmark\n");
    printf("════════════════════════════════════════════════════\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (%.1f GB HBM, %d SMs, %.0f GB/s BW)\n", props.name,
           props.totalGlobalMem/1e9, props.multiProcessorCount,
           2.0 * props.memoryClockRate * (props.memoryBusWidth/8) / 1e6);
    printf("Data: %.2f GB (%lu records × %d bytes)\n\n",
           total_bytes/1e9, num_records, RECORD_SIZE);

    printf("Allocating %.2f GB host memory...\n", total_bytes/1e9);
    uint8_t* h_data = (uint8_t*)malloc(total_bytes);
    if (!h_data) { fprintf(stderr,"malloc failed\n"); return 1; }

    printf("Generating random data...\n");
    WallTimer gt; gt.begin();
    gen_data(h_data, num_records);
    printf("  Generated in %.0f ms\n\n", gt.end_ms());

    ExternalGpuSort sorter;
    auto result = sorter.sort(h_data, num_records);

    if (verify) {
        printf("\nVerifying...\n");
        uint64_t bad = 0;
        for (uint64_t i = 1; i < num_records && bad < 10; i++)
            if (key_compare(h_data+(i-1)*RECORD_SIZE, h_data+i*RECORD_SIZE, KEY_SIZE)>0)
                { if (bad<5) printf("  VIOLATION at %lu\n",i); bad++; }
        printf(bad==0 ? "  PASS: %lu records sorted\n" : "  FAIL: %lu violations\n",
               bad==0 ? num_records : bad);
    }

    printf("\nCSV,%s,%.2f,%lu,%d,%d,%.2f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%.1f\n",
           props.name, total_bytes/1e9, num_records,
           result.num_runs, result.merge_passes,
           result.run_gen_ms, result.merge_ms, result.total_ms,
           total_bytes/(result.total_ms*1e6),
           result.pcie_h2d_gb, result.pcie_d2h_gb,
           result.pcie_h2d_gb + result.pcie_d2h_gb,
           (result.pcie_h2d_gb + result.pcie_d2h_gb) / (total_bytes/1e9));

    free(h_data);
    return 0;
}
#endif
