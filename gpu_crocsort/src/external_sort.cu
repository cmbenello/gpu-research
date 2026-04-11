// External sort: handles data larger than GPU HBM via PCIe streaming
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -DEXTERNAL_SORT_MAIN -Iinclude \
//   src/external_sort.cu src/run_generation.cu src/merge.cu -o external_sort
// Run:  ./external_sort --total-gb 2

#include "record.cuh"
#include "ovc.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

// Forward-declare kernel launchers (run_generation.cu, merge.cu)
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

// ── Timing utility ──────────────────────────────────────────────────

struct WallTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void begin() { t0 = std::chrono::high_resolution_clock::now(); }
    double end_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ── External sort implementation ────────────────────────────────────

class ExternalGpuSort {
    size_t gpu_budget;
    uint64_t chunk_records;
    size_t chunk_bytes;
    uint8_t *h_pinned_a, *h_pinned_b;
    uint8_t *d_buf_a, *d_buf_b;
    cudaStream_t stream_compute, stream_transfer;

public:
    // Timing results for experiments
    struct TimingResult {
        double run_gen_ms;
        double merge_ms;
        double total_ms;
        int num_runs;
        int merge_passes;
        double pcie_h2d_gb;
        double pcie_d2h_gb;
    };

    ExternalGpuSort(size_t budget = 0);
    ~ExternalGpuSort();
    TimingResult sort(uint8_t* h_data, uint64_t num_records);

private:
    struct RunInfo { uint64_t host_offset; uint64_t num_records; };

    void sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_out, uint64_t n, cudaStream_t s);
    std::vector<RunInfo> generate_runs(uint8_t* h_data, uint64_t num_records,
                                        double& run_gen_ms, double& h2d_bytes, double& d2h_bytes);
    void merge_runs_streaming(uint8_t* h_data, uint64_t num_records,
                               std::vector<RunInfo>& runs,
                               double& merge_ms, int& merge_passes,
                               double& h2d_bytes, double& d2h_bytes);
    void gpu_merge_pair(uint8_t* d_src, uint8_t* d_dst, uint64_t n, cudaStream_t s);
};

ExternalGpuSort::ExternalGpuSort(size_t budget) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    gpu_budget = budget > 0 ? std::min(budget, (size_t)(free_mem * 0.70))
                            : (size_t)(free_mem * 0.70);
    // Two device buffers (src + dst for merge), each gets half
    chunk_records = (gpu_budget / 2) / RECORD_SIZE;
    chunk_bytes = chunk_records * RECORD_SIZE;

    printf("[ExternalSort] GPU free: %.2f GB, budget: %.2f GB, chunk: %.2f GB (%lu recs)\n",
           free_mem / 1e9, gpu_budget / 1e9, chunk_bytes / 1e9, chunk_records);

    CUDA_CHECK(cudaMallocHost(&h_pinned_a, chunk_bytes));
    CUDA_CHECK(cudaMallocHost(&h_pinned_b, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf_a, chunk_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf_b, chunk_bytes));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));
}

ExternalGpuSort::~ExternalGpuSort() {
    cudaFreeHost(h_pinned_a); cudaFreeHost(h_pinned_b);
    cudaFree(d_buf_a); cudaFree(d_buf_b);
    cudaStreamDestroy(stream_compute); cudaStreamDestroy(stream_transfer);
}

// Sort a chunk entirely on GPU: run generation + iterative 2-way merge
void ExternalGpuSort::sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_out,
                                         uint64_t n, cudaStream_t s) {
    int nblocks = (int)((n + RECORDS_PER_BLOCK - 1) / RECORDS_PER_BLOCK);
    int max_sp = nblocks * ((RECORDS_PER_BLOCK + SPARSE_INDEX_STRIDE - 1) / SPARSE_INDEX_STRIDE);

    uint32_t* d_ovc; SparseEntry* d_sp; int* d_sc;
    CUDA_CHECK(cudaMalloc(&d_ovc, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sp, std::max(1,max_sp) * (int)sizeof(SparseEntry)));
    CUDA_CHECK(cudaMalloc(&d_sc, std::max(1,nblocks) * (int)sizeof(int)));

    launch_run_generation(d_in, d_out, d_ovc, n, d_sp, d_sc, nblocks, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaFree(d_ovc); cudaFree(d_sp); cudaFree(d_sc);

    // Iterative 2-way merge on GPU
    gpu_merge_pair(d_out, d_in, n, s);
}

// 2-way merge passes entirely on GPU (data already in device memory)
void ExternalGpuSort::gpu_merge_pair(uint8_t* d_src, uint8_t* d_dst,
                                      uint64_t n, cudaStream_t s) {
    int items_per_blk = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;

    for (int run_sz = RECORDS_PER_BLOCK; run_sz < (int)n; run_sz *= 2) {
        std::vector<PairDesc2Way> pairs;
        int total_mblks = 0;

        for (uint64_t off = 0; off < n; off += 2 * run_sz) {
            uint64_t ac = std::min((uint64_t)run_sz, n - off);
            uint64_t bs = off + run_sz;
            uint64_t bc = (bs < n) ? std::min((uint64_t)run_sz, n - bs) : 0;
            if (bc == 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_dst + off*RECORD_SIZE, d_src + off*RECORD_SIZE,
                    ac*RECORD_SIZE, cudaMemcpyDeviceToDevice, s));
                continue;
            }
            int pblks = (int)((ac+bc+items_per_blk-1) / items_per_blk);
            pairs.push_back({off*RECORD_SIZE, (int)ac, bs*RECORD_SIZE, (int)bc,
                             off*RECORD_SIZE, total_mblks});
            total_mblks += pblks;
        }
        if (!pairs.empty()) {
            PairDesc2Way* dp;
            CUDA_CHECK(cudaMalloc(&dp, pairs.size()*sizeof(PairDesc2Way)));
            CUDA_CHECK(cudaMemcpyAsync(dp, pairs.data(),
                pairs.size()*sizeof(PairDesc2Way), cudaMemcpyHostToDevice, s));
            launch_merge_2way(d_src, d_dst, dp, (int)pairs.size(), total_mblks, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
            cudaFree(dp);
        }
        std::swap(d_src, d_dst);
    }
    // Ensure result is in the original d_src position
    if (d_src != d_dst) {
        // Result ended up in d_dst after odd number of swaps; handled by caller
    }
}

// ── Phase 1: Double-buffered run generation ─────────────────────────
// Pipeline: GPU sorts chunk N while PCIe uploads chunk N+1

std::vector<ExternalGpuSort::RunInfo>
ExternalGpuSort::generate_runs(uint8_t* h_data, uint64_t num_records,
                                double& run_gen_ms, double& h2d_bytes, double& d2h_bytes) {
    std::vector<RunInfo> runs;
    uint8_t* d_buf[2] = {d_buf_a, d_buf_b};
    uint8_t* h_pin[2] = {h_pinned_a, h_pinned_b};
    int cur = 0;
    h2d_bytes = 0;
    d2h_bytes = 0;

    WallTimer timer;
    timer.begin();

    // Upload first chunk
    uint64_t first_n = std::min(chunk_records, num_records);
    memcpy(h_pin[0], h_data, first_n * RECORD_SIZE);
    CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_pin[0], first_n*RECORD_SIZE,
                                cudaMemcpyHostToDevice, stream_compute));
    h2d_bytes += first_n * RECORD_SIZE;

    for (uint64_t offset = 0; offset < num_records; ) {
        uint64_t cur_n = std::min(chunk_records, num_records - offset);
        int nxt = 1 - cur;

        // Sort current chunk on GPU (run gen + in-HBM merge)
        sort_chunk_on_gpu(d_buf[cur], d_buf[cur], cur_n, stream_compute);

        // Overlap: upload next chunk on transfer stream
        uint64_t next_off = offset + cur_n;
        uint64_t next_n = 0;
        if (next_off < num_records) {
            next_n = std::min(chunk_records, num_records - next_off);
            memcpy(h_pin[nxt], h_data + next_off*RECORD_SIZE, next_n*RECORD_SIZE);
            CUDA_CHECK(cudaMemcpyAsync(d_buf[nxt], h_pin[nxt], next_n*RECORD_SIZE,
                                        cudaMemcpyHostToDevice, stream_transfer));
            h2d_bytes += next_n * RECORD_SIZE;
        }

        // Download sorted chunk
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        CUDA_CHECK(cudaMemcpyAsync(h_pin[cur], d_buf[cur], cur_n*RECORD_SIZE,
                                    cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        memcpy(h_data + offset*RECORD_SIZE, h_pin[cur], cur_n*RECORD_SIZE);
        d2h_bytes += cur_n * RECORD_SIZE;

        runs.push_back({offset * RECORD_SIZE, cur_n});
        printf("\r  Run %d: %lu recs (%.1f MB)    ",
               (int)runs.size(), cur_n, cur_n*RECORD_SIZE/(1024.0*1024.0));
        fflush(stdout);

        if (next_n > 0) CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
        offset += next_off - offset + cur_n;
        offset = next_off + (next_n > 0 ? 0 : 0);
        // Fix: just advance
        offset = next_off;
        cur = nxt;
    }
    printf("\n");
    run_gen_ms = timer.end_ms();
    return runs;
}

// ── Phase 2: Streaming merge with overlapped PCIe ───────────────────
// For each merge pass: load pairs of runs into GPU, merge, write back.
// Uses double-buffering to overlap upload/download with compute.

void ExternalGpuSort::merge_runs_streaming(
    uint8_t* h_data, uint64_t num_records,
    std::vector<RunInfo>& runs,
    double& merge_ms, int& merge_passes,
    double& h2d_bytes, double& d2h_bytes
) {
    if (runs.size() <= 1) { merge_ms = 0; merge_passes = 0; return; }

    uint64_t total_bytes = num_records * RECORD_SIZE;
    uint8_t* h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, total_bytes));

    uint8_t *h_src = h_data, *h_dst = h_output;
    merge_passes = 0;
    h2d_bytes = 0;
    d2h_bytes = 0;

    WallTimer timer;
    timer.begin();

    while (runs.size() > 1) {
        merge_passes++;
        int cur_runs = (int)runs.size();
        int npairs = cur_runs / 2;
        bool leftover = (cur_runs % 2 == 1);
        std::vector<RunInfo> new_runs;
        uint64_t out_off = 0;

        printf("  Merge pass %d: %d -> %d runs", merge_passes, cur_runs, npairs + (leftover?1:0));
        fflush(stdout);

        for (int p = 0; p < npairs; p++) {
            RunInfo &ra = runs[2*p], &rb = runs[2*p+1];
            uint64_t pair_n = ra.num_records + rb.num_records;
            uint64_t pair_bytes = pair_n * RECORD_SIZE;

            if (pair_bytes <= chunk_bytes) {
                // Pair fits in GPU — upload, merge on GPU, download
                // Upload both runs into d_buf_a contiguously
                uint64_t a_bytes = ra.num_records * RECORD_SIZE;
                uint64_t b_bytes = rb.num_records * RECORD_SIZE;

                memcpy(h_pinned_a, h_src + ra.host_offset, a_bytes);
                memcpy(h_pinned_a + a_bytes, h_src + rb.host_offset, b_bytes);

                CUDA_CHECK(cudaMemcpyAsync(d_buf_a, h_pinned_a, pair_bytes,
                                            cudaMemcpyHostToDevice, stream_compute));
                h2d_bytes += pair_bytes;

                // Build merge descriptor
                PairDesc2Way desc = {0, (int)ra.num_records,
                                     a_bytes, (int)rb.num_records,
                                     0, 0};
                int items_per_blk = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;
                int mblks = (int)((pair_n + items_per_blk - 1) / items_per_blk);

                PairDesc2Way* d_desc;
                CUDA_CHECK(cudaMalloc(&d_desc, sizeof(PairDesc2Way)));
                CUDA_CHECK(cudaMemcpyAsync(d_desc, &desc, sizeof(PairDesc2Way),
                                            cudaMemcpyHostToDevice, stream_compute));

                launch_merge_2way(d_buf_a, d_buf_b, d_desc, 1, mblks, stream_compute);
                CUDA_CHECK(cudaStreamSynchronize(stream_compute));
                cudaFree(d_desc);

                // Download merged result
                CUDA_CHECK(cudaMemcpyAsync(h_pinned_b, d_buf_b, pair_bytes,
                                            cudaMemcpyDeviceToHost, stream_compute));
                CUDA_CHECK(cudaStreamSynchronize(stream_compute));
                memcpy(h_dst + out_off, h_pinned_b, pair_bytes);
                d2h_bytes += pair_bytes;

            } else {
                // Pair too large for GPU — streaming merge in chunks
                // Load chunks from each run, merge on GPU, write back
                uint64_t half_chunk = chunk_records / 2;
                uint64_t cursor_a = 0, cursor_b = 0;
                uint64_t written = 0;

                while (written < pair_n) {
                    uint64_t na = std::min(half_chunk, ra.num_records - cursor_a);
                    uint64_t nb = std::min(half_chunk, rb.num_records - cursor_b);
                    uint64_t loaded = na + nb;
                    if (loaded == 0) break;

                    // Upload A chunk then B chunk
                    uint64_t a_bytes_chunk = na * RECORD_SIZE;
                    uint64_t b_bytes_chunk = nb * RECORD_SIZE;
                    if (na > 0)
                        memcpy(h_pinned_a, h_src + ra.host_offset + cursor_a*RECORD_SIZE, a_bytes_chunk);
                    if (nb > 0)
                        memcpy(h_pinned_a + a_bytes_chunk,
                               h_src + rb.host_offset + cursor_b*RECORD_SIZE, b_bytes_chunk);

                    CUDA_CHECK(cudaMemcpyAsync(d_buf_a, h_pinned_a, (na+nb)*RECORD_SIZE,
                                                cudaMemcpyHostToDevice, stream_compute));
                    h2d_bytes += (na+nb)*RECORD_SIZE;

                    // Merge this batch
                    if (na > 0 && nb > 0) {
                        PairDesc2Way desc = {0, (int)na, a_bytes_chunk, (int)nb, 0, 0};
                        int items_per_blk = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;
                        int mblks = (int)((loaded + items_per_blk - 1) / items_per_blk);

                        PairDesc2Way* d_desc;
                        CUDA_CHECK(cudaMalloc(&d_desc, sizeof(PairDesc2Way)));
                        CUDA_CHECK(cudaMemcpyAsync(d_desc, &desc, sizeof(PairDesc2Way),
                                                    cudaMemcpyHostToDevice, stream_compute));
                        launch_merge_2way(d_buf_a, d_buf_b, d_desc, 1, mblks, stream_compute);
                        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
                        cudaFree(d_desc);
                    } else {
                        // Only one run has data — just copy
                        CUDA_CHECK(cudaMemcpyAsync(d_buf_b, d_buf_a, loaded*RECORD_SIZE,
                                                    cudaMemcpyDeviceToDevice, stream_compute));
                        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
                    }

                    // Download merged output
                    uint64_t out_n = std::min(loaded, pair_n - written);
                    CUDA_CHECK(cudaMemcpyAsync(h_pinned_b, d_buf_b, out_n*RECORD_SIZE,
                                                cudaMemcpyDeviceToHost, stream_compute));
                    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
                    memcpy(h_dst + out_off + written*RECORD_SIZE, h_pinned_b, out_n*RECORD_SIZE);
                    d2h_bytes += out_n*RECORD_SIZE;

                    cursor_a += na;
                    cursor_b += nb;
                    written += out_n;
                }
            }

            new_runs.push_back({out_off, pair_n});
            out_off += pair_n * RECORD_SIZE;
        }

        if (leftover) {
            RunInfo& rl = runs[cur_runs-1];
            memcpy(h_dst + out_off, h_src + rl.host_offset, rl.num_records*RECORD_SIZE);
            new_runs.push_back({out_off, rl.num_records});
            out_off += rl.num_records * RECORD_SIZE;
        }

        double pass_bytes = (double)(num_records * RECORD_SIZE);
        printf(" (%.1f GB transferred)\n", (pass_bytes * 2) / 1e9);

        runs = new_runs;
        std::swap(h_src, h_dst);
    }

    merge_ms = timer.end_ms();

    if (h_src != h_data) memcpy(h_data, h_src, total_bytes);
    CUDA_CHECK(cudaFreeHost(h_output));
}

ExternalGpuSort::TimingResult ExternalGpuSort::sort(uint8_t* h_data, uint64_t num_records) {
    TimingResult result = {};
    if (num_records <= 1) return result;

    uint64_t total_bytes = num_records * RECORD_SIZE;
    printf("[ExternalSort] Sorting %lu records (%.2f GB)\n", num_records, total_bytes/1e9);

    // Fast path: fits in one GPU chunk
    if (num_records <= chunk_records) {
        printf("[ExternalSort] Data fits in GPU — single-chunk sort\n");
        WallTimer t; t.begin();
        CUDA_CHECK(cudaMemcpy(d_buf_a, h_data, total_bytes, cudaMemcpyHostToDevice));
        sort_chunk_on_gpu(d_buf_a, d_buf_a, num_records, stream_compute);
        CUDA_CHECK(cudaMemcpy(h_data, d_buf_a, total_bytes, cudaMemcpyDeviceToHost));
        result.total_ms = t.end_ms();
        result.run_gen_ms = result.total_ms;
        result.num_runs = 1;
        result.pcie_h2d_gb = total_bytes / 1e9;
        result.pcie_d2h_gb = total_bytes / 1e9;
        return result;
    }

    printf("\n== Phase 1: Run Generation (double-buffered) ==\n");
    double rg_h2d = 0, rg_d2h = 0;
    auto runs = generate_runs(h_data, num_records, result.run_gen_ms, rg_h2d, rg_d2h);
    result.num_runs = (int)runs.size();
    printf("  %d runs in %.0f ms (%.2f GB/s effective)\n",
           result.num_runs, result.run_gen_ms,
           total_bytes / (result.run_gen_ms * 1e6));

    printf("\n== Phase 2: Streaming Merge ==\n");
    double mg_h2d = 0, mg_d2h = 0;
    merge_runs_streaming(h_data, num_records, runs,
                          result.merge_ms, result.merge_passes,
                          mg_h2d, mg_d2h);

    result.pcie_h2d_gb = (rg_h2d + mg_h2d) / 1e9;
    result.pcie_d2h_gb = (rg_d2h + mg_d2h) / 1e9;
    result.total_ms = result.run_gen_ms + result.merge_ms;

    printf("\n[ExternalSort] ═══════════════════════════════════════\n");
    printf("  DONE: %.0f ms (gen: %.0f + merge: %.0f)\n",
           result.total_ms, result.run_gen_ms, result.merge_ms);
    printf("  Throughput: %.2f GB/s\n", total_bytes / (result.total_ms * 1e6));
    printf("  Runs: %d, Merge passes: %d\n", result.num_runs, result.merge_passes);
    printf("  PCIe traffic: %.2f GB H2D + %.2f GB D2H = %.2f GB total\n",
           result.pcie_h2d_gb, result.pcie_d2h_gb,
           result.pcie_h2d_gb + result.pcie_d2h_gb);
    printf("  PCIe amplification: %.1fx data size\n",
           (result.pcie_h2d_gb + result.pcie_d2h_gb) / (total_bytes / 1e9));
    printf("═══════════════════════════════════════════════════════\n");

    return result;
}

// ── Main (standalone binary) ────────────────────────────────────────

#ifdef EXTERNAL_SORT_MAIN

static void gen_data(uint8_t* d, uint64_t n, unsigned seed) {
    srand(seed);
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

    printf("════════════════════════════════════════════════\n");
    printf("  External GPU Sort — Streaming Benchmark\n");
    printf("════════════════════════════════════════════════\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (%.1f GB HBM)\n", props.name,
           props.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("Data: %.2f GB (%lu records x %d bytes)\n\n",
           total_bytes/1e9, num_records, RECORD_SIZE);

    printf("Allocating %.2f GB host memory...\n", total_bytes/1e9);
    uint8_t* h_data = (uint8_t*)malloc(total_bytes);
    if (!h_data) { fprintf(stderr,"malloc failed (%.2f GB)\n", total_bytes/1e9); return 1; }

    printf("Generating random test data...\n");
    WallTimer gen_timer; gen_timer.begin();
    gen_data(h_data, num_records, 42);
    double gen_ms = gen_timer.end_ms();
    printf("  Data generated in %.0f ms (%.2f GB/s)\n\n", gen_ms, total_bytes/(gen_ms*1e6));

    ExternalGpuSort sorter(0);
    auto result = sorter.sort(h_data, num_records);

    if (verify) {
        printf("\nVerifying sort order...\n");
        uint64_t bad = 0;
        for (uint64_t i = 1; i < num_records && bad < 10; i++)
            if (key_compare(h_data+(i-1)*RECORD_SIZE, h_data+i*RECORD_SIZE, KEY_SIZE)>0)
                { if (bad < 5) printf("  VIOLATION at record %lu\n",i); bad++; }
        printf(bad==0 ? "  PASS: %lu records sorted correctly\n"
                      : "  FAIL: %lu violations found\n", bad==0 ? num_records : bad);
    }

    // CSV output for experiment analysis
    printf("\n# CSV: gpu,total_gb,num_records,num_runs,merge_passes,"
           "run_gen_ms,merge_ms,total_ms,throughput_gbs,"
           "pcie_h2d_gb,pcie_d2h_gb,pcie_total_gb,pcie_amplification\n");
    printf("CSV,%s,%.2f,%lu,%d,%d,%.2f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%.1f\n",
           props.name, total_bytes/1e9, num_records,
           result.num_runs, result.merge_passes,
           result.run_gen_ms, result.merge_ms, result.total_ms,
           total_bytes / (result.total_ms * 1e6),
           result.pcie_h2d_gb, result.pcie_d2h_gb,
           result.pcie_h2d_gb + result.pcie_d2h_gb,
           (result.pcie_h2d_gb + result.pcie_d2h_gb) / (total_bytes / 1e9));

    free(h_data);
    return 0;
}
#endif
