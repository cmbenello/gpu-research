// External sort: handles data larger than GPU HBM via PCIe streaming
// Build: nvcc -O3 -std=c++17 -arch=sm_80 -DEXTERNAL_SORT_MAIN -Iinclude \
//   src/external_sort.cu src/run_generation.cu src/merge.cu -o external_sort
// Run:  ./external_sort --total-gb 2

#include "record.cuh"
#include "ovc.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

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

class ExternalGpuSort {
    size_t gpu_budget, chunk_size;
    uint64_t chunk_records;
    uint8_t *h_pinned_a, *h_pinned_b;   // Pinned host (2x PCIe BW vs pageable)
    uint8_t *d_buf_a, *d_buf_b;         // Device double-buffers
    cudaStream_t stream_compute, stream_transfer;
public:
    ExternalGpuSort(size_t gpu_budget_bytes);
    ~ExternalGpuSort();
    void sort(uint8_t* h_data, uint64_t num_records);
private:
    struct RunInfo { uint64_t host_offset; uint64_t num_records; };
    void sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_out, uint64_t n, cudaStream_t s);
    std::vector<RunInfo> generate_runs(uint8_t* h_data, uint64_t num_records);
    void merge_runs(uint8_t* h_data, uint64_t num_records, std::vector<RunInfo>& runs);
    void stream_merge_k(uint8_t* h_src, uint8_t* h_out,
                        const std::vector<RunInfo>& to_merge, uint64_t total_recs);
};

ExternalGpuSort::ExternalGpuSort(size_t budget) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    gpu_budget = budget > 0 ? std::min(budget, (size_t)(free_mem * 0.70))
                            : (size_t)(free_mem * 0.70);
    // Two device buffers -> each gets half
    chunk_records = (gpu_budget / 2) / RECORD_SIZE;
    chunk_size = chunk_records * RECORD_SIZE;

    printf("[ExternalSort] GPU free: %.2f GB, chunk: %.2f MB (%lu recs)\n",
           free_mem / 1e9, chunk_size / (1024.0*1024.0), chunk_records);

    CUDA_CHECK(cudaMallocHost(&h_pinned_a, chunk_size));
    CUDA_CHECK(cudaMallocHost(&h_pinned_b, chunk_size));
    CUDA_CHECK(cudaMalloc(&d_buf_a, chunk_size));
    CUDA_CHECK(cudaMalloc(&d_buf_b, chunk_size));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));
}

ExternalGpuSort::~ExternalGpuSort() {
    cudaFreeHost(h_pinned_a); cudaFreeHost(h_pinned_b);
    cudaFree(d_buf_a); cudaFree(d_buf_b);
    cudaStreamDestroy(stream_compute); cudaStreamDestroy(stream_transfer);
}

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

    // Iterative 2-way merge passes
    uint8_t *src = d_out, *dst = d_in;
    int items_per_blk = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;

    for (int run_sz = RECORDS_PER_BLOCK; run_sz < (int)n; run_sz *= 2) {
        std::vector<PairDesc2Way> pairs;
        int total_mblks = 0;

        for (uint64_t off = 0; off < n; off += 2 * run_sz) {
            uint64_t ac = std::min((uint64_t)run_sz, n - off);
            uint64_t bs = off + run_sz;
            uint64_t bc = (bs < n) ? std::min((uint64_t)run_sz, n - bs) : 0;
            if (bc == 0) {
                CUDA_CHECK(cudaMemcpyAsync(dst + off*RECORD_SIZE, src + off*RECORD_SIZE,
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
            launch_merge_2way(src, dst, dp, (int)pairs.size(), total_mblks, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
            cudaFree(dp);
        }
        std::swap(src, dst);
    }
    if (src != d_out) {
        CUDA_CHECK(cudaMemcpyAsync(d_out, src, n*RECORD_SIZE, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
}

// Phase 1: double-buffered run generation
// Pipeline: GPU sorts chunk N while PCIe transfers chunk N+1
std::vector<ExternalGpuSort::RunInfo>
ExternalGpuSort::generate_runs(uint8_t* h_data, uint64_t num_records) {
    std::vector<RunInfo> runs;
    uint8_t* d_buf[2] = {d_buf_a, d_buf_b};
    uint8_t* h_pin[2] = {h_pinned_a, h_pinned_b};
    int cur = 0;

    // Prefetch first chunk
    uint64_t first_n = std::min(chunk_records, num_records);
    memcpy(h_pin[0], h_data, first_n * RECORD_SIZE);
    CUDA_CHECK(cudaMemcpyAsync(d_buf[0], h_pin[0], first_n*RECORD_SIZE,
                                cudaMemcpyHostToDevice, stream_compute));

    for (uint64_t offset = 0; offset < num_records; ) {
        uint64_t cur_n = std::min(chunk_records, num_records - offset);
        int nxt = 1 - cur;

        // Sort current chunk on GPU
        sort_chunk_on_gpu(d_buf[cur], d_buf[cur], cur_n, stream_compute);

        // Overlap: start uploading next chunk on transfer stream
        uint64_t next_off = offset + cur_n;
        uint64_t next_n = 0;
        if (next_off < num_records) {
            next_n = std::min(chunk_records, num_records - next_off);
            memcpy(h_pin[nxt], h_data + next_off*RECORD_SIZE, next_n*RECORD_SIZE);
            CUDA_CHECK(cudaMemcpyAsync(d_buf[nxt], h_pin[nxt], next_n*RECORD_SIZE,
                                        cudaMemcpyHostToDevice, stream_transfer));
        }

        // Download sorted chunk
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        CUDA_CHECK(cudaMemcpyAsync(h_pin[cur], d_buf[cur], cur_n*RECORD_SIZE,
                                    cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        memcpy(h_data + offset*RECORD_SIZE, h_pin[cur], cur_n*RECORD_SIZE);

        runs.push_back({offset * RECORD_SIZE, cur_n});
        printf("  Run %d: %lu recs (%.1f MB)\r", (int)runs.size()-1,
               cur_n, cur_n*RECORD_SIZE/(1024.0*1024.0));

        if (next_n > 0) CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
        offset = next_off;
        cur = nxt;
    }
    printf("\n");
    return runs;
}

// Phase 2: iterative pairwise merge of runs via GPU
void ExternalGpuSort::merge_runs(uint8_t* h_data, uint64_t num_records,
                                  std::vector<RunInfo>& runs) {
    if (runs.size() <= 1) return;
    uint64_t total_bytes = num_records * RECORD_SIZE;

    uint8_t* h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, total_bytes));

    uint8_t *h_src = h_data, *h_dst = h_output;
    int pass = 0;

    while (runs.size() > 1) {
        pass++;
        int cur_runs = (int)runs.size();
        int npairs = cur_runs / 2;
        bool leftover = (cur_runs % 2 == 1);
        std::vector<RunInfo> new_runs;
        uint64_t out_off = 0;

        for (int p = 0; p < npairs; p++) {
            RunInfo &ra = runs[2*p], &rb = runs[2*p+1];
            uint64_t pair_n = ra.num_records + rb.num_records;
            stream_merge_k(h_src, h_dst + out_off, {ra, rb}, pair_n);
            new_runs.push_back({out_off, pair_n});
            out_off += pair_n * RECORD_SIZE;
        }
        if (leftover) {
            RunInfo& rl = runs[cur_runs-1];
            memcpy(h_dst+out_off, h_src+rl.host_offset, rl.num_records*RECORD_SIZE);
            new_runs.push_back({out_off, rl.num_records});
            out_off += rl.num_records * RECORD_SIZE;
        }
        printf("  Merge pass %d: %d -> %d runs\n", pass, cur_runs, (int)new_runs.size());
        runs = new_runs;
        std::swap(h_src, h_dst);
    }

    if (h_src != h_data) memcpy(h_data, h_src, total_bytes);
    CUDA_CHECK(cudaFreeHost(h_output));
}

// Stream-merge: load run heads to GPU, sort/merge, write back
void ExternalGpuSort::stream_merge_k(uint8_t* h_src, uint8_t* h_out,
                                      const std::vector<RunInfo>& to_merge,
                                      uint64_t total_recs) {
    int K = (int)to_merge.size();
    uint64_t per_run = std::max((uint64_t)1, (chunk_records / 2) / K);
    std::vector<uint64_t> cursor(K, 0);
    uint64_t written = 0;

    while (written < total_recs) {
        // Load heads of each run into pinned buffer, then to GPU
        uint64_t loaded = 0;
        std::vector<uint64_t> lcount(K);
        for (int k = 0; k < K; k++) {
            uint64_t rem = to_merge[k].num_records - cursor[k];
            uint64_t n = std::min(per_run, rem);
            lcount[k] = n;
            if (n > 0) {
                memcpy(h_pinned_a + loaded*RECORD_SIZE,
                       h_src + to_merge[k].host_offset + cursor[k]*RECORD_SIZE,
                       n * RECORD_SIZE);
            }
            loaded += n;
        }
        if (loaded == 0) break;

        CUDA_CHECK(cudaMemcpyAsync(d_buf_a, h_pinned_a, loaded*RECORD_SIZE,
                                    cudaMemcpyHostToDevice, stream_compute));

        // Sort/merge the loaded records on GPU
        sort_chunk_on_gpu(d_buf_a, d_buf_a, loaded, stream_compute);

        // Download merged output
        uint64_t out_n = std::min(loaded, total_recs - written);
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_b, d_buf_a, out_n*RECORD_SIZE,
                                    cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        memcpy(h_out + written*RECORD_SIZE, h_pinned_b, out_n*RECORD_SIZE);

        for (int k = 0; k < K; k++) cursor[k] += lcount[k];
        written += out_n;
    }
}

void ExternalGpuSort::sort(uint8_t* h_data, uint64_t num_records) {
    if (num_records <= 1) return;
    uint64_t total_bytes = num_records * RECORD_SIZE;
    printf("[ExternalSort] Sorting %lu records (%.2f GB)\n", num_records, total_bytes/1e9);

    // Fast path: data fits in one GPU chunk
    if (num_records <= chunk_records) {
        printf("[ExternalSort] Data fits in GPU — single-chunk sort\n");
        CUDA_CHECK(cudaMemcpy(d_buf_a, h_data, total_bytes, cudaMemcpyHostToDevice));
        sort_chunk_on_gpu(d_buf_a, d_buf_a, num_records, stream_compute);
        CUDA_CHECK(cudaMemcpy(h_data, d_buf_a, total_bytes, cudaMemcpyDeviceToHost));
        return;
    }

    GpuTimer timer;
    timer.begin();
    printf("\n== Phase 1: Run Generation (double-buffered) ==\n");
    auto runs = generate_runs(h_data, num_records);
    float p1 = timer.end();
    printf("  %d runs in %.2f ms (%.2f GB/s)\n",
           (int)runs.size(), p1, total_bytes/(p1*1e6));

    timer.begin();
    printf("\n== Phase 2: Streaming Merge ==\n");
    merge_runs(h_data, num_records, runs);
    float p2 = timer.end();

    printf("\n[ExternalSort] DONE: %.2f ms (gen %.2f + merge %.2f), %.2f GB/s\n",
           p1+p2, p1, p2, total_bytes/((p1+p2)*1e6));
}

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

    printf("========================================\n");
    printf("External GPU Sort — Streaming Test\n");
    printf("========================================\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (%.1f GB HBM)\n", props.name,
           props.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("Data: %.2f GB (%lu records)\n\n", total_bytes/1e9, num_records);

    uint8_t* h_data = (uint8_t*)malloc(total_bytes);
    if (!h_data) { fprintf(stderr,"malloc failed\n"); return 1; }

    printf("Generating test data...\n");
    gen_data(h_data, num_records, 42);

    ExternalGpuSort sorter(0);
    sorter.sort(h_data, num_records);

    if (verify) {
        printf("\nVerifying...\n");
        uint64_t bad = 0;
        for (uint64_t i = 1; i < num_records && bad < 10; i++)
            if (key_compare(h_data+(i-1)*RECORD_SIZE, h_data+i*RECORD_SIZE, KEY_SIZE)>0)
                { printf("  VIOLATION at %lu\n",i); bad++; }
        printf(bad==0 ? "  PASS\n" : "  FAIL: %lu violations\n", bad);
    }
    free(h_data);
    return 0;
}
#endif
