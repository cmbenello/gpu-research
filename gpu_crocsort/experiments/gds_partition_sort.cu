// 19.45 — GPUDirect Storage partition + sort.
// Reads input.bin via cuFile directly into GPU memory, classifies records on GPU,
// scatters compact records to GPU bucket buffers, periodically D2H to host.
// Then standard sort phase.
//
// Pipeline:
//   for each 4 GB chunk of input:
//     cuFileRead → GPU buffer (5-7 GB/s)
//     GPU kernel: classify + scatter (40-byte records into per-bucket GPU staging)
//     D2H to host bucket buffer (when staging full)
//   sort phase (4-GPU concurrent on host bucket buffers, with hugepages + pre-pin)
//
// Build: nvcc -O3 -arch=sm_90 -std=c++17 experiments/gds_partition_sort.cu -lcufile -o experiments/gds_partition_sort
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <cub/cub.cuh>

static constexpr int KEY_SIZE = 66;
static constexpr int RECORD_SIZE = 120;
static constexpr int COMPACT_KEY_SIZE = 32;
static constexpr int COMPACT_REC_SIZE = COMPACT_KEY_SIZE + 8;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// GPU classify + scatter kernel.
// For each record in d_chunk, find its bucket via splitter binary-ish search,
// atomic-add to per-bucket counter, write compact (key + global_offset) record.
__global__ void gds_classify_scatter(
    const uint8_t* __restrict__ d_chunk,
    uint64_t chunk_records,
    uint64_t chunk_global_offset,
    const uint8_t* __restrict__ d_splitters,  // (K-1) × KEY_SIZE
    int K,
    int n_cmap,
    const int* __restrict__ d_cmap,
    uint8_t* d_bucket_buf,                    // single big GPU buffer of K × bucket_capacity × 40 bytes
    uint64_t bucket_capacity_records,         // per-bucket capacity in records
    uint64_t* __restrict__ d_bucket_counts    // K counters
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_records) return;

    const uint8_t* rec = d_chunk + i * RECORD_SIZE;

    // Linear scan for bucket (K is small, e.g. 8)
    int b = K - 1;
    for (int s = 0; s < K - 1; s++) {
        const uint8_t* sp = d_splitters + (uint64_t)s * KEY_SIZE;
        // Compare rec vs sp lexicographically over KEY_SIZE bytes
        int cmp = 0;
        for (int j = 0; j < KEY_SIZE; j++) {
            int diff = (int)rec[j] - (int)sp[j];
            if (diff != 0) { cmp = diff; break; }
        }
        if (cmp <= 0) { b = s; break; }
    }

    // Atomic reservation in bucket
    uint64_t pos = atomicAdd((unsigned long long*)&d_bucket_counts[b], 1ULL);
    if (pos >= bucket_capacity_records) {
        // overflow — drop record (shouldn't happen if we size correctly + flush)
        return;
    }

    uint8_t* dst = d_bucket_buf + ((uint64_t)b * bucket_capacity_records + pos) * COMPACT_REC_SIZE;
    for (int j = 0; j < n_cmap; j++) dst[j] = rec[d_cmap[j]];
    for (int j = n_cmap; j < COMPACT_KEY_SIZE; j++) dst[j] = 0;
    *(uint64_t*)(dst + COMPACT_KEY_SIZE) = chunk_global_offset + i;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s INPUT.bin OUT_PREFIX K\n", argv[0]);
        return 1;
    }
    const char* input_path = argv[1];
    const char* out_prefix = argv[2];
    int K = atoi(argv[3]);
    if (K < 2 || K > 16) { fprintf(stderr, "K must be 2..16\n"); return 1; }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Stat input
    struct stat st;
    if (stat(input_path, &st) != 0) { perror("stat"); return 1; }
    uint64_t total_bytes = st.st_size;
    uint64_t total_records = total_bytes / RECORD_SIZE;
    printf("Input: %lu records (%.2f GB)\n", total_records, total_bytes/1e9);

    // ── cmap detection (CPU sample) ──
    int cmap_h[KEY_SIZE]; int n_cmap = 0;
    {
        // mmap a small portion for sampling
        int fd_samp = open(input_path, O_RDONLY);
        size_t samp_bytes = std::min((size_t)(1ULL << 30), (size_t)total_bytes);
        const uint8_t* samp = (const uint8_t*)mmap(nullptr, samp_bytes, PROT_READ, MAP_PRIVATE, fd_samp, 0);
        close(fd_samp);
        bool varies[KEY_SIZE] = {};
        const int SAMPLE = 100000;
        const uint8_t* r0 = samp;
        for (int i = 1; i < SAMPLE; i++) {
            uint64_t idx = (uint64_t)i * (samp_bytes / RECORD_SIZE) / SAMPLE;
            const uint8_t* ri = samp + idx * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++)
                if (!varies[b] && r0[b] != ri[b]) varies[b] = true;
        }
        for (int b = 0; b < KEY_SIZE; b++) if (varies[b]) cmap_h[n_cmap++] = b;
        munmap((void*)samp, samp_bytes);
    }
    printf("cmap: %d varying bytes\n", n_cmap);
    if (n_cmap > 32) { fprintf(stderr, "ERROR: %d > 32\n", n_cmap); return 1; }

    // ── splitters (CPU sample) ──
    int N_SAMPLE = 8192;
    std::vector<std::array<uint8_t, KEY_SIZE>> samples(N_SAMPLE);
    {
        int fd_samp = open(input_path, O_RDONLY);
        const uint8_t* samp = (const uint8_t*)mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd_samp, 0);
        close(fd_samp);
        for (int i = 0; i < N_SAMPLE; i++) {
            uint64_t idx = (uint64_t)i * total_records / N_SAMPLE;
            memcpy(samples[i].data(), samp + idx * RECORD_SIZE, KEY_SIZE);
        }
        munmap((void*)samp, total_bytes);
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) { return memcmp(a.data(), b.data(), KEY_SIZE) < 0; });
    std::vector<uint8_t> splitters_h((K - 1) * KEY_SIZE);
    for (int i = 1; i < K; i++) {
        int pos = i * N_SAMPLE / K;
        if (pos >= N_SAMPLE) pos = N_SAMPLE - 1;
        memcpy(splitters_h.data() + (i-1) * KEY_SIZE, samples[pos].data(), KEY_SIZE);
    }
    printf("Splitters built (K=%d)\n", K);

    // ── GPU setup ──
    CUDA_CHECK(cudaSetDevice(0));

    // Init cuFile
    cuFileDriverOpen();
    int fd = open(input_path, O_RDONLY | O_DIRECT);
    if (fd < 0) { fd = open(input_path, O_RDONLY); printf("WARN: O_DIRECT failed\n"); }
    if (fd < 0) { perror("open"); return 1; }
    CUfileDescr_t descr = {};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t fh;
    CUfileError_t err = cuFileHandleRegister(&fh, &descr);
    if (err.err != 0) { fprintf(stderr, "HandleRegister err=%d\n", err.err); return 1; }

    // Allocate GPU chunk buffer (4 GB)
    uint64_t chunk_bytes = 4ULL << 30;
    uint64_t chunk_records = chunk_bytes / RECORD_SIZE;
    chunk_bytes = chunk_records * RECORD_SIZE;
    uint8_t* d_chunk;
    CUDA_CHECK(cudaMalloc(&d_chunk, chunk_bytes));
    err = cuFileBufRegister(d_chunk, chunk_bytes, 0);
    if (err.err != 0) { fprintf(stderr, "BufRegister err=%d\n", err.err); return 1; }

    // Allocate GPU bucket buffers + counters
    // Per-bucket capacity: total/K with 30% headroom, then split into chunks for periodic D2H
    uint64_t per_bucket_cap_records = (total_records / K * 13 / 10);
    // To keep GPU memory bounded, use a SMALLER per-bucket staging that gets D2H'd periodically.
    // Each chunk has chunk_records records → max per-bucket per-chunk = chunk_records.
    // Use staging = 2× chunk size per bucket, D2H when half full.
    uint64_t stage_records = chunk_records;  // per-bucket GPU staging capacity
    uint64_t stage_buf_bytes = (uint64_t)K * stage_records * COMPACT_REC_SIZE;
    printf("GPU stage buffer: %.2f GB (%lu records × %d B × %d buckets)\n",
           stage_buf_bytes/1e9, stage_records, COMPACT_REC_SIZE, K);
    uint8_t* d_bucket_buf;
    CUDA_CHECK(cudaMalloc(&d_bucket_buf, stage_buf_bytes));

    uint64_t* d_bucket_counts;
    CUDA_CHECK(cudaMalloc(&d_bucket_counts, K * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_bucket_counts, 0, K * sizeof(uint64_t)));

    // Splitters + cmap on GPU
    uint8_t* d_splitters;
    CUDA_CHECK(cudaMalloc(&d_splitters, splitters_h.size()));
    CUDA_CHECK(cudaMemcpy(d_splitters, splitters_h.data(), splitters_h.size(), cudaMemcpyHostToDevice));
    int* d_cmap;
    CUDA_CHECK(cudaMalloc(&d_cmap, n_cmap * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cmap, cmap_h, n_cmap * sizeof(int), cudaMemcpyHostToDevice));

    // Host bucket buffers (using hugepages if available)
    uint64_t per_bucket_max_bytes = per_bucket_cap_records * COMPACT_REC_SIZE;
    uint64_t hugepage_sz = 2 * 1024 * 1024;
    per_bucket_max_bytes = ((per_bucket_max_bytes + hugepage_sz - 1) / hugepage_sz) * hugepage_sz;
    std::vector<uint8_t*> bucket_bufs(K, nullptr);
    std::vector<uint64_t> host_bucket_offset(K, 0);
    for (int b = 0; b < K; b++) {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT);
        bucket_bufs[b] = (uint8_t*)mmap(nullptr, per_bucket_max_bytes, PROT_READ|PROT_WRITE, flags, -1, 0);
        if (bucket_bufs[b] == MAP_FAILED) {
            bucket_bufs[b] = (uint8_t*)mmap(nullptr, per_bucket_max_bytes, PROT_READ|PROT_WRITE,
                                            MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
            madvise(bucket_bufs[b], per_bucket_max_bytes, MADV_HUGEPAGE);
        }
    }
    printf("Host bucket buffers: %d × %.2f GB (hugepages where available)\n",
           K, per_bucket_max_bytes/1e9);

    // ── Streaming partition ──
    auto t_part0 = std::chrono::high_resolution_clock::now();
    uint64_t total_read = 0;
    while (total_read < total_bytes) {
        uint64_t want = std::min(chunk_bytes, total_bytes - total_read);
        // Round down to record boundary
        want = (want / RECORD_SIZE) * RECORD_SIZE;
        if (want == 0) break;
        uint64_t n_recs = want / RECORD_SIZE;

        // cuFileRead chunk → GPU
        ssize_t n = cuFileRead(fh, d_chunk, want, total_read, 0);
        if (n != (ssize_t)want) {
            fprintf(stderr, "cuFileRead short %zd vs %lu\n", n, want);
            break;
        }

        // Reset bucket counters
        CUDA_CHECK(cudaMemset(d_bucket_counts, 0, K * sizeof(uint64_t)));

        // Classify + scatter
        int threads = 256;
        int blocks = (int)((n_recs + threads - 1) / threads);
        gds_classify_scatter<<<blocks, threads>>>(
            d_chunk, n_recs, total_read / RECORD_SIZE,
            d_splitters, K, n_cmap, d_cmap,
            d_bucket_buf, stage_records, d_bucket_counts);
        CUDA_CHECK(cudaDeviceSynchronize());

        // D2H per-bucket staging buffers to host bucket buffers
        uint64_t bucket_counts_h[16];
        CUDA_CHECK(cudaMemcpy(bucket_counts_h, d_bucket_counts, K * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (int b = 0; b < K; b++) {
            uint64_t bytes = bucket_counts_h[b] * COMPACT_REC_SIZE;
            if (bytes == 0) continue;
            uint64_t off = host_bucket_offset[b];
            CUDA_CHECK(cudaMemcpy(bucket_bufs[b] + off,
                                  d_bucket_buf + (uint64_t)b * stage_records * COMPACT_REC_SIZE,
                                  bytes, cudaMemcpyDeviceToHost));
            host_bucket_offset[b] += bytes;
        }

        total_read += want;
    }
    auto t_part1 = std::chrono::high_resolution_clock::now();
    double part_ms = std::chrono::duration<double, std::milli>(t_part1 - t_part0).count();
    uint64_t total_part_bytes = 0;
    for (int b = 0; b < K; b++) {
        total_part_bytes += host_bucket_offset[b];
        printf("  bucket[%d]: %lu records (%.2f GB)\n",
               b, host_bucket_offset[b] / COMPACT_REC_SIZE, host_bucket_offset[b]/1e9);
    }
    printf("Partition (cuFile + GPU classify): %.0f ms (%.2f GB/s effective)\n",
           part_ms, total_bytes / (part_ms * 1e6));

    // Cleanup GPU buffers used for partition
    cuFileBufDeregister(d_chunk);
    cudaFree(d_chunk);
    cudaFree(d_bucket_buf);
    cudaFree(d_bucket_counts);
    cudaFree(d_splitters);
    cudaFree(d_cmap);
    cuFileHandleDeregister(fh);
    close(fd);
    cuFileDriverClose();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total partition wall: %.0f ms (%.2f GB/s end-to-end)\n",
           total_ms, total_bytes/(total_ms*1e6));
    printf("CSV,gds_partition,records=%lu,K=%d,part_ms=%.0f,total_ms=%.0f,gb_per_s=%.2f\n",
           total_records, K, part_ms, total_ms, total_bytes/(total_ms*1e6));

    // Free host bucket buffers (not running sort phase in this prototype)
    for (int b = 0; b < K; b++) munmap(bucket_bufs[b], per_bucket_max_bytes);

    return 0;
}
