// 19.25 — Streaming partition + sort: NO intermediate NVMe write.
//
// Reads input.bin, classifies records into per-bucket host buffers,
// then runs 4-GPU concurrent × 4 rounds sort directly from RAM.
// Writes only the final 8-byte sorted offsets per bucket (~68 GB total).
//
// vs 19.23 compact pipeline:
//   - eliminates the 360 GB intermediate bucket-file write
//   - eliminates the cudaHostRegister of bucket files (already in RAM)
//   - sort phase reads from in-RAM buffers, not via fread
//
// Projected: ~6m total (vs 7m56s in 19.23).
//
// Build: nvcc -O3 -arch=sm_90 -std=c++17 -Xcompiler -fopenmp \
//             experiments/stream_partition_sort.cu -o experiments/stream_partition_sort
//
// Run:   ./stream_partition_sort INPUT.bin OUT_PREFIX K
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
#include <cub/cub.cuh>

static constexpr int KEY_SIZE = 66;
static constexpr int RECORD_SIZE = 120;
static constexpr int COMPACT_KEY_SIZE = 32;
static constexpr int COMPACT_REC_SIZE = COMPACT_KEY_SIZE + 8;

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void extract_chunk_k(const uint8_t* records, const uint32_t* perm,
                                 uint64_t* out, uint64_t n, int byte_off, int chunk_bytes) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t idx = perm[i];
    const uint8_t* k = records + (uint64_t)idx * COMPACT_REC_SIZE + byte_off;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    out[i] = v;
}

__global__ void init_identity_k(uint32_t* perm, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) perm[i] = (uint32_t)i;
}

__global__ void gather_offsets_k(const uint8_t* records, const uint32_t* perm,
                                  uint64_t* out_offsets, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t idx = perm[i];
    out_offsets[i] = *(const uint64_t*)(records + (uint64_t)idx * COMPACT_REC_SIZE + COMPACT_KEY_SIZE);
}

static int find_bucket_static(const std::vector<std::array<uint8_t, KEY_SIZE>>& splitters,
                               const uint8_t* key) {
    int K = (int)splitters.size() + 1;
    for (int i = 0; i < (int)splitters.size(); i++) {
        if (memcmp(key, splitters[i].data(), KEY_SIZE) <= 0) return i;
    }
    return K - 1;
}

struct SortJob {
    int bucket_id;
    int gpu_id;
    uint8_t* h_records;     // pinned host buffer
    uint64_t n_records;
    char output_path[512];
};

static void sort_one_bucket(const SortJob& job) {
    CUDA_CHECK(cudaSetDevice(job.gpu_id));
    uint64_t n = job.n_records;
    uint64_t bytes = n * COMPACT_REC_SIZE;

    // Pin the bucket buffer for fast H2D
    cudaHostRegister(job.h_records, bytes, cudaHostRegisterDefault);

    // Upload to GPU
    uint8_t* d_records;
    CUDA_CHECK(cudaMalloc(&d_records, bytes));
    CUDA_CHECK(cudaMemcpy(d_records, job.h_records, bytes, cudaMemcpyHostToDevice));
    cudaHostUnregister(job.h_records);

    // Allocate sort buffers
    uint32_t *d_perm_a, *d_perm_b;
    CUDA_CHECK(cudaMalloc(&d_perm_a, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_perm_b, n * sizeof(uint32_t)));
    init_identity_k<<<(n+255)/256, 256>>>(d_perm_a, n);

    uint64_t *d_keys_a, *d_keys_b;
    CUDA_CHECK(cudaMalloc(&d_keys_a, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_b, n * sizeof(uint64_t)));

    size_t cub_temp_bytes = 0;
    cub::DoubleBuffer<uint64_t> kb(d_keys_a, d_keys_b);
    cub::DoubleBuffer<uint32_t> vb(d_perm_a, d_perm_b);
    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, kb, vb, (int)n, 0, 64);
    void* d_cub_temp;
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

    // 4 LSD passes
    for (int chunk = 3; chunk >= 0; chunk--) {
        int byte_off = chunk * 8;
        extract_chunk_k<<<(n+255)/256, 256>>>(d_records, vb.Current(), kb.Current(), n, byte_off, 8);
        cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_temp_bytes, kb, vb, (int)n, 0, 64);
    }

    // Gather offsets
    uint64_t* d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, n * sizeof(uint64_t)));
    gather_offsets_k<<<(n+255)/256, 256>>>(d_records, vb.Current(), d_offsets, n);

    uint64_t* h_offsets;
    CUDA_CHECK(cudaMallocHost(&h_offsets, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(h_offsets, d_offsets, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Write output
    FILE* fp = fopen(job.output_path, "wb");
    if (fp) { fwrite(h_offsets, sizeof(uint64_t), n, fp); fclose(fp); }

    cudaFreeHost(h_offsets);
    cudaFree(d_records);
    cudaFree(d_perm_a);
    cudaFree(d_perm_b);
    cudaFree(d_keys_a);
    cudaFree(d_keys_b);
    cudaFree(d_offsets);
    cudaFree(d_cub_temp);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s INPUT.bin OUT_PREFIX K\n", argv[0]);
        return 1;
    }
    const char* input_path = argv[1];
    const char* out_prefix = argv[2];
    int K = atoi(argv[3]);
    if (K < 4 || K > 64) { fprintf(stderr, "K must be 4..64\n"); return 1; }

    auto t0 = std::chrono::high_resolution_clock::now();

    struct stat st;
    if (stat(input_path, &st) != 0) { perror("stat"); return 1; }
    uint64_t total_bytes = st.st_size;
    uint64_t total_records = total_bytes / RECORD_SIZE;
    int fd_in = open(input_path, O_RDONLY);
    if (fd_in < 0) { perror("open input"); return 1; }
    const uint8_t* input = (const uint8_t*)mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd_in, 0);
    close(fd_in);
    if (input == MAP_FAILED) { perror("mmap input"); return 1; }
    madvise((void*)input, total_bytes, MADV_RANDOM);
    printf("Input: %s, %lu records (%.2f GB)\n", input_path, total_records, total_bytes/1e9);

    // ── cmap detection ──
    auto t_cmap0 = std::chrono::high_resolution_clock::now();
    int cmap[KEY_SIZE]; int n_cmap = 0;
    {
        bool varies[KEY_SIZE] = {};
        const int SAMPLE = 100000;
        const uint8_t* r0 = input;
        for (int i = 1; i < SAMPLE; i++) {
            uint64_t idx = (uint64_t)i * total_records / SAMPLE;
            const uint8_t* ri = input + idx * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++)
                if (!varies[b] && r0[b] != ri[b]) varies[b] = true;
        }
        for (int b = 0; b < KEY_SIZE; b++) if (varies[b]) cmap[n_cmap++] = b;
    }
    auto t_cmap1 = std::chrono::high_resolution_clock::now();
    printf("cmap: %d bytes in %.0f ms\n", n_cmap,
           std::chrono::duration<double, std::milli>(t_cmap1 - t_cmap0).count());
    if (n_cmap > 32) { fprintf(stderr, "ERROR: %d > 32 varying bytes\n", n_cmap); return 1; }

    // ── splitters ──
    int N_SAMPLE = 8192;
    std::vector<std::array<uint8_t, KEY_SIZE>> samples(N_SAMPLE);
    for (int i = 0; i < N_SAMPLE; i++) {
        uint64_t idx = (uint64_t)i * total_records / N_SAMPLE;
        memcpy(samples[i].data(), input + idx * RECORD_SIZE, KEY_SIZE);
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) { return memcmp(a.data(), b.data(), KEY_SIZE) < 0; });
    std::vector<std::array<uint8_t, KEY_SIZE>> splitters(K - 1);
    for (int i = 1; i < K; i++) {
        int pos = i * N_SAMPLE / K;
        if (pos >= N_SAMPLE) pos = N_SAMPLE - 1;
        memcpy(splitters[i-1].data(), samples[pos].data(), KEY_SIZE);
    }

    madvise((void*)input, total_bytes, MADV_SEQUENTIAL);

    // ── Allocate per-bucket pinned host buffers ──
    // Estimate per-bucket size with 10% headroom over uniform: total/K * 1.10
    uint64_t per_bucket_max_records = (total_records + K - 1) / K * 11 / 10;
    uint64_t per_bucket_max_bytes = per_bucket_max_records * COMPACT_REC_SIZE;
    printf("Allocating %d × %.2f GB pinned bucket buffers = %.2f GB total\n",
           K, per_bucket_max_bytes/1e9, K * per_bucket_max_bytes / 1e9);

    auto t_alloc0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t*> bucket_bufs(K, nullptr);
    std::vector<std::atomic<uint64_t>> bucket_counts(K);
    for (int b = 0; b < K; b++) bucket_counts[b].store(0);
    for (int b = 0; b < K; b++) {
        // Unpinned mmap; pin lazily before GPU sort. Saves ~3 min at SF1500.
        bucket_bufs[b] = (uint8_t*)mmap(nullptr, per_bucket_max_bytes,
                                        PROT_READ|PROT_WRITE,
                                        MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (bucket_bufs[b] == MAP_FAILED) {
            fprintf(stderr, "mmap bucket %d (%.2f GB) failed\n", b, per_bucket_max_bytes/1e9);
            return 1;
        }
        madvise(bucket_bufs[b], per_bucket_max_bytes, MADV_HUGEPAGE);
    }
    auto t_alloc1 = std::chrono::high_resolution_clock::now();
    printf("Unpinned mmap alloc: %.0f ms (will pin lazily per bucket)\n",
           std::chrono::duration<double, std::milli>(t_alloc1 - t_alloc0).count());

    // ── Partition pass: classify + write to RAM bucket buffers ──
    auto t_part0 = std::chrono::high_resolution_clock::now();
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);
    uint64_t per_t = (total_records + n_threads - 1) / n_threads;

    // Per-thread per-bucket scratch buffers (256K records × 40 B = 10 MB each)
    constexpr int BUF_RECS = 256 * 1024;
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = (uint64_t)t * per_t;
                uint64_t hi = std::min(lo + per_t, total_records);
                std::vector<std::vector<uint8_t>> tbuf(K);
                std::vector<uint64_t> tcnt(K, 0);
                for (int b = 0; b < K; b++) tbuf[b].resize(BUF_RECS * COMPACT_REC_SIZE);

                auto flush = [&](int b) {
                    if (tcnt[b] == 0) return;
                    uint64_t bytes = tcnt[b] * COMPACT_REC_SIZE;
                    uint64_t off = bucket_counts[b].fetch_add(bytes, std::memory_order_relaxed);
                    if (off + bytes > per_bucket_max_bytes) {
                        fprintf(stderr, "ERROR: bucket %d overflow (%lu + %lu > %lu)\n",
                                b, off, bytes, per_bucket_max_bytes);
                        std::exit(1);
                    }
                    memcpy(bucket_bufs[b] + off, tbuf[b].data(), bytes);
                    tcnt[b] = 0;
                };

                for (uint64_t r = lo; r < hi; r++) {
                    const uint8_t* src = input + r * RECORD_SIZE;
                    int b = find_bucket_static(splitters, src);
                    uint8_t* dst = tbuf[b].data() + tcnt[b] * COMPACT_REC_SIZE;
                    for (int i = 0; i < n_cmap; i++) dst[i] = src[cmap[i]];
                    for (int i = n_cmap; i < COMPACT_KEY_SIZE; i++) dst[i] = 0;
                    *(uint64_t*)(dst + COMPACT_KEY_SIZE) = r;
                    tcnt[b]++;
                    if (tcnt[b] == BUF_RECS) flush(b);
                }
                for (int b = 0; b < K; b++) flush(b);
            });
        }
        for (auto& th : threads) th.join();
    }
    auto t_part1 = std::chrono::high_resolution_clock::now();
    double partition_ms = std::chrono::duration<double, std::milli>(t_part1 - t_part0).count();
    uint64_t total_partitioned_bytes = 0;
    for (int b = 0; b < K; b++) {
        uint64_t cnt = bucket_counts[b].load();
        total_partitioned_bytes += cnt;
        printf("  bucket[%d]: %lu records (%.2f GB)\n",
               b, cnt / COMPACT_REC_SIZE, cnt / 1e9);
    }
    printf("Partition (in-RAM): %.0f ms (%.2f GB read, %.2f GB partitioned)\n",
           partition_ms, total_bytes/1e9, total_partitioned_bytes/1e9);

    // We can release input.bin mmap now — sort phase only needs the bucket buffers
    munmap((void*)input, total_bytes);

    // ── Sort phase: 4-GPU concurrent × ceil(K/4) rounds ──
    auto t_sort0 = std::chrono::high_resolution_clock::now();
    int rounds = (K + 3) / 4;
    for (int round = 0; round < rounds; round++) {
        int b0 = round * 4;
        std::vector<std::thread> sort_threads;
        for (int slot = 0; slot < 4 && (b0 + slot) < K; slot++) {
            int b = b0 + slot;
            int gpu = slot;
            uint64_t n_recs = bucket_counts[b].load() / COMPACT_REC_SIZE;
            SortJob job{ b, gpu, bucket_bufs[b], n_recs, "" };
            snprintf(job.output_path, sizeof(job.output_path), "%s.sorted_%d.bin", out_prefix, b);
            sort_threads.emplace_back([job]() { sort_one_bucket(job); });
        }
        for (auto& t : sort_threads) t.join();
        // Free pinned bucket buffers as we finish them
        for (int slot = 0; slot < 4 && (b0 + slot) < K; slot++) {
            int b = b0 + slot;
            munmap(bucket_bufs[b], per_bucket_max_bytes);
            bucket_bufs[b] = nullptr;
        }
        printf("  round %d done\n", round + 1);
    }
    auto t_sort1 = std::chrono::high_resolution_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t_sort1 - t_sort0).count();
    printf("Sort phase: %.0f ms (%d rounds)\n", sort_ms, rounds);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("\nTotal wall: %.0f ms (%.2f GB/s end-to-end)\n",
           total_ms, total_bytes / (total_ms * 1e6));
    printf("CSV,stream_partition_sort,records=%lu,K=%d,n_cmap=%d,partition_ms=%.0f,sort_ms=%.0f,total_ms=%.0f\n",
           total_records, K, n_cmap, partition_ms, sort_ms, total_ms);

    return 0;
}
