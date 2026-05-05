// 19.22 — SINGLE-PASS compact partition.
// Eliminates pass 1 (counting) by streaming directly to bucket files via
// per-thread per-bucket RAM buffers and atomic offset counters.
//
// Saves the ~180s pass-1 time. At SF1500 this drops partition wall from
// 7m32s to ~4m → total wall projected ~9m.
//
// Output identical to v1: 40-byte (32-byte compact key + 8-byte offset) records.
//
// Build: g++ -O3 -std=c++17 -pthread experiments/partition_by_range_compact_v2.cpp -o experiments/partition_by_range_compact_v2
// Run:   ./partition_by_range_compact_v2 INPUT.bin OUT_PREFIX K
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
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static constexpr int KEY_SIZE   = 66;
static constexpr int RECORD_SIZE = 120;
static constexpr int COMPACT_KEY_SIZE = 32;
static constexpr int COMPACT_REC_SIZE = COMPACT_KEY_SIZE + 8;

// Per-thread per-bucket buffer size (records, not bytes).
// 64 threads × 16 buckets × (BUF_RECORDS × 40 B) = total RAM.
// 256K records × 40 B = 10 MB per buffer × 1024 buffers = 10 GB total.
static constexpr int BUF_RECORDS = 256 * 1024;

static int find_bucket(const std::vector<std::array<uint8_t, KEY_SIZE>>& splitters,
                       const uint8_t* key) {
    int K = (int)splitters.size() + 1;
    for (int i = 0; i < (int)splitters.size(); i++) {
        if (memcmp(key, splitters[i].data(), KEY_SIZE) <= 0) return i;
    }
    return K - 1;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s INPUT.bin OUT_PREFIX K\n", argv[0]);
        return 1;
    }
    const char* input_path = argv[1];
    const char* out_prefix = argv[2];
    int K = atoi(argv[3]);
    if (K < 2 || K > 64) { fprintf(stderr, "K must be 2..64\n"); return 1; }

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
    auto t_cmap_start = std::chrono::high_resolution_clock::now();
    int cmap[KEY_SIZE];
    int n_cmap = 0;
    {
        bool varies[KEY_SIZE] = {};
        const int SAMPLE_RECS = 100000;
        const uint8_t* r0 = input;
        for (int i = 1; i < SAMPLE_RECS; i++) {
            uint64_t idx = (uint64_t)i * total_records / SAMPLE_RECS;
            const uint8_t* ri = input + idx * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++) {
                if (!varies[b] && r0[b] != ri[b]) varies[b] = true;
            }
        }
        for (int b = 0; b < KEY_SIZE; b++) if (varies[b]) cmap[n_cmap++] = b;
    }
    auto t_cmap_end = std::chrono::high_resolution_clock::now();
    double cmap_ms = std::chrono::duration<double, std::milli>(t_cmap_end - t_cmap_start).count();
    if (n_cmap > 32) {
        fprintf(stderr, "ERROR: %d varying bytes > 32\n", n_cmap);
        return 1;
    }
    printf("Compact map: %d bytes detected in %.0f ms\n", n_cmap, cmap_ms);

    // ── splitters ──
    auto t_sample_start = std::chrono::high_resolution_clock::now();
    int N_SAMPLE = 8192;
    std::vector<std::array<uint8_t, KEY_SIZE>> samples(N_SAMPLE);
    for (int i = 0; i < N_SAMPLE; i++) {
        uint64_t idx = (uint64_t)i * total_records / N_SAMPLE;
        memcpy(samples[i].data(), input + idx * RECORD_SIZE, KEY_SIZE);
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) {
                  return memcmp(a.data(), b.data(), KEY_SIZE) < 0;
              });
    std::vector<std::array<uint8_t, KEY_SIZE>> splitters(K - 1);
    for (int i = 1; i < K; i++) {
        int pos = i * N_SAMPLE / K;
        if (pos >= N_SAMPLE) pos = N_SAMPLE - 1;
        memcpy(splitters[i-1].data(), samples[pos].data(), KEY_SIZE);
    }
    auto t_sample_end = std::chrono::high_resolution_clock::now();
    double sample_ms = std::chrono::duration<double, std::milli>(t_sample_end - t_sample_start).count();
    printf("Splitter sort: %.0f ms\n", sample_ms);

    madvise((void*)input, total_bytes, MADV_SEQUENTIAL);

    // ── Open per-bucket output files (write mode, no pre-truncate) ──
    std::vector<int> bucket_fds(K, -1);
    std::vector<std::atomic<uint64_t>> bucket_offsets(K);
    for (int b = 0; b < K; b++) bucket_offsets[b].store(0);
    for (int b = 0; b < K; b++) {
        char path[512];
        snprintf(path, sizeof(path), "%s.bucket_%d.bin", out_prefix, b);
        bucket_fds[b] = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (bucket_fds[b] < 0) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    }

    // ── Single-pass: classify + flush ──
    auto t_pass_start = std::chrono::high_resolution_clock::now();
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);
    uint64_t per_t = (total_records + n_threads - 1) / n_threads;

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            uint64_t lo = (uint64_t)t * per_t;
            uint64_t hi = std::min(lo + per_t, total_records);
            // Per-thread per-bucket buffer (record-count-indexed)
            std::vector<std::vector<uint8_t>> buf(K);
            std::vector<uint64_t> buf_count(K, 0);
            for (int b = 0; b < K; b++) buf[b].resize(BUF_RECORDS * COMPACT_REC_SIZE);

            auto flush_bucket = [&](int b) {
                if (buf_count[b] == 0) return;
                uint64_t bytes = buf_count[b] * COMPACT_REC_SIZE;
                // Atomically reserve a region in bucket file
                uint64_t offset = bucket_offsets[b].fetch_add(bytes, std::memory_order_relaxed);
                // Write buffer at that region
                ssize_t wrote = pwrite(bucket_fds[b], buf[b].data(), bytes, offset);
                if ((uint64_t)wrote != bytes) {
                    fprintf(stderr, "thread %d: pwrite bucket %d failed (%zd vs %lu)\n", t, b, wrote, bytes);
                }
                buf_count[b] = 0;
            };

            for (uint64_t r = lo; r < hi; r++) {
                const uint8_t* src = input + r * RECORD_SIZE;
                int b = find_bucket(splitters, src);
                uint8_t* dst = buf[b].data() + buf_count[b] * COMPACT_REC_SIZE;
                for (int i = 0; i < n_cmap; i++) dst[i] = src[cmap[i]];
                for (int i = n_cmap; i < COMPACT_KEY_SIZE; i++) dst[i] = 0;
                *(uint64_t*)(dst + COMPACT_KEY_SIZE) = r;
                buf_count[b]++;
                if (buf_count[b] == BUF_RECORDS) flush_bucket(b);
            }
            // Flush all remaining
            for (int b = 0; b < K; b++) flush_bucket(b);
        });
    }
    for (auto& th : threads) th.join();
    auto t_pass_end = std::chrono::high_resolution_clock::now();
    double pass_ms = std::chrono::duration<double, std::milli>(t_pass_end - t_pass_start).count();

    uint64_t total_out_bytes = 0;
    for (int b = 0; b < K; b++) {
        uint64_t sz = bucket_offsets[b].load();
        total_out_bytes += sz;
        printf("  bucket[%d]: %lu records (%.2f GB compact)\n",
               b, sz / COMPACT_REC_SIZE, sz / 1e9);
    }
    printf("Single-pass (classify + write): %.0f ms (%.2f GB read, %.2f GB write)\n",
           pass_ms, total_bytes/1e9, total_out_bytes/1e9);

    for (int b = 0; b < K; b++) close(bucket_fds[b]);
    munmap((void*)input, total_bytes);

    {
        char path[512];
        snprintf(path, sizeof(path), "%s.cmap.bin", out_prefix);
        FILE* fp = fopen(path, "wb");
        if (fp) {
            fwrite(&n_cmap, sizeof(int), 1, fp);
            fwrite(cmap, sizeof(int), n_cmap, fp);
            fclose(fp);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total partition wall: %.0f ms (%.2f GB/s effective on input read)\n",
           total_ms, total_bytes / (total_ms * 1e6));
    printf("CSV,partition_compact_v2,records=%lu,K=%d,n_cmap=%d,sample_ms=%.0f,pass_ms=%.0f,"
           "total_ms=%.0f,read_gb=%.2f,write_gb=%.2f\n",
           total_records, K, n_cmap, sample_ms, pass_ms, total_ms,
           total_bytes/1e9, total_out_bytes/1e9);
    return 0;
}
