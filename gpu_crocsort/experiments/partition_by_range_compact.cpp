// 19.21 — Range-partition input file but emit 40-byte (compact-key + offset) records
// instead of full 120-byte records. Reduces partition write traffic by 3×.
//
// Output bucket entry layout:
//   [0..31]  = 32-byte compact key (26 varying bytes + 6 zero padding)
//   [32..39] = 8-byte uint64 offset into INPUT.bin (record index)
//
// At SF1500: bucket files are 22.8 GB each (vs 67 GB), partition wall projected
// 4.7m (vs 10.9m).
//
// Build: g++ -O3 -std=c++17 -pthread experiments/partition_by_range_compact.cpp -o partition_by_range_compact
// Run:   ./partition_by_range_compact INPUT.bin OUT_PREFIX K
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
static constexpr int COMPACT_KEY_SIZE = 32;  // 26 varying + 6 padding
static constexpr int COMPACT_REC_SIZE = COMPACT_KEY_SIZE + 8;  // + 8B offset

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
    int fd = open(input_path, O_RDONLY);
    if (fd < 0) { perror("open input"); return 1; }
    const uint8_t* input = (const uint8_t*)mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (input == MAP_FAILED) { perror("mmap input"); return 1; }
    madvise((void*)input, total_bytes, MADV_RANDOM);
    printf("Input: %s, %lu records (%.2f GB)\n", input_path, total_records, total_bytes/1e9);

    // ── Step 0: detect varying bytes (cmap) by scanning a sample ──
    // For TPC-H lineitem this finds 26 varying byte positions out of 66.
    auto t_cmap_start = std::chrono::high_resolution_clock::now();
    int cmap[KEY_SIZE];
    int n_cmap = 0;
    {
        bool varies[KEY_SIZE] = {};
        const int SAMPLE_RECS = 100000;
        // Compare to record 0
        const uint8_t* r0 = input;
        for (int i = 1; i < SAMPLE_RECS; i++) {
            uint64_t idx = (uint64_t)i * total_records / SAMPLE_RECS;
            const uint8_t* ri = input + idx * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++) {
                if (!varies[b] && r0[b] != ri[b]) varies[b] = true;
            }
        }
        for (int b = 0; b < KEY_SIZE; b++) {
            if (varies[b]) cmap[n_cmap++] = b;
        }
    }
    auto t_cmap_end = std::chrono::high_resolution_clock::now();
    double cmap_ms = std::chrono::duration<double, std::milli>(t_cmap_end - t_cmap_start).count();
    if (n_cmap > 32) {
        fprintf(stderr, "ERROR: detected %d varying bytes, > 32 (compact key won't fit). Use full key tool.\n", n_cmap);
        return 1;
    }
    printf("Compact map: %d varying bytes detected in %.0f ms\n", n_cmap, cmap_ms);
    printf("  cmap = ");
    for (int i = 0; i < n_cmap; i++) printf("%d%s", cmap[i], i+1<n_cmap?",":"\n");

    // ── Step 1: sample N=8192 records for splitters ──
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
    printf("Sample + sort + splitters: %.0f ms\n", sample_ms);

    madvise((void*)input, total_bytes, MADV_SEQUENTIAL);

    // ── Pass 1: count per-bucket records ──
    auto t_pass1_start = std::chrono::high_resolution_clock::now();
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);
    uint64_t per_t = (total_records + n_threads - 1) / n_threads;
    std::vector<std::vector<uint64_t>> tcounts(n_threads, std::vector<uint64_t>(K, 0));
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = (uint64_t)t * per_t;
                uint64_t hi = std::min(lo + per_t, total_records);
                auto& c = tcounts[t];
                for (uint64_t r = lo; r < hi; r++) {
                    int b = find_bucket(splitters, input + r * RECORD_SIZE);
                    c[b]++;
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    auto t_pass1_end = std::chrono::high_resolution_clock::now();
    double pass1_ms = std::chrono::duration<double, std::milli>(t_pass1_end - t_pass1_start).count();

    std::vector<uint64_t> bucket_total(K, 0);
    std::vector<std::vector<uint64_t>> tbucket_offsets(n_threads, std::vector<uint64_t>(K, 0));
    for (int b = 0; b < K; b++) {
        uint64_t cum = 0;
        for (int t = 0; t < n_threads; t++) {
            tbucket_offsets[t][b] = cum;
            cum += tcounts[t][b];
        }
        bucket_total[b] = cum;
    }
    printf("Pass 1 (count): %.0f ms\n", pass1_ms);
    for (int b = 0; b < K; b++) {
        printf("  bucket[%d]: %lu records (%.2f GB compact)\n",
               b, bucket_total[b], bucket_total[b] * COMPACT_REC_SIZE / 1e9);
    }

    // ── Open output bucket files (compact format) ──
    std::vector<uint8_t*> bucket_out(K, nullptr);
    std::vector<int> bucket_fds(K, -1);
    for (int b = 0; b < K; b++) {
        char path[512];
        snprintf(path, sizeof(path), "%s.bucket_%d.bin", out_prefix, b);
        bucket_fds[b] = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (bucket_fds[b] < 0) { fprintf(stderr, "cannot open %s\n", path); return 1; }
        uint64_t bytes = bucket_total[b] * COMPACT_REC_SIZE;
        if (bytes > 0) {
            if (ftruncate(bucket_fds[b], bytes) != 0) { perror("ftruncate"); return 1; }
            bucket_out[b] = (uint8_t*)mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                                            MAP_SHARED, bucket_fds[b], 0);
            if (bucket_out[b] == MAP_FAILED) { perror("mmap output"); return 1; }
            madvise(bucket_out[b], bytes, MADV_SEQUENTIAL);
        }
    }

    // ── Pass 2: route + emit compact records ──
    auto t_pass2_start = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = (uint64_t)t * per_t;
                uint64_t hi = std::min(lo + per_t, total_records);
                std::vector<uint64_t> wp(K, 0);
                for (int b = 0; b < K; b++) wp[b] = tbucket_offsets[t][b];
                for (uint64_t r = lo; r < hi; r++) {
                    const uint8_t* src = input + r * RECORD_SIZE;
                    int b = find_bucket(splitters, src);
                    uint8_t* dst = bucket_out[b] + wp[b] * COMPACT_REC_SIZE;
                    // Copy compact key: src[cmap[i]] → dst[i] for i in [0..n_cmap)
                    for (int i = 0; i < n_cmap; i++) dst[i] = src[cmap[i]];
                    // Pad to 32 bytes
                    for (int i = n_cmap; i < COMPACT_KEY_SIZE; i++) dst[i] = 0;
                    // Append 8-byte global offset
                    *(uint64_t*)(dst + COMPACT_KEY_SIZE) = r;
                    wp[b]++;
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    auto t_pass2_end = std::chrono::high_resolution_clock::now();
    double pass2_ms = std::chrono::duration<double, std::milli>(t_pass2_end - t_pass2_start).count();
    uint64_t total_out_bytes = 0;
    for (int b = 0; b < K; b++) total_out_bytes += bucket_total[b] * COMPACT_REC_SIZE;
    printf("Pass 2 (route + emit compact): %.0f ms (%.2f GB read, %.2f GB write)\n",
           pass2_ms, total_bytes/1e9, total_out_bytes/1e9);

    for (int b = 0; b < K; b++) {
        if (bucket_out[b]) {
            uint64_t bytes = bucket_total[b] * COMPACT_REC_SIZE;
            munmap(bucket_out[b], bytes);
        }
        close(bucket_fds[b]);
    }
    munmap((void*)input, total_bytes);

    // Write the cmap to a sidecar file so the sort phase can use it for verification
    {
        char path[512];
        snprintf(path, sizeof(path), "%s.cmap.bin", out_prefix);
        FILE* fp = fopen(path, "wb");
        if (fp) {
            fwrite(&n_cmap, sizeof(int), 1, fp);
            fwrite(cmap, sizeof(int), n_cmap, fp);
            fclose(fp);
            printf("Wrote cmap to %s (%d entries)\n", path, n_cmap);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total partition wall: %.0f ms (%.2f GB/s effective on input read)\n",
           total_ms, total_bytes / (total_ms * 1e6));
    printf("CSV,partition_compact,records=%lu,K=%d,n_cmap=%d,sample_ms=%.0f,pass1_ms=%.0f,"
           "pass2_ms=%.0f,total_ms=%.0f,read_gb=%.2f,write_gb=%.2f\n",
           total_records, K, n_cmap, sample_ms, pass1_ms, pass2_ms, total_ms,
           total_bytes/1e9, total_out_bytes/1e9);

    return 0;
}
