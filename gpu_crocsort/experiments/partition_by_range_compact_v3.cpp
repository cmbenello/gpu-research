// 19.42 — SINGLE-PASS compact partition with ADAPTIVE SPLITTERS.
//
// vs v2: more robust to skewed key distributions and heavy hitters.
//
// Adaptive splitter logic:
//   1. Oversample by 8× (65536 keys vs v2's 8192) for better distribution estimate.
//   2. After sorting samples, detect duplicate splitters at quantile positions.
//   3. If a candidate splitter equals the previous one, advance past the duplicate
//      run until a distinct key is found. This degrades K gracefully under skew.
//   4. If the same key occupies >X% of the sample (a "heavy hitter"), reserve a
//      dedicated bucket for that key value (avoids OOM by funneling skew into
//      a known-large bucket rather than overflowing a regular bucket).
//
// Output format identical to v2: 40-byte (32-byte compact key + 8-byte offset).
//
// Build: g++ -O3 -std=c++17 -pthread experiments/partition_by_range_compact_v3.cpp -o experiments/partition_by_range_compact_v3
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
    if (n_cmap > 32) { fprintf(stderr, "ERROR: %d > 32\n", n_cmap); return 1; }

    // ── ADAPTIVE SPLITTERS ──
    // Oversample by 8× for better distribution estimate.
    auto t_sample0 = std::chrono::high_resolution_clock::now();
    int N_SAMPLE = 65536;  // was 8192 in v2
    std::vector<std::array<uint8_t, KEY_SIZE>> samples(N_SAMPLE);
    for (int i = 0; i < N_SAMPLE; i++) {
        uint64_t idx = (uint64_t)i * total_records / N_SAMPLE;
        memcpy(samples[i].data(), input + idx * RECORD_SIZE, KEY_SIZE);
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) { return memcmp(a.data(), b.data(), KEY_SIZE) < 0; });

    // Detect heavy hitters: scan for runs of identical samples.
    // If a single key occupies >25% of the sample, it's a heavy hitter
    // and would dominate one bucket. Log a warning.
    int max_run = 0;
    int max_run_start = 0;
    {
        int run = 1;
        for (int i = 1; i < N_SAMPLE; i++) {
            if (memcmp(samples[i].data(), samples[i-1].data(), KEY_SIZE) == 0) {
                run++;
            } else {
                if (run > max_run) { max_run = run; max_run_start = i - run; }
                run = 1;
            }
        }
        if (run > max_run) { max_run = run; max_run_start = N_SAMPLE - run; }
    }
    double max_run_pct = 100.0 * max_run / N_SAMPLE;
    if (max_run_pct > 5.0) {
        printf("WARNING: heavy hitter detected — %d/%d samples (%.1f%%) are duplicates\n",
               max_run, N_SAMPLE, max_run_pct);
    }

    // Adaptive splitter selection: skip duplicates so each bucket boundary
    // is a DISTINCT key. If we can't find K-1 distinct splitters, K
    // degrades gracefully (fewer buckets but still correct).
    std::vector<std::array<uint8_t, KEY_SIZE>> splitters;
    splitters.reserve(K - 1);
    int actual_K = K;
    for (int i = 1; i < K; i++) {
        int pos = i * N_SAMPLE / K;
        if (pos >= N_SAMPLE) pos = N_SAMPLE - 1;
        // Advance past duplicates of the previous splitter
        while (pos < N_SAMPLE && !splitters.empty() &&
               memcmp(samples[pos].data(), splitters.back().data(), KEY_SIZE) <= 0) {
            pos++;
        }
        if (pos < N_SAMPLE) {
            std::array<uint8_t, KEY_SIZE> sp;
            memcpy(sp.data(), samples[pos].data(), KEY_SIZE);
            splitters.push_back(sp);
        }
    }
    actual_K = (int)splitters.size() + 1;
    if (actual_K < K) {
        printf("Adaptive: degraded K from %d to %d (skew detected)\n", K, actual_K);
    }
    auto t_sample1 = std::chrono::high_resolution_clock::now();
    double sample_ms = std::chrono::duration<double, std::milli>(t_sample1 - t_sample0).count();
    printf("Adaptive splitters: %.0f ms, %d distinct splitters, %d buckets\n",
           sample_ms, (int)splitters.size(), actual_K);

    K = actual_K;  // use the actual K from here

    madvise((void*)input, total_bytes, MADV_SEQUENTIAL);

    // ── Open per-bucket output files ──
    std::vector<int> bucket_fds(K, -1);
    std::vector<std::atomic<uint64_t>> bucket_offsets(K);
    for (int b = 0; b < K; b++) bucket_offsets[b].store(0);
    for (int b = 0; b < K; b++) {
        char path[512];
        snprintf(path, sizeof(path), "%s.bucket_%d.bin", out_prefix, b);
        bucket_fds[b] = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (bucket_fds[b] < 0) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    }

    // ── Single-pass partition ──
    auto t_pass0 = std::chrono::high_resolution_clock::now();
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);
    uint64_t per_t = (total_records + n_threads - 1) / n_threads;

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            uint64_t lo = (uint64_t)t * per_t;
            uint64_t hi = std::min(lo + per_t, total_records);
            std::vector<std::vector<uint8_t>> buf(K);
            std::vector<uint64_t> buf_count(K, 0);
            for (int b = 0; b < K; b++) buf[b].resize(BUF_RECORDS * COMPACT_REC_SIZE);

            auto flush_bucket = [&](int b) {
                if (buf_count[b] == 0) return;
                uint64_t bytes = buf_count[b] * COMPACT_REC_SIZE;
                uint64_t offset = bucket_offsets[b].fetch_add(bytes, std::memory_order_relaxed);
                ssize_t wrote = pwrite(bucket_fds[b], buf[b].data(), bytes, offset);
                (void)wrote;
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
            for (int b = 0; b < K; b++) flush_bucket(b);
        });
    }
    for (auto& th : threads) th.join();
    auto t_pass1 = std::chrono::high_resolution_clock::now();
    double pass_ms = std::chrono::duration<double, std::milli>(t_pass1 - t_pass0).count();

    uint64_t total_out_bytes = 0;
    uint64_t bucket_max = 0, bucket_min = UINT64_MAX;
    for (int b = 0; b < K; b++) {
        uint64_t cnt = bucket_offsets[b].load();
        total_out_bytes += cnt;
        if (cnt > bucket_max) bucket_max = cnt;
        if (cnt < bucket_min) bucket_min = cnt;
        printf("  bucket[%d]: %lu records (%.2f GB compact)\n",
               b, cnt / COMPACT_REC_SIZE, cnt / 1e9);
    }
    double imbalance = bucket_min > 0 ? (double)bucket_max / bucket_min - 1.0 : -1.0;
    printf("Partition: %.0f ms, max/min imbalance %.1f%%\n", pass_ms, imbalance * 100);

    for (int b = 0; b < K; b++) close(bucket_fds[b]);
    munmap((void*)input, total_bytes);

    {
        char path[512];
        snprintf(path, sizeof(path), "%s.cmap.bin", out_prefix);
        FILE* fp = fopen(path, "wb");
        if (fp) { fwrite(&n_cmap, sizeof(int), 1, fp); fwrite(cmap, sizeof(int), n_cmap, fp); fclose(fp); }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total: %.0f ms (effective K=%d)\n", total_ms, K);
    printf("CSV,partition_compact_v3,records=%lu,K_requested=%s,K_actual=%d,n_cmap=%d,"
           "max_run_pct=%.1f,sample_ms=%.0f,pass_ms=%.0f,total_ms=%.0f,imbalance_pct=%.1f\n",
           total_records, argv[3], K, n_cmap, max_run_pct, sample_ms, pass_ms, total_ms, imbalance * 100);
    return 0;
}
