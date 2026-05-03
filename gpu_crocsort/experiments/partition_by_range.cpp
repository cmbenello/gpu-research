// 15.5.3 — Range-partition input file into K bucket files using sample sort.
//
// After this tool runs, each bucket i contains exactly the records whose
// (compact) key falls in (splitter[i-1], splitter[i]]. So if K=4:
//   bucket0: keys ≤ splitter[0]
//   bucket1: splitter[0] < keys ≤ splitter[1]
//   bucket2: splitter[1] < keys ≤ splitter[2]
//   bucket3: keys > splitter[2]
//
// Then 4 GPUs each sort one bucket. Concatenating the sorted buckets in
// order gives a globally-sorted output — NO MERGE PHASE NEEDED.
//
// This is the standard parallel sample-sort partitioner, in CPU C++.
//
// Build: g++ -O3 -std=c++17 -pthread experiments/partition_by_range.cu -o partition_by_range
// Run:   ./partition_by_range INPUT.bin OUT_PREFIX K
//
// Memory: peak = num_records × RECORD_SIZE × 2 (input mmap + write mmap)
//         + per-thread local bucket buffers (~256 MB each).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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

static int find_bucket(const std::vector<std::array<uint8_t, KEY_SIZE>>& splitters,
                       const uint8_t* key) {
    // Linear search is fine for K ≤ 8; binary search not needed at this scale.
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

    // mmap input
    struct stat st;
    if (stat(input_path, &st) != 0) { perror("stat"); return 1; }
    uint64_t total_bytes = st.st_size;
    uint64_t total_records = total_bytes / RECORD_SIZE;
    int fd = open(input_path, O_RDONLY);
    if (fd < 0) { perror("open input"); return 1; }
    const uint8_t* input = (const uint8_t*)mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (input == MAP_FAILED) { perror("mmap input"); return 1; }
    // We do a stride-sample first (each access is random within a span),
    // then linear scan in passes 1+2. After sampling, hint sequential.
    madvise((void*)input, total_bytes, MADV_RANDOM);
    printf("Input: %s, %lu records (%.2f GB)\n", input_path, total_records, total_bytes/1e9);

    // ── Sample N=8192 records uniformly ─────────────────────────────────
    auto t_sample_start = std::chrono::high_resolution_clock::now();
    int N_SAMPLE = 8192;
    std::vector<std::array<uint8_t, KEY_SIZE>> samples(N_SAMPLE);
    {
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<uint64_t> dist(0, total_records - 1);
        for (int i = 0; i < N_SAMPLE; i++) {
            uint64_t idx = (uint64_t)i * total_records / N_SAMPLE;  // deterministic stride
            memcpy(samples[i].data(), input + idx * RECORD_SIZE, KEY_SIZE);
        }
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) {
                  return memcmp(a.data(), b.data(), KEY_SIZE) < 0;
              });
    // Pick K-1 splitters at uniform quantiles
    std::vector<std::array<uint8_t, KEY_SIZE>> splitters(K - 1);
    for (int i = 1; i < K; i++) {
        int pos = i * N_SAMPLE / K;
        if (pos >= N_SAMPLE) pos = N_SAMPLE - 1;
        memcpy(splitters[i-1].data(), samples[pos].data(), KEY_SIZE);
    }
    auto t_sample_end = std::chrono::high_resolution_clock::now();
    double sample_ms = std::chrono::duration<double, std::milli>(t_sample_end - t_sample_start).count();
    printf("Sample + sort + splitters: %.0f ms\n", sample_ms);

    // After sampling, switch hint to sequential — pass 1 + 2 stream through
    // the input linearly within each thread's chunk.
    madvise((void*)input, total_bytes, MADV_SEQUENTIAL);

    // ── Pass 1: count records per bucket (multi-threaded) ───────────────
    auto t_pass1_start = std::chrono::high_resolution_clock::now();
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);  // sweet spot from 15.5.2 / gather sweep
    uint64_t per_t = (total_records + n_threads - 1) / n_threads;

    // Per-thread, per-bucket counts
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

    // Aggregate per-bucket totals + per-thread offsets within each bucket
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
        printf("  bucket[%d]: %lu records (%.2f GB)\n",
               b, bucket_total[b], bucket_total[b] * RECORD_SIZE / 1e9);
    }

    // ── Open output bucket files, mmap each ─────────────────────────────
    std::vector<uint8_t*> bucket_out(K, nullptr);
    std::vector<int> bucket_fds(K, -1);
    for (int b = 0; b < K; b++) {
        char path[512];
        snprintf(path, sizeof(path), "%s.bucket_%d.bin", out_prefix, b);
        bucket_fds[b] = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (bucket_fds[b] < 0) { fprintf(stderr, "cannot open %s\n", path); return 1; }
        uint64_t bytes = bucket_total[b] * RECORD_SIZE;
        if (bytes > 0) {
            if (ftruncate(bucket_fds[b], bytes) != 0) { perror("ftruncate"); return 1; }
            bucket_out[b] = (uint8_t*)mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                                            MAP_SHARED, bucket_fds[b], 0);
            if (bucket_out[b] == MAP_FAILED) { perror("mmap output"); return 1; }
            madvise(bucket_out[b], bytes, MADV_SEQUENTIAL);
        }
    }

    // ── Pass 2: route each record to its bucket file ────────────────────
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
                    int b = find_bucket(splitters, input + r * RECORD_SIZE);
                    memcpy(bucket_out[b] + wp[b] * RECORD_SIZE,
                           input + r * RECORD_SIZE, RECORD_SIZE);
                    wp[b]++;
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    auto t_pass2_end = std::chrono::high_resolution_clock::now();
    double pass2_ms = std::chrono::duration<double, std::milli>(t_pass2_end - t_pass2_start).count();
    printf("Pass 2 (route): %.0f ms (%.2f GB/s)\n", pass2_ms,
           total_bytes / (pass2_ms * 1e6));

    // Sync + close output buffers
    for (int b = 0; b < K; b++) {
        if (bucket_out[b]) {
            uint64_t bytes = bucket_total[b] * RECORD_SIZE;
            munmap(bucket_out[b], bytes);
        }
        close(bucket_fds[b]);
    }
    munmap((void*)input, total_bytes);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total partition wall: %.0f ms (%.2f GB/s end-to-end)\n",
           total_ms, total_bytes / (total_ms * 1e6));
    printf("CSV,partition_by_range,records=%lu,K=%d,sample_ms=%.0f,pass1_ms=%.0f,"
           "pass2_ms=%.0f,total_ms=%.0f,gb_per_s=%.2f\n",
           total_records, K, sample_ms, pass1_ms, pass2_ms, total_ms,
           total_bytes/(total_ms*1e6));

    return 0;
}
