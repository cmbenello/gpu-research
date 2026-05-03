// 17.2 — Standalone gather microbenchmark.
//
// Mimics the gather phase of external_sort.cu without the surrounding
// machinery. Goal: characterize gather GB/s vs thread count to validate
// the 64-thread default and find the scaling cliff.
//
// Build: g++ -O3 -std=c++17 -pthread experiments/gather_bench.cpp \
//        -o experiments/gather_bench
// Run:   ./gather_bench --records N --threads T [--prefetch-block N]
//
// Default: 100 M records of 120 B each = 12 GB working set, sweeps
// {1,2,4,8,16,32,48,64,96,128} threads if --threads not specified.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <sys/mman.h>

constexpr int RECORD_SIZE = 120;

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
}

static void run_gather(uint8_t* in, uint8_t* out, const uint32_t* perm,
                       uint64_t num_records, int num_threads, int BLOCK) {
    auto worker = [&](uint64_t start, uint64_t end) {
        std::vector<const uint8_t*> src_ptrs(BLOCK);
        for (uint64_t base = start; base < end; base += BLOCK) {
            int count = (int)std::min((uint64_t)BLOCK, end - base);
            for (int j = 0; j < count; j++) {
                uint32_t s = perm[base + j];
                src_ptrs[j] = in + (uint64_t)s * RECORD_SIZE;
                __builtin_prefetch(src_ptrs[j], 0, 0);
            }
            for (int j = 0; j < count; j++) {
                std::memcpy(out + (base + j) * (uint64_t)RECORD_SIZE,
                            src_ptrs[j], RECORD_SIZE);
            }
        }
    };
    std::vector<std::thread> threads;
    uint64_t chunk = (num_records + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        uint64_t s = (uint64_t)t * chunk;
        uint64_t e = std::min(s + chunk, num_records);
        if (s < e) threads.emplace_back(worker, s, e);
    }
    for (auto& t : threads) t.join();
}

int main(int argc, char** argv) {
    uint64_t num_records = 100ULL * 1000 * 1000;
    int num_threads = -1;  // -1 = sweep
    int BLOCK = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--records") && i+1 < argc)
            num_records = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc)
            num_threads = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefetch-block") && i+1 < argc)
            BLOCK = std::atoi(argv[++i]);
    }

    uint64_t total_bytes = num_records * (uint64_t)RECORD_SIZE;
    fprintf(stderr, "# records=%lu, total=%.2f GB, prefetch_block=%d\n",
            num_records, total_bytes / 1e9, BLOCK);

    // Allocate input + output via mmap (anon, MADV_HUGEPAGE)
    auto alloc = [](size_t bytes) -> uint8_t* {
        void* p = mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { perror("mmap"); std::exit(1); }
        madvise(p, bytes, MADV_HUGEPAGE);
        return (uint8_t*)p;
    };
    uint8_t* in = alloc(total_bytes);
    uint8_t* out = alloc(total_bytes);

    fprintf(stderr, "# touching input...\n");
    // Touch input (sequential write to get page tables built + into cache once)
    #pragma omp parallel for
    for (uint64_t i = 0; i < total_bytes; i += 4096) in[i] = (uint8_t)(i & 0xff);

    // Generate random permutation
    fprintf(stderr, "# generating random permutation...\n");
    std::vector<uint32_t> perm(num_records);
    for (uint64_t i = 0; i < num_records; i++) perm[i] = (uint32_t)i;
    std::mt19937 rng(42);
    std::shuffle(perm.begin(), perm.end(), rng);

    // Warmup: only on small sets where it's cheap (< 1 GB equivalent).
    // For larger sets, skip — page tables get warmed by the first measurement.
    if (total_bytes < 8ULL * 1024 * 1024 * 1024) {
        fprintf(stderr, "# warmup gather (1 thread)...\n");
        run_gather(in, out, perm.data(), num_records, 1, BLOCK);
    } else {
        fprintf(stderr, "# skipping warmup (working set too large)\n");
    }

    bool large = total_bytes >= 8ULL * 1024 * 1024 * 1024;

    auto bench_at = [&](int T) {
        // Skip the cleanup memset on large sets (gather overwrites every byte
        // anyway; memset just thrashes the page cache).
        if (!large) std::memset(out, 0, total_bytes);
        double t0 = now_ms();
        run_gather(in, out, perm.data(), num_records, T, BLOCK);
        double dt = now_ms() - t0;
        double gbs = total_bytes / (dt * 1e6);
        printf("%d,%lu,%.0f,%.2f\n", T, num_records, dt, gbs);
        fflush(stdout);
    };

    printf("threads,records,wall_ms,gbs\n");
    if (num_threads > 0) {
        bench_at(num_threads);
    } else {
        // For large sweeps, skip the slow 1/2/4 thread points (each takes
        // many minutes at SF300). The interesting range is 16-128.
        if (large) {
            for (int T : {8, 16, 32, 48, 64, 96, 128}) bench_at(T);
        } else {
            for (int T : {1, 2, 4, 8, 16, 32, 48, 64, 96, 128}) bench_at(T);
        }
    }
    return 0;
}
