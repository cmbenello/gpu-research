// 18.6 — Generate synthetic 120 B records with controllable distribution.
//
//   --distribution random      uniform random keys
//                  sorted      already in ascending order
//                  reversed    descending
//                  all_equal   every record has the same key
//                  zipfian     skewed (1% of keys account for 50% of records)
//
// Build: g++ -O3 -std=c++17 -pthread experiments/gen_synthetic.cpp -o experiments/gen_synthetic
// Run: ./gen_synthetic --records 100000000 --output /mnt/data/synth_random_60M.bin --distribution random
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <random>
#include <thread>
#include <vector>
#include <algorithm>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

constexpr int RECORD_SIZE = 120;
constexpr int KEY_SIZE = 66;

void fill_record(uint8_t* rec, uint64_t key_high, uint64_t key_low, uint64_t i) {
    // First 16 bytes are the sort prefix (big-endian for lex order)
    for (int b = 0; b < 8; b++) rec[b]     = (key_high >> (56 - 8*b)) & 0xff;
    for (int b = 0; b < 8; b++) rec[8 + b] = (key_low  >> (56 - 8*b)) & 0xff;
    // Pad rest with i so each record is unique even with same key
    for (int b = 16; b < 66; b++) rec[b] = (i >> ((b - 16) * 4)) & 0xff;
    for (int b = 66; b < 120; b++) rec[b] = 0;
}

int main(int argc, char** argv) {
    uint64_t N = 60000000;
    const char* output = "/tmp/synth.bin";
    const char* dist = "random";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--records") && i+1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "--distribution") && i+1 < argc) dist = argv[++i];
    }

    fprintf(stderr, "Generating %lu records (%.1f GB) → %s, distribution=%s\n",
            N, N * 120.0 / 1e9, output, dist);

    int fd = open(output, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 1; }
    if (ftruncate(fd, N * RECORD_SIZE) < 0) { perror("ftruncate"); return 1; }

    uint8_t* mem = (uint8_t*)mmap(nullptr, N * RECORD_SIZE, PROT_READ|PROT_WRITE,
                                   MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(mem, N * RECORD_SIZE, MADV_SEQUENTIAL);

    int hwc = std::max(1, (int)std::thread::hardware_concurrency());
    int T = std::min(hwc, 64);
    std::vector<std::thread> threads;
    uint64_t per_t = (N + T - 1) / T;

    auto fill = [&](uint64_t s, uint64_t e, int tid) {
        std::mt19937_64 rng(42 + tid);
        for (uint64_t i = s; i < e; i++) {
            uint64_t kh, kl;
            if (!strcmp(dist, "random")) {
                kh = rng();
                kl = rng();
            } else if (!strcmp(dist, "sorted")) {
                kh = i;
                kl = 0;
            } else if (!strcmp(dist, "reversed")) {
                kh = ~i;
                kl = ~(uint64_t)0;
            } else if (!strcmp(dist, "all_equal")) {
                kh = 0xdeadbeefdeadbeefULL;
                kl = 0xcafebabecafebabeULL;
            } else if (!strcmp(dist, "zipfian")) {
                // Zipfian: 1% of values account for 50% of records.
                // Approximate by drawing rank from rng then mapping nonlinearly.
                uint64_t r = rng();
                // 50% of the time, pick from a small set of "popular" keys
                if (r % 2 == 0) {
                    kh = r % 1000000;  // 1M popular keys
                    kl = 0;
                } else {
                    kh = rng();
                    kl = rng();
                }
            } else {
                fprintf(stderr, "Unknown distribution: %s\n", dist);
                std::exit(1);
            }
            fill_record(mem + i * (uint64_t)RECORD_SIZE, kh, kl, i);
        }
    };

    for (int t = 0; t < T; t++) {
        uint64_t s = (uint64_t)t * per_t;
        uint64_t e = std::min(s + per_t, N);
        if (s < e) threads.emplace_back(fill, s, e, t);
    }
    for (auto& th : threads) th.join();

    munmap(mem, N * RECORD_SIZE);
    close(fd);
    fprintf(stderr, "Done.\n");
    return 0;
}
