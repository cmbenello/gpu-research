// Take a TPC-H lineitem file and inject heavy-hitter skew by replacing
// SKEW_PCT% of records with copies of record 0 (creating a heavy duplicate).
// Output is a new file at OUT_PATH.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <random>
#include <thread>
#include <vector>

constexpr int RECORD_SIZE = 120;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s INPUT.bin OUT.bin SKEW_PCT (e.g., 30 for 30%%)\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    int skew_pct = atoi(argv[3]);
    if (skew_pct < 0 || skew_pct > 99) { fprintf(stderr, "skew_pct in [0,99]\n"); return 1; }

    struct stat st;
    if (stat(in_path, &st) != 0) { perror("stat"); return 1; }
    uint64_t total_bytes = st.st_size;
    uint64_t n = total_bytes / RECORD_SIZE;
    int fdi = open(in_path, O_RDONLY);
    const uint8_t* in = (const uint8_t*)mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fdi, 0);
    close(fdi);
    if (in == MAP_FAILED) { perror("mmap input"); return 1; }

    int fdo = open(out_path, O_RDWR|O_CREAT|O_TRUNC, 0644);
    ftruncate(fdo, total_bytes);
    uint8_t* out = (uint8_t*)mmap(nullptr, total_bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fdo, 0);
    if (out == MAP_FAILED) { perror("mmap out"); return 1; }

    printf("Injecting %d%% skew (record 0's key) into %lu records (%.2f GB)\n",
           skew_pct, n, total_bytes/1e9);

    // Copy record 0's full content into a buffer for fast duplication
    uint8_t skew_rec[RECORD_SIZE];
    memcpy(skew_rec, in, RECORD_SIZE);

    int n_threads = std::min(64u, std::thread::hardware_concurrency());
    uint64_t per_t = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937_64 rng(0xdeadbeef + t);
            std::uniform_int_distribution<int> dist(0, 99);
            uint64_t lo = (uint64_t)t * per_t;
            uint64_t hi = std::min(lo + per_t, n);
            for (uint64_t r = lo; r < hi; r++) {
                if (dist(rng) < skew_pct) {
                    memcpy(out + r * RECORD_SIZE, skew_rec, RECORD_SIZE);
                } else {
                    memcpy(out + r * RECORD_SIZE, in + r * RECORD_SIZE, RECORD_SIZE);
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    munmap((void*)in, total_bytes);
    munmap(out, total_bytes);
    close(fdo);
    printf("Done. Output: %s\n", out_path);
    return 0;
}
