// Standalone parallel sort verifier.
// Reads a binary record file (RECORD_SIZE bytes per record) and verifies that
// every adjacent pair is in non-decreasing order by the first KEY_SIZE bytes
// (lexicographic unsigned byte compare). Parallel over record partitions.
//
// Build:
//   g++ -O3 -std=c++17 -pthread tools/verify_sorted.cpp -o verify_sorted
//
// Usage:
//   ./verify_sorted <file> <record_size> <key_size> [num_threads]
//
// Exit code 0 = PASS, 1 = FAIL (violation found), 2 = usage error.
//
// Design notes:
//  - Memory-maps the file, so we work in page-cache without copying.
//  - Each worker scans records [lo..hi) checking adjacent pairs. We also
//    compare the boundary pair (record hi-1 vs record hi) so no pairs are
//    missed between partitions.
//  - A global atomic holds the first-found violation index; workers stop
//    once any violation is observed. This gives O(1) early exit.
//  - No dependency on the sort library — this is an INDEPENDENT correctness
//    check, not the same code path as the in-sort verify.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file> <record_size> <key_size> [threads]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    int record_size = atoi(argv[2]);
    int key_size = atoi(argv[3]);
    int nthreads = (argc >= 5) ? atoi(argv[4]) : (int)std::thread::hardware_concurrency();
    if (record_size <= 0 || key_size <= 0 || key_size > record_size) {
        fprintf(stderr, "bad record_size / key_size\n");
        return 2;
    }
    if (nthreads < 1) nthreads = 1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 2; }
    uint64_t file_bytes = (uint64_t)st.st_size;
    if (file_bytes % record_size != 0) {
        fprintf(stderr, "file size %lu not a multiple of record_size %d\n",
                file_bytes, record_size);
        close(fd);
        return 2;
    }
    uint64_t nrec = file_bytes / record_size;
    void* addr = mmap(nullptr, file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) { perror("mmap"); close(fd); return 2; }
    // Hint kernel: sequential access, willing to read ahead.
    madvise(addr, file_bytes, MADV_SEQUENTIAL);

    const uint8_t* data = (const uint8_t*)addr;

    printf("Verifying %lu records × %dB (key %dB) using %d threads\n",
           nrec, record_size, key_size, nthreads);

    // First-found violation (record index i such that records[i-1] > records[i]).
    std::atomic<uint64_t> violation{UINT64_MAX};
    std::atomic<uint64_t> compared{0};

    uint64_t per_t = (nrec + nthreads - 1) / nthreads;

    auto worker = [&](int tid) {
        uint64_t lo = (uint64_t)tid * per_t;
        uint64_t hi = std::min(lo + per_t, nrec);
        if (lo == 0) lo = 1;  // first worker starts at 1 (we compare i-1, i)
        // Also compare pair (lo-1, lo) for contiguity — other workers do this
        // too via their own range starting at per_t offsets.
        uint64_t local_cmp = 0;
        for (uint64_t i = lo; i < hi; i++) {
            if (violation.load(std::memory_order_relaxed) != UINT64_MAX) break;
            const uint8_t* a = data + (i - 1) * record_size;
            const uint8_t* b = data + i * record_size;
            int c = memcmp(a, b, key_size);
            local_cmp++;
            if (c > 0) {
                // CAS-min into the global violation index.
                uint64_t cur = violation.load(std::memory_order_relaxed);
                while (i < cur &&
                       !violation.compare_exchange_weak(cur, i,
                                                        std::memory_order_relaxed)) {}
                break;
            }
        }
        compared.fetch_add(local_cmp, std::memory_order_relaxed);
    };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; t++) threads.emplace_back(worker, t);
    for (auto& t : threads) t.join();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    uint64_t v = violation.load();
    uint64_t c = compared.load();
    double gbs = (file_bytes / 1e9) / (ms / 1e3);

    munmap(addr, file_bytes);
    close(fd);

    if (v == UINT64_MAX) {
        printf("PASS: all %lu adjacent pairs in non-decreasing order "
               "(%lu compares, %.0f ms, %.2f GB/s)\n", c, c, ms, gbs);
        return 0;
    } else {
        fprintf(stderr, "FAIL: violation at record %lu — records[%lu] > records[%lu] by first %d bytes\n",
                v, v - 1, v, key_size);
        // Print the offending bytes for diagnostic.
        const uint8_t* a_tmp = (const uint8_t*)mmap(nullptr, file_bytes, PROT_READ,
                                                     MAP_PRIVATE, open(path, O_RDONLY), 0);
        if (a_tmp != MAP_FAILED) {
            fprintf(stderr, "  a = "); for (int k = 0; k < key_size; k++) fprintf(stderr, "%02x ", a_tmp[(v-1)*record_size+k]);
            fprintf(stderr, "\n  b = "); for (int k = 0; k < key_size; k++) fprintf(stderr, "%02x ", a_tmp[v*record_size+k]);
            fprintf(stderr, "\n");
        }
        return 1;
    }
}
