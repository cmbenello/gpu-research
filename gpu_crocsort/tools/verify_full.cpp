// Comprehensive sort verifier.
// Given an input file and a sort output file, verifies:
//   (1) Same byte size / same record count
//   (2) Output is in non-decreasing order by the first KEY_SIZE bytes
//   (3) Output is a permutation of input (multiset preservation) — checked
//       via order-independent sum of per-record FNV-1a 64-bit hashes.
//
// Build:
//   g++ -O3 -std=c++17 -pthread tools/verify_full.cpp -o verify_full
// Usage:
//   ./verify_full --input <file> --output <file> --record-size N --key-size N [--threads N]
//
// Exit code 0 = ALL PASS, 1 = FAIL on any check, 2 = usage error.
//
// The multiset check uses sum-of-hashes mod 2^64, which is order-independent
// and provably equal iff the two files contain the same multiset of records
// (modulo a 1/2^64 collision probability per disagreement, which is the
// FNV-1a hash collision rate).

#include <atomic>
#include <chrono>
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

static inline uint64_t fnv1a64(const uint8_t* data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

struct MMap {
    void* addr = nullptr;
    size_t bytes = 0;
    int fd = -1;
    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) { perror(path); return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { perror("fstat"); return false; }
        bytes = (size_t)st.st_size;
        addr = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { perror("mmap"); return false; }
        madvise(addr, bytes, MADV_SEQUENTIAL);
        return true;
    }
    ~MMap() {
        if (addr && addr != MAP_FAILED) munmap(addr, bytes);
        if (fd >= 0) close(fd);
    }
};

int main(int argc, char** argv) {
    const char* input_path = nullptr;
    const char* output_path = nullptr;
    int record_size = 0, key_size = 0;
    int nthreads = std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--input") && i+1 < argc) input_path = argv[++i];
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output_path = argv[++i];
        else if (!strcmp(argv[i], "--record-size") && i+1 < argc) record_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--key-size") && i+1 < argc) key_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) nthreads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!input_path || !output_path || record_size <= 0 || key_size <= 0 || key_size > record_size) {
        fprintf(stderr, "usage: %s --input FILE --output FILE --record-size N --key-size N [--threads N]\n", argv[0]);
        return 2;
    }

    MMap inp, out;
    if (!inp.open(input_path) || !out.open(output_path)) return 2;

    bool ok = true;
    auto print_check = [&](const char* name, bool pass, const char* extra = "") {
        printf("  %-40s %s%s\n", name, pass ? "PASS" : "FAIL", extra);
        if (!pass) ok = false;
    };

    // ── Check 1: file sizes match and are record-aligned ──
    printf("Check 1: byte size + record count\n");
    {
        char buf[256];
        bool size_match = inp.bytes == out.bytes;
        snprintf(buf, sizeof(buf), "  (input %zu B, output %zu B)", inp.bytes, out.bytes);
        print_check("byte sizes equal", size_match, buf);
        bool aligned = (inp.bytes % record_size == 0) && (out.bytes % record_size == 0);
        print_check("size divides record_size", aligned);
        if (!size_match || !aligned) {
            fprintf(stderr, "Cannot proceed with mismatched/misaligned files\n");
            return 1;
        }
    }
    uint64_t nrec = inp.bytes / record_size;
    printf("  → %lu records (%dB each, %dB key)\n", nrec, record_size, key_size);

    // ── Check 2: output is in non-decreasing order ──
    printf("Check 2: output is sorted (parallel adjacent-pair scan)\n");
    {
        auto t0 = std::chrono::steady_clock::now();
        std::atomic<uint64_t> bad{UINT64_MAX};
        uint64_t per_t = (nrec + nthreads - 1) / nthreads;
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = (uint64_t)t * per_t;
                uint64_t hi = std::min(lo + per_t, nrec);
                if (lo == 0) lo = 1;
                const uint8_t* d = (const uint8_t*)out.addr;
                for (uint64_t i = lo; i < hi; i++) {
                    if (bad.load(std::memory_order_relaxed) != UINT64_MAX) break;
                    if (memcmp(d + (i-1)*record_size, d + i*record_size, key_size) > 0) {
                        uint64_t cur = bad.load();
                        while (i < cur && !bad.compare_exchange_weak(cur, i)) {}
                        break;
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        char buf[256];
        if (bad.load() == UINT64_MAX) {
            snprintf(buf, sizeof(buf), "  (%lu adjacent pairs in %.0f ms, %.1f GB/s)",
                     nrec - 1, ms, out.bytes / (ms * 1e6));
            print_check("non-decreasing by first key bytes", true, buf);
        } else {
            uint64_t v = bad.load();
            snprintf(buf, sizeof(buf), "  (first violation at record %lu)", v);
            print_check("non-decreasing by first key bytes", false, buf);
            const uint8_t* a = (const uint8_t*)out.addr + (v-1)*record_size;
            const uint8_t* b = (const uint8_t*)out.addr + v*record_size;
            fprintf(stderr, "    a[0..%d] = ", key_size);
            for (int k = 0; k < key_size; k++) fprintf(stderr, "%02x ", a[k]);
            fprintf(stderr, "\n    b[0..%d] = ", key_size);
            for (int k = 0; k < key_size; k++) fprintf(stderr, "%02x ", b[k]);
            fprintf(stderr, "\n");
        }
    }

    // ── Check 3: output is a permutation of input (multiset hash) ──
    // Per-record FNV-1a 64-bit hash, summed across all records mod 2^64.
    // Order-independent — equal iff the two files contain the same multiset
    // of records (with collision probability ~1/2^64 per distinct record).
    printf("Check 3: multiset preservation (sum of FNV-1a 64 hashes per record)\n");
    {
        auto t0 = std::chrono::steady_clock::now();
        auto hash_sum = [&](const uint8_t* data) {
            std::vector<std::atomic<uint64_t>> partials(nthreads);
            for (int t = 0; t < nthreads; t++) partials[t].store(0);
            uint64_t per_t = (nrec + nthreads - 1) / nthreads;
            std::vector<std::thread> threads;
            for (int t = 0; t < nthreads; t++) {
                threads.emplace_back([&, t]() {
                    uint64_t lo = (uint64_t)t * per_t;
                    uint64_t hi = std::min(lo + per_t, nrec);
                    uint64_t local = 0;
                    for (uint64_t i = lo; i < hi; i++)
                        local += fnv1a64(data + i*record_size, record_size);
                    partials[t].store(local);
                });
            }
            for (auto& t : threads) t.join();
            uint64_t sum = 0;
            for (int t = 0; t < nthreads; t++) sum += partials[t].load();
            return sum;
        };
        uint64_t in_sum = hash_sum((const uint8_t*)inp.addr);
        uint64_t out_sum = hash_sum((const uint8_t*)out.addr);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        char buf[256];
        snprintf(buf, sizeof(buf), "  (input=0x%016lx, output=0x%016lx, %.0f ms)",
                 in_sum, out_sum, ms);
        print_check("multiset hash match", in_sum == out_sum, buf);
        if (in_sum != out_sum) {
            fprintf(stderr, "    multiset differs — output is NOT a permutation of input\n");
            fprintf(stderr, "    (record was added, removed, or modified during sort)\n");
        }
    }

    printf("\n%s\n", ok ? "── ALL CHECKS PASSED ──" : "── SOME CHECKS FAILED ──");
    return ok ? 0 : 1;
}
