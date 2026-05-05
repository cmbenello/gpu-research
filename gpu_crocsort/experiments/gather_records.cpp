// 19.26 — Gather sorted records from input.bin + sorted_offsets files.
// Produces the "full e2e" sorted-records output equivalent to 49m baseline.
//
// Reads:  INPUT.bin (1.08 TB at SF1500) + N×sorted_OFFSETS.bin (68 GB total)
// Writes: SORTED_RECORDS.bin (1.08 TB)
//
// The gather is sequential-write to output, random-read from input.
// At SF1500 with input.bin in OS cache: random reads from RAM at ~50 GB/s.
// Output write at NVMe peak 1.14 GB/s → 1.08 TB / 1.14 = 16 min wall.
//
// Build: g++ -O3 -std=c++17 -pthread experiments/gather_records.cpp -o experiments/gather_records
// Run:   ./gather_records INPUT.bin OUT.bin SORTED_OFFSETS_0.bin [SORTED_OFFSETS_1.bin ...]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr int RECORD_SIZE = 120;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s INPUT.bin OUT.bin OFFSETS_0.bin [OFFSETS_1.bin ...]\n", argv[0]);
        return 1;
    }
    const char* input_path = argv[1];
    const char* out_path = argv[2];
    int n_off_files = argc - 3;

    auto t0 = std::chrono::high_resolution_clock::now();

    // mmap input
    struct stat sin;
    if (stat(input_path, &sin) != 0) { perror("stat input"); return 1; }
    uint64_t total_input = sin.st_size;
    uint64_t total_records = total_input / RECORD_SIZE;
    int fdi = open(input_path, O_RDONLY);
    if (fdi < 0) { perror("open input"); return 1; }
    const uint8_t* input = (const uint8_t*)mmap(nullptr, total_input, PROT_READ, MAP_PRIVATE, fdi, 0);
    close(fdi);
    if (input == MAP_FAILED) { perror("mmap input"); return 1; }
    madvise((void*)input, total_input, MADV_RANDOM);
    printf("Input: %lu records (%.2f GB)\n", total_records, total_input/1e9);

    // Pre-compute total output size
    std::vector<uint64_t> off_records(n_off_files);
    uint64_t total_off_records = 0;
    for (int i = 0; i < n_off_files; i++) {
        struct stat so;
        if (stat(argv[3 + i], &so) != 0) { perror("stat offsets"); return 1; }
        off_records[i] = so.st_size / sizeof(uint64_t);
        total_off_records += off_records[i];
    }
    if (total_off_records != total_records) {
        fprintf(stderr, "WARN: offsets total %lu != input records %lu\n",
                total_off_records, total_records);
    }
    uint64_t out_bytes = total_off_records * RECORD_SIZE;
    printf("Output: %lu records (%.2f GB) across %d offsets files\n",
           total_off_records, out_bytes/1e9, n_off_files);

    // Open output, ftruncate, mmap
    int fdo = open(out_path, O_RDWR|O_CREAT|O_TRUNC, 0644);
    if (fdo < 0) { perror("open out"); return 1; }
    if (ftruncate(fdo, out_bytes) != 0) { perror("ftruncate"); return 1; }
    uint8_t* output = (uint8_t*)mmap(nullptr, out_bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fdo, 0);
    if (output == MAP_FAILED) { perror("mmap output"); return 1; }
    madvise(output, out_bytes, MADV_SEQUENTIAL);

    // For each offsets file: gather records to (cumulative output offset)
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);

    uint64_t out_pos = 0;
    auto t_gather0 = std::chrono::high_resolution_clock::now();
    for (int f = 0; f < n_off_files; f++) {
        struct stat so;
        stat(argv[3 + f], &so);
        int fdf = open(argv[3 + f], O_RDONLY);
        const uint64_t* offs = (const uint64_t*)mmap(nullptr, so.st_size, PROT_READ,
                                                       MAP_PRIVATE, fdf, 0);
        close(fdf);
        if (offs == MAP_FAILED) { perror("mmap offsets"); return 1; }
        madvise((void*)offs, so.st_size, MADV_SEQUENTIAL);

        uint64_t n = off_records[f];
        uint64_t per_t = (n + n_threads - 1) / n_threads;
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = (uint64_t)t * per_t;
                uint64_t hi = std::min(lo + per_t, n);
                constexpr int BLOCK = 64;
                const uint8_t* src_ptrs[BLOCK];
                for (uint64_t base = lo; base < hi; base += BLOCK) {
                    int count = std::min((uint64_t)BLOCK, hi - base);
                    for (int j = 0; j < count; j++) {
                        uint64_t src_idx = offs[base + j];
                        src_ptrs[j] = input + src_idx * RECORD_SIZE;
                        __builtin_prefetch(src_ptrs[j], 0, 0);
                    }
                    for (int j = 0; j < count; j++) {
                        memcpy(output + (out_pos + base + j) * RECORD_SIZE,
                               src_ptrs[j], RECORD_SIZE);
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
        munmap((void*)offs, so.st_size);
        out_pos += n;
        printf("  gathered file %d: %lu records (cumulative %lu)\n", f, n, out_pos);
    }
    auto t_gather1 = std::chrono::high_resolution_clock::now();
    double gather_ms = std::chrono::duration<double, std::milli>(t_gather1 - t_gather0).count();
    printf("Gather: %.0f ms (%.2f GB/s effective)\n", gather_ms, out_bytes/(gather_ms*1e6));

    munmap(output, out_bytes);
    close(fdo);
    munmap((void*)input, total_input);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total wall: %.0f ms (%.2f GB/s)\n", total_ms, out_bytes/(total_ms*1e6));
    printf("CSV,gather_records,records=%lu,gather_ms=%.0f,total_ms=%.0f,gb_per_s=%.2f\n",
           total_off_records, gather_ms, total_ms, out_bytes/(total_ms*1e6));
    return 0;
}
