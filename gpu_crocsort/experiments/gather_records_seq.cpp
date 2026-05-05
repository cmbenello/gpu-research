// 19.27 — Sequential-read gather. Read input.bin sequentially, scatter to
// random output positions via inv_perm.
//
// vs random-read gather (gather_records.cpp): trades random reads for random
// writes. NVMe random writes (especially with kernel write coalescing) can
// be faster than random reads.
//
// Build: g++ -O3 -std=c++17 -pthread experiments/gather_records_seq.cpp -o experiments/gather_records_seq
// Run:   ./gather_records_seq INPUT.bin OUT.bin OFFSETS_0.bin [...]
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

    struct stat sin;
    if (stat(input_path, &sin) != 0) { perror("stat input"); return 1; }
    uint64_t total_input = sin.st_size;
    uint64_t total_records = total_input / RECORD_SIZE;
    int fdi = open(input_path, O_RDONLY);
    const uint8_t* input = (const uint8_t*)mmap(nullptr, total_input, PROT_READ, MAP_PRIVATE, fdi, 0);
    close(fdi);
    if (input == MAP_FAILED) { perror("mmap input"); return 1; }
    madvise((void*)input, total_input, MADV_SEQUENTIAL);  // KEY: sequential read hint
    printf("Input: %lu records (%.2f GB)\n", total_records, total_input/1e9);

    // Build inv_perm: inv_perm[input_offset] = sorted_position
    // Read all sorted-offset files in order, increment cumulative sorted pos
    auto t_inv0 = std::chrono::high_resolution_clock::now();
    uint32_t* inv_perm = (uint32_t*)mmap(nullptr, total_records * sizeof(uint32_t),
                                           PROT_READ|PROT_WRITE,
                                           MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (inv_perm == MAP_FAILED) { perror("mmap inv_perm"); return 1; }
    madvise(inv_perm, total_records * sizeof(uint32_t), MADV_HUGEPAGE);

    // Sanity: inv_perm holds 4-byte indices, but total_records can be 9B → needs 5 bytes.
    // For now use uint32_t with TRUNCATED bucket-local indexing.
    // Better: use uint64_t. 36 GB array.
    munmap(inv_perm, total_records * sizeof(uint32_t));

    uint64_t* inv_perm64 = (uint64_t*)mmap(nullptr, total_records * sizeof(uint64_t),
                                             PROT_READ|PROT_WRITE,
                                             MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (inv_perm64 == MAP_FAILED) { perror("mmap inv_perm64"); return 1; }
    madvise(inv_perm64, total_records * sizeof(uint64_t), MADV_HUGEPAGE);

    uint64_t cum = 0;
    for (int f = 0; f < n_off_files; f++) {
        struct stat so; stat(argv[3 + f], &so);
        int fdf = open(argv[3 + f], O_RDONLY);
        const uint64_t* offs = (const uint64_t*)mmap(nullptr, so.st_size, PROT_READ,
                                                       MAP_PRIVATE, fdf, 0);
        close(fdf);
        uint64_t n = so.st_size / sizeof(uint64_t);
        for (uint64_t i = 0; i < n; i++) {
            inv_perm64[offs[i]] = cum + i;
        }
        munmap((void*)offs, so.st_size);
        cum += n;
        printf("  built inv_perm from file %d: %lu cumulative\n", f, cum);
    }
    auto t_inv1 = std::chrono::high_resolution_clock::now();
    double inv_ms = std::chrono::duration<double, std::milli>(t_inv1 - t_inv0).count();
    printf("inv_perm: %.0f ms\n", inv_ms);

    // Open output file (sparse), mmap it
    int fdo = open(out_path, O_RDWR|O_CREAT|O_TRUNC, 0644);
    uint64_t out_bytes = total_records * RECORD_SIZE;
    if (ftruncate(fdo, out_bytes) != 0) { perror("ftruncate"); return 1; }
    uint8_t* output = (uint8_t*)mmap(nullptr, out_bytes, PROT_READ|PROT_WRITE,
                                       MAP_SHARED, fdo, 0);
    if (output == MAP_FAILED) { perror("mmap output"); return 1; }
    madvise(output, out_bytes, MADV_RANDOM);  // hint: random writes

    // Stream input sequentially, scatter to output via inv_perm
    auto t_scat0 = std::chrono::high_resolution_clock::now();
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);
    uint64_t per_t = (total_records + n_threads - 1) / n_threads;

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            uint64_t lo = (uint64_t)t * per_t;
            uint64_t hi = std::min(lo + per_t, total_records);
            for (uint64_t r = lo; r < hi; r++) {
                uint64_t target = inv_perm64[r];
                memcpy(output + target * RECORD_SIZE, input + r * RECORD_SIZE, RECORD_SIZE);
            }
        });
    }
    for (auto& th : threads) th.join();
    auto t_scat1 = std::chrono::high_resolution_clock::now();
    double scat_ms = std::chrono::duration<double, std::milli>(t_scat1 - t_scat0).count();
    printf("Sequential-read scatter: %.0f ms (%.2f GB/s)\n", scat_ms, out_bytes/(scat_ms*1e6));

    munmap(output, out_bytes);
    close(fdo);
    munmap(inv_perm64, total_records * sizeof(uint64_t));
    munmap((void*)input, total_input);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total wall: %.0f ms (%.2f GB/s)\n", total_ms, out_bytes/(total_ms*1e6));
    printf("CSV,gather_records_seq,records=%lu,inv_ms=%.0f,scat_ms=%.0f,total_ms=%.0f\n",
           total_records, inv_ms, scat_ms, total_ms);
    return 0;
}
