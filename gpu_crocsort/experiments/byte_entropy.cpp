// 19.8 — Per-byte entropy analysis of TPC-H lineitem
//
// For each of the 66 sort-key bytes, count distinct values across
// a sample of records. Outputs which bytes have low entropy (good
// compaction candidate) and which have high entropy (must keep).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <set>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

constexpr int RECORD_SIZE = 120;
constexpr int KEY_SIZE = 66;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.bin> [n_sample=1000000]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    uint64_t N_SAMPLE = (argc >= 3) ? std::strtoull(argv[2], nullptr, 10) : 1000000;

    struct stat st;
    if (stat(path, &st) != 0) { perror(path); return 1; }
    uint64_t total_bytes = st.st_size;
    uint64_t total_records = total_bytes / RECORD_SIZE;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 1; }
    const uint8_t* in = (const uint8_t*)mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (in == MAP_FAILED) { perror("mmap"); return 1; }
    madvise((void*)in, total_bytes, MADV_RANDOM);

    if (N_SAMPLE > total_records) N_SAMPLE = total_records;
    fprintf(stderr, "Sampling %lu of %lu records (%.2fGB)\n",
            N_SAMPLE, total_records, total_bytes/1e9);

    // Per-byte distinct counts
    std::vector<std::set<uint8_t>> byte_distinct(KEY_SIZE);

    std::mt19937_64 rng(42);
    for (uint64_t i = 0; i < N_SAMPLE; i++) {
        uint64_t idx = (i * total_records) / N_SAMPLE;  // stride sample
        const uint8_t* rec = in + idx * RECORD_SIZE;
        for (int b = 0; b < KEY_SIZE; b++) {
            byte_distinct[b].insert(rec[b]);
        }
    }

    // Output
    printf("byte_position,distinct_values,fraction_full,bits_used\n");
    int low_entropy_count = 0;  // <= 16 distinct
    int mid_entropy_count = 0;  // 17 - 64
    int high_entropy_count = 0; // > 64
    for (int b = 0; b < KEY_SIZE; b++) {
        size_t d = byte_distinct[b].size();
        double frac = d / 256.0;
        // bits = ceil(log2(d))
        int bits = 0;
        while ((1u << bits) < d) bits++;
        printf("%d,%zu,%.3f,%d\n", b, d, frac, bits);
        if (d <= 16) low_entropy_count++;
        else if (d <= 64) mid_entropy_count++;
        else high_entropy_count++;
    }

    fprintf(stderr, "\nSummary:\n");
    fprintf(stderr, "  Low entropy (<=16 distinct):  %d bytes\n", low_entropy_count);
    fprintf(stderr, "  Mid entropy (17-64):          %d bytes\n", mid_entropy_count);
    fprintf(stderr, "  High entropy (>64):           %d bytes\n", high_entropy_count);
    fprintf(stderr, "  Total bytes scanned:          %d\n", KEY_SIZE);

    // Entropy budget — bits per record assuming each byte uses log2(d) bits
    double total_bits = 0;
    for (int b = 0; b < KEY_SIZE; b++) {
        size_t d = byte_distinct[b].size();
        if (d <= 1) continue;
        // ideal bits per byte = log2(distinct values)
        double bits = std::log2((double)d);
        total_bits += bits;
    }
    fprintf(stderr, "  Total entropy bits per record: %.1f (vs %d if all bytes were full)\n",
            total_bits, KEY_SIZE * 8);
    fprintf(stderr, "  Theoretical compact key size:  %.1f bytes\n", total_bits / 8.0);
    fprintf(stderr, "  Compact map decision (bytes used):  %d of %d\n",
            low_entropy_count + mid_entropy_count + high_entropy_count, KEY_SIZE);
    return 0;
}
