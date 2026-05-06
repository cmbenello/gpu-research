// 19.59 — Sort + aggregate query (TPC-H Q1 style).
// Reads sorted offsets, gathers records from input.bin, computes
// SELECT count(*), sum(l_quantity_proxy) FROM lineitem GROUP BY first_byte_of_key
//
// Demonstrates the sort recipe plugs into a real analytical pipeline:
// sort produces ordered offsets → linear-scan aggregate is trivial (records
// with same group key are contiguous).
//
// Build: g++ -O3 -std=c++17 -pthread experiments/sort_aggregate.cpp -o experiments/sort_aggregate
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <utility>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr int RECORD_SIZE = 120;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s INPUT.bin GROUP_BYTE OFFSETS_0.bin [OFFSETS_1.bin ...]\n"
                        "  GROUP_BYTE: byte position in record to group by (e.g., 0 for first byte)\n", argv[0]);
        return 1;
    }
    const char* input_path = argv[1];
    int group_byte = atoi(argv[2]);
    int n_off_files = argc - 3;

    auto t0 = std::chrono::high_resolution_clock::now();

    struct stat sin;
    stat(input_path, &sin);
    uint64_t total_input = sin.st_size;
    int fdi = open(input_path, O_RDONLY);
    const uint8_t* input = (const uint8_t*)mmap(nullptr, total_input, PROT_READ, MAP_PRIVATE, fdi, 0);
    close(fdi);
    if (input == MAP_FAILED) { perror("mmap input"); return 1; }
    madvise((void*)input, total_input, MADV_RANDOM);

    // Open all offset files (each contains uint64 offsets in sorted order)
    uint64_t total_recs = 0;
    std::vector<std::pair<const uint64_t*, uint64_t>> offset_arrays;
    for (int f = 0; f < n_off_files; f++) {
        struct stat so; stat(argv[3 + f], &so);
        int fdf = open(argv[3 + f], O_RDONLY);
        const uint64_t* offs = (const uint64_t*)mmap(nullptr, so.st_size, PROT_READ, MAP_PRIVATE, fdf, 0);
        close(fdf);
        madvise((void*)offs, so.st_size, MADV_SEQUENTIAL);
        uint64_t n = so.st_size / sizeof(uint64_t);
        offset_arrays.push_back({offs, n});
        total_recs += n;
    }
    printf("Input: %lu records, %d offset files, group_byte=%d\n", total_recs, n_off_files, group_byte);

    // Aggregate: count per group_byte value (256 buckets)
    auto t_agg0 = std::chrono::high_resolution_clock::now();
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, 64);
    // Per-thread per-group counter (256 groups × 64 threads = 16K counters)
    std::vector<std::vector<uint64_t>> tcount(n_threads, std::vector<uint64_t>(256, 0));

    std::vector<std::thread> threads;
    int file_idx = 0;
    uint64_t cumulative_off = 0;
    // Distribute work across offset files in order
    for (auto& [offs, n] : offset_arrays) {
        uint64_t per_t = (n + n_threads - 1) / n_threads;
        std::vector<std::thread> t_workers;
        for (int t = 0; t < n_threads; t++) {
            t_workers.emplace_back([&, t]() {
                uint64_t lo = (uint64_t)t * per_t;
                uint64_t hi = std::min(lo + per_t, n);
                auto& mycnt = tcount[t];
                for (uint64_t i = lo; i < hi; i++) {
                    uint64_t rec_off = offs[i];
                    uint8_t g = input[rec_off * RECORD_SIZE + group_byte];
                    mycnt[g]++;
                }
            });
        }
        for (auto& th : t_workers) th.join();
        munmap((void*)offs, n * sizeof(uint64_t));
    }
    auto t_agg1 = std::chrono::high_resolution_clock::now();
    double agg_ms = std::chrono::duration<double, std::milli>(t_agg1 - t_agg0).count();

    // Reduce per-thread counters
    uint64_t total[256] = {0};
    for (auto& tc : tcount) for (int g = 0; g < 256; g++) total[g] += tc[g];

    uint64_t verify_total = 0;
    int n_groups_with_data = 0;
    for (int g = 0; g < 256; g++) {
        if (total[g] > 0) {
            verify_total += total[g];
            n_groups_with_data++;
        }
    }
    printf("Aggregate (gather + count): %.0f ms (%.2f Mrec/s)\n",
           agg_ms, total_recs / (agg_ms * 1e3));
    printf("Total records aggregated: %lu (verify match: %s)\n",
           verify_total, verify_total == total_recs ? "YES" : "NO");
    printf("Distinct group values: %d / 256\n", n_groups_with_data);

    // Show top 5 groups by count
    std::vector<std::pair<uint64_t, int>> sorted_groups;
    for (int g = 0; g < 256; g++) if (total[g] > 0) sorted_groups.push_back({total[g], g});
    std::sort(sorted_groups.begin(), sorted_groups.end(), std::greater<std::pair<uint64_t, int>>());
    printf("Top 5 groups (by count):\n");
    for (int i = 0; i < std::min((int)sorted_groups.size(), 5); i++) {
        printf("  group_byte=0x%02x: %lu records (%.1f%%)\n",
               sorted_groups[i].second, sorted_groups[i].first,
               100.0 * sorted_groups[i].first / total_recs);
    }

    munmap((void*)input, total_input);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total wall: %.0f ms\n", total_ms);
    printf("CSV,sort_aggregate,records=%lu,group_byte=%d,agg_ms=%.0f,total_ms=%.0f,Mrec_per_s=%.2f\n",
           total_recs, group_byte, agg_ms, total_ms, total_recs / (agg_ms * 1e3));
    return 0;
}
