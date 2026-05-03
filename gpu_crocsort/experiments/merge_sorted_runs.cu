// 15.5 — Multi-way merge of sorted TPC-H lineitem run files.
//
// Reads K already-sorted .bin files (each contains 120 B normalized records
// in sorted order by KEY_SIZE-byte key prefix) and produces a single
// globally-sorted output file.
//
// Algorithm: tournament-style k-way merge using a min-heap over the heads of
// the K input streams. memcmp on the first KEY_SIZE bytes of each record.
//
// Build: nvcc -O3 -std=c++17 experiments/merge_sorted_runs.cu -o merge_sorted_runs
//        (or g++ — no GPU here; nvcc accepts .cu by default for consistency)
// Run:   ./merge_sorted_runs --output OUT IN1 IN2 ... INK
//
// Memory: only mmaps; no buffered allocations beyond the output write buffer.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static constexpr int KEY_SIZE   = 66;
static constexpr int RECORD_SIZE = 120;

struct Stream {
    const uint8_t* base;
    uint64_t n_records;
    uint64_t pos;
    int idx;
};

struct CompareHeads {
    const std::vector<Stream>* streams;
    bool operator()(int a, int b) const {
        const auto& A = (*streams)[a];
        const auto& B = (*streams)[b];
        const uint8_t* ka = A.base + A.pos * RECORD_SIZE;
        const uint8_t* kb = B.base + B.pos * RECORD_SIZE;
        // Min-heap → "less" means "a should be popped LATER than b"
        // i.e., a is bigger. So we want "ka > kb" returning true.
        return memcmp(ka, kb, KEY_SIZE) > 0;
    }
};

int main(int argc, char** argv) {
    const char* output = nullptr;
    std::vector<const char*> inputs;
    bool verify = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output") && i+1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "--verify")) verify = true;
        else inputs.push_back(argv[i]);
    }
    if (!output || inputs.empty()) {
        fprintf(stderr, "Usage: %s --output OUT [--verify] IN1 IN2 ... INK\n", argv[0]);
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // mmap all inputs read-only
    std::vector<Stream> streams(inputs.size());
    uint64_t total_records = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        struct stat st;
        if (stat(inputs[i], &st) != 0) {
            fprintf(stderr, "stat(%s) failed\n", inputs[i]); return 1;
        }
        size_t n_bytes = st.st_size;
        uint64_t n_rec = n_bytes / RECORD_SIZE;
        int fd = open(inputs[i], O_RDONLY);
        void* p = mmap(nullptr, n_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) { fprintf(stderr, "mmap %s failed\n", inputs[i]); return 1; }
        madvise(p, n_bytes, MADV_SEQUENTIAL);
        streams[i] = { (const uint8_t*)p, n_rec, 0, (int)i };
        total_records += n_rec;
        printf("  input[%zu]: %s, %lu records (%.2f GB)\n",
               i, inputs[i], n_rec, n_bytes/1e9);
    }
    uint64_t total_bytes = total_records * RECORD_SIZE;
    printf("Merging %zu streams, %lu total records (%.2f GB) → %s\n",
           inputs.size(), total_records, total_bytes/1e9, output);

    // Open output, pre-allocate. mmap with PROT_READ|PROT_WRITE + MAP_SHARED
    // requires the fd to be O_RDWR (O_WRONLY is rejected by mmap).
    int outfd = open(output, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (outfd < 0) { perror("open output"); return 1; }
    if (ftruncate(outfd, total_bytes) != 0) { perror("ftruncate"); return 1; }
    uint8_t* out = (uint8_t*)mmap(nullptr, total_bytes, PROT_READ|PROT_WRITE,
                                  MAP_SHARED, outfd, 0);
    close(outfd);
    if (out == MAP_FAILED) { perror("mmap output"); return 1; }
    madvise(out, total_bytes, MADV_SEQUENTIAL);

    // Init priority queue with one head from each non-empty stream
    CompareHeads cmp{&streams};
    std::priority_queue<int, std::vector<int>, CompareHeads> pq(cmp);
    for (size_t i = 0; i < streams.size(); i++) {
        if (streams[i].n_records > 0) pq.push((int)i);
    }

    auto t_setup = std::chrono::high_resolution_clock::now();
    printf("Setup: %.0f ms. Beginning merge.\n",
           std::chrono::duration<double, std::milli>(t_setup - t0).count());

    // K-way merge loop. For very large total_records and small K (4), the
    // priority queue's overhead can dominate; fall back to inline compare
    // when K <= 4.
    uint64_t out_pos = 0;
    if (streams.size() <= 4) {
        // Inline 4-way (or smaller) compare. Faster than std::priority_queue.
        // Track current head pointers for each stream.
        const uint8_t* heads[4] = {nullptr, nullptr, nullptr, nullptr};
        uint64_t remaining[4] = {0,0,0,0};
        for (size_t i = 0; i < streams.size(); i++) {
            heads[i] = streams[i].base;
            remaining[i] = streams[i].n_records;
        }
        int K = (int)streams.size();
        uint64_t last_log = 0;
        while (true) {
            int min_i = -1;
            for (int i = 0; i < K; i++) {
                if (remaining[i] == 0) continue;
                if (min_i < 0 || memcmp(heads[i], heads[min_i], KEY_SIZE) < 0) {
                    min_i = i;
                }
            }
            if (min_i < 0) break;
            memcpy(out + out_pos * RECORD_SIZE, heads[min_i], RECORD_SIZE);
            heads[min_i] += RECORD_SIZE;
            remaining[min_i]--;
            out_pos++;
            if (out_pos - last_log >= total_records / 10) {
                printf("  %5.1f%% (%lu / %lu)\n",
                       100.0 * out_pos / total_records, out_pos, total_records);
                last_log = out_pos;
            }
        }
    } else {
        // Generic priority-queue path for K > 4.
        uint64_t last_log = 0;
        while (!pq.empty()) {
            int top = pq.top(); pq.pop();
            const uint8_t* src = streams[top].base + streams[top].pos * RECORD_SIZE;
            memcpy(out + out_pos * RECORD_SIZE, src, RECORD_SIZE);
            streams[top].pos++;
            out_pos++;
            if (streams[top].pos < streams[top].n_records) pq.push(top);
            if (out_pos - last_log >= total_records / 10) {
                printf("  %5.1f%% (%lu / %lu)\n",
                       100.0 * out_pos / total_records, out_pos, total_records);
                last_log = out_pos;
            }
        }
    }

    auto t_merge = std::chrono::high_resolution_clock::now();
    double merge_ms = std::chrono::duration<double, std::milli>(t_merge - t_setup).count();
    printf("Merged %lu records in %.0f ms (%.2f GB/s)\n",
           out_pos, merge_ms, total_bytes/(merge_ms*1e6));

    if (verify) {
        printf("Verifying global sortedness...\n");
        uint64_t bad = 0;
        for (uint64_t i = 1; i < total_records && bad < 5; i++) {
            const uint8_t* a = out + (i-1)*RECORD_SIZE;
            const uint8_t* b = out + i*RECORD_SIZE;
            if (memcmp(a, b, KEY_SIZE) > 0) {
                if (bad < 3) printf("  VIOLATION at %lu\n", i);
                bad++;
            }
        }
        printf(bad == 0 ? "  PASS: %lu records globally sorted\n"
                        : "  FAIL: %lu violations\n",
               bad == 0 ? total_records : bad);
    }

    msync(out, total_bytes, MS_SYNC);
    munmap(out, total_bytes);
    for (size_t i = 0; i < streams.size(); i++) {
        munmap((void*)streams[i].base, streams[i].n_records * RECORD_SIZE);
    }
    return 0;
}
