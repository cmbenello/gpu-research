// 15.5 — Multi-way merge of sorted TPC-H lineitem run files.
//
// Reads K already-sorted .bin files (each contains 120 B normalized records
// in sorted order by KEY_SIZE-byte key prefix) and produces a single
// globally-sorted output file.
//
// Two algorithms:
//   - Single-threaded: tournament-style k-way merge over the head of each
//     stream. Used when --threads is 1 or unset.
//   - Multi-threaded (15.5.2): partition the merged output into N stripes
//     using sample splitters. Each thread merges its stripe independently.
//     Used when --threads > 1. ~4-16× faster on this 192-core box.
//
// Build: nvcc -O3 -std=c++17 experiments/merge_sorted_runs.cu -o merge_sorted_runs
// Run:   ./merge_sorted_runs --output OUT [--verify] [--threads N] IN1 IN2 ... INK
//
// Memory: mmap'd inputs/output; per-thread O(K) state for stream pointers.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <thread>
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

// Binary-search the position in a SORTED stream where `target_key` would be
// inserted (returns the first record index with key >= target).
// Used to partition each input stream by sample splitters.
static uint64_t lower_bound_key(const uint8_t* base, uint64_t n_records,
                                const uint8_t* target_key) {
    uint64_t lo = 0, hi = n_records;
    while (lo < hi) {
        uint64_t mid = (lo + hi) >> 1;
        if (memcmp(base + mid * RECORD_SIZE, target_key, KEY_SIZE) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int main(int argc, char** argv) {
    const char* output = nullptr;
    std::vector<const char*> inputs;
    bool verify = false;
    int n_threads = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output") && i+1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "--verify")) verify = true;
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) n_threads = atoi(argv[++i]);
        else inputs.push_back(argv[i]);
    }
    if (!output || inputs.empty()) {
        fprintf(stderr, "Usage: %s --output OUT [--verify] [--threads N] IN1 IN2 ... INK\n", argv[0]);
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

    // (15.5.2) Multi-threaded merge via splitter-based partitioning.
    // Sample 256 keys uniformly from each input, sort the union, pick
    // n_threads-1 splitters at quantile positions. For each input
    // stream, binary-search for the offset of each splitter. Each
    // thread merges its slice of all K streams into a contiguous
    // chunk of the output buffer (positions are pre-computed so no
    // synchronization needed during merge).
    uint64_t out_pos = 0;
    if (n_threads > 1 && streams.size() > 1) {
        const int K = (int)streams.size();
        const int T = n_threads;
        const int S = 256;          // samples per stream
        const int total_samples = K * S;

        // Collect samples
        std::vector<std::vector<uint8_t>> samples(total_samples,
                                                  std::vector<uint8_t>(KEY_SIZE));
        for (int k = 0; k < K; k++) {
            uint64_t step = streams[k].n_records / S;
            if (step == 0) step = 1;
            for (int s = 0; s < S; s++) {
                uint64_t idx = std::min((uint64_t)s * step, streams[k].n_records - 1);
                memcpy(samples[k*S + s].data(),
                       streams[k].base + idx * RECORD_SIZE, KEY_SIZE);
            }
        }
        std::sort(samples.begin(), samples.end(),
                  [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
                      return memcmp(a.data(), b.data(), KEY_SIZE) < 0;
                  });

        // Pick T-1 splitters at uniform quantiles
        std::vector<std::vector<uint8_t>> splitters(T - 1, std::vector<uint8_t>(KEY_SIZE));
        for (int t = 1; t < T; t++) {
            uint64_t pos = (uint64_t)t * total_samples / T;
            if (pos >= (uint64_t)total_samples) pos = total_samples - 1;
            memcpy(splitters[t-1].data(), samples[pos].data(), KEY_SIZE);
        }

        // Per-stream, per-thread offsets: stream_offsets[s][t] = first record
        // in stream s that belongs to thread t (i.e., key >= splitter[t-1]).
        std::vector<std::vector<uint64_t>> stream_offsets(K, std::vector<uint64_t>(T+1, 0));
        for (int s = 0; s < K; s++) {
            stream_offsets[s][0] = 0;
            stream_offsets[s][T] = streams[s].n_records;
            for (int t = 1; t < T; t++) {
                stream_offsets[s][t] = lower_bound_key(streams[s].base,
                                                       streams[s].n_records,
                                                       splitters[t-1].data());
            }
        }

        // Compute per-thread output start positions
        std::vector<uint64_t> out_start(T+1, 0);
        for (int t = 0; t < T; t++) {
            uint64_t cnt = 0;
            for (int s = 0; s < K; s++) {
                cnt += stream_offsets[s][t+1] - stream_offsets[s][t];
            }
            out_start[t+1] = out_start[t] + cnt;
        }
        if (out_start[T] != total_records) {
            fprintf(stderr, "splitter accounting bug: %lu vs %lu\n",
                    out_start[T], total_records);
            return 1;
        }

        // Spawn merge workers
        std::vector<std::thread> workers;
        for (int t = 0; t < T; t++) {
            workers.emplace_back([&, t]() {
                // For each stream, current head and end of this thread's slice
                std::vector<const uint8_t*> heads(K);
                std::vector<uint64_t> remaining(K);
                for (int s = 0; s < K; s++) {
                    heads[s] = streams[s].base + stream_offsets[s][t] * RECORD_SIZE;
                    remaining[s] = stream_offsets[s][t+1] - stream_offsets[s][t];
                }
                uint64_t op = out_start[t];
                while (true) {
                    int min_i = -1;
                    for (int s = 0; s < K; s++) {
                        if (remaining[s] == 0) continue;
                        if (min_i < 0 ||
                            memcmp(heads[s], heads[min_i], KEY_SIZE) < 0) {
                            min_i = s;
                        }
                    }
                    if (min_i < 0) break;
                    memcpy(out + op * RECORD_SIZE, heads[min_i], RECORD_SIZE);
                    heads[min_i] += RECORD_SIZE;
                    remaining[min_i]--;
                    op++;
                }
            });
        }
        for (auto& w : workers) w.join();
        out_pos = total_records;
        printf("  Multi-threaded merge: %d threads, %d-way streams\n", T, K);
    } else if (streams.size() <= 4) {
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
        // (15.5.4) Inline verify: instead of reading back the entire
        // 360 GB output from disk for verification (which takes ~2 min on
        // NVMe), check sortedness as we wrote it.
        // For multi-threaded merge, check boundaries between adjacent
        // thread slices PLUS sample within-slice pairs. Each thread's
        // slice is internally sorted by construction (single-stream
        // pop) — only the slice boundaries need an explicit check.
        printf("Verifying global sortedness (inline post-merge scan)...\n");
        auto v0 = std::chrono::high_resolution_clock::now();
        uint64_t bad = 0;
        for (uint64_t i = 1; i < total_records && bad < 5; i++) {
            const uint8_t* a = out + (i-1)*RECORD_SIZE;
            const uint8_t* b = out + i*RECORD_SIZE;
            if (memcmp(a, b, KEY_SIZE) > 0) {
                if (bad < 3) printf("  VIOLATION at %lu\n", i);
                bad++;
            }
        }
        auto v1 = std::chrono::high_resolution_clock::now();
        double v_ms = std::chrono::duration<double, std::milli>(v1 - v0).count();
        printf(bad == 0 ? "  PASS: %lu records globally sorted (%.0f ms verify)\n"
                        : "  FAIL: %lu violations\n",
               bad == 0 ? total_records : bad, v_ms);
    }

    // (15.5.4) Skip explicit msync — let the kernel flush at unmap or
    // process exit. Saves ~120 s at SF500 where 360 GB of dirty pages
    // get sync'd to NVMe at ~3 GB/s. The data is correct in memory;
    // the disk copy will be consistent once the kernel flushes.
    munmap(out, total_bytes);
    for (size_t i = 0; i < streams.size(); i++) {
        munmap((void*)streams[i].base, streams[i].n_records * RECORD_SIZE);
    }
    return 0;
}
