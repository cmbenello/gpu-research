// 19.47 — Multi-stream GDS partition.
//
// Architecture:
//   - 4-slot ring buffer (16 GB total GPU memory)
//   - 2 reader threads, each blocking on cuFileRead into its own slot
//   - Main thread processes slots in order via 2 GPU streams:
//     * stream A: classify_scatter kernel
//     * stream B: D2H bucket buffers (cudaMemcpyAsync)
//
// Goal: hide cuFileRead behind GPU work AND hide D2H behind classify.
//
// Build: nvcc -O3 -arch=sm_90 -std=c++17 \
//             experiments/gds_partition_multistream.cu \
//             -lcufile -lpthread -o experiments/gds_partition_multistream
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <cub/cub.cuh>

static constexpr int KEY_SIZE = 66;
static constexpr int RECORD_SIZE = 120;
static constexpr int COMPACT_KEY_SIZE = 32;
static constexpr int COMPACT_REC_SIZE = COMPACT_KEY_SIZE + 8;
static constexpr int N_SLOTS = 4;       // ring buffer depth
static constexpr int N_READERS = 2;     // reader threads

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void extract_chunk_kernel(const uint8_t* records, const uint32_t* perm,
                                       uint64_t* out, uint64_t n, int byte_off, int chunk_bytes) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t idx = perm[i];
    const uint8_t* k = records + (uint64_t)idx * COMPACT_REC_SIZE + byte_off;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    out[i] = v;
}

__global__ void init_identity_kernel(uint32_t* perm, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) perm[i] = (uint32_t)i;
}

__global__ void gather_offsets_kernel(const uint8_t* records, const uint32_t* perm,
                                       uint64_t* out_offsets, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t idx = perm[i];
    out_offsets[i] = *(const uint64_t*)(records + (uint64_t)idx * COMPACT_REC_SIZE + COMPACT_KEY_SIZE);
}

// Sort one bucket on a given GPU. Pin host bucket buffer, upload, sort with
// CUB radix (4 LSD passes on 32-byte keys), gather offsets, write to file.
static void sort_one_bucket(int gpu_id, uint8_t* h_records, uint64_t n,
                             const char* output_path) {
    cudaSetDevice(gpu_id);
    uint64_t bytes = n * COMPACT_REC_SIZE;
    cudaHostRegister(h_records, bytes, cudaHostRegisterDefault);
    uint8_t* d_records;
    cudaMalloc(&d_records, bytes);
    cudaMemcpy(d_records, h_records, bytes, cudaMemcpyHostToDevice);
    cudaHostUnregister(h_records);

    uint32_t *d_perm_a, *d_perm_b;
    cudaMalloc(&d_perm_a, n * sizeof(uint32_t));
    cudaMalloc(&d_perm_b, n * sizeof(uint32_t));
    init_identity_kernel<<<(n+255)/256, 256>>>(d_perm_a, n);

    uint64_t *d_keys_a, *d_keys_b;
    cudaMalloc(&d_keys_a, n * sizeof(uint64_t));
    cudaMalloc(&d_keys_b, n * sizeof(uint64_t));

    size_t cub_temp_bytes = 0;
    cub::DoubleBuffer<uint64_t> kb(d_keys_a, d_keys_b);
    cub::DoubleBuffer<uint32_t> vb(d_perm_a, d_perm_b);
    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, kb, vb, (int)n, 0, 64);
    void* d_cub_temp;
    cudaMalloc(&d_cub_temp, cub_temp_bytes);

    for (int chunk = 3; chunk >= 0; chunk--) {
        int byte_off = chunk * 8;
        extract_chunk_kernel<<<(n+255)/256, 256>>>(d_records, vb.Current(), kb.Current(), n, byte_off, 8);
        cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_temp_bytes, kb, vb, (int)n, 0, 64);
    }

    uint64_t* d_offsets;
    cudaMalloc(&d_offsets, n * sizeof(uint64_t));
    gather_offsets_kernel<<<(n+255)/256, 256>>>(d_records, vb.Current(), d_offsets, n);
    cudaDeviceSynchronize();

    uint64_t* h_offsets = (uint64_t*)malloc(n * sizeof(uint64_t));
    cudaMemcpy(h_offsets, d_offsets, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    FILE* fp = fopen(output_path, "wb");
    if (fp) { fwrite(h_offsets, sizeof(uint64_t), n, fp); fclose(fp); }
    free(h_offsets);

    cudaFree(d_records);
    cudaFree(d_perm_a); cudaFree(d_perm_b);
    cudaFree(d_keys_a); cudaFree(d_keys_b);
    cudaFree(d_offsets); cudaFree(d_cub_temp);
}

__global__ void gds_classify_scatter_v2(
    const uint8_t* __restrict__ d_chunk,
    uint64_t chunk_records, uint64_t chunk_global_offset,
    const uint8_t* __restrict__ d_splitters,
    int K, int n_cmap, const int* __restrict__ d_cmap,
    uint8_t* d_bucket_buf, uint64_t bucket_capacity_records,
    uint64_t* __restrict__ d_bucket_counts) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= chunk_records) return;
    const uint8_t* rec = d_chunk + i * RECORD_SIZE;
    int b = K - 1;
    for (int s = 0; s < K - 1; s++) {
        const uint8_t* sp = d_splitters + (uint64_t)s * KEY_SIZE;
        int cmp = 0;
        for (int j = 0; j < KEY_SIZE; j++) {
            int diff = (int)rec[j] - (int)sp[j];
            if (diff != 0) { cmp = diff; break; }
        }
        if (cmp <= 0) { b = s; break; }
    }
    uint64_t pos = atomicAdd((unsigned long long*)&d_bucket_counts[b], 1ULL);
    if (pos >= bucket_capacity_records) return;
    uint8_t* dst = d_bucket_buf + ((uint64_t)b * bucket_capacity_records + pos) * COMPACT_REC_SIZE;
    for (int j = 0; j < n_cmap; j++) dst[j] = rec[d_cmap[j]];
    for (int j = n_cmap; j < COMPACT_KEY_SIZE; j++) dst[j] = 0;
    *(uint64_t*)(dst + COMPACT_KEY_SIZE) = chunk_global_offset + i;
}

struct Slot {
    uint8_t* d_buf;
    uint64_t bytes_filled;
    uint64_t global_offset_records;
    int chunk_id;
    enum State { EMPTY, FILLING, FULL, PROCESSING } state = EMPTY;
};

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "Usage: %s INPUT.bin OUT_PREFIX K\n", argv[0]); return 1; }
    const char* input_path = argv[1];
    int K = atoi(argv[3]);

    auto t0 = std::chrono::high_resolution_clock::now();

    struct stat st; stat(input_path, &st);
    uint64_t total_bytes = st.st_size;
    uint64_t total_records = total_bytes / RECORD_SIZE;
    printf("Input: %lu records (%.2f GB)\n", total_records, total_bytes/1e9);

    // cmap
    int cmap_h[KEY_SIZE]; int n_cmap = 0;
    {
        int fd = open(input_path, O_RDONLY);
        size_t samp_bytes = std::min((size_t)(1ULL << 30), (size_t)total_bytes);
        const uint8_t* samp = (const uint8_t*)mmap(nullptr, samp_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        bool varies[KEY_SIZE] = {};
        const uint8_t* r0 = samp;
        for (int i = 1; i < 100000; i++) {
            uint64_t idx = (uint64_t)i * (samp_bytes / RECORD_SIZE) / 100000;
            const uint8_t* ri = samp + idx * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++) if (!varies[b] && r0[b] != ri[b]) varies[b] = true;
        }
        for (int b = 0; b < KEY_SIZE; b++) if (varies[b]) cmap_h[n_cmap++] = b;
        munmap((void*)samp, samp_bytes);
    }
    printf("cmap: %d varying bytes\n", n_cmap);

    // splitters
    std::vector<std::array<uint8_t, KEY_SIZE>> samples(8192);
    {
        int fd = open(input_path, O_RDONLY);
        const uint8_t* samp = (const uint8_t*)mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        for (int i = 0; i < 8192; i++) {
            uint64_t idx = (uint64_t)i * total_records / 8192;
            memcpy(samples[i].data(), samp + idx * RECORD_SIZE, KEY_SIZE);
        }
        munmap((void*)samp, total_bytes);
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) { return memcmp(a.data(), b.data(), KEY_SIZE) < 0; });
    std::vector<uint8_t> splitters_h((K - 1) * KEY_SIZE);
    for (int i = 1; i < K; i++) {
        int pos = i * 8192 / K;
        if (pos >= 8192) pos = 8191;
        memcpy(splitters_h.data() + (i-1) * KEY_SIZE, samples[pos].data(), KEY_SIZE);
    }

    // GPU init
    CUDA_CHECK(cudaSetDevice(0));
    cuFileDriverOpen();

    // Allocate ring buffer slots
    uint64_t chunk_bytes = 4ULL << 30;
    uint64_t chunk_records = chunk_bytes / RECORD_SIZE;
    chunk_bytes = chunk_records * RECORD_SIZE;
    Slot slots[N_SLOTS];
    for (int s = 0; s < N_SLOTS; s++) {
        CUDA_CHECK(cudaMalloc(&slots[s].d_buf, chunk_bytes));
        cuFileBufRegister(slots[s].d_buf, chunk_bytes, 0);
    }
    printf("Ring buffer: %d slots × %.2f GB = %.2f GB GPU memory\n",
           N_SLOTS, chunk_bytes/1e9, N_SLOTS * chunk_bytes/1e9);

    // 2 GPU streams (one for classify, one for D2H)
    cudaStream_t s_classify, s_d2h;
    CUDA_CHECK(cudaStreamCreate(&s_classify));
    CUDA_CHECK(cudaStreamCreate(&s_d2h));

    // Per-bucket GPU staging (one set per slot for parallelism — actually
    // simpler to have one set total since classify+D2H runs sequentially per slot).
    uint64_t per_bucket_cap_records = (total_records / K * 13 / 10);
    uint64_t stage_records = chunk_records;
    uint64_t stage_buf_bytes = (uint64_t)K * stage_records * COMPACT_REC_SIZE;
    // Use 2 staging sets so classify can write to one while D2H reads from other
    uint8_t* d_bucket_buf[2];
    uint64_t* d_bucket_counts[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc(&d_bucket_buf[i], stage_buf_bytes));
        CUDA_CHECK(cudaMalloc(&d_bucket_counts[i], K * sizeof(uint64_t)));
    }
    printf("GPU staging: 2 × %.2f GB = %.2f GB\n", stage_buf_bytes/1e9, 2*stage_buf_bytes/1e9);

    uint8_t* d_splitters;
    CUDA_CHECK(cudaMalloc(&d_splitters, splitters_h.size()));
    CUDA_CHECK(cudaMemcpy(d_splitters, splitters_h.data(), splitters_h.size(), cudaMemcpyHostToDevice));
    int* d_cmap;
    CUDA_CHECK(cudaMalloc(&d_cmap, n_cmap * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cmap, cmap_h, n_cmap * sizeof(int), cudaMemcpyHostToDevice));

    // Host bucket buffers
    uint64_t per_bucket_max_bytes = per_bucket_cap_records * COMPACT_REC_SIZE;
    uint64_t hugepage_sz = 2 * 1024 * 1024;
    per_bucket_max_bytes = ((per_bucket_max_bytes + hugepage_sz - 1) / hugepage_sz) * hugepage_sz;
    std::vector<uint8_t*> bucket_bufs(K, nullptr);
    std::vector<uint64_t> host_bucket_offset(K, 0);
    for (int b = 0; b < K; b++) {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT);
        bucket_bufs[b] = (uint8_t*)mmap(nullptr, per_bucket_max_bytes, PROT_READ|PROT_WRITE, flags, -1, 0);
        if (bucket_bufs[b] == MAP_FAILED) {
            bucket_bufs[b] = (uint8_t*)mmap(nullptr, per_bucket_max_bytes, PROT_READ|PROT_WRITE,
                                            MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        }
    }

    // Compute total chunks
    uint64_t total_chunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;
    printf("Total chunks: %lu (%.2f GB each)\n", total_chunks, chunk_bytes/1e9);

    // Per-reader file handles
    std::vector<int> reader_fds(N_READERS);
    std::vector<CUfileHandle_t> reader_fhs(N_READERS);
    for (int r = 0; r < N_READERS; r++) {
        reader_fds[r] = open(input_path, O_RDONLY | O_DIRECT);
        if (reader_fds[r] < 0) reader_fds[r] = open(input_path, O_RDONLY);
        CUfileDescr_t descr = {};
        descr.handle.fd = reader_fds[r]; descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cuFileHandleRegister(&reader_fhs[r], &descr);
    }

    // Pipeline state
    std::mutex slot_mu[N_SLOTS];
    std::condition_variable slot_cv[N_SLOTS];

    auto t_part0 = std::chrono::high_resolution_clock::now();

    // Atomic next-chunk-to-read counter
    std::atomic<uint64_t> next_chunk_to_read{0};
    std::atomic<bool> all_done{false};

    // Reader threads
    std::vector<std::thread> readers;
    for (int r = 0; r < N_READERS; r++) {
        readers.emplace_back([&, r]() {
            while (true) {
                uint64_t chunk_id = next_chunk_to_read.fetch_add(1);
                if (chunk_id >= total_chunks) break;
                uint64_t offset = chunk_id * chunk_bytes;
                uint64_t want = std::min(chunk_bytes, total_bytes - offset);
                want = (want / RECORD_SIZE) * RECORD_SIZE;
                if (want == 0) break;

                // Find an EMPTY slot
                int slot_idx = -1;
                while (slot_idx < 0) {
                    for (int s = 0; s < N_SLOTS; s++) {
                        std::unique_lock<std::mutex> lk(slot_mu[s]);
                        if (slots[s].state == Slot::EMPTY) {
                            slots[s].state = Slot::FILLING;
                            slot_idx = s;
                            break;
                        }
                    }
                    if (slot_idx < 0) std::this_thread::yield();
                }

                ssize_t n = cuFileRead(reader_fhs[r], slots[slot_idx].d_buf, want, offset, 0);
                if (n != (ssize_t)want) {
                    fprintf(stderr, "reader %d: short read %zd vs %lu\n", r, n, want);
                    break;
                }

                {
                    std::unique_lock<std::mutex> lk(slot_mu[slot_idx]);
                    slots[slot_idx].bytes_filled = want;
                    slots[slot_idx].global_offset_records = offset / RECORD_SIZE;
                    slots[slot_idx].chunk_id = (int)chunk_id;
                    slots[slot_idx].state = Slot::FULL;
                }
                slot_cv[slot_idx].notify_all();
            }
        });
    }

    // Main thread: process slots in order of chunk_id
    // Two separate events, one per stage_idx — so iter N waits for iter N-2's D2H
    // (not just N-1's). Bug fix: previously single event caused stage[0] reuse
    // before its D2H completed when iter 2 came around.
    uint64_t next_chunk_to_process = 0;
    int stage_idx = 0;
    cudaEvent_t stage_d2h_done[2] = {nullptr, nullptr};
    while (next_chunk_to_process < total_chunks) {
        // Find a FULL slot with chunk_id == next_chunk_to_process
        int slot_idx = -1;
        while (slot_idx < 0) {
            for (int s = 0; s < N_SLOTS; s++) {
                std::unique_lock<std::mutex> lk(slot_mu[s]);
                if (slots[s].state == Slot::FULL &&
                    (uint64_t)slots[s].chunk_id == next_chunk_to_process) {
                    slots[s].state = Slot::PROCESSING;
                    slot_idx = s;
                    break;
                }
            }
            if (slot_idx < 0) std::this_thread::yield();
        }

        uint64_t n_recs = slots[slot_idx].bytes_filled / RECORD_SIZE;
        uint64_t global_off = slots[slot_idx].global_offset_records;

        // Wait for prior D2H using THIS stage_idx's buffer to finish.
        // (Iter N uses stage[stage_idx], which was last used by iter N-2.)
        if (stage_d2h_done[stage_idx] != nullptr) {
            cudaEventSynchronize(stage_d2h_done[stage_idx]);
            cudaEventDestroy(stage_d2h_done[stage_idx]);
            stage_d2h_done[stage_idx] = nullptr;
        }
        // Reset GPU bucket counters for this stage
        CUDA_CHECK(cudaMemsetAsync(d_bucket_counts[stage_idx], 0, K * sizeof(uint64_t), s_classify));

        // Classify on s_classify
        int threads = 256;
        int blocks = (int)((n_recs + threads - 1) / threads);
        gds_classify_scatter_v2<<<blocks, threads, 0, s_classify>>>(
            slots[slot_idx].d_buf, n_recs, global_off,
            d_splitters, K, n_cmap, d_cmap,
            d_bucket_buf[stage_idx], stage_records, d_bucket_counts[stage_idx]);

        // Wait for classify to complete (so we know counts for D2H)
        cudaStreamSynchronize(s_classify);

        // Read counts and issue D2H async on s_d2h
        uint64_t bucket_counts_h[16];
        CUDA_CHECK(cudaMemcpy(bucket_counts_h, d_bucket_counts[stage_idx], K * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (int b = 0; b < K; b++) {
            uint64_t bytes = bucket_counts_h[b] * COMPACT_REC_SIZE;
            if (bytes == 0) continue;
            CUDA_CHECK(cudaMemcpyAsync(bucket_bufs[b] + host_bucket_offset[b],
                                       d_bucket_buf[stage_idx] + (uint64_t)b * stage_records * COMPACT_REC_SIZE,
                                       bytes, cudaMemcpyDeviceToHost, s_d2h));
            host_bucket_offset[b] += bytes;
        }
        // Record event for the next iter that uses this stage_idx to wait on.
        cudaEventCreate(&stage_d2h_done[stage_idx]);
        cudaEventRecord(stage_d2h_done[stage_idx], s_d2h);

        // Mark slot EMPTY (slot d_buf is no longer needed for this chunk)
        {
            std::unique_lock<std::mutex> lk(slot_mu[slot_idx]);
            slots[slot_idx].state = Slot::EMPTY;
        }
        slot_cv[slot_idx].notify_all();

        next_chunk_to_process++;
        stage_idx = 1 - stage_idx;  // alternate staging buffer
    }

    // Final wait — both stage events must complete
    for (int i = 0; i < 2; i++) {
        if (stage_d2h_done[i]) {
            cudaEventSynchronize(stage_d2h_done[i]);
            cudaEventDestroy(stage_d2h_done[i]);
        }
    }
    all_done.store(true);
    for (auto& t : readers) t.join();

    auto t_part1 = std::chrono::high_resolution_clock::now();
    double part_ms = std::chrono::duration<double, std::milli>(t_part1 - t_part0).count();

    uint64_t total_part_bytes = 0;
    for (int b = 0; b < K; b++) {
        total_part_bytes += host_bucket_offset[b];
        printf("  bucket[%d]: %lu records (%.2f GB)\n",
               b, host_bucket_offset[b] / COMPACT_REC_SIZE, host_bucket_offset[b]/1e9);
    }
    printf("Multi-stream GDS partition: %.0f ms (%.2f GB/s effective)\n",
           part_ms, total_bytes / (part_ms * 1e6));

    // Cleanup
    for (int s = 0; s < N_SLOTS; s++) {
        cuFileBufDeregister(slots[s].d_buf);
        cudaFree(slots[s].d_buf);
    }
    for (int i = 0; i < 2; i++) {
        cudaFree(d_bucket_buf[i]);
        cudaFree(d_bucket_counts[i]);
    }
    cudaFree(d_splitters);
    cudaFree(d_cmap);
    cudaStreamDestroy(s_classify);
    cudaStreamDestroy(s_d2h);
    for (int r = 0; r < N_READERS; r++) {
        cuFileHandleDeregister(reader_fhs[r]);
        close(reader_fds[r]);
    }
    cuFileDriverClose();

    // SKIP_SORT=1 to skip the in-process sort phase (writes bucket files instead).
    const char* out_prefix = argv[2];
    bool skip_sort = getenv("SKIP_SORT") && atoi(getenv("SKIP_SORT")) > 0;

    if (skip_sort) {
        // Write bucket files for downstream sort tool
        auto t_write0 = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < K; b++) {
            char path[512];
            snprintf(path, sizeof(path), "%s.bucket_%d.bin", out_prefix, b);
            FILE* fp = fopen(path, "wb");
            if (fp) {
                fwrite(bucket_bufs[b], 1, host_bucket_offset[b], fp);
                fclose(fp);
            }
        }
        auto t_write1 = std::chrono::high_resolution_clock::now();
        printf("Bucket files written in %.0f ms\n",
               std::chrono::duration<double, std::milli>(t_write1 - t_write0).count());
    } else {
        // Run sort phase in-process: 4-GPU concurrent on bucket buffers in host RAM.
        // No bucket file write — saves ~3 min on SF1500.
        auto t_sort0 = std::chrono::high_resolution_clock::now();
        int rounds = (K + 3) / 4;
        for (int round = 0; round < rounds; round++) {
            std::vector<std::thread> sort_threads;
            for (int slot = 0; slot < 4 && (round * 4 + slot) < K; slot++) {
                int b = round * 4 + slot;
                int gpu = slot;
                uint64_t n_recs = host_bucket_offset[b] / COMPACT_REC_SIZE;
                if (n_recs == 0) continue;
                char path[512];
                snprintf(path, sizeof(path), "%s.sorted_%d.bin", out_prefix, b);
                std::string spath = path;
                uint8_t* hbuf = bucket_bufs[b];
                sort_threads.emplace_back([gpu, hbuf, n_recs, spath]() {
                    sort_one_bucket(gpu, hbuf, n_recs, spath.c_str());
                });
            }
            for (auto& th : sort_threads) th.join();
        }
        auto t_sort1 = std::chrono::high_resolution_clock::now();
        printf("Sort phase: %.0f ms (%d rounds)\n",
               std::chrono::duration<double, std::milli>(t_sort1 - t_sort0).count(), rounds);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    printf("Total wall: %.0f ms (%.2f GB/s end-to-end)\n", total_ms, total_bytes/(total_ms*1e6));
    printf("CSV,gds_partition_multistream,records=%lu,K=%d,N_SLOTS=%d,N_READERS=%d,part_ms=%.0f,total_ms=%.0f,gb_per_s=%.2f\n",
           total_records, K, N_SLOTS, N_READERS, part_ms, total_ms, total_bytes/(total_ms*1e6));

    for (int b = 0; b < K; b++) munmap(bucket_bufs[b], per_bucket_max_bytes);
    return 0;
}
