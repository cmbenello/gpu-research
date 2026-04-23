// ============================================================================
// GPU Sample Sort — Partitioned Radix Sort
//
// Instead of:  chunk1-sort → chunk2-sort → ... → merge all
// Do:          sample keys → pick splitters → partition → sort each partition
//
// Advantages:
//   - No merge phase (partitions are already globally ordered)
//   - Better GPU utilization (each partition sorts independently)
//   - Natural multi-GPU extension (one partition per GPU)
//
// Usage: SAMPLE_SORT=1 ./external_sort_tpch_compact ...
// ============================================================================

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thread>
#include <numeric>
#include <cub/cub.cuh>

// ── GPU kernel: extract uint64 sort key from keyed records via permutation ──
__global__ void ss_extract_uint64_kernel(
    const uint8_t* __restrict__ keys,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ out,
    uint64_t n,
    uint32_t key_stride,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t idx = perm[i];
    const uint8_t* k = keys + (uint64_t)idx * key_stride + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    out[i] = v;
}

// ── GPU kernel: init identity permutation ──
__global__ void ss_init_identity_kernel(uint32_t* perm, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) perm[i] = (uint32_t)i;
}

// ── Sample Sort Engine ──────────────────────────────────────────────

struct SampleSortResult {
    uint32_t* h_perm;            // global sorted permutation (caller frees)
    double classify_ms;
    double partition_ms;
    double sort_ms;
    double total_ms;
    int num_partitions;
};

// CPU-side sample sort orchestrator.
// h_data: host records (N × record_stride). Keys are the first key_size bytes.
// Produces a sorted permutation using only key_size bytes for comparison.
// GPU upload extracts only key bytes (padded to 8B alignment) — not full records.
static SampleSortResult sample_sort_keys(
    const uint8_t* h_data,       // host record buffer (N × record_stride)
    uint32_t key_size,           // meaningful key bytes per record
    uint32_t record_stride,      // stride between records (>= key_size)
    uint64_t num_records,
    int target_partitions = 0    // 0 = auto
) {
    SampleSortResult result = {};

    auto now = []() { return std::chrono::high_resolution_clock::now(); };
    auto elapsed = [](auto t0) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };
    auto total_t0 = now();

    // GPU key stride: pad key_size to 8-byte alignment for efficient CUB access
    uint32_t gpu_key_stride = ((key_size + 7) / 8) * 8;

    // ── Step 1: Determine number of partitions ──
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    // Per-record GPU memory: keys + 2×sort_keys + 2×perm
    size_t per_record_gpu = gpu_key_stride + 2*sizeof(uint64_t) + 2*sizeof(uint32_t);
    uint64_t max_per_partition = (free_mem - 500*1024*1024) / per_record_gpu;

    if (target_partitions <= 0) {
        target_partitions = std::max(2, (int)((num_records + max_per_partition - 1) / max_per_partition));
        target_partitions = std::min(target_partitions, 256);
    }
    int P = target_partitions;
    result.num_partitions = P;
    printf("  [SampleSort] %d partitions for %lu records (%.1fM avg, GPU fits %.1fM)\n",
           P, num_records, num_records / (double)P / 1e6, max_per_partition / 1e6);
    printf("  [SampleSort] Key: %dB → %dB GPU stride, %d LSD passes\n",
           key_size, gpu_key_stride, (key_size + 7) / 8);

    // ── Step 2: Sample keys and compute splitters ──
    int sample_size = std::min((uint64_t)(P * 10000), num_records);
    uint64_t step = num_records / sample_size;
    std::vector<std::vector<uint8_t>> samples(sample_size);
    for (int s = 0; s < sample_size; s++) {
        uint64_t idx = s * step;
        samples[s].assign(h_data + idx * record_stride, h_data + idx * record_stride + key_size);
    }
    std::sort(samples.begin(), samples.end());

    std::vector<std::vector<uint8_t>> splitter_keys(P - 1);
    for (int i = 0; i < P - 1; i++) {
        int idx = (int)((i + 1) * (double)sample_size / P);
        idx = std::min(idx, sample_size - 1);
        splitter_keys[i] = samples[idx];
    }

    // ── Step 3: Classify (CPU, multi-threaded) ──
    auto classify_t0 = now();
    std::vector<uint32_t> part_ids(num_records);
    std::vector<uint64_t> part_counts(P, 0);

    int hw = std::max(1, (int)std::thread::hardware_concurrency());
    std::vector<std::vector<uint64_t>> thread_counts(hw, std::vector<uint64_t>(P, 0));

    uint64_t per_thread = (num_records + hw - 1) / hw;
    std::vector<std::thread> threads;
    for (int t = 0; t < hw; t++) {
        threads.emplace_back([&, t]() {
            uint64_t lo = t * per_thread;
            uint64_t hi = std::min(lo + per_thread, num_records);
            for (uint64_t i = lo; i < hi; i++) {
                const uint8_t* k = h_data + i * record_stride;
                int slo = 0, shi = P - 1;
                while (slo < shi) {
                    int mid = (slo + shi) / 2;
                    int cmp = memcmp(splitter_keys[mid].data(), k, key_size);
                    if (cmp <= 0) slo = mid + 1;
                    else shi = mid;
                }
                part_ids[i] = (uint32_t)slo;
                thread_counts[t][slo]++;
            }
        });
    }
    for (auto& t : threads) t.join();

    for (int t = 0; t < hw; t++)
        for (int p = 0; p < P; p++)
            part_counts[p] += thread_counts[t][p];

    result.classify_ms = elapsed(classify_t0);
    printf("  [SampleSort] Classify: %.0f ms\n", result.classify_ms);

    uint64_t min_p = *std::min_element(part_counts.begin(), part_counts.end());
    uint64_t max_p = *std::max_element(part_counts.begin(), part_counts.end());
    printf("  [SampleSort] Partition sizes: min=%.1fM, max=%.1fM, skew=%.2f\n",
           min_p/1e6, max_p/1e6, (double)max_p / (num_records / P));

    // ── Step 4: Build partition-ordered permutation ──
    auto partition_t0 = now();

    std::vector<uint64_t> part_offsets(P + 1, 0);
    for (int p = 0; p < P; p++) part_offsets[p + 1] = part_offsets[p] + part_counts[p];

    std::vector<uint32_t> perm(num_records);
    std::vector<uint64_t> write_pos(P);
    for (int p = 0; p < P; p++) write_pos[p] = part_offsets[p];
    for (uint64_t i = 0; i < num_records; i++) {
        uint32_t p = part_ids[i];
        perm[write_pos[p]++] = (uint32_t)i;
    }
    part_ids.clear();
    part_ids.shrink_to_fit();

    result.partition_ms = elapsed(partition_t0);
    printf("  [SampleSort] Partition: %.0f ms\n", result.partition_ms);

    // ── Step 5: Sort each partition independently on GPU ──
    auto sort_t0 = now();

    uint64_t largest = *std::max_element(part_counts.begin(), part_counts.end());

    // Allocate GPU workspace (sized for largest partition)
    uint8_t* d_keys;
    uint64_t* d_sort_keys, *d_sort_keys_alt;
    uint32_t* d_perm_in, *d_perm_out;
    cudaMalloc(&d_keys, largest * gpu_key_stride);
    cudaMalloc(&d_sort_keys, largest * sizeof(uint64_t));
    cudaMalloc(&d_sort_keys_alt, largest * sizeof(uint64_t));
    cudaMalloc(&d_perm_in, largest * sizeof(uint32_t));
    cudaMalloc(&d_perm_out, largest * sizeof(uint32_t));

    size_t cub_temp_bytes = 0;
    {
        cub::DoubleBuffer<uint64_t> kb(nullptr, nullptr);
        cub::DoubleBuffer<uint32_t> vb(nullptr, nullptr);
        cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, kb, vb,
                                         (int)largest, 0, key_size * 8);
    }
    void* d_temp;
    cudaMalloc(&d_temp, cub_temp_bytes);

    // Host staging buffer: extract just key bytes (at gpu_key_stride), not full records
    uint8_t* h_staging = (uint8_t*)malloc(largest * gpu_key_stride);
    if (!h_staging) {
        fprintf(stderr, "malloc failed for staging buffer (%.1f GB)\n",
                largest * gpu_key_stride / 1e9);
        result.total_ms = elapsed(total_t0);
        return result;
    }

    int num_lsd_chunks = (key_size + 7) / 8;
    double h2d_bytes = 0, d2h_bytes = 0;

    for (int p = 0; p < P; p++) {
        uint64_t off = part_offsets[p];
        uint64_t count = part_counts[p];
        if (count <= 1) continue;

        // Gather this partition's KEY bytes into staging (multi-threaded)
        {
            int gt = std::max(1, (int)std::thread::hardware_concurrency());
            uint64_t per_t = (count + gt - 1) / gt;
            std::vector<std::thread> gthreads;
            for (int t = 0; t < gt; t++) {
                gthreads.emplace_back([&, t]() {
                    uint64_t lo = t * per_t, hi = std::min(lo + per_t, count);
                    for (uint64_t i = lo; i < hi; i++) {
                        uint8_t* dst = h_staging + i * gpu_key_stride;
                        const uint8_t* src = h_data + (uint64_t)perm[off + i] * record_stride;
                        memcpy(dst, src, key_size);
                        // Zero padding
                        if (gpu_key_stride > key_size)
                            memset(dst + key_size, 0, gpu_key_stride - key_size);
                    }
                });
            }
            for (auto& t : gthreads) t.join();
        }

        // Upload to GPU (only key bytes, not full records)
        cudaMemcpy(d_keys, h_staging, count * gpu_key_stride, cudaMemcpyHostToDevice);
        h2d_bytes += count * gpu_key_stride;

        // Init identity perm on GPU
        int nt = 256;
        int nb = (count + nt - 1) / nt;
        ss_init_identity_kernel<<<nb, nt>>>(d_perm_in, count);

        uint32_t* pi = d_perm_in;
        uint32_t* po = d_perm_out;

        for (int chunk = num_lsd_chunks - 1; chunk >= 0; chunk--) {
            int byte_offset = chunk * 8;
            int chunk_bytes = std::min(8, (int)key_size - byte_offset);

            ss_extract_uint64_kernel<<<nb, nt>>>(d_keys, pi, d_sort_keys, count,
                                                   gpu_key_stride, byte_offset, chunk_bytes);

            cub::DoubleBuffer<uint64_t> k_buf(d_sort_keys, d_sort_keys_alt);
            cub::DoubleBuffer<uint32_t> p_buf(pi, po);
            size_t temp = cub_temp_bytes;
            cub::DeviceRadixSort::SortPairs(d_temp, temp, k_buf, p_buf,
                                             (int)count, 0, chunk_bytes * 8);
            pi = p_buf.Current();
            po = p_buf.Alternate();
        }

        // Download sorted local perm and remap to global indices
        std::vector<uint32_t> local_perm(count);
        cudaMemcpy(local_perm.data(), pi, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        d2h_bytes += count * sizeof(uint32_t);

        std::vector<uint32_t> global_sorted(count);
        for (uint64_t i = 0; i < count; i++) {
            global_sorted[i] = perm[off + local_perm[i]];
        }
        memcpy(perm.data() + off, global_sorted.data(), count * sizeof(uint32_t));

        printf("\r  [SampleSort] Sorted partition %d/%d (%.1fM records)    ", p+1, P, count/1e6);
        fflush(stdout);
    }
    cudaDeviceSynchronize();
    printf("\n");

    free(h_staging);

    result.sort_ms = elapsed(sort_t0);
    printf("  [SampleSort] GPU sort: %.0f ms (H2D %.1f GB, D2H %.1f GB)\n",
           result.sort_ms, h2d_bytes/1e9, d2h_bytes/1e9);

    // Return sorted permutation
    result.h_perm = (uint32_t*)malloc(num_records * sizeof(uint32_t));
    memcpy(result.h_perm, perm.data(), num_records * sizeof(uint32_t));

    cudaFree(d_keys);
    cudaFree(d_sort_keys);
    cudaFree(d_sort_keys_alt);
    cudaFree(d_perm_in);
    cudaFree(d_perm_out);
    cudaFree(d_temp);

    result.total_ms = elapsed(total_t0);
    printf("  [SampleSort] Total: %.0f ms (classify %.0f + partition %.0f + sort %.0f)\n",
           result.total_ms, result.classify_ms, result.partition_ms, result.sort_ms);

    return result;
}
