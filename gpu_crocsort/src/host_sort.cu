#include "record.cuh"
#include "ovc.cuh"
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// ============================================================================
// Host orchestration: supports both 2-way and K-way merge strategies
//
// The fundamental tradeoff:
//   Passes × BW_per_pass = Total HBM Traffic = THE thing that determines speed
//
//   2-way merge path: log2(N) passes, each at ~80% bandwidth → lots of traffic
//   8-way merge tree: log8(N) passes, each at ~80% bandwidth → 3x less traffic
//
// We implement both and let the user benchmark.
// ============================================================================

// Merge descriptors (must match merge.cu)
struct PairDesc2Way {
    uint64_t a_byte_offset;
    int      a_count;
    uint64_t b_byte_offset;
    int      b_count;
    uint64_t out_byte_offset;
    int      first_block;
};

static constexpr int KWAY_K = 8;

struct KWayPartition {
    int      src_rec_start[KWAY_K];
    int      src_rec_count[KWAY_K];
    uint64_t src_byte_off[KWAY_K];
    uint64_t out_byte_offset;
    int      total_records;
};

// External kernel launchers
extern "C" void launch_run_generation(
    const uint8_t* d_input, uint8_t* d_output, uint32_t* d_ovc,
    uint64_t total_records, SparseEntry* d_sparse, int* d_sparse_counts,
    int num_blocks, cudaStream_t stream);

extern "C" void launch_merge_2way(
    const uint8_t* d_input, uint8_t* d_output,
    const PairDesc2Way* d_pairs, int num_pairs, int total_blocks,
    cudaStream_t stream);

extern "C" void launch_merge_kway(
    const uint8_t* d_input, uint8_t* d_output,
    const KWayPartition* d_partitions, int num_partitions,
    int max_records_per_partition, cudaStream_t stream);

// ── Run descriptor ─────────────────────────────────────────────────

struct Run {
    uint64_t byte_offset;
    uint64_t num_records;
};

// ── Strategy selector ──────────────────────────────────────────────

enum MergeStrategy {
    STRATEGY_2WAY,      // 2-way merge path: max parallelism, many passes
    STRATEGY_KWAY,      // K-way merge tree: fewer passes, still parallel
};

// ── Shared memory limit for K-way merge tree ───────────────────────

static int max_records_per_kway_partition() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    // Use the per-block optin limit (cudaFuncSetAttribute in merge.cu handles opt-in)
    int max_smem = props.sharedMemPerBlockOptin;
    // Leave 1KB for static shared memory (seq_start, seq arrays)
    int usable_smem = max_smem - 1024;
    // Need 2 buffers (ping-pong) of total_records × RECORD_SIZE
    return usable_smem / (2 * RECORD_SIZE);
}

// ============================================================================
// Sample-based partitioning for K-way merge
// Ported from experiments/sample_partition.cu
// ============================================================================

// Phase 1: Sample every S-th key from each run (one block per run)
__global__ void sample_keys_kernel(
    const uint8_t* __restrict__ d_runs, const uint64_t* __restrict__ d_run_offsets,
    const int* __restrict__ d_run_lengths, int K, int S,
    SortKey* __restrict__ d_samples, int* __restrict__ d_sample_counts
) {
    int run_id = blockIdx.x;
    if (run_id >= K) return;
    int run_len = d_run_lengths[run_id];
    uint64_t base = d_run_offsets[run_id];
    int num_samples = run_len / S;
    // Compute prefix offset for this run's samples
    int sample_base = 0;
    for (int r = 0; r < run_id; r++) sample_base += d_run_lengths[r] / S;
    if (threadIdx.x == 0) d_sample_counts[run_id] = num_samples;
    for (int i = threadIdx.x; i < num_samples; i += blockDim.x) {
        const uint8_t* rec = d_runs + base + (uint64_t)(i * S) * RECORD_SIZE;
        d_samples[sample_base + i] = make_sort_key(rec);
    }
}

// Binary search: find first record in run >= target key
__device__ int lower_bound_run(const uint8_t* run_data, int run_len, SortKey target) {
    int lo = 0, hi = run_len;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (make_sort_key(run_data + (uint64_t)mid * RECORD_SIZE) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Phase 4: Binary search each run for each boundary (P blocks, K threads)
__global__ void compute_partition_ranges_kernel(
    const uint8_t* __restrict__ d_runs, const uint64_t* __restrict__ d_run_offsets,
    const int* __restrict__ d_run_lengths, int K,
    const SortKey* __restrict__ d_boundaries, int P,
    int* __restrict__ d_starts, int* __restrict__ d_counts
) {
    int p = blockIdx.x, k = threadIdx.x;
    if (p >= P || k >= K) return;
    int run_len = d_run_lengths[k];
    const uint8_t* run_data = d_runs + d_run_offsets[k];
    int lo = (p == 0)     ? 0       : lower_bound_run(run_data, run_len, d_boundaries[p - 1]);
    int hi = (p == P - 1) ? run_len : lower_bound_run(run_data, run_len, d_boundaries[p]);
    d_starts[p * K + k] = lo;
    d_counts[p * K + k] = hi - lo;
}

// Host orchestrator: compute sample-based partitions for a group of K runs
static void compute_sample_partitions(
    const uint8_t* d_runs,
    const std::vector<Run>& group_runs,
    int K, int P,
    uint64_t out_base_offset,
    std::vector<KWayPartition>& out_partitions
) {
    // Build host arrays for run offsets and lengths
    std::vector<uint64_t> h_run_offsets(K);
    std::vector<int> h_run_lengths(K);
    uint64_t total = 0;
    for (int i = 0; i < K; i++) {
        h_run_offsets[i] = group_runs[i].byte_offset;
        h_run_lengths[i] = (int)group_runs[i].num_records;
        total += group_runs[i].num_records;
    }

    // Compute sampling rate: ~10 samples per partition
    int target_samples = 10 * P;
    int S = std::max(1, (int)(total / target_samples));

    int total_samples = 0;
    for (int i = 0; i < K; i++) total_samples += h_run_lengths[i] / S;

    if (total_samples < 2) {
        // Too few records — single partition gets everything
        out_partitions.resize(1);
        KWayPartition& kp = out_partitions[0];
        kp.out_byte_offset = out_base_offset;
        kp.total_records = (int)total;
        for (int k = 0; k < K; k++) {
            kp.src_rec_start[k] = 0;
            kp.src_rec_count[k] = h_run_lengths[k];
            kp.src_byte_off[k] = h_run_offsets[k];
        }
        for (int k = K; k < KWAY_K; k++) {
            kp.src_rec_start[k] = 0;
            kp.src_rec_count[k] = 0;
            kp.src_byte_off[k] = 0;
        }
        return;
    }

    // Upload run metadata
    uint64_t* d_run_offsets_dev;
    int* d_run_lengths_dev;
    CUDA_CHECK(cudaMalloc(&d_run_offsets_dev, K * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_run_lengths_dev, K * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_run_offsets_dev, h_run_offsets.data(), K * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_run_lengths_dev, h_run_lengths.data(), K * sizeof(int), cudaMemcpyHostToDevice));

    // Phase 1: Sample keys from each run
    SortKey* d_samples;
    int* d_sample_counts;
    CUDA_CHECK(cudaMalloc(&d_samples, std::max(1, total_samples) * sizeof(SortKey)));
    CUDA_CHECK(cudaMalloc(&d_sample_counts, K * sizeof(int)));

    sample_keys_kernel<<<K, 256>>>(
        d_runs, d_run_offsets_dev, d_run_lengths_dev, K, S, d_samples, d_sample_counts);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: Sort samples using Thrust
    thrust::device_ptr<SortKey> dp_samples(d_samples);
    thrust::sort(dp_samples, dp_samples + total_samples);

    // Phase 3: Select P-1 evenly spaced boundaries from sorted samples
    std::vector<SortKey> h_samples(total_samples);
    CUDA_CHECK(cudaMemcpy(h_samples.data(), d_samples,
                          total_samples * sizeof(SortKey), cudaMemcpyDeviceToHost));

    int num_boundaries = P - 1;
    std::vector<SortKey> h_boundaries(num_boundaries);
    for (int i = 0; i < num_boundaries; i++) {
        int idx = (int)(((uint64_t)(i + 1) * total_samples) / P);
        idx = std::min(idx, total_samples - 1);
        h_boundaries[i] = h_samples[idx];
    }

    SortKey* d_boundaries;
    CUDA_CHECK(cudaMalloc(&d_boundaries, num_boundaries * sizeof(SortKey)));
    CUDA_CHECK(cudaMemcpy(d_boundaries, h_boundaries.data(),
                          num_boundaries * sizeof(SortKey), cudaMemcpyHostToDevice));

    // Phase 4: Binary search each run for each boundary
    int* d_starts;
    int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_starts, P * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, P * K * sizeof(int)));

    compute_partition_ranges_kernel<<<P, K>>>(
        d_runs, d_run_offsets_dev, d_run_lengths_dev, K,
        d_boundaries, P, d_starts, d_counts);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back
    std::vector<int> h_starts(P * K), h_counts(P * K);
    CUDA_CHECK(cudaMemcpy(h_starts.data(), d_starts, P * K * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, P * K * sizeof(int), cudaMemcpyDeviceToHost));

    // Phase 5: Build KWayPartition descriptors
    out_partitions.resize(P);
    uint64_t out_offset = out_base_offset;

    for (int p = 0; p < P; p++) {
        KWayPartition& kp = out_partitions[p];
        kp.out_byte_offset = out_offset;
        kp.total_records = 0;
        for (int k = 0; k < K; k++) {
            kp.src_rec_start[k] = h_starts[p * K + k];
            kp.src_rec_count[k] = h_counts[p * K + k];
            kp.src_byte_off[k]  = h_run_offsets[k];
            kp.total_records += kp.src_rec_count[k];
        }
        for (int k = K; k < KWAY_K; k++) {
            kp.src_rec_start[k] = 0;
            kp.src_rec_count[k] = 0;
            kp.src_byte_off[k]  = 0;
        }
        out_offset += (uint64_t)kp.total_records * RECORD_SIZE;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_run_offsets_dev));
    CUDA_CHECK(cudaFree(d_run_lengths_dev));
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_sample_counts));
    CUDA_CHECK(cudaFree(d_boundaries));
    CUDA_CHECK(cudaFree(d_starts));
    CUDA_CHECK(cudaFree(d_counts));
}

// ── Main sort function ─────────────────────────────────────────────

void gpu_crocsort_in_hbm(
    uint8_t* d_data,
    uint64_t num_records,
    bool verify,
    MergeStrategy strategy
) {
    if (num_records <= 1) return;

    uint64_t total_bytes = num_records * RECORD_SIZE;
    int max_recs_per_part = max_records_per_kway_partition();

    printf("GPU CrocSort: %lu records (%.2f MB)\n",
           num_records, (double)total_bytes / (1024.0 * 1024.0));
    printf("  K-way merge tree: K=%d, max %d records/partition (%.1f KB smem)\n",
           KWAY_K, max_recs_per_part, max_recs_per_part * RECORD_SIZE * 2.0 / 1024.0);

    // ════════════════════════════════════════
    // Phase 1: Run Generation
    // ════════════════════════════════════════

    int num_runs = (int)((num_records + RECORDS_PER_BLOCK - 1) / RECORDS_PER_BLOCK);

    uint8_t* d_sorted;
    uint32_t* d_ovc;
    SparseEntry* d_sparse;
    int* d_sparse_counts;

    CUDA_CHECK(cudaMalloc(&d_sorted, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_ovc, num_records * sizeof(uint32_t)));

    int max_sparse = num_runs * ((RECORDS_PER_BLOCK + SPARSE_INDEX_STRIDE - 1) / SPARSE_INDEX_STRIDE);
    CUDA_CHECK(cudaMalloc(&d_sparse, std::max(1, max_sparse) * sizeof(SparseEntry)));
    CUDA_CHECK(cudaMalloc(&d_sparse_counts, std::max(1, num_runs) * (int)sizeof(int)));

    GpuTimer timer;
    timer.begin();
    launch_run_generation(d_data, d_sorted, d_ovc, num_records,
                          d_sparse, d_sparse_counts, num_runs, 0);
    float run_gen_ms = timer.end();

    int expected_passes_2way = (int)ceil(log2((double)num_runs));
    int expected_passes_kway = (int)ceil(log((double)num_runs) / log((double)KWAY_K));

    printf("  Phase 1: %d runs in %.2f ms (%.2f GB/s), %d blocks\n",
           num_runs, run_gen_ms, total_bytes / (run_gen_ms * 1e6), num_runs);
    printf("  Merge passes: 2-way=%d, %d-way=%d\n",
           expected_passes_2way, KWAY_K, expected_passes_kway);
    printf("  HBM traffic: 2-way=%.1f GB, %d-way=%.1f GB\n",
           2.0 * total_bytes * expected_passes_2way / 1e9,
           KWAY_K, 2.0 * total_bytes * expected_passes_kway / 1e9);

    CUDA_CHECK(cudaFree(d_ovc));
    CUDA_CHECK(cudaFree(d_sparse));
    CUDA_CHECK(cudaFree(d_sparse_counts));

    // Build run list
    std::vector<Run> runs(num_runs);
    for (int i = 0; i < num_runs; i++) {
        runs[i].byte_offset = (uint64_t)i * RECORDS_PER_BLOCK * RECORD_SIZE;
        runs[i].num_records = std::min((uint64_t)RECORDS_PER_BLOCK,
                                       num_records - (uint64_t)i * RECORDS_PER_BLOCK);
    }

    // ════════════════════════════════════════
    // Phase 2: Multi-pass merge
    // ════════════════════════════════════════

    uint8_t* d_merge_buf;
    CUDA_CHECK(cudaMalloc(&d_merge_buf, total_bytes));

    uint8_t* d_src = d_sorted;
    uint8_t* d_dst = d_merge_buf;

    int pass = 0;
    float total_merge_ms = 0;

    if (strategy == STRATEGY_KWAY) {
        // ── K-way merge tree strategy with sample-based partitioning ──
        printf("\n  Using %d-way shared-memory merge tree (sample partitioning)\n", KWAY_K);

        while (runs.size() > 1) {
            pass++;
            int current_runs = (int)runs.size();
            int group_size = std::min(KWAY_K, current_runs);
            int num_groups = (current_runs + group_size - 1) / group_size;

            std::vector<Run> new_runs;
            uint64_t out_offset = 0;
            float pass_ms = 0;

            for (int g = 0; g < num_groups; g++) {
                int g_start = g * group_size;
                int g_end = std::min(g_start + group_size, current_runs);
                int g_size = g_end - g_start;

                if (g_size == 1) {
                    // Single run — just copy
                    Run& r = runs[g_start];
                    CUDA_CHECK(cudaMemcpy(d_dst + out_offset, d_src + r.byte_offset,
                                          r.num_records * RECORD_SIZE, cudaMemcpyDeviceToDevice));
                    new_runs.push_back({out_offset, r.num_records});
                    out_offset += r.num_records * RECORD_SIZE;
                    continue;
                }

                // Build group runs
                std::vector<Run> group_runs(runs.begin() + g_start, runs.begin() + g_end);

                // Compute total records to determine partition count
                uint64_t group_total = 0;
                for (auto& r : group_runs) group_total += r.num_records;

                // Use 2x safety margin since sample partitioning doesn't guarantee sizes
                int min_partitions = (int)((group_total + max_recs_per_part - 1) / max_recs_per_part);
                int num_partitions = std::max(min_partitions * 2, 64);

                // Compute partitions, retry with more partitions if any exceed smem limit
                std::vector<KWayPartition> partitions;
                int max_rec;
                for (int attempt = 0; attempt < 4; attempt++) {
                    partitions.clear();
                    compute_sample_partitions(d_src, group_runs, g_size,
                                              num_partitions, out_offset, partitions);
                    max_rec = 0;
                    for (auto& p : partitions) max_rec = std::max(max_rec, p.total_records);
                    if (max_rec <= max_recs_per_part) break;
                    num_partitions *= 2; // Double and retry
                }

                // Verify partition integrity: each source must be fully covered
                for (int k = 0; k < g_size; k++) {
                    int total_from_src = 0;
                    int prev_end = 0;
                    for (auto& p : partitions) {
                        if (p.src_rec_start[k] != prev_end) {
                            printf("  WARNING: source %d gap at partition, start=%d prev_end=%d\n",
                                   k, p.src_rec_start[k], prev_end);
                        }
                        total_from_src += p.src_rec_count[k];
                        prev_end = p.src_rec_start[k] + p.src_rec_count[k];
                    }
                    if (total_from_src != (int)group_runs[k].num_records) {
                        printf("  WARNING: source %d: partitions cover %d records, expected %d\n",
                               k, total_from_src, (int)group_runs[k].num_records);
                    }
                }

                // Upload partition descriptors
                KWayPartition* d_parts;
                CUDA_CHECK(cudaMalloc(&d_parts, partitions.size() * sizeof(KWayPartition)));
                CUDA_CHECK(cudaMemcpy(d_parts, partitions.data(),
                                      partitions.size() * sizeof(KWayPartition),
                                      cudaMemcpyHostToDevice));

                timer.begin();
                launch_merge_kway(d_src, d_dst, d_parts, (int)partitions.size(), max_rec, 0);
                pass_ms += timer.end();

                CUDA_CHECK(cudaFree(d_parts));

                // Merged run
                uint64_t merged_records = 0;
                for (auto& r : group_runs) merged_records += r.num_records;
                new_runs.push_back({out_offset, merged_records});
                out_offset += merged_records * RECORD_SIZE;
            }

            total_merge_ms += pass_ms;
            printf("  Pass %d: %d runs -> %d (%d-way), %d partitions, %.2f ms (%.2f GB/s)\n",
                   pass, current_runs, (int)new_runs.size(), group_size,
                   (int)(current_runs / group_size) * 64,
                   pass_ms, 2.0 * total_bytes / (pass_ms * 1e6));

            runs = new_runs;
            std::swap(d_src, d_dst);
        }

    } else {
        // ── 2-way merge path strategy ──
        printf("\n  Using 2-way merge path\n");

        while (runs.size() > 1) {
            pass++;
            int current_runs = (int)runs.size();
            int num_pairs = current_runs / 2;
            bool leftover = (current_runs % 2 == 1);

            std::vector<PairDesc2Way> pairs(num_pairs);
            int total_blocks = 0;
            uint64_t out_offset = 0;
            int items_per_block = 8 * 256;

            for (int p = 0; p < num_pairs; p++) {
                Run& ra = runs[2*p];
                Run& rb = runs[2*p+1];
                int pair_total = (int)(ra.num_records + rb.num_records);
                int pair_blocks = (pair_total + items_per_block - 1) / items_per_block;

                pairs[p] = {ra.byte_offset, (int)ra.num_records,
                           rb.byte_offset, (int)rb.num_records,
                           out_offset, total_blocks};
                total_blocks += pair_blocks;
                out_offset += (uint64_t)pair_total * RECORD_SIZE;
            }

            uint64_t leftover_off = out_offset;

            PairDesc2Way* d_pairs;
            CUDA_CHECK(cudaMalloc(&d_pairs, num_pairs * sizeof(PairDesc2Way)));
            CUDA_CHECK(cudaMemcpy(d_pairs, pairs.data(),
                                  num_pairs * sizeof(PairDesc2Way), cudaMemcpyHostToDevice));

            timer.begin();
            launch_merge_2way(d_src, d_dst, d_pairs, num_pairs, total_blocks, 0);
            if (leftover) {
                Run& rl = runs[current_runs - 1];
                CUDA_CHECK(cudaMemcpyAsync(d_dst + leftover_off, d_src + rl.byte_offset,
                                           rl.num_records * RECORD_SIZE,
                                           cudaMemcpyDeviceToDevice, 0));
            }
            float pass_ms = timer.end();
            total_merge_ms += pass_ms;

            printf("  Pass %d: %d->%d runs, %d blocks, %.2f ms (%.2f GB/s)\n",
                   pass, current_runs, num_pairs + (leftover?1:0),
                   total_blocks, pass_ms, 2.0*total_bytes/(pass_ms*1e6));

            CUDA_CHECK(cudaFree(d_pairs));

            // Build new runs
            std::vector<Run> new_runs;
            uint64_t nr_off = 0;
            for (int p = 0; p < num_pairs; p++) {
                uint64_t c = runs[2*p].num_records + runs[2*p+1].num_records;
                new_runs.push_back({nr_off, c});
                nr_off += c * RECORD_SIZE;
            }
            if (leftover) new_runs.push_back({leftover_off, runs[current_runs-1].num_records});

            runs = new_runs;
            std::swap(d_src, d_dst);
        }
    }

    // ════════════════════════════════════════
    // Results
    // ════════════════════════════════════════

    if (d_src != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, d_src, total_bytes, cudaMemcpyDeviceToDevice));
    }

    float total_ms = run_gen_ms + total_merge_ms;
    printf("\n  ═══════════════════════════════════════════\n");
    printf("  SORT COMPLETE: %.2f ms (gen: %.2f + merge: %.2f)\n",
           total_ms, run_gen_ms, total_merge_ms);
    printf("  Throughput: %.2f GB/s | %.2f M records/sec\n",
           total_bytes / (total_ms * 1e6), num_records / (total_ms * 1e3));
    printf("  Merge passes: %d | Total HBM traffic: ~%.1f GB\n",
           pass, 2.0 * total_bytes * pass / 1e9);
    printf("  Strategy: %s\n", strategy == STRATEGY_KWAY ? "K-way merge tree" : "2-way merge path");
    printf("  ═══════════════════════════════════════════\n");

    // ════════════════════════════════════════
    // Verify
    // ════════════════════════════════════════

    if (verify) {
        printf("\nVerifying...\n");
        uint8_t* h = (uint8_t*)malloc(total_bytes);
        CUDA_CHECK(cudaMemcpy(h, d_data, total_bytes, cudaMemcpyDeviceToHost));

        uint64_t violations = 0;
        for (uint64_t i = 1; i < num_records && violations < 10; i++) {
            if (key_compare(h + (i-1)*RECORD_SIZE, h + i*RECORD_SIZE, KEY_SIZE) > 0)
                violations++;
        }
        printf(violations == 0 ? "  PASS: sorted correctly\n" : "  FAIL: %lu violations\n", violations);
        free(h);
    }

    CUDA_CHECK(cudaFree(d_sorted));
    CUDA_CHECK(cudaFree(d_merge_buf));
}
