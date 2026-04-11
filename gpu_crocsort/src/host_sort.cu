#include "record.cuh"
#include "ovc.cuh"
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>

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

// ── Shared memory limit for K-way merge tree ───────────────────────
// Each partition needs 2 × total_records × RECORD_SIZE bytes in shared mem
// Query device to find actual limit

static int get_max_smem_per_block() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    // Can configure up to this much on modern GPUs (A100: 164KB, H100: 228KB)
    return props.sharedMemPerMultiprocessor; // Conservative: use SM-level max
}

static int max_records_per_kway_partition() {
    int smem = get_max_smem_per_block();
    // Need 2 buffers (ping-pong) of total_records × RECORD_SIZE
    // Cap at 99KB to leave room for other shared mem usage
    int usable_smem = std::min(smem, 99 * 1024); // Stay under 100KB
    return usable_smem / (2 * RECORD_SIZE);       // Records per partition
}

// ── K-way merge: partition computation ─────────────────────────────
// Splits K runs into P partitions by sampling boundaries.
// Each partition gets a roughly equal number of total records.
// Within each partition, binary search each run to find its sub-range.

static void compute_kway_partitions(
    const uint8_t* h_sorted_data,  // Sorted run data (host copy for boundary search)
    const std::vector<Run>& group_runs,
    int K,
    int max_records_per_part,
    std::vector<KWayPartition>& out_partitions,
    uint64_t out_base_offset
) {
    // Total records across all runs in this group
    uint64_t total = 0;
    for (auto& r : group_runs) total += r.num_records;

    // Number of partitions: enough so each fits in shared memory
    int min_partitions = (int)((total + max_records_per_part - 1) / max_records_per_part);
    int num_partitions = std::max(min_partitions, 64); // At least 64 for parallelism

    // Simple count-balanced partitioning: each partition gets ~total/P records
    int records_per_part = (int)((total + num_partitions - 1) / num_partitions);

    // For each partition, determine how many records from each source
    // Simple approach: proportional split (each run contributes proportionally)
    out_partitions.resize(num_partitions);

    for (int p = 0; p < num_partitions; p++) {
        KWayPartition& kp = out_partitions[p];
        uint64_t part_start = (uint64_t)p * records_per_part;
        uint64_t part_end = std::min(part_start + (uint64_t)records_per_part, total);
        kp.total_records = (int)(part_end - part_start);
        kp.out_byte_offset = out_base_offset + part_start * RECORD_SIZE;

        // Distribute records from each source
        // Simple proportional: each source contributes (source_records / total) * partition_records
        uint64_t remaining = kp.total_records;
        for (int k = 0; k < K; k++) {
            uint64_t run_recs = group_runs[k].num_records;
            int this_source;
            if (k == K - 1) {
                this_source = (int)remaining; // Last source gets remainder
            } else {
                this_source = (int)((run_recs * kp.total_records + total - 1) / total);
                this_source = std::min(this_source, (int)remaining);
                this_source = std::min(this_source, (int)run_recs);
            }

            // Compute start offset within this source's run
            uint64_t src_part_start = (run_recs * p) / num_partitions;
            int src_count = std::min((uint64_t)this_source,
                                     run_recs - src_part_start);

            kp.src_rec_start[k] = (int)src_part_start;
            kp.src_rec_count[k] = src_count;
            kp.src_byte_off[k] = group_runs[k].byte_offset;
            remaining -= src_count;
        }

        // Zero unused sources
        for (int k = K; k < KWAY_K; k++) {
            kp.src_rec_start[k] = 0;
            kp.src_rec_count[k] = 0;
            kp.src_byte_off[k] = 0;
        }
    }
}

// ── Strategy selector ──────────────────────────────────────────────

enum MergeStrategy {
    STRATEGY_2WAY,      // 2-way merge path: max parallelism, many passes
    STRATEGY_KWAY,      // K-way merge tree: fewer passes, still parallel
};

// ── Main sort function ─────────────────────────────────────────────

void gpu_crocsort_in_hbm(
    uint8_t* d_data,
    uint64_t num_records,
    bool verify
) {
    if (num_records <= 1) return;

    uint64_t total_bytes = num_records * RECORD_SIZE;
    int max_recs_per_part = max_records_per_kway_partition();

    printf("GPU CrocSort: %lu records (%.2f MB)\n",
           num_records, (double)total_bytes / (1024.0 * 1024.0));
    printf("  K-way merge tree: K=%d, max %d records/partition (%.1f KB smem)\n",
           KWAY_K, max_recs_per_part, max_recs_per_part * RECORD_SIZE * 2.0 / 1024.0);

    // Choose strategy
    MergeStrategy strategy = STRATEGY_KWAY; // Default: fewer passes

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
        // ── K-way merge tree strategy ──
        printf("\n  Using %d-way shared-memory merge tree\n", KWAY_K);

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

                // Compute partitions
                std::vector<KWayPartition> partitions;
                compute_kway_partitions(nullptr, group_runs, g_size,
                                        max_recs_per_part, partitions, out_offset);

                int max_rec = 0;
                for (auto& p : partitions) max_rec = std::max(max_rec, p.total_records);

                // Upload
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
            printf("  Pass %d: %d runs -> %d (%d-way), %d+ partitions, %.2f ms (%.2f GB/s)\n",
                   pass, current_runs, (int)new_runs.size(), group_size,
                   (int)(current_runs / group_size) * 64, // Approximate partition count
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
