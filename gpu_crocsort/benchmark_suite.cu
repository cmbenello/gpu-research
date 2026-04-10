// ============================================================================
// GPU CrocSort -- Comprehensive Benchmark Suite
//
// Tests multiple data sizes x distributions x merge strategies.
// Outputs human-readable tables and CSV for plotting.
//
// Usage:
//   ./benchmark_suite [--quick] [--full] [--csv results.csv] [--seed S]
//   --quick   : Fewer sizes (10K, 1M, 10M) and fewer distributions
//   --full    : All sizes up to 50M and all distributions
//   (default) : Moderate set (10K, 100K, 1M, 5M, 10M)
//
// Build:
//   make benchmark-suite
// ============================================================================

#include "record.cuh"
#include "ovc.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <climits>

// ── Forward declarations from the project ────────────────────────────

extern "C" void launch_run_generation(
    const uint8_t* d_input, uint8_t* d_output, uint32_t* d_ovc,
    uint64_t total_records, SparseEntry* d_sparse, int* d_sparse_counts,
    int num_blocks, cudaStream_t stream);

// ── Merge strategy enum ─────────────────────────────────────────────

enum BenchMergeStrategy {
    BENCH_STRATEGY_2WAY = 0,
    BENCH_STRATEGY_KWAY = 1,
};

static const char* strategy_name(BenchMergeStrategy s) {
    return s == BENCH_STRATEGY_2WAY ? "2-way" : "8-way";
}

// ── Data distribution enum ───────────────────────────────────────────

enum DataDistribution {
    DIST_RANDOM = 0,
    DIST_SORTED,
    DIST_REVERSE_SORTED,
    DIST_NEARLY_SORTED,
    DIST_ALL_DUPLICATES,
    DIST_FEW_UNIQUE,
    DIST_ZIPF,
    DIST_COUNT
};

static const char* dist_name(DataDistribution d) {
    switch (d) {
        case DIST_RANDOM:          return "random";
        case DIST_SORTED:          return "sorted";
        case DIST_REVERSE_SORTED:  return "reverse-sorted";
        case DIST_NEARLY_SORTED:   return "nearly-sorted-95";
        case DIST_ALL_DUPLICATES:  return "all-duplicates";
        case DIST_FEW_UNIQUE:      return "few-unique-1000";
        case DIST_ZIPF:            return "zipf";
        default:                   return "unknown";
    }
}

// ── Data generation helpers ──────────────────────────────────────────

// Simple LCG for fast, reproducible random bytes
struct FastRng {
    uint64_t state;
    FastRng(uint64_t seed) : state(seed ^ 0x6c62272e07bb0142ULL) {}
    uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (uint32_t)(state >> 32);
    }
    uint8_t next_byte() { return (uint8_t)(next() & 0xFF); }
};

// Write a uint64_t value as a big-endian 10-byte key
static void write_key_from_u64(uint8_t* key, uint64_t val) {
    for (int i = 7; i >= 0; i--) {
        key[i] = (uint8_t)(val & 0xFF);
        val >>= 8;
    }
    key[8] = 0;
    key[9] = 0;
}

static void fill_payload(uint8_t* rec, uint64_t idx) {
    memset(rec + KEY_SIZE, 0, VALUE_SIZE);
    memcpy(rec + KEY_SIZE, &idx, sizeof(uint64_t));
}

static void generate_data(uint8_t* h_data, uint64_t num_records,
                           DataDistribution dist, unsigned seed)
{
    FastRng rng(seed);

    switch (dist) {
    case DIST_RANDOM:
        for (uint64_t i = 0; i < num_records; i++) {
            uint8_t* rec = h_data + i * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++)
                rec[b] = rng.next_byte();
            fill_payload(rec, i);
        }
        break;

    case DIST_SORTED:
        for (uint64_t i = 0; i < num_records; i++) {
            uint8_t* rec = h_data + i * RECORD_SIZE;
            write_key_from_u64(rec, i);
            fill_payload(rec, i);
        }
        break;

    case DIST_REVERSE_SORTED:
        for (uint64_t i = 0; i < num_records; i++) {
            uint8_t* rec = h_data + i * RECORD_SIZE;
            write_key_from_u64(rec, num_records - 1 - i);
            fill_payload(rec, i);
        }
        break;

    case DIST_NEARLY_SORTED: {
        // 95% sorted: generate sorted, then swap 5% of adjacent pairs
        for (uint64_t i = 0; i < num_records; i++) {
            uint8_t* rec = h_data + i * RECORD_SIZE;
            write_key_from_u64(rec, i);
            fill_payload(rec, i);
        }
        uint64_t num_swaps = num_records / 20; // 5%
        for (uint64_t s = 0; s < num_swaps; s++) {
            uint64_t idx = (uint64_t)rng.next() % (num_records > 1 ? num_records - 1 : 1);
            uint8_t tmp[RECORD_SIZE];
            memcpy(tmp, h_data + idx * RECORD_SIZE, RECORD_SIZE);
            memcpy(h_data + idx * RECORD_SIZE, h_data + (idx + 1) * RECORD_SIZE, RECORD_SIZE);
            memcpy(h_data + (idx + 1) * RECORD_SIZE, tmp, RECORD_SIZE);
        }
        break;
    }

    case DIST_ALL_DUPLICATES: {
        uint8_t fixed_key[KEY_SIZE];
        for (int b = 0; b < KEY_SIZE; b++)
            fixed_key[b] = rng.next_byte();
        for (uint64_t i = 0; i < num_records; i++) {
            uint8_t* rec = h_data + i * RECORD_SIZE;
            memcpy(rec, fixed_key, KEY_SIZE);
            fill_payload(rec, i);
        }
        break;
    }

    case DIST_FEW_UNIQUE: {
        static constexpr int NUM_UNIQUE = 1000;
        uint8_t unique_keys[NUM_UNIQUE][KEY_SIZE];
        for (int k = 0; k < NUM_UNIQUE; k++)
            for (int b = 0; b < KEY_SIZE; b++)
                unique_keys[k][b] = rng.next_byte();
        for (uint64_t i = 0; i < num_records; i++) {
            uint8_t* rec = h_data + i * RECORD_SIZE;
            int which = rng.next() % NUM_UNIQUE;
            memcpy(rec, unique_keys[which], KEY_SIZE);
            fill_payload(rec, i);
        }
        break;
    }

    case DIST_ZIPF: {
        // Zipf distribution: rank ~ N / (u*N + 1)
        for (uint64_t i = 0; i < num_records; i++) {
            uint8_t* rec = h_data + i * RECORD_SIZE;
            double u = (double)(rng.next()) / (double)UINT32_MAX;
            uint64_t rank = (uint64_t)(num_records / (u * num_records + 1.0));
            write_key_from_u64(rec, rank);
            fill_payload(rec, i);
        }
        break;
    }

    default:
        break;
    }
}

// ── GPU hardware info ────────────────────────────────────────────────

struct GpuInfo {
    char name[256];
    int sm_count;
    int smem_per_sm_kb;
    int l2_kb;
    double hbm_gb;
    double peak_bw_gbs;
    int compute_major;
    int compute_minor;
    int clock_mhz;
    int mem_clock_mhz;
    int bus_width;
};

static GpuInfo get_gpu_info() {
    GpuInfo info;
    memset(&info, 0, sizeof(info));
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    strncpy(info.name, props.name, sizeof(info.name) - 1);
    info.name[sizeof(info.name) - 1] = '\0';
    info.sm_count = props.multiProcessorCount;
    info.smem_per_sm_kb = (int)(props.sharedMemPerMultiprocessor / 1024);
    info.l2_kb = (int)(props.l2CacheSize / 1024);
    info.hbm_gb = props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    info.peak_bw_gbs = 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6;
    info.compute_major = props.major;
    info.compute_minor = props.minor;
    info.clock_mhz = props.clockRate / 1000;
    info.mem_clock_mhz = props.memoryClockRate / 1000;
    info.bus_width = props.memoryBusWidth;

    return info;
}

static void print_gpu_info_block(const GpuInfo& info) {
    printf("================================================================\n");
    printf("  GPU Hardware\n");
    printf("================================================================\n");
    printf("  Device       : %s\n", info.name);
    printf("  SMs          : %d\n", info.sm_count);
    printf("  SMEM/SM      : %d KB\n", info.smem_per_sm_kb);
    printf("  L2 Cache     : %d KB\n", info.l2_kb);
    printf("  HBM          : %.1f GB\n", info.hbm_gb);
    printf("  Peak BW      : %.0f GB/s\n", info.peak_bw_gbs);
    printf("  Core Clock   : %d MHz\n", info.clock_mhz);
    printf("  Mem Clock    : %d MHz\n", info.mem_clock_mhz);
    printf("  Bus Width    : %d bits\n", info.bus_width);
    printf("  Compute      : %d.%d\n", info.compute_major, info.compute_minor);
    printf("================================================================\n\n");
}

// ── Benchmark result struct ──────────────────────────────────────────

struct BenchResult {
    uint64_t num_records;
    DataDistribution distribution;
    BenchMergeStrategy strategy;
    float total_ms;
    float run_gen_ms;
    float merge_ms;
    double throughput_gbs;
    double throughput_mrec_s;
    int merge_passes;
    double hbm_traffic_gb;
    bool verify_pass;
};

// ── Merge infrastructure (mirrors host_sort.cu) ─────────────────────

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

extern "C" void launch_merge_2way(
    const uint8_t* d_input, uint8_t* d_output,
    const PairDesc2Way* d_pairs, int num_pairs, int total_blocks,
    cudaStream_t stream);

extern "C" void launch_merge_kway(
    const uint8_t* d_input, uint8_t* d_output,
    const KWayPartition* d_partitions, int num_partitions,
    int max_records_per_partition, cudaStream_t stream);

struct Run {
    uint64_t byte_offset;
    uint64_t num_records;
};

static int bench_max_records_per_kway_partition() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int smem = props.sharedMemPerMultiprocessor;
    int usable_smem = std::min(smem, 99 * 1024);
    return usable_smem / (2 * RECORD_SIZE);
}

static void compute_kway_partitions_bench(
    const std::vector<Run>& group_runs,
    int K,
    int max_records_per_part,
    std::vector<KWayPartition>& out_partitions,
    uint64_t out_base_offset)
{
    uint64_t total = 0;
    for (auto& r : group_runs) total += r.num_records;

    int min_partitions = (int)((total + max_records_per_part - 1) / max_records_per_part);
    int num_partitions = std::max(min_partitions, 64);
    int records_per_part = (int)((total + num_partitions - 1) / num_partitions);

    out_partitions.resize(num_partitions);

    for (int p = 0; p < num_partitions; p++) {
        KWayPartition& kp = out_partitions[p];
        uint64_t part_start = (uint64_t)p * records_per_part;
        uint64_t part_end = std::min(part_start + (uint64_t)records_per_part, total);
        kp.total_records = (int)(part_end - part_start);
        kp.out_byte_offset = out_base_offset + part_start * RECORD_SIZE;

        uint64_t remaining = kp.total_records;
        for (int k = 0; k < K; k++) {
            uint64_t run_recs = group_runs[k].num_records;
            int this_source;
            if (k == K - 1) {
                this_source = (int)remaining;
            } else {
                this_source = (int)((run_recs * kp.total_records + total - 1) / total);
                this_source = std::min(this_source, (int)remaining);
                this_source = std::min(this_source, (int)run_recs);
            }
            uint64_t src_part_start = (run_recs * p) / num_partitions;
            int src_count = std::min((uint64_t)this_source, run_recs - src_part_start);
            kp.src_rec_start[k] = (int)src_part_start;
            kp.src_rec_count[k] = src_count;
            kp.src_byte_off[k] = group_runs[k].byte_offset;
            remaining -= src_count;
        }
        for (int k = K; k < KWAY_K; k++) {
            kp.src_rec_start[k] = 0;
            kp.src_rec_count[k] = 0;
            kp.src_byte_off[k] = 0;
        }
    }
}

// ── Single benchmark run with phase-level timing ─────────────────────

static BenchResult run_benchmark(
    uint8_t* h_data,
    uint64_t num_records,
    DataDistribution dist,
    BenchMergeStrategy strategy,
    bool do_verify)
{
    BenchResult result;
    result.num_records = num_records;
    result.distribution = dist;
    result.strategy = strategy;
    result.verify_pass = true;

    uint64_t total_bytes = num_records * RECORD_SIZE;

    // Allocate device memory
    uint8_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice));

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

    // Warmup run (discard timing)
    launch_run_generation(d_data, d_sorted, d_ovc, num_records,
                          d_sparse, d_sparse_counts, num_runs, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-upload fresh data for the timed run
    CUDA_CHECK(cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice));

    // ── Phase 1: Run generation (timed) ──
    timer.begin();
    launch_run_generation(d_data, d_sorted, d_ovc, num_records,
                          d_sparse, d_sparse_counts, num_runs, 0);
    result.run_gen_ms = timer.end();

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

    // ── Phase 2: Merge (timed) ──
    uint8_t* d_merge_buf;
    CUDA_CHECK(cudaMalloc(&d_merge_buf, total_bytes));

    uint8_t* d_src = d_sorted;
    uint8_t* d_dst = d_merge_buf;

    int pass = 0;
    float total_merge_ms = 0;

    if (strategy == BENCH_STRATEGY_KWAY) {
        int max_recs_per_part = bench_max_records_per_kway_partition();

        while (runs.size() > 1) {
            pass++;
            int current_runs = (int)runs.size();
            int group_size = std::min(KWAY_K, current_runs);
            int num_groups = (current_runs + group_size - 1) / group_size;

            std::vector<Run> new_runs;
            uint64_t out_offset = 0;

            for (int g = 0; g < num_groups; g++) {
                int g_start = g * group_size;
                int g_end = std::min(g_start + group_size, current_runs);
                int g_size = g_end - g_start;

                if (g_size == 1) {
                    Run& r = runs[g_start];
                    CUDA_CHECK(cudaMemcpy(d_dst + out_offset, d_src + r.byte_offset,
                                          r.num_records * RECORD_SIZE, cudaMemcpyDeviceToDevice));
                    new_runs.push_back({out_offset, r.num_records});
                    out_offset += r.num_records * RECORD_SIZE;
                    continue;
                }

                std::vector<Run> group_runs_vec(runs.begin() + g_start, runs.begin() + g_end);
                std::vector<KWayPartition> partitions;
                compute_kway_partitions_bench(group_runs_vec, g_size,
                                              max_recs_per_part, partitions, out_offset);

                int max_rec = 0;
                for (auto& pt : partitions) max_rec = std::max(max_rec, pt.total_records);

                KWayPartition* d_parts;
                CUDA_CHECK(cudaMalloc(&d_parts, partitions.size() * sizeof(KWayPartition)));
                CUDA_CHECK(cudaMemcpy(d_parts, partitions.data(),
                                      partitions.size() * sizeof(KWayPartition),
                                      cudaMemcpyHostToDevice));

                timer.begin();
                launch_merge_kway(d_src, d_dst, d_parts, (int)partitions.size(), max_rec, 0);
                total_merge_ms += timer.end();

                CUDA_CHECK(cudaFree(d_parts));

                uint64_t merged_records = 0;
                for (auto& rr : group_runs_vec) merged_records += rr.num_records;
                new_runs.push_back({out_offset, merged_records});
                out_offset += merged_records * RECORD_SIZE;
            }
            runs = new_runs;
            std::swap(d_src, d_dst);
        }
    } else {
        // 2-way merge path
        int items_per_block = MERGE_ITEMS_PER_THREAD_CFG * MERGE_BLOCK_THREADS_CFG;

        while (runs.size() > 1) {
            pass++;
            int current_runs = (int)runs.size();
            int num_pairs = current_runs / 2;
            bool leftover = (current_runs % 2 == 1);

            std::vector<PairDesc2Way> pairs(num_pairs);
            int total_blocks = 0;
            uint64_t out_offset = 0;

            for (int p = 0; p < num_pairs; p++) {
                Run& ra = runs[2 * p];
                Run& rb = runs[2 * p + 1];
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
            total_merge_ms += timer.end();

            CUDA_CHECK(cudaFree(d_pairs));

            std::vector<Run> new_runs;
            uint64_t nr_off = 0;
            for (int p = 0; p < num_pairs; p++) {
                uint64_t c = runs[2 * p].num_records + runs[2 * p + 1].num_records;
                new_runs.push_back({nr_off, c});
                nr_off += c * RECORD_SIZE;
            }
            if (leftover) new_runs.push_back({leftover_off, runs[current_runs - 1].num_records});
            runs = new_runs;
            std::swap(d_src, d_dst);
        }
    }

    result.merge_ms = total_merge_ms;
    result.total_ms = result.run_gen_ms + result.merge_ms;
    result.merge_passes = pass;
    result.throughput_gbs = total_bytes / (result.total_ms * 1e6);
    result.throughput_mrec_s = num_records / (result.total_ms * 1e3);
    result.hbm_traffic_gb = 2.0 * total_bytes * pass / 1e9;

    // ── Optional verification ──
    if (do_verify && num_records > 1) {
        if (d_src != d_data) {
            CUDA_CHECK(cudaMemcpy(d_data, d_src, total_bytes, cudaMemcpyDeviceToDevice));
        }
        uint8_t* h_check = (uint8_t*)malloc(total_bytes);
        CUDA_CHECK(cudaMemcpy(h_check, d_data, total_bytes, cudaMemcpyDeviceToHost));

        for (uint64_t i = 1; i < num_records; i++) {
            if (key_compare(h_check + (i - 1) * RECORD_SIZE,
                            h_check + i * RECORD_SIZE, KEY_SIZE) > 0) {
                result.verify_pass = false;
                break;
            }
        }
        free(h_check);
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_sorted));
    CUDA_CHECK(cudaFree(d_merge_buf));

    return result;
}

// ── Formatting helpers ───────────────────────────────────────────────

static const char* human_size(uint64_t n) {
    static char buf[32];
    if (n >= 1000000) snprintf(buf, sizeof(buf), "%.0fM", n / 1e6);
    else if (n >= 1000) snprintf(buf, sizeof(buf), "%.0fK", n / 1e3);
    else snprintf(buf, sizeof(buf), "%lu", (unsigned long)n);
    return buf;
}

// ── Print human-readable results table ───────────────────────────────

static void print_results_table(const std::vector<BenchResult>& results) {
    printf("\n");
    printf("================================================================"
           "================================================================\n");
    printf("  BENCHMARK RESULTS\n");
    printf("================================================================"
           "================================================================\n");
    printf("%-10s %-18s %-8s %10s %10s %10s %10s %12s %7s %10s %6s\n",
           "Records", "Distribution", "Strategy",
           "Total(ms)", "RunGen", "Merge",
           "GB/s", "Mrec/s", "Passes", "HBM(GB)", "OK");
    printf("---------- ------------------ -------- "
           "---------- ---------- ---------- "
           "---------- ------------ ------- ---------- ------\n");

    for (auto& r : results) {
        printf("%-10s %-18s %-8s %10.2f %10.2f %10.2f %10.3f %12.2f %7d %10.3f %6s\n",
               human_size(r.num_records),
               dist_name(r.distribution),
               strategy_name(r.strategy),
               r.total_ms,
               r.run_gen_ms,
               r.merge_ms,
               r.throughput_gbs,
               r.throughput_mrec_s,
               r.merge_passes,
               r.hbm_traffic_gb,
               r.verify_pass ? "PASS" : "FAIL");
    }

    printf("================================================================"
           "================================================================\n");
}

// ── Write CSV output ─────────────────────────────────────────────────

static void write_csv(const std::vector<BenchResult>& results,
                      const GpuInfo& gpu,
                      const char* filename)
{
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(f, "gpu,num_records,distribution,strategy,"
               "total_ms,run_gen_ms,merge_ms,"
               "throughput_gbs,throughput_mrec_s,"
               "merge_passes,hbm_traffic_gb,verify\n");

    for (auto& r : results) {
        fprintf(f, "%s,%lu,%s,%s,"
                   "%.4f,%.4f,%.4f,"
                   "%.6f,%.4f,"
                   "%d,%.6f,%s\n",
                gpu.name,
                (unsigned long)r.num_records,
                dist_name(r.distribution),
                strategy_name(r.strategy),
                r.total_ms,
                r.run_gen_ms,
                r.merge_ms,
                r.throughput_gbs,
                r.throughput_mrec_s,
                r.merge_passes,
                r.hbm_traffic_gb,
                r.verify_pass ? "pass" : "FAIL");
    }

    fclose(f);
    printf("\nCSV results written to: %s\n", filename);
}

// ── Print strategy comparison (2-way vs 8-way) ──────────────────────

static void print_strategy_comparison(const std::vector<BenchResult>& results) {
    printf("\n");
    printf("================================================================\n");
    printf("  2-WAY vs 8-WAY COMPARISON\n");
    printf("================================================================\n");

    struct Key {
        uint64_t num_records;
        DataDistribution dist;
        bool operator<(const Key& o) const {
            if (num_records != o.num_records) return num_records < o.num_records;
            return dist < o.dist;
        }
        bool operator==(const Key& o) const {
            return num_records == o.num_records && dist == o.dist;
        }
    };

    std::vector<Key> keys;
    for (auto& r : results) {
        Key k{r.num_records, r.distribution};
        bool found = false;
        for (auto& ek : keys) { if (ek == k) { found = true; break; } }
        if (!found) keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end());

    printf("%-10s %-18s %10s %10s %10s\n",
           "Records", "Distribution", "2-way(ms)", "8-way(ms)", "Speedup");
    printf("---------- ------------------ ---------- ---------- ----------\n");

    for (auto& k : keys) {
        float time_2way = -1, time_8way = -1;
        for (auto& r : results) {
            if (r.num_records == k.num_records && r.distribution == k.dist) {
                if (r.strategy == BENCH_STRATEGY_2WAY) time_2way = r.total_ms;
                else time_8way = r.total_ms;
            }
        }
        if (time_2way > 0 && time_8way > 0) {
            printf("%-10s %-18s %10.2f %10.2f %9.2fx\n",
                   human_size(k.num_records),
                   dist_name(k.dist),
                   time_2way, time_8way,
                   time_2way / time_8way);
        } else if (time_2way > 0) {
            printf("%-10s %-18s %10.2f %10s %10s\n",
                   human_size(k.num_records), dist_name(k.dist),
                   time_2way, "N/A", "N/A");
        } else if (time_8way > 0) {
            printf("%-10s %-18s %10s %10.2f %10s\n",
                   human_size(k.num_records), dist_name(k.dist),
                   "N/A", time_8way, "N/A");
        }
    }
    printf("================================================================\n");
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // ── Parse arguments ──
    const char* csv_file = "benchmark_results.csv";
    unsigned seed = 42;
    bool quick_mode = false;
    bool full_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = true;
        } else if (strcmp(argv[i], "--full") == 0) {
            full_mode = true;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_file = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--quick] [--full] [--csv FILE] [--seed S]\n", argv[0]);
            printf("  --quick   Fewer sizes and distributions (fast testing)\n");
            printf("  --full    All sizes up to 50M and all distributions\n");
            printf("  --csv F   Output CSV to file F (default: benchmark_results.csv)\n");
            printf("  --seed S  Random seed (default: 42)\n");
            return 0;
        }
    }

    // ── GPU info ──
    GpuInfo gpu = get_gpu_info();

    printf("\n");
    printf("================================================================\n");
    printf("  GPU CrocSort -- Comprehensive Benchmark Suite\n");
    printf("================================================================\n");
    print_gpu_info_block(gpu);

    // ── Configure test matrix ──
    std::vector<uint64_t> sizes;
    std::vector<DataDistribution> distributions;

    if (quick_mode) {
        sizes = {10000, 1000000, 10000000};
        distributions = {DIST_RANDOM, DIST_SORTED, DIST_ALL_DUPLICATES};
    } else if (full_mode) {
        sizes = {10000, 100000, 1000000, 5000000, 10000000, 50000000};
        distributions = {DIST_RANDOM, DIST_SORTED, DIST_REVERSE_SORTED,
                         DIST_NEARLY_SORTED, DIST_ALL_DUPLICATES,
                         DIST_FEW_UNIQUE, DIST_ZIPF};
    } else {
        // Default: moderate
        sizes = {10000, 100000, 1000000, 5000000, 10000000};
        distributions = {DIST_RANDOM, DIST_SORTED, DIST_REVERSE_SORTED,
                         DIST_NEARLY_SORTED, DIST_ALL_DUPLICATES,
                         DIST_FEW_UNIQUE, DIST_ZIPF};
    }

    std::vector<BenchMergeStrategy> strategies = {BENCH_STRATEGY_2WAY, BENCH_STRATEGY_KWAY};

    int total_tests = (int)(sizes.size() * distributions.size() * strategies.size());
    printf("Test matrix: %lu sizes x %lu distributions x %lu strategies = %d tests\n",
           (unsigned long)sizes.size(),
           (unsigned long)distributions.size(),
           (unsigned long)strategies.size(),
           total_tests);
    printf("Mode: %s\n", quick_mode ? "QUICK" : (full_mode ? "FULL" : "DEFAULT"));
    printf("CSV output: %s\n", csv_file);
    printf("Seed: %u\n\n", seed);

    // ── Check available GPU memory ──
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.2f GB free / %.2f GB total\n\n",
           free_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0));

    // ── Run benchmarks ──
    std::vector<BenchResult> results;
    int test_num = 0;

    for (uint64_t nrec : sizes) {
        uint64_t total_bytes = nrec * RECORD_SIZE;

        // Check if this size fits in GPU memory (need ~3x for sort buffers)
        uint64_t required = total_bytes * 3 + nrec * sizeof(uint32_t) * 2;
        if (required > free_mem) {
            printf("SKIPPING %s records: requires %.2f GB, only %.2f GB free\n",
                   human_size(nrec),
                   required / (1024.0 * 1024.0 * 1024.0),
                   free_mem / (1024.0 * 1024.0 * 1024.0));
            continue;
        }

        // Allocate host buffer once per size
        uint8_t* h_data = (uint8_t*)malloc(total_bytes);
        if (!h_data) {
            fprintf(stderr, "SKIPPING %s records: host alloc failed\n", human_size(nrec));
            continue;
        }

        for (DataDistribution dist : distributions) {
            // Generate data
            generate_data(h_data, nrec, dist, seed);

            for (BenchMergeStrategy strat : strategies) {
                test_num++;
                printf("[%3d/%3d] %10s | %-18s | %-5s ... ",
                       test_num, total_tests,
                       human_size(nrec), dist_name(dist), strategy_name(strat));
                fflush(stdout);

                // Verify only on the smallest size to save time
                bool do_verify = (nrec <= 100000);

                BenchResult r = run_benchmark(h_data, nrec, dist, strat, do_verify);
                results.push_back(r);

                printf("%8.2f ms | %7.3f GB/s | %7.2f Mrec/s | %d passes%s\n",
                       r.total_ms, r.throughput_gbs, r.throughput_mrec_s,
                       r.merge_passes,
                       (do_verify && !r.verify_pass) ? " | FAIL" : "");
            }
        }

        free(h_data);
    }

    // ── Output results ──
    print_results_table(results);
    print_strategy_comparison(results);
    write_csv(results, gpu, csv_file);

    // ── Summary ──
    if (!results.empty()) {
        double best_gbs = 0, worst_gbs = 1e12;
        double best_mrec = 0;
        int fails = 0;
        for (auto& r : results) {
            if (r.throughput_gbs > best_gbs) best_gbs = r.throughput_gbs;
            if (r.throughput_gbs < worst_gbs) worst_gbs = r.throughput_gbs;
            if (r.throughput_mrec_s > best_mrec) best_mrec = r.throughput_mrec_s;
            if (!r.verify_pass) fails++;
        }
        printf("\n");
        printf("================================================================\n");
        printf("  SUMMARY\n");
        printf("================================================================\n");
        printf("  Tests run      : %d\n", (int)results.size());
        printf("  Best throughput : %.3f GB/s (%.2f Mrec/s)\n", best_gbs, best_mrec);
        printf("  Worst throughput: %.3f GB/s\n", worst_gbs);
        printf("  Peak BW util   : %.1f%%\n", 100.0 * best_gbs / gpu.peak_bw_gbs);
        printf("  Verify failures: %d\n", fails);
        printf("================================================================\n");
    }

    printf("\nDone.\n");
    return 0;
}
