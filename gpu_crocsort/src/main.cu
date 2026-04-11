#include "record.cuh"
#include "ovc.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

// ============================================================================
// GPU CrocSort — Main Entry Point
//
// Usage:
//   ./gpu_crocsort [--num-records N] [--verify] [--seed S]
//
// Generates random test data, sorts on GPU using CrocSort algorithms
// (OVC encoding, loser tree merge), and optionally verifies correctness.
// ============================================================================

// Strategy enum (must match host_sort.cu)
enum MergeStrategy { STRATEGY_2WAY, STRATEGY_KWAY };

// Forward declaration
void gpu_crocsort_in_hbm(uint8_t* d_data, uint64_t num_records, bool verify, MergeStrategy strategy);

// ── Random data generation ─────────────────────────────────────────
// Generates GenSort-style records: 10-byte random key + 90-byte payload

static void generate_random_records(uint8_t* h_data, uint64_t num_records, unsigned seed) {
    srand(seed);
    for (uint64_t i = 0; i < num_records; i++) {
        uint8_t* rec = h_data + i * RECORD_SIZE;
        // Random 10-byte key
        for (int b = 0; b < KEY_SIZE; b++) {
            rec[b] = (uint8_t)(rand() & 0xFF);
        }
        // Payload: store record index for verification
        memset(rec + KEY_SIZE, 0, VALUE_SIZE);
        uint64_t idx = i;
        memcpy(rec + KEY_SIZE, &idx, sizeof(uint64_t));
    }
}

// Generate records with many duplicate keys (tests OVC duplicate shortcut)
static void generate_duplicate_heavy_records(uint8_t* h_data, uint64_t num_records, unsigned seed) {
    srand(seed);
    // Only 1000 unique keys -> many duplicates
    int num_unique = 1000;
    uint8_t unique_keys[1000][KEY_SIZE];
    for (int i = 0; i < num_unique; i++) {
        for (int b = 0; b < KEY_SIZE; b++) {
            unique_keys[i][b] = (uint8_t)(rand() & 0xFF);
        }
    }
    for (uint64_t i = 0; i < num_records; i++) {
        uint8_t* rec = h_data + i * RECORD_SIZE;
        int which = rand() % num_unique;
        memcpy(rec, unique_keys[which], KEY_SIZE);
        memset(rec + KEY_SIZE, 0, VALUE_SIZE);
        uint64_t idx = i;
        memcpy(rec + KEY_SIZE, &idx, sizeof(uint64_t));
    }
}

// ── Print GPU info ─────────────────────────────────────────────────

static void print_gpu_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("GPU: %s\n", props.name);
    printf("  SMs: %d, SMEM/SM: %d KB, L2: %d KB\n",
           props.multiProcessorCount,
           (int)(props.sharedMemPerMultiprocessor / 1024),
           (int)(props.l2CacheSize / 1024));
    printf("  HBM: %.1f GB, Bandwidth: %.0f GB/s\n",
           props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);
    printf("  Compute capability: %d.%d\n", props.major, props.minor);
}

// ── Main ───────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Parse arguments
    uint64_t num_records = 1000000;  // Default: 1M records = ~100MB
    bool verify = false;
    unsigned seed = 42;
    bool duplicates = false;
    MergeStrategy strategy = STRATEGY_KWAY;  // Default: K-way (fewer passes)

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--num-records") == 0 && i + 1 < argc) {
            num_records = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--duplicates") == 0) {
            duplicates = true;
        } else if (strcmp(argv[i], "--strategy") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "2way") == 0) strategy = STRATEGY_2WAY;
            else if (strcmp(argv[i], "kway") == 0) strategy = STRATEGY_KWAY;
            else { fprintf(stderr, "Unknown strategy: %s (use 2way or kway)\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--num-records N] [--verify] [--seed S] [--duplicates] [--strategy 2way|kway]\n", argv[0]);
            printf("  --num-records N       Number of records to sort (default: 1000000)\n");
            printf("  --verify              Verify sort correctness after sorting\n");
            printf("  --seed S              Random seed (default: 42)\n");
            printf("  --duplicates          Generate data with many duplicate keys\n");
            printf("  --strategy 2way|kway  Merge strategy (default: kway)\n");
            return 0;
        }
    }

    uint64_t total_bytes = num_records * RECORD_SIZE;

    printf("========================================\n");
    printf("GPU CrocSort — External Merge Sort\n");
    printf("========================================\n");
    print_gpu_info();
    printf("\nConfiguration:\n");
    printf("  Records: %lu\n", num_records);
    printf("  Record size: %d bytes (key=%d, value=%d)\n", RECORD_SIZE, KEY_SIZE, VALUE_SIZE);
    printf("  Total data: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
    uint64_t expected_runs = (num_records + RECORDS_PER_BLOCK - 1) / RECORDS_PER_BLOCK;
    int passes_2way = (int)ceil(log2((double)expected_runs));
    int passes_8way = (int)ceil(log((double)expected_runs) / log(8.0));
    printf("  Records per block: %d\n", RECORDS_PER_BLOCK);
    printf("  Expected runs: %lu\n", expected_runs);
    printf("  Merge strategy: %s\n", strategy == STRATEGY_KWAY
           ? "8-way shared-memory merge tree (sample partitioning)"
           : "2-way merge path");
    printf("  Merge passes: %d (vs %d for 2-way) = %.1fx less HBM traffic\n",
           passes_8way, passes_2way, (double)passes_2way / passes_8way);
    printf("  Mode: %s\n", duplicates ? "duplicate-heavy" : "random");
    printf("\n");

    // Check GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.2f GB free / %.2f GB total\n",
           free_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0));

    // We need ~3x the data size (input + 2 merge buffers)
    uint64_t required = total_bytes * 3 + num_records * sizeof(uint32_t) * 2;
    if (required > free_mem) {
        printf("WARNING: Estimated memory requirement (%.2f GB) exceeds free GPU memory.\n",
               required / (1024.0 * 1024.0 * 1024.0));
        printf("  Consider reducing --num-records or using external sort (not yet implemented).\n");
    }

    // ── Generate test data on host ──
    printf("Generating %s test data...\n", duplicates ? "duplicate-heavy" : "random");
    uint8_t* h_data = (uint8_t*)malloc(total_bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate %lu bytes on host\n", total_bytes);
        return 1;
    }

    GpuTimer overall_timer;
    if (duplicates) {
        generate_duplicate_heavy_records(h_data, num_records, seed);
    } else {
        generate_random_records(h_data, num_records, seed);
    }

    // ── Transfer to GPU ──
    printf("Transferring to GPU...\n");
    uint8_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));

    overall_timer.begin();
    CUDA_CHECK(cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice));

    // ── Sort ──
    gpu_crocsort_in_hbm(d_data, num_records, verify, strategy);

    float total_ms = overall_timer.end();

    printf("\n========================================\n");
    printf("Total time (including H2D transfer): %.2f ms\n", total_ms);
    printf("Effective throughput: %.2f GB/s\n",
           (double)total_bytes / (total_ms * 1e6));
    printf("Records/second: %.2f M\n",
           (double)num_records / (total_ms * 1e3));
    printf("========================================\n");

    // Cleanup
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
