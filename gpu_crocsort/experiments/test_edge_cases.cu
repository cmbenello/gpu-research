// ============================================================================
// GPU CrocSort -- Edge Case & Adversarial Input Tests
//
// Build (from gpu_crocsort/):
//   nvcc -O2 -std=c++17 -arch=sm_80 \
//        -I include \
//        experiments/test_edge_cases.cu src/run_generation.cu src/merge.cu \
//        src/host_sort.cu -o test_edge_cases
//
// Run:
//   ./test_edge_cases
// ============================================================================

#include "record.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>
#include <cuda_runtime.h>

extern void gpu_crocsort_in_hbm(uint8_t* d_data, uint64_t num_records, bool verify);

// ── Verification helper ──────────────────────────────────────────────
// Returns true if: (a) output is sorted by key, (b) record count matches,
// (c) multiset of keys is identical to input.

static bool verify_sort(const uint8_t* input, const uint8_t* output,
                         uint64_t num_records) {
    if (num_records == 0) return true;

    // (a) Check sorted order
    for (uint64_t i = 1; i < num_records; i++) {
        const uint8_t* prev = output + (i - 1) * RECORD_SIZE;
        const uint8_t* curr = output + i * RECORD_SIZE;
        if (memcmp(prev, curr, KEY_SIZE) > 0) {
            fprintf(stderr, "    FAIL: not sorted at index %lu\n", (unsigned long)i);
            return false;
        }
    }

    // (b) + (c) Build multiset of keys from input and output, compare
    // Use a map from 10-byte key (as string) to count
    std::map<std::vector<uint8_t>, int64_t> key_counts;
    for (uint64_t i = 0; i < num_records; i++) {
        const uint8_t* k = input + i * RECORD_SIZE;
        std::vector<uint8_t> key(k, k + KEY_SIZE);
        key_counts[key]++;
    }
    for (uint64_t i = 0; i < num_records; i++) {
        const uint8_t* k = output + i * RECORD_SIZE;
        std::vector<uint8_t> key(k, k + KEY_SIZE);
        key_counts[key]--;
    }
    for (auto& [key, count] : key_counts) {
        if (count != 0) {
            fprintf(stderr, "    FAIL: key multiset mismatch (delta=%ld)\n",
                    (long)count);
            return false;
        }
    }
    return true;
}

// ── Fill a record with a given key and index payload ─────────────────

static void fill_record(uint8_t* rec, const uint8_t key[KEY_SIZE], uint64_t idx) {
    memcpy(rec, key, KEY_SIZE);
    memset(rec + KEY_SIZE, 0, VALUE_SIZE);
    memcpy(rec + KEY_SIZE, &idx, sizeof(uint64_t));
}

// ── Run one test case ────────────────────────────────────────────────
// Allocates GPU memory, copies data, sorts, copies back, verifies.
// Returns true on pass.

static bool run_test(const char* name, int test_num, uint8_t* h_input,
                     uint64_t num_records, float* out_ms) {
    uint64_t total_bytes = num_records * RECORD_SIZE;

    // Save a copy of input for verification
    uint8_t* h_input_copy = (uint8_t*)malloc(total_bytes);
    memcpy(h_input_copy, h_input, total_bytes);

    uint8_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_input, total_bytes, cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer.begin();
    gpu_crocsort_in_hbm(d_data, num_records, false);
    *out_ms = timer.end();

    uint8_t* h_output = (uint8_t*)malloc(total_bytes);
    CUDA_CHECK(cudaMemcpy(h_output, d_data, total_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    bool ok = verify_sort(h_input_copy, h_output, num_records);

    free(h_input_copy);
    free(h_output);
    return ok;
}

// ── Key constants ────────────────────────────────────────────────────

static void key_zero(uint8_t k[KEY_SIZE])  { memset(k, 0x00, KEY_SIZE); }
static void key_ff(uint8_t k[KEY_SIZE])    { memset(k, 0xFF, KEY_SIZE); }
static void key_from_u64(uint8_t k[KEY_SIZE], uint64_t v) {
    // Big-endian so lexicographic order matches numeric order
    memset(k, 0, KEY_SIZE);
    for (int i = 7; i >= 0; i--) {
        k[i + 2] = (uint8_t)(v & 0xFF);  // offset by 2 since KEY_SIZE=10
        v >>= 8;
    }
    // k[0], k[1] stay 0
}

// ── Test generators ──────────────────────────────────────────────────

static void gen_already_sorted(uint8_t* buf, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint8_t k[KEY_SIZE];
        key_from_u64(k, i);
        fill_record(buf + i * RECORD_SIZE, k, i);
    }
}

static void gen_reverse_sorted(uint8_t* buf, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint8_t k[KEY_SIZE];
        key_from_u64(k, n - 1 - i);
        fill_record(buf + i * RECORD_SIZE, k, i);
    }
}

static void gen_all_identical(uint8_t* buf, uint64_t n) {
    uint8_t k[KEY_SIZE] = {0x42, 0x13, 0x37, 0xAB, 0xCD,
                           0xEF, 0x01, 0x23, 0x45, 0x67};
    for (uint64_t i = 0; i < n; i++)
        fill_record(buf + i * RECORD_SIZE, k, i);
}

static void gen_two_distinct(uint8_t* buf, uint64_t n) {
    uint8_t kA[KEY_SIZE], kB[KEY_SIZE];
    memset(kA, 0x11, KEY_SIZE);
    memset(kB, 0xEE, KEY_SIZE);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* k = (i < n / 2) ? kA : kB;
        fill_record(buf + i * RECORD_SIZE, k, i);
    }
}

static void gen_alternating(uint8_t* buf, uint64_t n) {
    uint8_t kA[KEY_SIZE], kB[KEY_SIZE];
    memset(kA, 0x11, KEY_SIZE);
    memset(kB, 0xEE, KEY_SIZE);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* k = (i % 2 == 0) ? kA : kB;
        fill_record(buf + i * RECORD_SIZE, k, i);
    }
}

static void gen_all_zero_keys(uint8_t* buf, uint64_t n) {
    uint8_t k[KEY_SIZE];
    key_zero(k);
    for (uint64_t i = 0; i < n; i++)
        fill_record(buf + i * RECORD_SIZE, k, i);
}

static void gen_all_ff_keys(uint8_t* buf, uint64_t n) {
    uint8_t k[KEY_SIZE];
    key_ff(k);
    for (uint64_t i = 0; i < n; i++)
        fill_record(buf + i * RECORD_SIZE, k, i);
}

static void gen_skewed(uint8_t* buf, uint64_t n) {
    uint8_t k_common[KEY_SIZE];
    memset(k_common, 0x55, KEY_SIZE);
    srand(12345);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t k[KEY_SIZE];
        if ((uint64_t)(rand() % 10) < 9) {
            memcpy(k, k_common, KEY_SIZE);
        } else {
            for (int b = 0; b < KEY_SIZE; b++)
                k[b] = (uint8_t)(rand() & 0xFF);
        }
        fill_record(buf + i * RECORD_SIZE, k, i);
    }
}

static void gen_sequential(uint8_t* buf, uint64_t n) {
    gen_already_sorted(buf, n);  // key = index, same thing
}

// ── Test table ───────────────────────────────────────────────────────

struct TestCase {
    const char* name;
    uint64_t    num_records;
    void        (*generate)(uint8_t*, uint64_t);
};

int main() {
    TestCase tests[] = {
        {"already sorted",       10000, gen_already_sorted},
        {"reverse sorted",       10000, gen_reverse_sorted},
        {"all identical",        10000, gen_all_identical},
        {"two distinct keys",    10000, gen_two_distinct},
        {"single record",            1, gen_already_sorted},
        {"two records",               2, gen_reverse_sorted},
        {"power-of-2 (512)",        512, gen_reverse_sorted},
        {"power-of-2 (1024)",      1024, gen_already_sorted},
        {"2^n-1 (511)",             511, gen_reverse_sorted},
        {"2^n-1 (1023)",           1023, gen_already_sorted},
        {"very few (3)",              3, gen_alternating},
        {"very few (7)",              7, gen_alternating},
        {"very few (15)",            15, gen_alternating},
        {"all-zero keys",        10000, gen_all_zero_keys},
        {"all-0xFF keys",        10000, gen_all_ff_keys},
        {"alternating keys",     10000, gen_alternating},
        {"skewed 90/10",         10000, gen_skewed},
        {"sequential keys",      10000, gen_sequential},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    printf("========================================\n");
    printf("GPU CrocSort -- Edge Case Tests\n");
    printf("========================================\n\n");

    int passed = 0;
    for (int t = 0; t < num_tests; t++) {
        TestCase& tc = tests[t];
        uint64_t total_bytes = tc.num_records * RECORD_SIZE;

        uint8_t* h_data = (uint8_t*)malloc(total_bytes);
        tc.generate(h_data, tc.num_records);

        float ms = 0;
        bool ok = run_test(tc.name, t + 1, h_data, tc.num_records, &ms);
        free(h_data);

        if (ok) passed++;
        printf("Test %2d (%-22s %5lu recs):  %s  (%.1f ms)\n",
               t + 1, tc.name, (unsigned long)tc.num_records,
               ok ? "PASS" : "FAIL", ms);
    }

    printf("\nResults: %d/%d passed\n", passed, num_tests);
    return (passed == num_tests) ? 0 : 1;
}
