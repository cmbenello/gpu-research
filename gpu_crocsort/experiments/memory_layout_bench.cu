// ============================================================================
// Memory Access Pattern Benchmark
//
// Measures how different record layouts affect merge bandwidth.
// This is the most underexplored axis in GPU sort optimization.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_80 experiments/memory_layout_bench.cu -o layout_bench
// Run:   ./layout_bench
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Timer {
    cudaEvent_t start, stop;
    Timer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end_ms() {
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
    }
};

static const int KEY_SIZE = 10;
static const int VALUE_SIZE = 90;
static const int RECORD_SIZE = 100;

// ── Layout 1: Array of Structs (AoS) — current approach ───────────
// [key0|val0][key1|val1][key2|val2]...
// Merge comparison reads 100B but only uses first 10B

__global__ void merge_aos_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ out,
    int a_len, int b_len, int items_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = a_len + b_len;
    int start = tid * items_per_thread;
    int end = min(start + items_per_thread, total);
    if (start >= total) return;

    // Merge path search (simplified — compare first 8 bytes as uint64)
    int lo = max(0, start - b_len), hi = min(start, a_len);
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        int bm = start - 1 - m;
        // Compare: read 100B record but only use first 10B
        bool a_gt = false;
        if (m < a_len && bm >= 0 && bm < b_len) {
            for (int i = 0; i < KEY_SIZE; i++) {
                uint8_t ak = A[m * RECORD_SIZE + i];
                uint8_t bk = B[bm * RECORD_SIZE + i];
                if (ak != bk) { a_gt = (ak > bk); break; }
            }
        }
        if (a_gt) hi = m; else lo = m + 1;
    }
    int ai = lo, bi = start - lo;

    for (int i = start; i < end; i++) {
        bool take_a = (ai < a_len) && (bi >= b_len || ({
            bool less = false;
            for (int j = 0; j < KEY_SIZE; j++) {
                uint8_t ak = A[ai * RECORD_SIZE + j];
                uint8_t bk = B[bi * RECORD_SIZE + j];
                if (ak != bk) { less = (ak <= bk); break; }
                if (j == KEY_SIZE - 1) less = true;
            }
            less;
        }));

        const uint8_t* src = take_a ? A + ai * RECORD_SIZE : B + bi * RECORD_SIZE;
        if (take_a) ai++; else bi++;
        uint8_t* dst = out + i * RECORD_SIZE;
        // Copy full 100B record
        for (int b = 0; b < RECORD_SIZE; b += 4)
            *reinterpret_cast<uint32_t*>(dst+b) = *reinterpret_cast<const uint32_t*>(src+b);
    }
}

// ── Layout 2: Struct of Arrays (SoA) — keys and values separate ───
// Keys: [key0][key1][key2]...
// Vals: [val0][val1][val2]...
// Merge comparison reads only 10B!

__global__ void merge_soa_kernel(
    const uint8_t* __restrict__ keys_A,
    const uint8_t* __restrict__ keys_B,
    const uint8_t* __restrict__ vals_A,
    const uint8_t* __restrict__ vals_B,
    uint8_t* __restrict__ out_keys,
    uint8_t* __restrict__ out_vals,
    int a_len, int b_len, int items_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = a_len + b_len;
    int start = tid * items_per_thread;
    int end = min(start + items_per_thread, total);
    if (start >= total) return;

    // Merge path search — only reads 10B keys!
    int lo = max(0, start - b_len), hi = min(start, a_len);
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        int bm = start - 1 - m;
        bool a_gt = false;
        if (m < a_len && bm >= 0 && bm < b_len) {
            for (int i = 0; i < KEY_SIZE; i++) {
                uint8_t ak = keys_A[m * KEY_SIZE + i];
                uint8_t bk = keys_B[bm * KEY_SIZE + i];
                if (ak != bk) { a_gt = (ak > bk); break; }
            }
        }
        if (a_gt) hi = m; else lo = m + 1;
    }
    int ai = lo, bi = start - lo;

    for (int i = start; i < end; i++) {
        bool take_a = (ai < a_len) && (bi >= b_len || ({
            bool less = false;
            for (int j = 0; j < KEY_SIZE; j++) {
                uint8_t ak = keys_A[ai * KEY_SIZE + j];
                uint8_t bk = keys_B[bi * KEY_SIZE + j];
                if (ak != bk) { less = (ak <= bk); break; }
                if (j == KEY_SIZE - 1) less = true;
            }
            less;
        }));

        // Copy key (10B)
        const uint8_t* sk = take_a ? keys_A + ai * KEY_SIZE : keys_B + bi * KEY_SIZE;
        for (int b = 0; b < KEY_SIZE; b++)
            out_keys[i * KEY_SIZE + b] = sk[b];

        // Copy value (90B) — separate from comparison
        const uint8_t* sv = take_a ? vals_A + ai * VALUE_SIZE : vals_B + bi * VALUE_SIZE;
        for (int b = 0; b < VALUE_SIZE; b += 4)
            *reinterpret_cast<uint32_t*>(out_vals + i * VALUE_SIZE + b) =
                *reinterpret_cast<const uint32_t*>(sv + b);

        if (take_a) ai++; else bi++;
    }
}

// ── Layout 3: Key + Index (sort keys, scatter values) ──────────────
// Phase 1: merge keys only (10B each), produce sorted indices
// Phase 2: gather values by indices (separate kernel, coalesced writes)

__global__ void merge_keys_only_kernel(
    const uint8_t* __restrict__ keys_A,
    const uint8_t* __restrict__ keys_B,
    uint8_t* __restrict__ out_keys,
    uint32_t* __restrict__ out_indices, // encodes (source, index)
    int a_len, int b_len, int items_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = a_len + b_len;
    int start = tid * items_per_thread;
    int end = min(start + items_per_thread, total);
    if (start >= total) return;

    int lo = max(0, start - b_len), hi = min(start, a_len);
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        int bm = start - 1 - m;
        bool a_gt = false;
        if (m < a_len && bm >= 0 && bm < b_len) {
            for (int i = 0; i < KEY_SIZE; i++) {
                uint8_t ak = keys_A[m * KEY_SIZE + i];
                uint8_t bk = keys_B[bm * KEY_SIZE + i];
                if (ak != bk) { a_gt = (ak > bk); break; }
            }
        }
        if (a_gt) hi = m; else lo = m + 1;
    }
    int ai = lo, bi = start - lo;

    for (int i = start; i < end; i++) {
        bool take_a = (ai < a_len) && (bi >= b_len || ({
            bool less = false;
            for (int j = 0; j < KEY_SIZE; j++) {
                uint8_t ak = keys_A[ai * KEY_SIZE + j];
                uint8_t bk = keys_B[bi * KEY_SIZE + j];
                if (ak != bk) { less = (ak <= bk); break; }
                if (j == KEY_SIZE - 1) less = true;
            }
            less;
        }));

        const uint8_t* sk = take_a ? keys_A + ai * KEY_SIZE : keys_B + bi * KEY_SIZE;
        for (int b = 0; b < KEY_SIZE; b++)
            out_keys[i * KEY_SIZE + b] = sk[b];

        // Encode source+index: bit 31 = source (0=A, 1=B), bits 0-30 = index
        uint32_t idx = take_a ? (uint32_t)ai : ((uint32_t)bi | 0x80000000u);
        out_indices[i] = idx;

        if (take_a) ai++; else bi++;
    }
}

__global__ void gather_values_kernel(
    const uint8_t* __restrict__ vals_A,
    const uint8_t* __restrict__ vals_B,
    uint8_t* __restrict__ out_vals,
    const uint32_t* __restrict__ indices,
    int total
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    uint32_t idx = indices[tid];
    bool from_b = (idx & 0x80000000u) != 0;
    int src_idx = idx & 0x7FFFFFFFu;

    const uint8_t* src = from_b
        ? vals_B + src_idx * VALUE_SIZE
        : vals_A + src_idx * VALUE_SIZE;
    uint8_t* dst = out_vals + tid * VALUE_SIZE;

    for (int b = 0; b < VALUE_SIZE; b += 4)
        *reinterpret_cast<uint32_t*>(dst + b) =
            *reinterpret_cast<const uint32_t*>(src + b);
}

// ── Benchmark ──────────────────────────────────────────────────────

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("═══ Memory Layout Benchmark ═══\n");
    printf("GPU: %s, Peak BW: %.0f GB/s\n\n",
           props.name,
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);

    int num_records = 10000000; // 10M per run, merge two runs
    int a_len = num_records, b_len = num_records;
    int total = a_len + b_len;

    // Generate sorted keys
    uint8_t* h_keys_a = (uint8_t*)malloc(a_len * KEY_SIZE);
    uint8_t* h_keys_b = (uint8_t*)malloc(b_len * KEY_SIZE);
    srand(42);
    for (int i = 0; i < a_len; i++) {
        uint64_t v = 2 * (uint64_t)i;
        for (int b = 0; b < 8; b++) h_keys_a[i*KEY_SIZE + 7-b] = (v >> (b*8)) & 0xFF;
        h_keys_a[i*KEY_SIZE+8] = h_keys_a[i*KEY_SIZE+9] = 0;
    }
    for (int i = 0; i < b_len; i++) {
        uint64_t v = 2 * (uint64_t)i + 1;
        for (int b = 0; b < 8; b++) h_keys_b[i*KEY_SIZE + 7-b] = (v >> (b*8)) & 0xFF;
        h_keys_b[i*KEY_SIZE+8] = h_keys_b[i*KEY_SIZE+9] = 0;
    }

    int threads = 256;
    int items_per_thread = 8;
    int blocks = (total + threads * items_per_thread - 1) / (threads * items_per_thread);
    Timer t;

    printf("Merging 2 × %dM records (%dB each)\n\n", num_records/1000000, RECORD_SIZE);

    // ── AoS benchmark ──
    {
        uint8_t *d_a, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_a, (uint64_t)a_len * RECORD_SIZE));
        CUDA_CHECK(cudaMalloc(&d_b, (uint64_t)b_len * RECORD_SIZE));
        CUDA_CHECK(cudaMalloc(&d_out, (uint64_t)total * RECORD_SIZE));
        // Fill with key data at record stride
        uint8_t* h_aos = (uint8_t*)calloc(a_len, RECORD_SIZE);
        for (int i = 0; i < a_len; i++) memcpy(h_aos + i*RECORD_SIZE, h_keys_a + i*KEY_SIZE, KEY_SIZE);
        CUDA_CHECK(cudaMemcpy(d_a, h_aos, (uint64_t)a_len*RECORD_SIZE, cudaMemcpyHostToDevice));
        for (int i = 0; i < b_len; i++) memcpy(h_aos + i*RECORD_SIZE, h_keys_b + i*KEY_SIZE, KEY_SIZE);
        CUDA_CHECK(cudaMemcpy(d_b, h_aos, (uint64_t)b_len*RECORD_SIZE, cudaMemcpyHostToDevice));
        free(h_aos);

        // Warmup
        merge_aos_kernel<<<blocks, threads>>>(d_a, d_b, d_out, a_len, b_len, items_per_thread);
        CUDA_CHECK(cudaDeviceSynchronize());
        t.begin();
        merge_aos_kernel<<<blocks, threads>>>(d_a, d_b, d_out, a_len, b_len, items_per_thread);
        float ms = t.end_ms();

        double data_gb = (double)total * RECORD_SIZE / 1e9;
        printf("Layout 1 — AoS (current):     %7.2f ms  %6.1f GB/s  [reads %dB per compare]\n",
               ms, 2*data_gb/(ms/1000), RECORD_SIZE);

        CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    }

    // ── SoA benchmark ──
    {
        uint8_t *d_ka, *d_kb, *d_va, *d_vb, *d_ok, *d_ov;
        CUDA_CHECK(cudaMalloc(&d_ka, (uint64_t)a_len * KEY_SIZE));
        CUDA_CHECK(cudaMalloc(&d_kb, (uint64_t)b_len * KEY_SIZE));
        CUDA_CHECK(cudaMalloc(&d_va, (uint64_t)a_len * VALUE_SIZE));
        CUDA_CHECK(cudaMalloc(&d_vb, (uint64_t)b_len * VALUE_SIZE));
        CUDA_CHECK(cudaMalloc(&d_ok, (uint64_t)total * KEY_SIZE));
        CUDA_CHECK(cudaMalloc(&d_ov, (uint64_t)total * VALUE_SIZE));
        CUDA_CHECK(cudaMemcpy(d_ka, h_keys_a, (uint64_t)a_len*KEY_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kb, h_keys_b, (uint64_t)b_len*KEY_SIZE, cudaMemcpyHostToDevice));

        merge_soa_kernel<<<blocks, threads>>>(d_ka, d_kb, d_va, d_vb, d_ok, d_ov,
                                               a_len, b_len, items_per_thread);
        CUDA_CHECK(cudaDeviceSynchronize());
        t.begin();
        merge_soa_kernel<<<blocks, threads>>>(d_ka, d_kb, d_va, d_vb, d_ok, d_ov,
                                               a_len, b_len, items_per_thread);
        float ms = t.end_ms();

        double data_gb = (double)total * RECORD_SIZE / 1e9;
        printf("Layout 2 — SoA (keys+vals):   %7.2f ms  %6.1f GB/s  [reads %dB per compare]\n",
               ms, 2*data_gb/(ms/1000), KEY_SIZE);

        CUDA_CHECK(cudaFree(d_ka)); CUDA_CHECK(cudaFree(d_kb));
        CUDA_CHECK(cudaFree(d_va)); CUDA_CHECK(cudaFree(d_vb));
        CUDA_CHECK(cudaFree(d_ok)); CUDA_CHECK(cudaFree(d_ov));
    }

    // ── Key+Index benchmark ──
    {
        uint8_t *d_ka, *d_kb, *d_ok;
        uint32_t *d_indices;
        uint8_t *d_va, *d_vb, *d_ov;
        CUDA_CHECK(cudaMalloc(&d_ka, (uint64_t)a_len * KEY_SIZE));
        CUDA_CHECK(cudaMalloc(&d_kb, (uint64_t)b_len * KEY_SIZE));
        CUDA_CHECK(cudaMalloc(&d_ok, (uint64_t)total * KEY_SIZE));
        CUDA_CHECK(cudaMalloc(&d_indices, (uint64_t)total * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_va, (uint64_t)a_len * VALUE_SIZE));
        CUDA_CHECK(cudaMalloc(&d_vb, (uint64_t)b_len * VALUE_SIZE));
        CUDA_CHECK(cudaMalloc(&d_ov, (uint64_t)total * VALUE_SIZE));
        CUDA_CHECK(cudaMemcpy(d_ka, h_keys_a, (uint64_t)a_len*KEY_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kb, h_keys_b, (uint64_t)b_len*KEY_SIZE, cudaMemcpyHostToDevice));

        int gather_blocks = (total + 255) / 256;

        // Warmup
        merge_keys_only_kernel<<<blocks, threads>>>(d_ka, d_kb, d_ok, d_indices,
                                                     a_len, b_len, items_per_thread);
        gather_values_kernel<<<gather_blocks, 256>>>(d_va, d_vb, d_ov, d_indices, total);
        CUDA_CHECK(cudaDeviceSynchronize());

        t.begin();
        merge_keys_only_kernel<<<blocks, threads>>>(d_ka, d_kb, d_ok, d_indices,
                                                     a_len, b_len, items_per_thread);
        gather_values_kernel<<<gather_blocks, 256>>>(d_va, d_vb, d_ov, d_indices, total);
        float ms = t.end_ms();

        double data_gb = (double)total * RECORD_SIZE / 1e9;
        printf("Layout 3 — Key+Index+Gather:  %7.2f ms  %6.1f GB/s  [merge %dB, gather %dB]\n",
               ms, 2*data_gb/(ms/1000), KEY_SIZE, VALUE_SIZE);

        CUDA_CHECK(cudaFree(d_ka)); CUDA_CHECK(cudaFree(d_kb)); CUDA_CHECK(cudaFree(d_ok));
        CUDA_CHECK(cudaFree(d_indices));
        CUDA_CHECK(cudaFree(d_va)); CUDA_CHECK(cudaFree(d_vb)); CUDA_CHECK(cudaFree(d_ov));
    }

    printf("\nKey insight: AoS wastes 90%% of merge bandwidth reading values\n");
    printf("that aren't used for comparison. SoA reads only keys during\n");
    printf("merge, then gathers values in a separate pass.\n");

    free(h_keys_a);
    free(h_keys_b);
    return 0;
}
