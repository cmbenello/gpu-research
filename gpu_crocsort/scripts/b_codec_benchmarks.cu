/**
 * B1/B2/B3: GPU codec microbenchmarks
 *
 * B1: FOR decode throughput on GPU
 * B2: Bit-pack decode throughput on GPU
 * B3: Direct-sort on FOR-encoded keys (radix sort on compressed vs uncompressed)
 *
 * Build: nvcc -O3 -arch=sm_75 -o b_codec_bench scripts/b_codec_benchmarks.cu
 * Run:   ./b_codec_bench
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <cub/cub.cuh>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ═══════════════════════════════════════════════════════════════
// FOR decode kernel: out[i] = in[i] + min_val
// ═══════════════════════════════════════════════════════════════
__global__ void for_decode_u32(const uint32_t* __restrict__ encoded,
                                uint32_t* __restrict__ decoded,
                                uint32_t min_val, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) decoded[i] = encoded[i] + min_val;
}

__global__ void for_decode_u16(const uint16_t* __restrict__ encoded,
                                uint32_t* __restrict__ decoded,
                                uint32_t min_val, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) decoded[i] = (uint32_t)encoded[i] + min_val;
}

__global__ void for_decode_u8(const uint8_t* __restrict__ encoded,
                               uint32_t* __restrict__ decoded,
                               uint32_t min_val, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) decoded[i] = (uint32_t)encoded[i] + min_val;
}

// ═══════════════════════════════════════════════════════════════
// Bit-pack decode kernel: extract bit_width bits per value
// ═══════════════════════════════════════════════════════════════
__global__ void bitpack_decode(const uint32_t* __restrict__ packed,
                                uint32_t* __restrict__ decoded,
                                int bit_width, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        size_t bit_offset = (size_t)i * bit_width;
        size_t word_idx = bit_offset / 32;
        int bit_pos = (int)(bit_offset % 32);
        uint32_t mask = (1U << bit_width) - 1;

        uint32_t val = (packed[word_idx] >> bit_pos) & mask;
        // Handle crossing a word boundary
        if (bit_pos + bit_width > 32) {
            int remaining = bit_pos + bit_width - 32;
            val |= (packed[word_idx + 1] & ((1U << remaining) - 1)) << (bit_width - remaining);
        }
        decoded[i] = val;
    }
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════
struct Timer {
    cudaEvent_t start, stop;
    Timer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

void generate_data_u32(uint32_t* h_data, size_t n, uint32_t range) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, range - 1);
    for (size_t i = 0; i < n; i++) h_data[i] = dist(rng);
}

void run_b1_for_decode(FILE* csv) {
    printf("\n=== B1: FOR Decode Throughput ===\n");
    fprintf(csv, "experiment,N,width_bytes,time_ms,throughput_GBs\n");

    size_t sizes[] = {100000000, 300000000};
    int widths[] = {1, 2, 4};  // u8, u16, u32

    for (size_t N : sizes) {
        // Generate source data
        uint32_t* h_data = new uint32_t[N];
        generate_data_u32(h_data, N, 1000000);  // range < 2^20

        uint32_t* d_decoded;
        CUDA_CHECK(cudaMalloc(&d_decoded, N * sizeof(uint32_t)));

        for (int w : widths) {
            void* d_encoded;
            size_t encoded_bytes = N * w;
            CUDA_CHECK(cudaMalloc(&d_encoded, encoded_bytes));

            // Pack to narrower width on host and upload
            if (w == 4) {
                CUDA_CHECK(cudaMemcpy(d_encoded, h_data, encoded_bytes, cudaMemcpyHostToDevice));
            } else if (w == 2) {
                uint16_t* h_enc = new uint16_t[N];
                for (size_t i = 0; i < N; i++) h_enc[i] = (uint16_t)(h_data[i] & 0xFFFF);
                CUDA_CHECK(cudaMemcpy(d_encoded, h_enc, encoded_bytes, cudaMemcpyHostToDevice));
                delete[] h_enc;
            } else {
                uint8_t* h_enc = new uint8_t[N];
                for (size_t i = 0; i < N; i++) h_enc[i] = (uint8_t)(h_data[i] & 0xFF);
                CUDA_CHECK(cudaMemcpy(d_encoded, h_enc, encoded_bytes, cudaMemcpyHostToDevice));
                delete[] h_enc;
            }

            int nthreads = 256;
            int nblocks = (int)((N + nthreads - 1) / nthreads);

            // Warmup
            if (w == 4) for_decode_u32<<<nblocks, nthreads>>>((uint32_t*)d_encoded, d_decoded, 1000, N);
            else if (w == 2) for_decode_u16<<<nblocks, nthreads>>>((uint16_t*)d_encoded, d_decoded, 1000, N);
            else for_decode_u8<<<nblocks, nthreads>>>((uint8_t*)d_encoded, d_decoded, 1000, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Benchmark (5 runs)
            Timer t;
            float total_ms = 0;
            int runs = 5;
            for (int r = 0; r < runs; r++) {
                t.begin();
                if (w == 4) for_decode_u32<<<nblocks, nthreads>>>((uint32_t*)d_encoded, d_decoded, 1000, N);
                else if (w == 2) for_decode_u16<<<nblocks, nthreads>>>((uint16_t*)d_encoded, d_decoded, 1000, N);
                else for_decode_u8<<<nblocks, nthreads>>>((uint8_t*)d_encoded, d_decoded, 1000, N);
                total_ms += t.end_ms();
            }
            float avg_ms = total_ms / runs;
            // Throughput: input (encoded) + output (decoded) bytes
            double total_bytes = (double)encoded_bytes + N * sizeof(uint32_t);
            double throughput = total_bytes / (avg_ms / 1000.0) / 1e9;

            printf("  N=%zuM, width=%dB: %.2f ms (%.1f GB/s)\n", N/1000000, w, avg_ms, throughput);
            fprintf(csv, "B1_FOR_decode,%zu,%d,%.2f,%.2f\n", N, w, avg_ms, throughput);

            cudaFree(d_encoded);
        }
        cudaFree(d_decoded);
        delete[] h_data;
    }
}

void run_b2_bitpack_decode(FILE* csv) {
    printf("\n=== B2: Bit-Pack Decode Throughput ===\n");

    size_t N = 100000000;
    uint32_t* h_data = new uint32_t[N];
    generate_data_u32(h_data, N, 1 << 20);

    uint32_t* d_decoded;
    CUDA_CHECK(cudaMalloc(&d_decoded, N * sizeof(uint32_t)));

    int bit_widths[] = {8, 12, 16, 20, 24, 32};

    for (int bw : bit_widths) {
        // Pack bits on host
        size_t total_bits = (size_t)N * bw;
        size_t packed_words = (total_bits + 31) / 32;
        uint32_t* h_packed = (uint32_t*)calloc(packed_words + 1, sizeof(uint32_t));
        uint32_t mask = (bw == 32) ? 0xFFFFFFFF : ((1U << bw) - 1);
        for (size_t i = 0; i < N; i++) {
            size_t bit_offset = i * bw;
            size_t word_idx = bit_offset / 32;
            int bit_pos = (int)(bit_offset % 32);
            uint32_t val = h_data[i] & mask;
            h_packed[word_idx] |= val << bit_pos;
            if (bit_pos + bw > 32)
                h_packed[word_idx + 1] |= val >> (32 - bit_pos);
        }

        uint32_t* d_packed;
        size_t packed_bytes = (packed_words + 1) * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
        CUDA_CHECK(cudaMemcpy(d_packed, h_packed, packed_bytes, cudaMemcpyHostToDevice));

        int nthreads = 256;
        int nblocks = (int)((N + nthreads - 1) / nthreads);

        // Warmup
        bitpack_decode<<<nblocks, nthreads>>>(d_packed, d_decoded, bw, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        Timer t;
        float total_ms = 0;
        int runs = 5;
        for (int r = 0; r < runs; r++) {
            t.begin();
            bitpack_decode<<<nblocks, nthreads>>>(d_packed, d_decoded, bw, N);
            total_ms += t.end_ms();
        }
        float avg_ms = total_ms / runs;
        double input_gb = packed_bytes / 1e9;
        double output_gb = N * sizeof(uint32_t) / 1e9;
        double throughput = (input_gb + output_gb) / (avg_ms / 1000.0);

        printf("  %d-bit: %.2f ms (%.1f GB/s, input %.2f GB)\n", bw, avg_ms, throughput, input_gb);
        fprintf(csv, "B2_bitpack_decode,%zu,%d,%.2f,%.2f\n", N, bw, avg_ms, throughput);

        cudaFree(d_packed);
        free(h_packed);
    }

    cudaFree(d_decoded);
    delete[] h_data;
}

void run_b3_direct_sort(FILE* csv) {
    printf("\n=== B3: Direct Sort on FOR-Encoded Keys ===\n");

    size_t N = 100000000;

    // Generate data with known range
    uint32_t* h_data = new uint32_t[N];
    generate_data_u32(h_data, N, 1 << 20);  // 20-bit range

    // Test: sort full 32-bit keys vs FOR-encoded narrower keys
    struct TestCase {
        const char* name;
        int key_bits;
    };
    TestCase cases[] = {
        {"raw_32bit", 32},
        {"FOR_24bit", 24},
        {"FOR_20bit", 20},
        {"FOR_16bit", 16},
        {"FOR_8bit",  8},
    };

    for (auto& tc : cases) {
        uint32_t* d_keys, *d_keys_alt;
        uint32_t* d_vals, *d_vals_alt;
        CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_alt, N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_vals, N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_vals_alt, N * sizeof(uint32_t)));

        // FOR encode: mask to key_bits
        uint32_t mask = (tc.key_bits == 32) ? 0xFFFFFFFF : ((1U << tc.key_bits) - 1);
        uint32_t* h_encoded = new uint32_t[N];
        for (size_t i = 0; i < N; i++) h_encoded[i] = h_data[i] & mask;

        CUDA_CHECK(cudaMemcpy(d_keys, h_encoded, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // CUB radix sort — only sort key_bits bits
        size_t temp_bytes = 0;
        cub::DoubleBuffer<uint32_t> kb(d_keys, d_keys_alt);
        cub::DoubleBuffer<uint32_t> vb(d_vals, d_vals_alt);
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, kb, vb, (int)N, 0, tc.key_bits);
        void* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

        // Warmup
        CUDA_CHECK(cudaMemcpy(d_keys, h_encoded, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        kb = cub::DoubleBuffer<uint32_t>(d_keys, d_keys_alt);
        vb = cub::DoubleBuffer<uint32_t>(d_vals, d_vals_alt);
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, kb, vb, (int)N, 0, tc.key_bits);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        Timer t;
        float total_ms = 0;
        int runs = 5;
        for (int r = 0; r < runs; r++) {
            CUDA_CHECK(cudaMemcpy(d_keys, h_encoded, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
            kb = cub::DoubleBuffer<uint32_t>(d_keys, d_keys_alt);
            vb = cub::DoubleBuffer<uint32_t>(d_vals, d_vals_alt);
            t.begin();
            cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, kb, vb, (int)N, 0, tc.key_bits);
            total_ms += t.end_ms();
        }
        float avg_ms = total_ms / runs;
        double throughput = N * sizeof(uint32_t) * 2.0 / (avg_ms / 1000.0) / 1e9;

        // Verify sort correctness
        uint32_t* h_sorted = new uint32_t[N];
        CUDA_CHECK(cudaMemcpy(h_sorted, kb.Current(), N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        bool correct = true;
        for (size_t i = 1; i < N; i++) {
            if (h_sorted[i] < h_sorted[i-1]) { correct = false; break; }
        }

        printf("  %s: %.2f ms (%.1f GB/s) %s\n",
               tc.name, avg_ms, throughput, correct ? "PASS" : "FAIL");
        fprintf(csv, "B3_direct_sort,%zu,%d,%.2f,%.2f,%s\n",
                N, tc.key_bits, avg_ms, throughput, correct ? "PASS" : "FAIL");

        delete[] h_sorted;
        delete[] h_encoded;
        cudaFree(d_keys); cudaFree(d_keys_alt);
        cudaFree(d_vals); cudaFree(d_vals_alt);
        cudaFree(d_temp);
    }

    delete[] h_data;
}

int main() {
    printf("GPU Codec Microbenchmarks\n");
    printf("========================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.1f GB)\n", prop.name, prop.totalGlobalMem / 1e9);

    FILE* csv_b1 = fopen("results/overnight/b1_for_decode.csv", "w");
    FILE* csv_b2 = fopen("results/overnight/b2_bitpack_decode.csv", "w");
    FILE* csv_b3 = fopen("results/overnight/b3_direct_sort.csv", "w");

    if (!csv_b1 || !csv_b2 || !csv_b3) {
        fprintf(stderr, "Failed to open output CSVs\n");
        return 1;
    }

    run_b1_for_decode(csv_b1);
    run_b2_bitpack_decode(csv_b2);
    run_b3_direct_sort(csv_b3);

    fclose(csv_b1);
    fclose(csv_b2);
    fclose(csv_b3);

    printf("\nAll codec benchmarks complete.\n");
    return 0;
}
