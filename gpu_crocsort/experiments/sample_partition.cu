// sample_partition.cu — Sample-based partitioning for K-way merge
// Build: nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_80
//        -I../include experiments/sample_partition.cu -o sample_partition
// Run:   ./sample_partition

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "record.cuh"

static constexpr int KWAY_K = 8;

struct KWayPartition {
    int      src_rec_start[KWAY_K];
    int      src_rec_count[KWAY_K];
    uint64_t src_byte_off[KWAY_K];
    uint64_t out_byte_offset;
    int      total_records;
};

// ── Phase 1: Sample every S-th key from each run (one block per run) ──

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

// ── Binary search helpers ────────────────────────────────────────

__device__ int lower_bound_run(const uint8_t* run_data, int run_len, SortKey target) {
    int lo = 0, hi = run_len;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (make_sort_key(run_data + (uint64_t)mid * RECORD_SIZE) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ── Phase 4: Binary search each run for each boundary (P blocks, K threads) ──

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

// ── Host orchestrator ────────────────────────────────────────────

void compute_sample_partitions(
    const uint8_t* d_runs,
    const uint64_t* h_run_offsets,
    const int*      h_run_lengths,
    int K, int P,
    uint64_t out_base_offset,
    std::vector<KWayPartition>& out_partitions
) {
    // Total records and compute sampling rate
    uint64_t total = 0;
    int total_samples = 0;
    for (int i = 0; i < K; i++) total += h_run_lengths[i];

    // S chosen so total samples ~= 10 * P
    int target_samples = 10 * P;
    int S = std::max(1, (int)(total / target_samples));

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
    uint64_t* d_run_offsets;
    int* d_run_lengths;
    CUDA_CHECK(cudaMalloc(&d_run_offsets, K * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_run_lengths, K * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_run_offsets, h_run_offsets, K * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_run_lengths, h_run_lengths, K * sizeof(int), cudaMemcpyHostToDevice));

    // Phase 1: Sample keys
    SortKey* d_samples;
    int* d_sample_counts;
    CUDA_CHECK(cudaMalloc(&d_samples, total_samples * sizeof(SortKey)));
    CUDA_CHECK(cudaMalloc(&d_sample_counts, K * sizeof(int)));

    sample_keys_kernel<<<K, 256>>>(
        d_runs, d_run_offsets, d_run_lengths, K, S, d_samples, d_sample_counts);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: Sort samples
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

    // Phase 4: Compute per-source ranges via binary search
    int* d_starts;
    int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_starts, P * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, P * K * sizeof(int)));

    compute_partition_ranges_kernel<<<P, K>>>(
        d_runs, d_run_offsets, d_run_lengths, K,
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
    CUDA_CHECK(cudaFree(d_run_offsets));
    CUDA_CHECK(cudaFree(d_run_lengths));
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_sample_counts));
    CUDA_CHECK(cudaFree(d_boundaries));
    CUDA_CHECK(cudaFree(d_starts));
    CUDA_CHECK(cudaFree(d_counts));
}

// ── Test ─────────────────────────────────────────────────────────

static void fill_sorted_run(uint8_t* buf, int n, uint64_t start, uint64_t step) {
    for (int i = 0; i < n; i++) {
        SortKey sk; sk.hi = start + (uint64_t)i * step; sk.lo = 0;
        write_sort_key(buf + (uint64_t)i * RECORD_SIZE, sk);
        memset(buf + (uint64_t)i * RECORD_SIZE + KEY_SIZE, 0, VALUE_SIZE);
    }
}

int main() {
    printf("=== Sample-based partition test ===\n");
    const int K = 4, P = 16;
    int rlen[K] = {10000, 5000, 20000, 8000};
    uint64_t total = 0;
    for (int i = 0; i < K; i++) total += rlen[i];

    // Skewed: runs 0,1,3 overlap in low range; run 2 in high range
    size_t total_bytes = total * RECORD_SIZE;
    uint8_t* h_data = (uint8_t*)malloc(total_bytes);
    uint64_t off[K]; off[0] = 0;
    for (int i = 1; i < K; i++) off[i] = off[i-1] + (uint64_t)rlen[i-1] * RECORD_SIZE;

    fill_sorted_run(h_data + off[0], rlen[0], 0, 1);
    fill_sorted_run(h_data + off[1], rlen[1], 0, 1);
    fill_sorted_run(h_data + off[2], rlen[2], 1000000, 1);
    fill_sorted_run(h_data + off[3], rlen[3], 0, 1);

    uint8_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice));

    std::vector<KWayPartition> parts;
    compute_sample_partitions(d_data, off, rlen, K, P, 0, parts);

    // Check 1: all records accounted for
    bool pass = true;
    int total_assigned = 0, per_src[K] = {};
    for (auto& kp : parts) {
        total_assigned += kp.total_records;
        for (int k = 0; k < K; k++) per_src[k] += kp.src_rec_count[k];
    }
    if (total_assigned != (int)total) { printf("FAIL: total %d != %d\n", total_assigned, (int)total); pass = false; }
    else printf("OK: total records = %d\n", total_assigned);
    for (int k = 0; k < K; k++)
        if (per_src[k] != rlen[k]) { printf("FAIL: src %d: %d != %d\n", k, per_src[k], rlen[k]); pass = false; }

    // Check 2: per-source ranges contiguous
    for (int k = 0; k < K; k++) {
        int prev = 0;
        for (auto& kp : parts) {
            if (kp.src_rec_start[k] != prev) { printf("FAIL: src %d gap\n", k); pass = false; break; }
            prev = kp.src_rec_start[k] + kp.src_rec_count[k];
        }
        if (prev != rlen[k]) { printf("FAIL: src %d end %d != %d\n", k, prev, rlen[k]); pass = false; }
    }

    // Check 3: boundary ordering (max key in part p <= min key in part p+1)
    for (int p = 0; p < (int)parts.size() - 1; p++) {
        SortKey mx = {0, 0}, mn = {UINT64_MAX, UINT16_MAX};
        for (int k = 0; k < K; k++) {
            if (parts[p].src_rec_count[k] > 0) {
                int last = parts[p].src_rec_start[k] + parts[p].src_rec_count[k] - 1;
                SortKey s = make_sort_key(h_data + off[k] + (uint64_t)last * RECORD_SIZE);
                if (mx < s) mx = s;
            }
            if (parts[p+1].src_rec_count[k] > 0) {
                SortKey s = make_sort_key(h_data + off[k] + (uint64_t)parts[p+1].src_rec_start[k] * RECORD_SIZE);
                if (s < mn) mn = s;
            }
        }
        if (mn < mx) { printf("FAIL: boundary violation at %d/%d\n", p, p+1); pass = false; }
    }

    // Summary
    printf("\nPartitions (K=%d, P=%d, S=%d):\n", K, P, std::max(1,(int)(total/(10*P))));
    for (int p = 0; p < (int)parts.size(); p++) {
        printf("  %2d: %5d [", p, parts[p].total_records);
        for (int k = 0; k < K; k++) printf(" %d:%d", parts[p].src_rec_start[k], parts[p].src_rec_count[k]);
        printf(" ]\n");
    }
    printf("\n%s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    free(h_data); CUDA_CHECK(cudaFree(d_data));
    return pass ? 0 : 1;
}
