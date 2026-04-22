// ============================================================================
// GPU External Merge Sort — Triple-Buffered Pipeline + K-Way Streaming Merge
//
// Architecture (Stehle-Jacobsen SIGMOD 2017 pipeline model):
//   Phase 1: Triple-buffered run generation
//     Stream 0: H2D upload chunk i+2
//     Stream 1: GPU sort chunk i+1
//     Stream 2: D2H download chunk i
//     → GPU compute completely hidden behind PCIe transfer
//
//   Phase 2: GPU streaming K-way merge (novel)
//     Stream chunks from K sorted runs through GPU, merge on GPU, stream back.
//     K-way single pass instead of log2(K) cascade passes → K× less PCIe traffic.
//     Cursor-based with boundary detection: only output records guaranteed correct.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_75 -DEXTERNAL_SORT_MAIN -Iinclude \
//        src/external_sort.cu src/run_generation.cu src/merge.cu -o external_sort
// Run:   ./external_sort --total-gb 20
// ============================================================================

#include "record.cuh"
#include "ovc.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <chrono>
#include <thread>
#include <array>
#include <atomic>
#include <mutex>
#include <cub/cub.cuh>
#include <sys/mman.h>  // madvise for huge pages
#include <fcntl.h>     // open
#include <unistd.h>    // close
#include <emmintrin.h> // _mm_stream_si64, _mm_sfence

// Forward-declare the in-HBM sort from host_sort.cu
enum MergeStrategy { STRATEGY_2WAY, STRATEGY_KWAY };
void gpu_crocsort_in_hbm(uint8_t* d_data, uint64_t num_records,
                           bool verify, MergeStrategy strategy);

// Forward-declare kernel launchers
struct PairDesc2Way {
    uint64_t a_byte_offset; int a_count;
    uint64_t b_byte_offset; int b_count;
    uint64_t out_byte_offset; int first_block;
};
extern "C" void launch_run_generation(
    const uint8_t*, uint8_t*, uint32_t*, uint64_t,
    SparseEntry*, int*, int, cudaStream_t);
extern "C" void launch_merge_2way(
    const uint8_t*, uint8_t*, const PairDesc2Way*, int, int, cudaStream_t);

// K-way merge tree (from merge.cu + host_sort.cu)
static constexpr int KWAY_K = 8;
struct KWayPartition {
    int      src_rec_start[KWAY_K];
    int      src_rec_count[KWAY_K];
    uint64_t src_byte_off[KWAY_K];
    uint64_t out_byte_offset;
    int      total_records;
};
extern "C" void launch_merge_kway(
    const uint8_t*, uint8_t*, const KWayPartition*, int, int, cudaStream_t);

// Run struct matching host_sort.cu's Run (identical layout)
struct Run { uint64_t byte_offset; uint64_t num_records; };

// Sample-based partitioning (from host_sort.cu, made non-static)
void compute_sample_partitions(
    const uint8_t* d_runs, const std::vector<Run>& group_runs,
    int K, int P, uint64_t out_base_offset,
    std::vector<KWayPartition>& out_partitions);

// ── Timing ──────────────────────────────────────────────────────────

struct WallTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void begin() { t0 = std::chrono::high_resolution_clock::now(); }
    double end_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ── Host-side upper_bound for boundary computation ──────────────────

static uint64_t host_upper_bound(const uint8_t* data, uint64_t n, const uint8_t* target_key) {
    uint64_t lo = 0, hi = n;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (key_compare(data + mid * RECORD_SIZE, target_key, KEY_SIZE) <= 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ── GPU kernels for CUB radix sort pipeline ─────────────────────────

// Extract 8-byte sort key (big-endian) from each record for CUB radix sort
__global__ void extract_sort_keys_kernel(
    const uint8_t* __restrict__ records,
    uint64_t* __restrict__ keys,
    uint32_t* __restrict__ indices,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* rec = records + i * RECORD_SIZE;
    // Big-endian 8-byte key for correct lexicographic ordering
    uint64_t k = 0;
    for (int b = 0; b < 8; b++) k = (k << 8) | rec[b];
    keys[i] = k;
    indices[i] = (uint32_t)i;
}

// Reorder records from src to dst using sorted index array
__global__ void reorder_records_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const uint32_t* __restrict__ sorted_indices,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t src_idx = sorted_indices[i];
    const uint8_t* s = src + (uint64_t)src_idx * RECORD_SIZE;
    uint8_t* d = dst + i * RECORD_SIZE;
    // Copy 100-byte record (25 × uint32)
    for (int b = 0; b < RECORD_SIZE; b += 4) {
        *reinterpret_cast<uint32_t*>(d + b) = *reinterpret_cast<const uint32_t*>(s + b);
    }
}

// Initialize uint32 array to identity [0, 1, 2, ..., N-1]
__global__ void init_identity_kernel(uint32_t* __restrict__ arr, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = (uint32_t)i;
}

// Extract uint32 sort key (top 4 bytes, big-endian) from KEY_SIZE-stride buffer
// For random 10B keys, 4 bytes covers 2^32 values vs 600M records — ~0 ties
__global__ void extract_uint32_from_keys_kernel(
    const uint8_t* __restrict__ key_buffer,
    uint32_t* __restrict__ sort_keys,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* k = key_buffer + i * KEY_SIZE;
    uint32_t v = ((uint32_t)k[0] << 24) | ((uint32_t)k[1] << 16) |
                 ((uint32_t)k[2] << 8) | (uint32_t)k[3];
    sort_keys[i] = v;
}

// Gather uint64 values by permutation: out[i] = src[perm[i]]
__global__ void gather_uint64_kernel(
    const uint64_t* __restrict__ src,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ out,
    uint64_t n
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[perm[i]];
}

// Extract uint64 from a specific 8-byte chunk of the key, in permutation order.
// Reads key[perm[i]][byte_offset : byte_offset + chunk_bytes] as big-endian uint64.
__global__ void extract_uint64_chunk_kernel(
    const uint8_t* __restrict__ key_buffer,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes  // 1-8
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = key_buffer + (uint64_t)orig_idx * KEY_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    // Left-align: shift so MSB of chunk is in MSB of uint64
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}

// ── Compact key for TPC-H: strip constant bytes, keep only varying ones ───
// The 66B TPC-H key has ~40 constant bytes and only 26 varying bytes.
// Packing the varying bytes into a 32B compact key reduces LSD from 9 to 4 passes.
//
// Compile with -DUSE_COMPACT_KEY to enable. The byte mapping is:
//   compact[0]  = record[0]   returnflag
//   compact[1]  = record[1]   linestatus
//   compact[2]  = record[4]   shipdate[2]
//   compact[3]  = record[5]   shipdate[3]
//   compact[4]  = record[8]   commitdate[2]
//   compact[5]  = record[9]   commitdate[3]
//   compact[6]  = record[12]  receiptdate[2]
//   compact[7]  = record[13]  receiptdate[3]
//   compact[8]  = record[19]  extprice[5]
//   compact[9]  = record[20]  extprice[6]
//   compact[10] = record[21]  extprice[7]
//   compact[11] = record[29]  discount[7]
//   compact[12] = record[37]  tax[7]
//   compact[13] = record[44]  quantity[6]
//   compact[14] = record[45]  quantity[7]
//   compact[15] = record[51]  orderkey[5]
//   compact[16] = record[52]  orderkey[6]
//   compact[17] = record[53]  orderkey[7]
//   compact[18-21] = record[54-57]  partkey
//   compact[22] = record[59]  suppkey[1]
//   compact[23] = record[60]  suppkey[2]
//   compact[24] = record[61]  suppkey[3]
//   compact[25] = record[65]  linenumber[3]
//   compact[26-31] = 0 (padding to 32B)

#ifdef USE_COMPACT_KEY
static constexpr int COMPACT_KEY_SIZE = 32;
// Runtime-detected compact map (no hardcoded TPC-H values).
// Host-side: filled by detect_compact_map() at sort entry.
// Device-side: a plain cudaMalloc'd buffer, NOT __constant__ memory.
// We deliberately avoid cudaMemcpyToSymbol on a __constant__ symbol because
// that forces eager CUDA module loading, which triggers a PTX→sm_75 JIT
// failure on Turing ("no kernel image available"). Using cudaMalloc +
// cudaMemcpy + pass-by-pointer keeps module loading lazy so only the kernels
// we actually launch get JIT'd.
static int g_compact_map[64];
static int g_compact_count = 0;
static int* d_compact_map_ptr = nullptr;   // device buffer, size 64*sizeof(int)
// Per-byte observed min/max from detection (full KEY_SIZE indexed).
// For bytes NOT in the compact map, g_sample_min[b] == g_sample_max[b] is the
// "expected constant value" V[b] that extraction verifies against every record.
static uint8_t g_sample_min[KEY_SIZE];
static uint8_t g_sample_max[KEY_SIZE];
// Set to the first (lowest) byte position where extraction found variation the
// sample missed, or -1 if none. Primarily diagnostic now — the hybrid
// canonical/exception fallback below handles the correctness side.
static std::atomic<int> g_violated_byte{-1};

// Hybrid exception tracking. During CPU extraction each record is classified
// as canonical (all non-mapped bytes == V[b]) or exception. Exceptions are
// accumulated into a global list (indexed by original record index) and after
// the GPU canonical sort we CPU-sort the exceptions by full key and merge-in
// at the permutation level. The GPU never sees exceptions' true keys, but
// canonicals' compact order == full-key order by construction, so the merge
// is correct.
//
// Global accumulator + mutex. Exceptions are expected to be rare so simple
// locking is fine; if we ever see >1% exception rate the merge becomes the
// bottleneck anyway.
static std::vector<uint64_t> g_exception_indices;  // original record indices
static std::mutex g_exception_mtx;

// Hybrid fallback flag. When the compact path detects exceptions we rerun
// sort() from the caller with this set, which takes the standard full-record
// upload path instead. That path uploads full 66-byte keys to the GPU, so no
// sampling-based assumption can affect correctness.
static bool g_force_no_compact = false;
static bool g_disable_compact = false;
// Streaming mode: use MAP_NORESERVE for output buffers to reduce peak RAM.
// Set by main() when --streaming is used or auto-detected.
static bool g_streaming_mode = false;
// true if we have verified against full data (sample was confirmed complete)
static bool g_map_full_scan_verified = false;

// TODO: measure the contribution of each stage independently:
//   (1) how much does compact-key compression alone shrink the key / speed up sort?
//   (2) how much additional compression/benefit does OVC give on top of that?
// Want per-stage numbers (key bytes, PCIe bytes, CUB pass count, wall time) so we
// can tell whether OVC is still pulling weight once the key is already compact.
//
// TODO: design a compression method specifically tailored to OVC.
// Current approach (byte-position varies-or-not) is order-preserving but coarse —
// keeps a whole byte if any bit varies. OVC compares adjacent sorted records and
// encodes the first differing position, so an OVC-aware compression could:
//   - use per-byte value ranges (min/max from sample) to pack at bit granularity
//   - detect correlated byte groups that always co-vary (encode jointly)
//   - exploit that OVC needs order-preserving equivalence, not exact bytes
// Goal: shrink compact key below 32B (fewer CUB passes) while staying OVC-compatible.
//
// Detect which byte positions vary across a SAMPLE of records.
//
// Strategy: stratified random sampling. Divide the data into N buckets of size
// stride; pick one random index from each bucket. Combines stride coverage
// (sees all regions) with random offset (avoids periodic-alignment pathology).
//
// Probabilistic: a byte that varies in only K records (K << total) has
// (K/total)^N probability of being missed at sample size N. For 1M sample
// of 600M records, missing a byte that varies in 1% of records has prob
// (0.99)^1M ≈ 0. Missing one that varies in 0.0001% (60 records) has prob
// (0.9999999)^1M ≈ 90%. So sub-rare variations CAN be missed.
//
// Tunable via COMPACT_SAMPLE env var. Future work: characterize miss
// probability vs. sample size on real datasets, compare strategies:
//   - pure stride        (cheap, vulnerable to alignment)
//   - stratified random  (current default — balanced)
//   - reservoir sampling (uniform but more compute)
//   - full scan          (deterministic, ~1-2s for 72GB)
// Also returns, for every KEY_SIZE byte position:
//   mn_out[b], mx_out[b] — min/max observed in the sample.
// For bytes NOT in the compact map, mn==mx==V[b] is the "expected constant value"
// the caller will later verify against every record in a full scan.
static int detect_compact_map(const uint8_t* h_data, uint64_t num_records,
                               int* h_map_out,
                               uint8_t* mn_out = nullptr,
                               uint8_t* mx_out = nullptr) {
    uint64_t sample_target = 1000000;
    if (const char* e = getenv("COMPACT_SAMPLE")) {
        long v = atol(e);
        if (v > 0) sample_target = (uint64_t)v;
    }
    // Full scan for datasets under 50M records (~6 GB). Sampling 1M from 19M
    // can miss rare byte variations (e.g. timestamp high bytes that barely
    // change over a few months). Full scan adds ~30-50ms — negligible.
    if (num_records <= 50000000) sample_target = num_records;
    uint64_t sample_n = std::min(num_records, sample_target);
    uint64_t stride = std::max((uint64_t)1, num_records / sample_n);
    uint64_t n_buckets = (num_records + stride - 1) / stride;

    uint64_t seed0 = 0x12345678ULL;
    if (const char* e = getenv("COMPACT_SEED")) seed0 = (uint64_t)atoll(e);

    int nthreads = std::max(1u, std::thread::hardware_concurrency());
    if (const char* e = getenv("COMPACT_THREADS")) {
        long v = atol(e);
        if (v > 0) nthreads = (int)v;
    }
    if ((uint64_t)nthreads > n_buckets) nthreads = (int)n_buckets;
    if (nthreads < 1) nthreads = 1;

    // Per-thread min/max + per-byte distinct-value bitmap (256 bits = 32 bytes
    // per byte position). Distinct count = popcount, used as an entropy proxy
    // for byte selection when the candidate count exceeds the compact map
    // capacity (COMPACT_KEY_SIZE = 32). High-distinct-count bytes are
    // discriminating; low-count bytes (constants or near-constants) are not.
    std::vector<std::array<uint8_t, KEY_SIZE>> tmn(nthreads), tmx(nthreads);
    // 32 B per byte position per thread = ~2 KB per thread for KEY_SIZE=66.
    std::vector<std::array<std::array<uint32_t, 8>, KEY_SIZE>> tdistinct(nthreads);
    for (int t = 0; t < nthreads; t++) {
        for (int b = 0; b < KEY_SIZE; b++) {
            tmn[t][b] = 0xff; tmx[t][b] = 0;
            for (int w = 0; w < 8; w++) tdistinct[t][b][w] = 0;
        }
    }

    auto worker = [&](int tid, uint64_t bucket_lo, uint64_t bucket_hi) {
        uint64_t seed = seed0 ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1));
        uint8_t* mn = tmn[tid].data();
        uint8_t* mx = tmx[tid].data();
        auto& dis = tdistinct[tid];
        for (uint64_t bucket = bucket_lo; bucket < bucket_hi; bucket++) {
            seed = seed * 1664525ULL + 1013904223ULL;
            uint64_t b_lo = bucket * stride;
            uint64_t b_hi = std::min(b_lo + stride, num_records);
            uint64_t off = (b_hi > b_lo) ? (seed % (b_hi - b_lo)) : 0;
            uint64_t i = b_lo + off;
            const uint8_t* rec = h_data + i * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++) {
                uint8_t v = rec[b];
                if (v < mn[b]) mn[b] = v;
                if (v > mx[b]) mx[b] = v;
                dis[b][v >> 5] |= (1u << (v & 31));
            }
        }
    };

    std::vector<std::thread> threads;
    uint64_t per = (n_buckets + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; t++) {
        uint64_t lo = (uint64_t)t * per;
        uint64_t hi = std::min(lo + per, n_buckets);
        if (lo < hi) threads.emplace_back(worker, t, lo, hi);
    }
    for (auto& th : threads) th.join();

    uint8_t mn[KEY_SIZE], mx[KEY_SIZE];
    uint32_t merged_distinct[KEY_SIZE][8] = {{0}};
    for (int b = 0; b < KEY_SIZE; b++) { mn[b] = 0xff; mx[b] = 0; }
    for (int t = 0; t < nthreads; t++) {
        for (int b = 0; b < KEY_SIZE; b++) {
            if (tmn[t][b] < mn[b]) mn[b] = tmn[t][b];
            if (tmx[t][b] > mx[b]) mx[b] = tmx[t][b];
            for (int w = 0; w < 8; w++) merged_distinct[b][w] |= tdistinct[t][b][w];
        }
    }
    // Compute per-byte distinct count via popcount.
    int dc[KEY_SIZE];
    for (int b = 0; b < KEY_SIZE; b++) {
        int cnt = 0;
        for (int w = 0; w < 8; w++) cnt += __builtin_popcount(merged_distinct[b][w]);
        dc[b] = cnt;
    }

    // Build candidate list (bytes where min != max).
    int candidates[KEY_SIZE]; int ncand = 0;
    for (int b = 0; b < KEY_SIZE; b++) if (mn[b] != mx[b]) candidates[ncand++] = b;

    // SELECTION POLICY:
    //   "position" (DEFAULT, correctness-correct): take first 64 candidates by source position.
    //   "entropy"  (COMPACT_SELECT=entropy): rank candidates by distinct count desc, top N.
    //
    // ⚠ BUG (overnight 2026-04-15): entropy selection produces INCORRECT sort
    // order. If two records differ at byte B (NOT in top-32) AND byte P > B
    // (IN top-32), compact compares at P first and disagrees with full-key lex
    // which compares at B first. The only way compact-key order matches
    // full-key lex order on canonicals is if compact bytes form a SOURCE-
    // POSITION-ORDERED PREFIX of the varying bytes. Entropy mode violates this
    // and should NOT be used for correctness-sensitive sort. Retained as
    // COMPACT_SELECT=entropy for the paper's negative-result section only.
    bool entropy_select = false;
    if (const char* e = getenv("COMPACT_SELECT")) {
        if (std::string(e) == "entropy") entropy_select = true;
        else if (std::string(e) == "position") entropy_select = false;
    }

    // Entropy selection rule: when there are more candidates than the compact
    // KEY can hold (COMPACT_KEY_SIZE=32), pick the top 32 by distinct-value
    // count (entropy proxy). Sort those 32 by source byte position so the
    // compact-key lex order still matches a sub-projection of the full-key
    // lex order. The REMAINING candidates go into map[32..63] (also sorted by
    // position) so verification + fixup logic still knows about every varying
    // byte. The compact KEY is built from map[0..32] (caller caps at
    // COMPACT_KEY_SIZE), so high-entropy bytes land in the GPU prefix.
    if (entropy_select && ncand > COMPACT_KEY_SIZE) {
        std::vector<std::pair<int, int>> pairs(ncand);   // (-distinct, byte_pos)
        for (int i = 0; i < ncand; i++) pairs[i] = {-dc[candidates[i]], candidates[i]};
        std::sort(pairs.begin(), pairs.end());
        // Top 32 by entropy → mark as "in compact key".
        std::vector<int> top32; top32.reserve(COMPACT_KEY_SIZE);
        std::vector<int> rest;  rest.reserve(ncand - COMPACT_KEY_SIZE);
        for (int i = 0; i < ncand; i++) {
            if (i < COMPACT_KEY_SIZE) top32.push_back(pairs[i].second);
            else                       rest.push_back(pairs[i].second);
        }
        std::sort(top32.begin(), top32.end());
        std::sort(rest.begin(), rest.end());
        // Place top32 first (these will be the compact key bytes), then rest.
        int n_total = std::min(64, ncand);
        for (int i = 0; i < (int)top32.size() && i < 64; i++) h_map_out[i] = top32[i];
        int off = (int)top32.size();
        for (int i = 0; i < (int)rest.size() && off + i < 64; i++) h_map_out[off + i] = rest[i];

        if (mn_out) memcpy(mn_out, mn, KEY_SIZE);
        if (mx_out) memcpy(mx_out, mx, KEY_SIZE);

        printf("[CompactDetect] ENTROPY selection: %d candidates, top %zu by distinct count "
               "go in compact KEY (positions:", ncand, top32.size());
        for (int p : top32) printf(" %d", p);
        printf(")\n");
        return n_total;
    }

    // Default: position-order selection (existing behavior).
    int count = 0;
    for (int b = 0; b < KEY_SIZE; b++) {
        if (mn[b] != mx[b]) {
            if (count < 64) h_map_out[count] = b;
            count++;
        }
    }
    if (mn_out) memcpy(mn_out, mn, KEY_SIZE);
    if (mx_out) memcpy(mx_out, mx, KEY_SIZE);
    return count;
}

// Extract compact-key bytes [base..base+8) and [base+8..base+16) from records
// in current perm order. Used for the 32B-extension fast path: after the 16B
// merge, we need pfx3 (compact bytes 16..23) and pfx4 (compact bytes 24..31)
// already in 16B-sorted order so we can do segmented sort within tied groups.
//
// out_lo[i] = uint64 from compact bytes [base..base+8) of records[perm[i]]
// out_hi[i] = uint64 from compact bytes [base+8..base+16)
__global__ void build_compact_keys_kernel(
    const uint8_t* __restrict__ records,
    uint8_t* __restrict__ compact_keys,
    uint64_t num_records,
    const int* __restrict__ cmap,
    int cmap_n
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* rec = records + i * RECORD_SIZE;
    uint8_t* ck = compact_keys + i * COMPACT_KEY_SIZE;
    int n = cmap_n < COMPACT_KEY_SIZE ? cmap_n : COMPACT_KEY_SIZE;
    for (int b = 0; b < n; b++) ck[b] = rec[cmap[b]];
    for (int b = n; b < COMPACT_KEY_SIZE; b++) ck[b] = 0;
}

// Extract uint64 from compact key buffer at COMPACT_KEY_SIZE stride
__global__ void extract_uint64_from_compact_kernel(
    const uint8_t* __restrict__ compact_keys,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = compact_keys + (uint64_t)orig_idx * COMPACT_KEY_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}
// Extract compact key prefix directly from original records using the byte map.
// No separate compact key buffer needed — reads cmap positions from records.
// Gives first min(32, cmap_n) varying bytes — much higher entropy than raw record bytes 0-31.
// prefix3/prefix4 may be null; when non-null, we also emit compact bytes 16-23 and 24-31
// so the GPU merge can do 4-pass LSD on 32B (native 32B OVC path).
__global__ void extract_compact_prefix_from_records_kernel(
    const uint8_t* __restrict__ records,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ prefix1,   // compact positions 0-7
    uint64_t* __restrict__ prefix2,   // compact positions 8-15
    uint64_t* __restrict__ prefix3,   // compact positions 16-23 (may be null)
    uint64_t* __restrict__ prefix4,   // compact positions 24-31 (may be null)
    uint64_t num_records,
    const int* __restrict__ cmap,
    int cmap_n
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t idx = perm[i];
    const uint8_t* rec = records + (uint64_t)idx * RECORD_SIZE;
    uint64_t v1 = 0;
    for (int b = 0; b < 8; b++) v1 = (v1 << 8) | (b < cmap_n ? rec[cmap[b]] : 0);
    prefix1[i] = v1;
    uint64_t v2 = 0;
    for (int b = 8; b < 16; b++) v2 = (v2 << 8) | (b < cmap_n ? rec[cmap[b]] : 0);
    prefix2[i] = v2;
    if (prefix3 != nullptr) {
        uint64_t v3 = 0;
        for (int b = 16; b < 24; b++) v3 = (v3 << 8) | (b < cmap_n ? rec[cmap[b]] : 0);
        prefix3[i] = v3;
    }
    if (prefix4 != nullptr) {
        uint64_t v4 = 0;
        for (int b = 24; b < 32; b++) v4 = (v4 << 8) | (b < cmap_n ? rec[cmap[b]] : 0);
        prefix4[i] = v4;
    }
}
// Extract compact key prefix from compact key buffer (not from full records).
// Used when compact keys were pre-extracted on CPU and uploaded directly.
// prefix3/prefix4 may be null; when non-null, reads compact bytes 16-23 and 24-31.
__global__ void extract_prefix_from_compact_buffer_kernel(
    const uint8_t* __restrict__ compact_keys,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ prefix1,
    uint64_t* __restrict__ prefix2,
    uint64_t* __restrict__ prefix3,   // may be null
    uint64_t* __restrict__ prefix4,   // may be null
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t idx = perm[i];
    const uint8_t* ck = compact_keys + (uint64_t)idx * COMPACT_KEY_SIZE;
    uint64_t v1 = 0;
    for (int b = 0; b < 8; b++) v1 = (v1 << 8) | ck[b];
    prefix1[i] = v1;
    uint64_t v2 = 0;
    for (int b = 8; b < 16; b++) v2 = (v2 << 8) | ck[b];
    prefix2[i] = v2;
    if (prefix3 != nullptr) {
        uint64_t v3 = 0;
        for (int b = 16; b < 24; b++) v3 = (v3 << 8) | ck[b];
        prefix3[i] = v3;
    }
    if (prefix4 != nullptr) {
        uint64_t v4 = 0;
        for (int b = 24; b < 32; b++) v4 = (v4 << 8) | ck[b];
        prefix4[i] = v4;
    }
}
#endif // USE_COMPACT_KEY

// ── OVC Architecture kernels ──────────────────────────────────────
// Compute OVCs for a sorted run: compare adjacent sorted records.
// OVC[0] = OVC_INITIAL, OVC[i] = ovc_compute_delta(record[i-1], record[i]).
__global__ void compute_run_ovcs_kernel(
    const uint8_t* __restrict__ sorted_records,
    uint32_t* __restrict__ ovcs,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    if (i == 0) {
        ovcs[i] = OVC_INITIAL;
    } else {
        const uint8_t* prev = sorted_records + (i - 1) * RECORD_SIZE;
        const uint8_t* curr = sorted_records + i * RECORD_SIZE;
        ovcs[i] = ovc_compute_delta(prev, curr, KEY_SIZE);
    }
}

// Extract 8-byte big-endian prefix from sorted records (for OVC tiebreaking)
__global__ void extract_prefix8_kernel(
    const uint8_t* __restrict__ sorted_records,
    uint64_t* __restrict__ prefixes,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* k = sorted_records + i * RECORD_SIZE;
    uint64_t v = 0;
    for (int b = 0; b < 8; b++) v = (v << 8) | k[b];
    prefixes[i] = v;
}

// Build global permutation: map sorted position to original h_data index.
// After sort_chunk_on_gpu, the chunk's records are sorted in d_buf[cur].
// local_perm[i] = the original index within the chunk of the i-th sorted record.
// global_perm[run_offset + i] = chunk_start + local_perm[i].
__global__ void build_global_perm_kernel(
    const uint32_t* __restrict__ local_perm,
    uint32_t* __restrict__ global_perm,
    uint64_t chunk_start,    // record offset of this chunk in h_data
    uint64_t run_offset,     // position in the global OVC/perm arrays
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    global_perm[run_offset + i] = (uint32_t)(chunk_start + local_perm[i]);
}

// Extract 16B of key (two uint64 prefixes) via permutation for GPU prefix merge.
// prefix1[i] = big-endian uint64 of bytes 0-7 of record[perm[i]]
// prefix2[i] = big-endian uint64 of bytes 8-15 of record[perm[i]]
__global__ void extract_dual_prefix_kernel(
    const uint8_t* __restrict__ records,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ prefix1,   // bytes 0-7
    uint64_t* __restrict__ prefix2,   // bytes 8-15
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t idx = perm[i];
    const uint8_t* k = records + (uint64_t)idx * RECORD_SIZE;
    // Extract bytes 0-7
    uint64_t v1 = 0;
    for (int b = 0; b < 8; b++) v1 = (v1 << 8) | k[b];
    prefix1[i] = v1;
    // Extract bytes 8-15 (or up to KEY_SIZE)
    uint64_t v2 = 0;
    int end = (KEY_SIZE < 16) ? KEY_SIZE : 16;
    for (int b = 8; b < end; b++) v2 = (v2 << 8) | k[b];
    v2 <<= (8 - (end - 8)) * 8;  // left-align if fewer than 8 bytes
    prefix2[i] = v2;
}

// Check if any adjacent pair has BOTH pfx1 AND pfx2 equal (full 16B tie).
// Run between LSD passes when sorted pfx2 and gathered pfx1 are available.
__global__ void check_full_16B_ties_kernel(
    const uint64_t* __restrict__ sorted_pfx2,
    const uint64_t* __restrict__ pfx1_gathered,
    uint64_t num_records,
    uint32_t* __restrict__ has_ties
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i >= num_records) return;
    if (sorted_pfx2[i] == sorted_pfx2[i - 1] &&
        pfx1_gathered[i] == pfx1_gathered[i - 1])
        atomicOr(has_ties, 1u);
}

// Check if any adjacent pair has ALL FOUR pfx1/2/3/4 equal (full 32B tie).
// Run between pass 3 and pass 4 of 4-pass LSD when the array is stably sorted
// by (pfx2, pfx3, pfx4) and pfx1 is gathered to match. An equality on all four
// prefixes means 32 compact bytes are identical — CPU fixup will finish the
// ordering on the remaining record bytes.
__global__ void check_full_32B_ties_kernel(
    const uint64_t* __restrict__ pfx1,
    const uint64_t* __restrict__ pfx2,
    const uint64_t* __restrict__ pfx3,
    const uint64_t* __restrict__ pfx4,
    uint64_t num_records,
    uint32_t* __restrict__ has_ties
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i >= num_records) return;
    if (pfx1[i] == pfx1[i - 1] &&
        pfx2[i] == pfx2[i - 1] &&
        pfx3[i] == pfx3[i - 1] &&
        pfx4[i] == pfx4[i - 1])
        atomicOr(has_ties, 1u);
}

// Simple gather for uint32 (used in LSD merge satellite reordering)
__global__ void gather_uint32_kernel(const uint32_t* src, const uint32_t* idx,
                                      uint32_t* dst, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// Extract uint64 from records at RECORD_SIZE stride (for in-chunk LSD sort)
__global__ void extract_uint64_from_records_kernel(
    const uint8_t* __restrict__ records,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = records + (uint64_t)orig_idx * RECORD_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}

// Extract uint64 sort key (bytes 0-7 big-endian) from KEY_SIZE-stride key buffer.
__global__ void extract_uint64_from_keys_kernel(
    const uint8_t* __restrict__ key_buffer,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* k = key_buffer + i * KEY_SIZE;
    uint64_t v = 0;
    for (int b = 0; b < 8; b++) v = (v << 8) | k[b];
    sort_keys[i] = v;
}

// Extract 16-bit tiebreaker (bytes 8-9 big-endian) using permutation to lookup original keys
__global__ void extract_tiebreaker_kernel(
    const uint8_t* __restrict__ key_buffer,
    const uint32_t* __restrict__ perm,
    uint16_t* __restrict__ tiebreakers,
    uint64_t num_records,
    int byte_offset
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = key_buffer + (uint64_t)orig_idx * KEY_SIZE + byte_offset;
    tiebreakers[i] = ((uint16_t)k[0] << 8) | k[1];
}

// ── GPU key extraction kernel ────────────────────────────────────────
// Extract KEY_SIZE bytes from each sorted record into a contiguous key array.
// Runs at HBM bandwidth (~672 GB/s), essentially free compared to PCIe.

__global__ void extract_keys_kernel(
    const uint8_t* __restrict__ sorted_records,
    uint8_t* __restrict__ key_buffer,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* src = sorted_records + i * RECORD_SIZE;
    uint8_t* dst = key_buffer + i * KEY_SIZE;
    // Copy 10-byte key
    for (int b = 0; b < KEY_SIZE; b++) dst[b] = src[b];
}

// Extract all KEY_SIZE bytes from records into packed key buffer (for LSD)
__global__ void extract_all_keys_kernel(
    const uint8_t* __restrict__ records,
    uint8_t* __restrict__ packed_keys,
    uint64_t num_records
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    const uint8_t* src = records + i * RECORD_SIZE;
    uint8_t* dst = packed_keys + i * KEY_SIZE;
    for (int b = 0; b < KEY_SIZE; b++) dst[b] = src[b];
}

// Extract uint64 from packed key buffer at KEY_SIZE stride (for LSD passes 2+)
__global__ void extract_uint64_from_packed_keys_kernel(
    const uint8_t* __restrict__ packed_keys,
    const uint32_t* __restrict__ perm,
    uint64_t* __restrict__ sort_keys,
    uint64_t num_records,
    int byte_offset,
    int chunk_bytes
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_records) return;
    uint32_t orig_idx = perm[i];
    const uint8_t* k = packed_keys + (uint64_t)orig_idx * KEY_SIZE + byte_offset;
    uint64_t v = 0;
    for (int b = 0; b < chunk_bytes; b++) v = (v << 8) | k[b];
    v <<= (8 - chunk_bytes) * 8;
    sort_keys[i] = v;
}

// Pre-allocated workspace for sort_chunk_on_gpu (allocated once, reused per chunk)
struct SortWorkspace {
    // CUB radix sort workspace
    uint64_t* d_keys = nullptr;
    uint64_t* d_keys_alt = nullptr;
    uint32_t* d_indices = nullptr;
    uint32_t* d_indices_alt = nullptr;
    uint8_t* d_compact = nullptr;   // Compact key buffer (USE_COMPACT_KEY)
    uint64_t capacity = 0;

    void allocate(uint64_t max_records) {
        if (capacity >= max_records) return;
        free();
        capacity = max_records;
        CUDA_CHECK(cudaMalloc(&d_keys, max_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_alt, max_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_indices, max_records * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_indices_alt, max_records * sizeof(uint32_t)));
#ifdef USE_COMPACT_KEY
        // Skip compact key buffer if OVC mode is active (saves 1.4GB GPU memory)
        cudaError_t ck_err = cudaMalloc(&d_compact, max_records * COMPACT_KEY_SIZE);
        if (ck_err != cudaSuccess) d_compact = nullptr; // Graceful fallback
#endif
    }
    void free() {
        if (d_keys) { cudaFree(d_keys); d_keys = nullptr; }
        if (d_keys_alt) { cudaFree(d_keys_alt); d_keys_alt = nullptr; }
        if (d_indices) { cudaFree(d_indices); d_indices = nullptr; }
        if (d_indices_alt) { cudaFree(d_indices_alt); d_indices_alt = nullptr; }
        if (d_compact) { cudaFree(d_compact); d_compact = nullptr; }
        capacity = 0;
    }
};

// ============================================================================
// External Sort Engine
// ============================================================================

class ExternalGpuSort {
    static constexpr int NBUFS = 3;
    size_t gpu_budget;
    uint64_t buf_records;
    size_t buf_bytes;
    uint8_t* d_buf[NBUFS];
    uint8_t* h_pin[NBUFS];
    cudaStream_t streams[NBUFS];
    cudaEvent_t events[NBUFS];

    // Persistent key buffer (for small-key single-pass path)
    uint8_t* d_key_buffer;
    uint64_t key_buffer_capacity;
    std::vector<uint64_t> run_key_offsets;

    // Prefix merge buffers (for large-key path: persistent across run gen)
    uint64_t* d_ovc_buffer;        // 8B secondary prefix (bytes 8-15) per record
    uint64_t* d_prefix_buffer;     // 8B primary prefix (bytes 0-7) per record
    // Native 32B OVC extension: optional pfx3/pfx4 (compact bytes 16-23, 24-31).
    // Allocated only when the merge workspace fits in GPU memory (SF50: yes; SF100: no).
    // When non-null, run gen extracts all 32B of compact prefix and merge does 4-pass
    // LSD instead of 2-pass, preserving correctness (same source-position-order bytes).
    uint64_t* d_pfx3_buffer;       // 8B tertiary prefix (bytes 16-23) per record, or null
    uint64_t* d_pfx4_buffer;       // 8B quaternary prefix (bytes 24-31) per record, or null
    uint32_t* d_global_perm;       // global index per record
    uint64_t  ovc_buffer_records;  // capacity in records
    uint64_t  ovc_run_offset;     // current write position in prefix/perm buffers
    uint64_t  ovc_chunk_start;    // record offset of current chunk in h_data

    // Pre-allocated sort workspace (avoids cudaMalloc per chunk)
    SortWorkspace sort_ws;

    // Track whether compact key prefixes were used (for CPU fixup grouping)
    bool used_compact_prefix;

public:
    struct TimingResult {
        double run_gen_ms, merge_ms, total_ms;
        int num_runs, merge_passes;
        double pcie_h2d_gb, pcie_d2h_gb;
        uint8_t* sorted_output;  // pointer to sorted data
        uint64_t sorted_output_size;  // size in bytes (for munmap)
        bool sorted_output_is_mmap;   // true = munmap, false = free
        // Hybrid correctness fallback: set to true when the compact path
        // detected exception records (non-mapped bytes != V[]). Caller should
        // discard sorted_output, set g_force_no_compact = true, and retry.
        // Guaranteed correct because the non-compact path uploads full keys.
        bool needs_hybrid_retry;
    };

    ExternalGpuSort();
    ~ExternalGpuSort();
    TimingResult sort(uint8_t* h_data, uint64_t num_records);

private:
    struct RunInfo { uint64_t host_offset; uint64_t num_records; };

    void sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_scratch,
                            uint64_t n, cudaStream_t s,
                            bool compact_preloaded = false);

    std::vector<RunInfo> generate_runs_pipelined(
        uint8_t* h_data, uint64_t num_records,
        double& ms, double& h2d, double& d2h);

    uint8_t* streaming_merge(uint8_t* h_data, uint64_t num_records,
                              std::vector<RunInfo>& runs,
                              double& ms, int& passes, double& h2d, double& d2h,
                              uint32_t* h_perm_prealloc, uint8_t* h_output_prealloc);

};

ExternalGpuSort::ExternalGpuSort() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Sort buffers get 65% of GPU memory (3 × ~5.5GB each)
    gpu_budget = (size_t)(free_mem * 0.65);
    key_buffer_capacity = 0;
    d_key_buffer = nullptr;
    d_ovc_buffer = nullptr;
    d_prefix_buffer = nullptr;
    d_pfx3_buffer = nullptr;
    d_pfx4_buffer = nullptr;
    d_global_perm = nullptr;
    ovc_buffer_records = 0;
    ovc_run_offset = 0;
    ovc_chunk_start = 0;
    used_compact_prefix = false;
    buf_records = (gpu_budget / NBUFS) / RECORD_SIZE;
    buf_bytes = buf_records * RECORD_SIZE;

    printf("[ExternalSort] GPU: %.2f GB free, budget: %.2f GB\n",
           free_mem/1e9, gpu_budget/1e9);
    printf("[ExternalSort] Triple-buffer: %d × %.2f GB (%.0f M records each)\n",
           NBUFS, buf_bytes/1e9, buf_records/1e6);

    // Allocate host pinned buffers and streams/events (not GPU buffers yet — lazy alloc)
    for (int i = 0; i < NBUFS; i++) {
        d_buf[i] = nullptr;  // allocated lazily in sort()
        CUDA_CHECK(cudaMallocHost(&h_pin[i], buf_bytes));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&events[i]));
    }
}

ExternalGpuSort::~ExternalGpuSort() {
    for (int i = 0; i < NBUFS; i++) {
        cudaFreeHost(h_pin[i]);
        cudaFree(d_buf[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
    if (d_key_buffer) cudaFree(d_key_buffer);
    if (d_ovc_buffer) cudaFree(d_ovc_buffer);
    if (d_prefix_buffer) cudaFree(d_prefix_buffer);
    if (d_pfx3_buffer) cudaFree(d_pfx3_buffer);
    if (d_pfx4_buffer) cudaFree(d_pfx4_buffer);
    if (d_global_perm) cudaFree(d_global_perm);
    sort_ws.free();
}

// Sort a chunk on GPU using LSD radix sort on full KEY_SIZE-byte key + record reorder.
// For KEY_SIZE ≤ 8: single CUB pass on uint64.
// For KEY_SIZE > 8: ceil(KEY_SIZE/8) LSD passes, each sorting 8 bytes of key.
void ExternalGpuSort::sort_chunk_on_gpu(uint8_t* d_in, uint8_t* d_scratch,
                                         uint64_t n, cudaStream_t s,
                                         bool compact_preloaded) {
    int nthreads = 256;
    int nblks = (n + nthreads - 1) / nthreads;

    // Initialize identity permutation
    init_identity_kernel<<<nblks, nthreads, 0, s>>>(sort_ws.d_indices, n);

    uint32_t* perm_in = sort_ws.d_indices;
    uint32_t* perm_out = sort_ws.d_indices_alt;

#ifdef USE_COMPACT_KEY
    if (sort_ws.d_compact) {
    // Compact key path: build 32B keys from varying bytes, then 4 LSD passes
    if (!compact_preloaded)
        build_compact_keys_kernel<<<nblks, nthreads, 0, s>>>(
            d_in, sort_ws.d_compact, n, d_compact_map_ptr, g_compact_count);

    int num_chunks = (COMPACT_KEY_SIZE + 7) / 8;  // 4 for 32B
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int byte_offset = chunk * 8;
        int chunk_bytes = std::min(8, COMPACT_KEY_SIZE - byte_offset);

        extract_uint64_from_compact_kernel<<<nblks, nthreads, 0, s>>>(
            sort_ws.d_compact, perm_in, sort_ws.d_keys, n, byte_offset, chunk_bytes);

        cub::DoubleBuffer<uint64_t> keys_buf(sort_ws.d_keys, sort_ws.d_keys_alt);
        cub::DoubleBuffer<uint32_t> idx_buf(perm_in, perm_out);
        size_t temp_bytes = buf_bytes;
        cub::DeviceRadixSort::SortPairs(d_scratch, temp_bytes,
            keys_buf, idx_buf, (int)n, 0, chunk_bytes * 8, s);
        perm_in = idx_buf.Current();
        perm_out = idx_buf.Alternate();
    }
    } else {
    // Compact buffer unavailable — fall through to standard LSD
#else
    {
#endif
    // Standard LSD path: ceil(KEY_SIZE/8) passes on full key
    int num_chunks = (KEY_SIZE + 7) / 8;
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int byte_offset = chunk * 8;
        int chunk_bytes = std::min(8, KEY_SIZE - byte_offset);

        extract_uint64_from_records_kernel<<<nblks, nthreads, 0, s>>>(
            d_in, perm_in, sort_ws.d_keys, n, byte_offset, chunk_bytes);

        cub::DoubleBuffer<uint64_t> keys_buf(sort_ws.d_keys, sort_ws.d_keys_alt);
        cub::DoubleBuffer<uint32_t> idx_buf(perm_in, perm_out);
        size_t temp_bytes = buf_bytes;
        cub::DeviceRadixSort::SortPairs(d_scratch, temp_bytes,
            keys_buf, idx_buf, (int)n, 0, chunk_bytes * 8, s);
        perm_in = idx_buf.Current();
        perm_out = idx_buf.Alternate();
    }
    } // end standard LSD / compact fallback

    // OVC mode: extract OVCs + prefixes + global perm instead of reordering records.
    // This avoids the D2H of sorted records — only OVCs + perm are downloaded.
    if (d_ovc_buffer && ovc_run_offset + n <= ovc_buffer_records) {
        // Compute OVCs by comparing adjacent records via permutation
        // OVC[0] = INITIAL, OVC[i] = delta(record[perm[i-1]], record[perm[i]])
        // We need a kernel that reads records via permutation, not from sorted output
        // (because we're NOT reordering records in OVC mode)

        // Extract 16B prefix for merge. With USE_COMPACT_KEY, extract compact
        // key positions directly from records (16 of 26 varying bytes — much higher
        // entropy than raw bytes 0-15 which include ~40 constant bytes).
#ifdef USE_COMPACT_KEY
        // Native 32B OVC path: if pfx3/pfx4 buffers were allocated (SF50-class: merge
        // workspace fits), extract all 32B of compact prefix during run gen — zero
        // extra PCIe cost, reads already happen on the record for pfx1/pfx2. Merge
        // will do 4-pass LSD instead of 2-pass.
        uint64_t* pfx3_out = d_pfx3_buffer ? d_pfx3_buffer + ovc_run_offset : nullptr;
        uint64_t* pfx4_out = d_pfx4_buffer ? d_pfx4_buffer + ovc_run_offset : nullptr;
        if (compact_preloaded) {
            // Compact keys already in sort_ws.d_compact — read prefix from there
            extract_prefix_from_compact_buffer_kernel<<<nblks, nthreads, 0, s>>>(
                sort_ws.d_compact, perm_in,
                d_prefix_buffer + ovc_run_offset,
                d_ovc_buffer + ovc_run_offset,
                pfx3_out, pfx4_out,
                n);
        } else {
            extract_compact_prefix_from_records_kernel<<<nblks, nthreads, 0, s>>>(
                d_in, perm_in,
                d_prefix_buffer + ovc_run_offset,
                d_ovc_buffer + ovc_run_offset,
                pfx3_out, pfx4_out,
                n, d_compact_map_ptr, g_compact_count);
        }
        used_compact_prefix = true;
#else
        extract_dual_prefix_kernel<<<nblks, nthreads, 0, s>>>(
            d_in, perm_in,
            d_prefix_buffer + ovc_run_offset,
            d_ovc_buffer + ovc_run_offset,
            n);
#endif

        // Build global perm: global_perm[run_offset + i] = chunk_start + perm[i]
        build_global_perm_kernel<<<nblks, nthreads, 0, s>>>(
            perm_in, d_global_perm, ovc_chunk_start, ovc_run_offset, n);

        ovc_run_offset += n;
        return; // Don't reorder records — they stay unsorted in d_buf
    }

    // Standard path: reorder full records using the final sorted permutation
    reorder_records_kernel<<<nblks, nthreads, 0, s>>>(
        d_in, d_scratch, perm_in, n);

    // Copy sorted result back to d_in (async — no CPU sync needed)
    CUDA_CHECK(cudaMemcpyAsync(d_in, d_scratch, n * RECORD_SIZE,
                                cudaMemcpyDeviceToDevice, s));
    // Don't sync here — let the caller manage dependencies via events
}

// gpu_merge_inplace removed — CUB radix sort replaced bitonic+K-way merge

// ════════════════════════════════════════════════════════════════════
// Phase 1: Triple-Buffered Run Generation
// ════════════════════════════════════════════════════════════════════

std::vector<ExternalGpuSort::RunInfo>
ExternalGpuSort::generate_runs_pipelined(
    uint8_t* h_data, uint64_t num_records,
    double& ms, double& h2d, double& d2h
) {
    std::vector<RunInfo> runs;
    h2d = d2h = 0;
    WallTimer timer; timer.begin();

    int total_chunks = (num_records + buf_records - 1) / buf_records;

#ifdef USE_COMPACT_KEY
    // ── Compact key upload path ──────────────────────────────────────
    // Instead of uploading 120B records (72GB for SF100), extract 32B compact
    // keys on CPU and upload only those. 3.75× less PCIe traffic.
    // Pipeline: CPU extract → H2D compact keys → GPU sort (from d_compact)
    // Compact upload: pack the first min(count, COMPACT_KEY_SIZE) varying bytes
    // into the compact buffer. If count > 32 the compact prefix won't cover the
    // full key, so CPU fixup will resolve prefix ties afterwards.
    bool use_compact_upload = (!g_force_no_compact && !g_disable_compact && d_ovc_buffer && sort_ws.d_compact && g_compact_count > 0);
    if (use_compact_upload) {
        const int* h_cmap = g_compact_map;
        const int cmap_n = std::min(g_compact_count, (int)COMPACT_KEY_SIZE);

        // Use h_pin[0] and h_pin[1] as double-buffered staging for compact keys.
        // They're allocated at buf_bytes (3.7GB each), plenty for compact keys (1GB).
        // Reinterpret as compact key staging buffers.
        uint8_t* h_compact[2] = { h_pin[0], h_pin[1] };

        // Sort completion event: ensure d_compact isn't overwritten during sort
        cudaEvent_t sort_done;
        CUDA_CHECK(cudaEventCreate(&sort_done));

        int hw = std::max(1, (int)std::thread::hardware_concurrency());

        // Build a dense list of non-mapped byte positions + expected values V[].
        // Iterating only the positions we care about (not all 66) keeps the
        // per-record inner loop tight and branch-free.
        int nonmap[KEY_SIZE]; uint8_t V[KEY_SIZE]; int nonmap_n = 0;
        {
            bool mapped[KEY_SIZE]; for (int b = 0; b < (int)KEY_SIZE; b++) mapped[b] = false;
            int stored = std::min(g_compact_count, 64);
            for (int i = 0; i < stored; i++) if (g_compact_map[i] < (int)KEY_SIZE) mapped[g_compact_map[i]] = true;
            for (int b = 0; b < (int)KEY_SIZE; b++) {
                if (!mapped[b]) { nonmap[nonmap_n] = b; V[nonmap_n] = g_sample_min[b]; nonmap_n++; }
            }
        }

        // Helper lambda: multi-threaded compact key extraction + per-record
        // canonical/exception classification. For every record we:
        //   1. Write the compact key (mapped bytes → compact buffer).
        //   2. Check non-mapped bytes against V[k]. If any mismatch, the record
        //      is an EXCEPTION — its full key is not captured by the compact
        //      representation. Append its original index to a per-thread list.
        //      Also note the smallest differing byte position (diagnostic).
        // After all threads join, the per-thread exception lists are flushed
        // into the global g_exception_indices under a single mutex take.
        auto do_extract = [&](uint64_t offset, uint64_t cur_n, int ping) {
            uint64_t per_t = (cur_n + hw - 1) / hw;
            std::vector<std::thread> threads;
            std::vector<std::vector<uint64_t>> thread_exc(hw);
            for (int t = 0; t < hw; t++) {
                threads.emplace_back([&, t, offset, cur_n, ping]() {
                    uint64_t lo = t * per_t, hi = std::min(lo + per_t, cur_n);
                    int local_violation = INT32_MAX;
                    auto& my_exc = thread_exc[t];
                    for (uint64_t j = lo; j < hi; j++) {
                        const uint8_t* rec = h_data + (offset + j) * RECORD_SIZE;
                        uint8_t* ck = h_compact[ping] + j * COMPACT_KEY_SIZE;
                        // Copy mapped bytes → compact key
                        for (int b = 0; b < cmap_n; b++) ck[b] = rec[h_cmap[b]];
                        for (int b = cmap_n; b < COMPACT_KEY_SIZE; b++) ck[b] = 0;
                        // Classify canonical vs exception by scanning only the
                        // non-mapped positions (e.g. 39 of 66).
                        bool is_exception = false;
                        for (int k = 0; k < nonmap_n; k++) {
                            int b = nonmap[k];
                            if (rec[b] != V[k]) {
                                is_exception = true;
                                if (b < local_violation) local_violation = b;
                                break;
                            }
                        }
                        if (is_exception) my_exc.push_back(offset + j);
                    }
                    if (local_violation != INT32_MAX) {
                        int cur = g_violated_byte.load(std::memory_order_relaxed);
                        while ((cur == -1 || local_violation < cur) &&
                               !g_violated_byte.compare_exchange_weak(cur, local_violation,
                                                                     std::memory_order_relaxed)) {}
                    }
                });
            }
            for (auto& t : threads) t.join();
            // Flush per-thread exception indices into the global list under mutex.
            size_t total_new = 0;
            for (auto& v : thread_exc) total_new += v.size();
            if (total_new) {
                std::lock_guard<std::mutex> lk(g_exception_mtx);
                g_exception_indices.reserve(g_exception_indices.size() + total_new);
                for (auto& v : thread_exc) {
                    g_exception_indices.insert(g_exception_indices.end(), v.begin(), v.end());
                }
            }
        };

        // Pipeline: overlap CPU extraction of chunk c+1 with GPU sort of chunk c.
        // Extract chunk 0 first, then loop with prefetch of c+1 running in parallel.
        {
            uint64_t cur_n0 = std::min(buf_records, num_records);
            do_extract(0, cur_n0, 0);
        }

        for (int c = 0; c < total_chunks; c++) {
            uint64_t offset = (uint64_t)c * buf_records;
            uint64_t cur_n = std::min(buf_records, num_records - offset);
            int ping = c % 2;

            // Wait for previous sort to finish reading d_compact
            if (c > 0) {
                CUDA_CHECK(cudaEventSynchronize(sort_done));
            }

            // H2D: upload compact keys to sort_ws.d_compact
            size_t compact_bytes = cur_n * COMPACT_KEY_SIZE;
            CUDA_CHECK(cudaMemcpyAsync(sort_ws.d_compact, h_compact[ping], compact_bytes,
                                        cudaMemcpyHostToDevice, streams[0]));
            h2d += compact_bytes;

            // Make sort stream wait for H2D
            CUDA_CHECK(cudaEventRecord(events[0], streams[0]));
            CUDA_CHECK(cudaStreamWaitEvent(streams[1], events[0], 0));

            // Sort from pre-loaded compact keys (skip build_compact_keys_kernel)
            ovc_chunk_start = offset;
            sort_chunk_on_gpu(nullptr, d_buf[2], cur_n, streams[1], /*compact_preloaded=*/true);

            // Record sort completion so next H2D can safely overwrite d_compact
            CUDA_CHECK(cudaEventRecord(sort_done, streams[1]));

            // In parallel with GPU work: extract compact keys for chunk c+1
            if (c + 1 < total_chunks) {
                uint64_t next_offset = (uint64_t)(c+1) * buf_records;
                uint64_t next_n = std::min(buf_records, num_records - next_offset);
                int next_ping = (c+1) % 2;
                do_extract(next_offset, next_n, next_ping);
            }

            runs.push_back({offset * RECORD_SIZE, cur_n});
            printf("\r  Run %d/%d: %.1f MB (compact upload %.1f MB)    ", c+1, total_chunks,
                   cur_n * RECORD_SIZE / (1024.0*1024.0),
                   compact_bytes / (1024.0*1024.0));
            fflush(stdout);
        }

        CUDA_CHECK(cudaEventSynchronize(sort_done));
        CUDA_CHECK(cudaEventDestroy(sort_done));
        printf("\n");
        ms = timer.end_ms();
        return runs;
    }
#endif // USE_COMPACT_KEY

    // ── Standard full-record upload path ─────────────────────────────
    // Pipeline: H2D full records → GPU sort → D2H (if non-OVC)

    // Per-buffer D2H completion events
    cudaEvent_t d2h_done[2];
    CUDA_CHECK(cudaEventCreate(&d2h_done[0]));
    CUDA_CHECK(cudaEventCreate(&d2h_done[1]));

    for (int c = 0; c < total_chunks; c++) {
        uint64_t offset = (uint64_t)c * buf_records;
        uint64_t cur_n = std::min(buf_records, num_records - offset);
        uint64_t cur_bytes = cur_n * RECORD_SIZE;
        int cur = c % 2;
        int scratch = 2;

        // Wait for previous D2H of this buffer to complete before overwriting
        if (c >= 2) {
            CUDA_CHECK(cudaStreamWaitEvent(streams[0], d2h_done[cur], 0));
        }

        // Start H2D upload on stream[0]
        CUDA_CHECK(cudaMemcpyAsync(d_buf[cur], h_data + offset * RECORD_SIZE, cur_bytes,
                                    cudaMemcpyHostToDevice, streams[0]));
        h2d += cur_bytes;

        // Record event when H2D completes, make sort stream wait for it
        CUDA_CHECK(cudaEventRecord(events[0], streams[0]));
        CUDA_CHECK(cudaStreamWaitEvent(streams[1], events[0], 0));

        // Sort (GPU-side wait for H2D via event — no CPU blocking)
        ovc_chunk_start = offset;
        sort_chunk_on_gpu(d_buf[cur], d_buf[scratch], cur_n, streams[1]);

        // Extract keys to persistent GPU key buffer (for small-key single-pass path)
        uint64_t key_off = run_key_offsets.empty() ? 0 :
            run_key_offsets.back() + (runs.empty() ? 0 : runs.back().num_records * KEY_SIZE);
        if (d_key_buffer && (key_off + cur_n * KEY_SIZE) <= key_buffer_capacity) {
            int nthreads = 256;
            int nblocks_k = (cur_n + nthreads - 1) / nthreads;
            extract_keys_kernel<<<nblocks_k, nthreads, 0, streams[1]>>>(
                d_buf[cur], d_key_buffer + key_off, cur_n);
            run_key_offsets.push_back(key_off);
        }

        // Make D2H stream wait for sort+OVC extraction on stream[1] to finish
        CUDA_CHECK(cudaEventRecord(events[1], streams[1]));
        CUDA_CHECK(cudaStreamWaitEvent(streams[2], events[1], 0));

        if (d_ovc_buffer) {
            CUDA_CHECK(cudaEventRecord(d2h_done[cur], streams[1]));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(h_data + offset * RECORD_SIZE, d_buf[cur], cur_bytes,
                                        cudaMemcpyDeviceToHost, streams[2]));
            d2h += cur_bytes;
            CUDA_CHECK(cudaEventRecord(d2h_done[cur], streams[2]));
        }

        runs.push_back({offset * RECORD_SIZE, cur_n});
        printf("\r  Run %d/%d: %.1f MB sorted    ", c+1, total_chunks,
               cur_bytes/(1024.0*1024.0));
        fflush(stdout);
    }

    // Finalize: wait for last D2H to complete
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));
    CUDA_CHECK(cudaEventDestroy(d2h_done[0]));
    CUDA_CHECK(cudaEventDestroy(d2h_done[1]));
    printf("\n");
    ms = timer.end_ms();
    return runs;
}

// ════════════════════════════════════════════════════════════════════
// Phase 2: Key-Only GPU Merge
//
// Instead of sending full 100B records over PCIe, send only 10B keys.
// GPU merges keys and produces a permutation array.
// CPU gathers full records using the permutation.
//
// For 60GB dataset: keys = 6GB (fits in GPU), perm = 2.4GB download.
// Total PCIe: ~8.4GB vs ~120GB with full-record approach. 14x less.
// ════════════════════════════════════════════════════════════════════

// Key-only merge descriptor (must match merge.cu)
struct KeyMergePair {
    uint64_t a_key_offset;
    int      a_count;
    uint64_t b_key_offset;
    int      b_count;
    uint64_t out_key_offset;
    uint64_t out_perm_offset;
    uint64_t a_perm_offset;
    uint64_t b_perm_offset;
    int      first_block;
};

extern "C" void launch_merge_ovc(
    const uint32_t*, uint32_t*, const uint64_t*, uint64_t*,
    const uint32_t*, uint32_t*,
    const void*, int, int, cudaStream_t);

extern "C" void launch_merge_keys_only(
    const uint8_t*, uint8_t*, const uint32_t*, uint32_t*,
    const KeyMergePair*, int, int, cudaStream_t);

uint8_t* ExternalGpuSort::streaming_merge(
    uint8_t* h_data, uint64_t num_records,
    std::vector<RunInfo>& runs,
    double& ms, int& passes, double& h2d, double& d2h,
    uint32_t* h_perm_prealloc, uint8_t* h_output_prealloc
) {
    if (runs.size() <= 1) { ms = 0; passes = 0; h2d = d2h = 0; return nullptr; }

    int K = (int)runs.size();
    uint64_t total_bytes = num_records * RECORD_SIZE;
    uint64_t total_keys_bytes = num_records * KEY_SIZE;
    uint64_t total_perm_bytes = num_records * sizeof(uint32_t);

    // Detailed merge phase profiling
    auto tpoint = [](const char* label) {
        static std::chrono::high_resolution_clock::time_point prev;
        auto now = std::chrono::high_resolution_clock::now();
        if (label[0] != '_')
            printf("    [merge] %-30s %6.0f ms\n", label,
                   std::chrono::duration<double,std::milli>(now - prev).count());
        prev = now;
    };
    tpoint("_start");

    WallTimer timer; timer.begin();

    // Build global index mapping
    std::vector<uint64_t> merge_key_offsets(K);
    std::vector<uint64_t> run_global_base(K);
    uint64_t global_idx = 0;
    for (int r = 0; r < K; r++) {
        run_global_base[r] = global_idx;
        global_idx += runs[r].num_records;
    }

    // ── Step 1+2: Get keys onto GPU ──
    // Check if keys were retained on GPU during run generation
    bool keys_retained = (d_key_buffer != nullptr &&
                          (int)run_key_offsets.size() == K);

    uint8_t *d_keys_in, *d_keys_out;
    uint32_t *d_perm_in, *d_perm_out;
    uint8_t* h_keys = nullptr;  // only allocated if keys not retained
    bool allocated_keys_gpu = false;

    if (keys_retained) {
        printf("  Keys already on GPU from run generation (%.2f GB, saved upload!)\n",
               total_keys_bytes / 1e9);
        // Keys are in d_key_buffer at known offsets — use directly as d_keys_in
        d_keys_in = d_key_buffer;
        for (int r = 0; r < K; r++) merge_key_offsets[r] = run_key_offsets[r];
    } else {
        // Fallback: extract keys on CPU and upload
        printf("  Extracting %lu keys (%.2f GB) on CPU...\n", num_records, total_keys_bytes/1e9);
        WallTimer ext_timer; ext_timer.begin();
        CUDA_CHECK(cudaMallocHost(&h_keys, total_keys_bytes));
        uint64_t key_off = 0;
        for (int r = 0; r < K; r++) {
            merge_key_offsets[r] = key_off;
            const uint8_t* run_data = h_data + runs[r].host_offset;
            uint8_t* key_dst = h_keys + key_off;
            for (uint64_t i = 0; i < runs[r].num_records; i++)
                memcpy(key_dst + i * KEY_SIZE, run_data + i * RECORD_SIZE, KEY_SIZE);
            key_off += runs[r].num_records * KEY_SIZE;
        }
        printf("    Extracted in %.0f ms\n", ext_timer.end_ms());

        printf("  Uploading %.2f GB keys to GPU...\n", total_keys_bytes/1e9);
        CUDA_CHECK(cudaMalloc(&d_keys_in, total_keys_bytes));
        allocated_keys_gpu = true;
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_keys, total_keys_bytes, cudaMemcpyHostToDevice));
        h2d += total_keys_bytes;
    }

    tpoint("keys ready");

    // ── Step 3: CUB radix sort ALL keys + permutation in ONE pass ──
    passes = 1;
    printf("  CUB radix sort on all %lu keys (single pass)...\n", num_records);

    // Try to reuse sort buffers as merge workspace. Falls back to fresh alloc if too small.
    size_t needed_buf0 = num_records * 2 * sizeof(uint64_t); // keys + keys_alt
    size_t needed_buf1 = num_records * 2 * sizeof(uint32_t) + 256*1024*1024; // perm + perm_alt + temp estimate
    bool reuse_bufs = (d_buf[0] && d_buf[1] && needed_buf0 <= buf_bytes && needed_buf1 <= buf_bytes);

    uint64_t* d_sort_keys; uint64_t* d_sort_keys_alt;
    void* d_temp; size_t cub_temp_bytes;
    uint8_t* d_merge_arena = nullptr;

    if (reuse_bufs) {
        d_sort_keys = (uint64_t*)d_buf[0];
        d_sort_keys_alt = d_sort_keys + num_records;
        d_perm_in = (uint32_t*)d_buf[1];
        d_perm_out = d_perm_in + num_records;
        d_temp = (void*)(d_perm_out + num_records);
        cub_temp_bytes = buf_bytes - num_records * 2 * sizeof(uint32_t);
    } else {
        // Split merge workspace across sort buffers + one extra alloc.
        // d_buf[0] (5.44GB): d_sort_keys (N×8B = 4.8GB)
        // d_buf[1] (5.44GB): d_sort_keys_alt (N×8B = 4.8GB)
        // d_buf[2] (5.44GB): d_perm_in (N×4B) + d_perm_out (N×4B) = 4.8GB
        // CUB temp: allocate separately (small, ~100MB)
        size_t keys_sz = num_records * sizeof(uint64_t);
        size_t perm_sz = num_records * sizeof(uint32_t);
        bool can_split = (d_buf[0] && d_buf[1] && d_buf[2] &&
                          keys_sz <= buf_bytes && 2 * perm_sz <= buf_bytes);
        if (can_split) {
            d_sort_keys = (uint64_t*)d_buf[0];
            d_sort_keys_alt = (uint64_t*)d_buf[1];
            d_perm_in = (uint32_t*)d_buf[2];
            d_perm_out = d_perm_in + num_records;
            // CUB temp fits after perm_out in d_buf[2] (0.6GB remaining)
            d_temp = (void*)(d_perm_out + num_records);
            cub_temp_bytes = buf_bytes - 2 * perm_sz;
            // ZERO cudaMalloc in merge phase!
        } else {
            // Fallback: single arena alloc
            cub::DoubleBuffer<uint64_t> dk(nullptr,nullptr);
            cub::DoubleBuffer<uint32_t> dp(nullptr,nullptr);
            cub_temp_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, dk, dp, (int)num_records, 0, 64, streams[0]);
            size_t arena_sz = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + cub_temp_bytes;
            for (int i = 0; i < NBUFS; i++) { if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; } }
            CUDA_CHECK(cudaMalloc(&d_merge_arena, arena_sz));
            d_sort_keys = (uint64_t*)d_merge_arena;
            d_sort_keys_alt = d_sort_keys + num_records;
            d_perm_in = (uint32_t*)(d_sort_keys_alt + num_records);
            d_perm_out = d_perm_in + num_records;
            d_temp = (void*)(d_perm_out + num_records);
        }
    }

    tpoint("alloc merge workspace");

    // Init perm + extract keys — both on GPU, overlapped on same stream
    {
        int nt = 256, nb = (num_records + nt - 1) / nt;
        init_identity_kernel<<<nb, nt, 0, streams[0]>>>(d_perm_in, num_records);
        extract_uint64_from_keys_kernel<<<nb, nt, 0, streams[0]>>>(
            d_keys_in, d_sort_keys, num_records);
    }

    // Free key buffers — no longer needed
    if (allocated_keys_gpu && d_keys_in) { CUDA_CHECK(cudaFree(d_keys_in)); d_keys_in = nullptr; }
    if (d_keys_out) { CUDA_CHECK(cudaFree(d_keys_out)); d_keys_out = nullptr; }
    if (d_key_buffer) { cudaFree(d_key_buffer); d_key_buffer = nullptr; }
    tpoint("init perm + extract keys + free bufs");

    // CUB sort (uint64 key, uint32 perm) pairs
    cub::DoubleBuffer<uint64_t> d_sortkey_buf(d_sort_keys, d_sort_keys_alt);
    cub::DoubleBuffer<uint32_t> d_perm_buf(d_perm_in, d_perm_out);

    cub::DeviceRadixSort::SortPairs(d_temp, cub_temp_bytes,
        d_sortkey_buf, d_perm_buf, (int)num_records, 0, 64, streams[0]);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    tpoint("CUB radix sort done");

    d_perm_in = d_perm_buf.Current();
    printf("  Downloading permutation (%.2f GB)...\n", total_perm_bytes/1e9);
    uint32_t* h_perm = h_perm_prealloc;
    if (!h_perm) { CUDA_CHECK(cudaMallocHost(&h_perm, total_perm_bytes)); }

    // Start async perm download — CPU continues with alloc/free work in parallel
    CUDA_CHECK(cudaMemcpyAsync(h_perm, d_perm_in, total_perm_bytes,
                                cudaMemcpyDeviceToHost, streams[0]));
    d2h += total_perm_bytes;

    // While perm downloads: allocate gather output buffer + free GPU memory
    // These are CPU operations that run concurrently with the D2H DMA
    int num_threads = std::min(48, (int)std::thread::hardware_concurrency());
    printf("  CPU value gather (%.2f GB, %d threads)...\n", total_bytes/1e9, num_threads);

    // Use pre-allocated output buffer if available (allocated before run gen)
    uint8_t* h_output = h_output_prealloc;
    if (!h_output) {
        h_output = (uint8_t*)malloc(total_bytes);
        if (!h_output) { fprintf(stderr, "malloc failed for gather output\n"); ms = timer.end_ms(); return nullptr; }
        madvise(h_output, total_bytes, MADV_HUGEPAGE);
    }

    // Free GPU memory while perm downloads
    if (d_merge_arena) { cudaFree(d_merge_arena); d_merge_arena = nullptr; }
    for (int i = 0; i < NBUFS; i++) {
        if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; }
    }
    if (h_keys) { cudaFreeHost(h_keys); h_keys = nullptr; }

    // NOW wait for perm download to complete before starting gather
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    tpoint("perm downloaded + alloc + GPU freed (overlapped)");

    WallTimer gather_timer; gather_timer.begin();

    // Pre-compute run lookup table for O(1) run_id lookup instead of O(K) scan
    // run_global_base is sorted, so we can use it directly
    // But for speed, build a direct lookup: global_idx → (run_id, idx_in_run)
    // For K <= 64, binary search is fast enough

    // Pre-compute source pointers for all records in a block, then copy.
    // This separates the "find source" phase (random permutation reads) from
    // the "copy data" phase (random source reads + sequential writes).
    // Deeper prefetch pipeline hides more DRAM latency.
    auto gather_worker = [&](uint64_t start, uint64_t end) {
        constexpr int BLOCK = 256;  // balanced: enough prefetch depth without cache pressure
        const uint8_t* src_ptrs[BLOCK];

        for (uint64_t base = start; base < end; base += BLOCK) {
            int count = std::min((uint64_t)BLOCK, end - base);

            // Phase 1: Resolve all source pointers + prefetch
            for (int j = 0; j < count; j++) {
                uint32_t src_global = h_perm[base + j];
                int run_id = K - 1;
                for (int r = 0; r < K - 1; r++) {
                    if (src_global < run_global_base[r+1]) { run_id = r; break; }
                }
                uint64_t idx_in_run = src_global - run_global_base[run_id];
                src_ptrs[j] = h_data + runs[run_id].host_offset + idx_in_run * RECORD_SIZE;
                __builtin_prefetch(src_ptrs[j], 0, 0);  // prefetch source record
            }

            // Phase 2: Copy all records (sources should be in cache from prefetch)
            for (int j = 0; j < count; j++) {
                memcpy(h_output + (base + j) * RECORD_SIZE, src_ptrs[j], RECORD_SIZE);
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    uint64_t chunk_sz = (num_records + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        uint64_t start = (uint64_t)t * chunk_sz;
        uint64_t end = std::min(start + chunk_sz, num_records);
        if (start < end) {
            threads.emplace_back(gather_worker, start, end);
        }
    }
    for (auto& t : threads) t.join();

    double gather_ms = gather_timer.end_ms();
    printf("    Gathered in %.0f ms (%.2f GB/s)\n",
           gather_ms, total_bytes / (gather_ms * 1e6));

    CUDA_CHECK(cudaFreeHost(h_perm));
    ms = timer.end_ms();
    return h_output;  // caller owns this buffer (allocated with malloc)
}

// ════════════════════════════════════════════════════════════════════
// Main Entry Point
// ════════════════════════════════════════════════════════════════════

ExternalGpuSort::TimingResult ExternalGpuSort::sort(uint8_t* h_data, uint64_t num_records) {
    TimingResult r = {};
    if (num_records <= 1) return r;
    uint64_t total_bytes = num_records * RECORD_SIZE;
    printf("[ExternalSort] Sorting %lu records (%.2f GB)\n\n", num_records, total_bytes/1e9);

#ifdef USE_COMPACT_KEY
    // Detect varying byte positions in the sort key from a 1M-record sample.
    // This makes the compact-key optimization general — works on any dataset,
    // not just TPC-H. Cost: ~10-50ms (one sequential scan over 0.4-2 GB).
    {
        // Memoize detection across repeated sort() calls on the same buffer
        // (--runs mode re-sorts the same h_data; no need to re-detect).
        static const uint8_t* s_cached_data = nullptr;
        static uint64_t s_cached_n = 0;
        WallTimer dt; dt.begin();
        double det_ms;
        if (h_data == s_cached_data && num_records == s_cached_n && g_compact_count > 0) {
            det_ms = 0;  // cache hit
        } else {
            g_compact_count = detect_compact_map(h_data, num_records, g_compact_map,
                                                 g_sample_min, g_sample_max);
            s_cached_data = h_data;
            s_cached_n = num_records;
            g_map_full_scan_verified = false;  // will be set true after extraction verifies
            g_violated_byte.store(-1);
            det_ms = dt.end_ms();
        }
        // Reset exception list for this sort call (independent of cache hit).
        {
            std::lock_guard<std::mutex> lk(g_exception_mtx);
            g_exception_indices.clear();
        }
        uint64_t s_target = 1000000;
        if (const char* e = getenv("COMPACT_SAMPLE")) { long v = atol(e); if (v > 0) s_target = (uint64_t)v; }
        if (num_records <= 50000000) s_target = num_records;
        uint64_t actual_sample = std::min(num_records, s_target);
        bool full_scan = (actual_sample == num_records);
        printf("[CompactDetect] %d/%d key bytes vary (%s %luK of %luM records, %.0f ms)\n",
               g_compact_count, KEY_SIZE,
               full_scan ? "full scan" : "stratified random sample",
               actual_sample / 1000, num_records / 1000000, det_ms);
        if (g_compact_count > 0) {
            printf("[CompactDetect] Varying byte positions:");
            for (int i = 0; i < g_compact_count && i < 32; i++) printf(" %d", g_compact_map[i]);
            if (g_compact_count > 32) printf(" ...");
            printf("\n");
        }
        // Decision: which sort path benefits most?
        double compactness = (double)g_compact_count / KEY_SIZE;
        const char* decision;
        if (g_compact_count == 0) decision = "ALL CONSTANT — degenerate (sort is identity)";
        else if (g_compact_count <= 16)
            decision = "≤16 varying bytes — compact prefix is full sort key (no ties possible)";
        else if (g_compact_count <= COMPACT_KEY_SIZE)
            decision = "compact upload WINS (≤32 varying bytes, fits in 32B compact buffer)";
        else if (g_compact_count <= 64)
            decision = "compact UPLOAD still wins, but compact PREFIX may have ties (CPU fixup)";
        else {
            decision = "too many varying bytes — no compression benefit, use full upload";
            g_disable_compact = true;
        }
        printf("[CompactDetect] Compactness: %d/%d = %.0f%% varying. Decision: %s\n",
               g_compact_count, KEY_SIZE, compactness * 100.0, decision);
        if (g_compact_count > 0) {
            int bits1 = 0, bits2 = 0, bits4 = 0, bits8 = 0;
            int packed_bits = 0;
            for (int i = 0; i < g_compact_count; i++) {
                int pos = g_compact_map[i];
                int range = (int)g_sample_max[pos] - (int)g_sample_min[pos] + 1;
                int bits = (range <= 2) ? 1 : (range <= 4) ? 2 : (range <= 16) ? 4 : 8;
                packed_bits += bits;
                if (bits == 1) bits1++;
                else if (bits == 2) bits2++;
                else if (bits == 4) bits4++;
                else bits8++;
            }
            int packed_bytes = (packed_bits + 7) / 8;
            printf("[CompactDetect] Per-position bit-widths: %d×1b  %d×2b  %d×4b  %d×8b"
                   " → %d bits = %d packed bytes (vs %d current)\n",
                   bits1, bits2, bits4, bits8, packed_bits, packed_bytes, g_compact_count);
            int covered_by_32B = 0, cum_bits = 0;
            for (int i = 0; i < g_compact_count && cum_bits < 256; i++) {
                int pos = g_compact_map[i];
                int range = (int)g_sample_max[pos] - (int)g_sample_min[pos] + 1;
                int bits = (range <= 2) ? 1 : (range <= 4) ? 2 : (range <= 16) ? 4 : 8;
                cum_bits += bits;
                if (cum_bits <= 256) covered_by_32B = i + 1;
            }
            printf("[CompactDetect] 32B prefix would cover %d/%d positions if nibble-packed"
                   " (vs %d currently)\n", covered_by_32B, g_compact_count,
                   std::min(g_compact_count, (int)COMPACT_KEY_SIZE));
        }
        // Allocate + upload the runtime compact map to a plain device buffer.
        // Using cudaMalloc + cudaMemcpy (not cudaMemcpyToSymbol on a __constant__
        // symbol) keeps CUDA module loading lazy — the buggy PTX-JIT path on
        // Turing only triggers if we eagerly touch a __constant__ symbol.
        if (!d_compact_map_ptr) {
            CUDA_CHECK(cudaMalloc(&d_compact_map_ptr, 64 * sizeof(int)));
        }
        CUDA_CHECK(cudaMemcpy(d_compact_map_ptr, g_compact_map,
                              64 * sizeof(int), cudaMemcpyHostToDevice));
    }
#endif

    // Lazy-allocate GPU buffers if needed
    if (!d_buf[0]) {
        for (int i = 0; i < NBUFS; i++)
            CUDA_CHECK(cudaMalloc(&d_buf[i], buf_bytes));
    }

    // Fast path: fits in one buffer
    if (num_records <= buf_records) {
        printf("  Data fits in GPU — single-chunk sort\n");

#ifdef USE_COMPACT_KEY
        // Correctness fallback on fast-path retry: sort_chunk_on_gpu uses
        // compact keys internally, so if the sample was proven bad (retry)
        // we can't use the GPU path at all. For fast-path data (≤5 GB) fall
        // back to a parallel CPU full-key sort — slow but guaranteed correct.
        if (g_force_no_compact) {
            printf("[Hybrid] Fast-path retry: parallel CPU std::sort (full-key, guaranteed correct)\n");
            WallTimer t; t.begin();
            std::vector<uint32_t> idx(num_records);
            for (uint64_t i = 0; i < num_records; i++) idx[i] = (uint32_t)i;
            std::sort(idx.begin(), idx.end(), [h_data](uint32_t a, uint32_t b) {
                return key_compare(h_data + (uint64_t)a * RECORD_SIZE,
                                   h_data + (uint64_t)b * RECORD_SIZE, KEY_SIZE) < 0;
            });
            uint8_t* h_out = (uint8_t*)mmap(nullptr, total_bytes, PROT_READ|PROT_WRITE,
                                            MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
            for (uint64_t i = 0; i < num_records; i++)
                memcpy(h_out + i * RECORD_SIZE, h_data + (uint64_t)idx[i] * RECORD_SIZE, RECORD_SIZE);
            r.total_ms = r.run_gen_ms = t.end_ms();
            r.num_runs = 1;
            r.pcie_h2d_gb = r.pcie_d2h_gb = 0;
            r.sorted_output = h_out;
            r.sorted_output_size = total_bytes;
            r.sorted_output_is_mmap = true;
            return r;
        }
        // The fast path sorts compact keys on the GPU without the CPU extraction
        // verification pass, so we need to verify the sample map ourselves here.
        // Parallel scan of non-mapped bytes — bandwidth-bound, <200ms for
        // anything under ~5 GB. If a violation is found, signal the caller to
        // retry on the full-key upload path (no compaction).
        if (g_compact_count > 0) {
            bool mapped[KEY_SIZE] = {false};
            int stored = std::min(g_compact_count, 64);
            for (int i = 0; i < stored; i++)
                if (g_compact_map[i] < (int)KEY_SIZE) mapped[g_compact_map[i]] = true;
            int nonmap[KEY_SIZE]; uint8_t V[KEY_SIZE]; int nonmap_n = 0;
            for (int b = 0; b < (int)KEY_SIZE; b++)
                if (!mapped[b]) { nonmap[nonmap_n] = b; V[nonmap_n] = g_sample_min[b]; nonmap_n++; }

            int hw = std::max(1, (int)std::thread::hardware_concurrency());
            std::atomic<int> vb{-1};
            uint64_t per_t = (num_records + hw - 1) / hw;
            std::vector<std::thread> vts;
            for (int t = 0; t < hw; t++) {
                vts.emplace_back([&, t]() {
                    uint64_t lo = (uint64_t)t * per_t, hi = std::min(lo + per_t, num_records);
                    int local = INT32_MAX;
                    for (uint64_t i = lo; i < hi; i++) {
                        if (local == 0) break;
                        const uint8_t* rec = h_data + i * RECORD_SIZE;
                        for (int k = 0; k < nonmap_n; k++) {
                            if (rec[nonmap[k]] != V[k]) {
                                if (nonmap[k] < local) local = nonmap[k];
                                break;
                            }
                        }
                    }
                    if (local != INT32_MAX) {
                        int cur = vb.load(std::memory_order_relaxed);
                        while ((cur == -1 || local < cur) &&
                               !vb.compare_exchange_weak(cur, local, std::memory_order_relaxed)) {}
                    }
                });
            }
            for (auto& t : vts) t.join();
            int v = vb.load();
            if (v != -1) {
                fprintf(stderr,
                    "[Hybrid] Sample missed varying byte %d on the fast path.\n"
                    "  Falling back to full-key upload for a guaranteed-correct sort.\n", v);
                r.needs_hybrid_retry = true;
                r.sorted_output = nullptr;
                r.sorted_output_size = 0;
                return r;
            }
            printf("[Correctness] Sample map verified against all %lu records "
                   "(fast-path pre-scan) — compact sort is FULL-KEY-EQUIVALENT.\n",
                   num_records);
        }
#endif

        sort_ws.allocate(num_records);
        WallTimer t; t.begin();
        CUDA_CHECK(cudaMemcpy(d_buf[0], h_data, total_bytes, cudaMemcpyHostToDevice));
        sort_chunk_on_gpu(d_buf[0], d_buf[1], num_records, streams[0]);

        // In streaming mode, h_data may be mmap'd read-only — write sorted data
        // to a separate output buffer instead of in-place.
        uint8_t* h_sorted_buf = h_data;
        bool h_sorted_buf_is_mmap = false;
        if (g_streaming_mode) {
            h_sorted_buf = (uint8_t*)mmap(nullptr, total_bytes, PROT_READ|PROT_WRITE,
                                           MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
            if (h_sorted_buf == MAP_FAILED) {
                h_sorted_buf = (uint8_t*)malloc(total_bytes);
            } else {
                h_sorted_buf_is_mmap = true;
            }
            madvise(h_sorted_buf, total_bytes, MADV_HUGEPAGE);
        }
        CUDA_CHECK(cudaMemcpy(h_sorted_buf, d_buf[0], total_bytes, cudaMemcpyDeviceToHost));
        double gpu_ms = t.end_ms();

#ifdef USE_COMPACT_KEY
        int sc_prefix = std::min(g_compact_count, (int)COMPACT_KEY_SIZE);
        if (g_compact_count > (int)COMPACT_KEY_SIZE) {
            WallTimer ft; ft.begin();
            int hw = std::max(1, (int)std::thread::hardware_concurrency());
            if (const char* e = getenv("FIXUP_THREADS")) { int v = atoi(e); if (v > 0) hw = v; }

            // Group detection: find runs of identical compact-key prefix
            std::vector<std::vector<uint64_t>> bnd_per_t(hw);
            uint64_t rpt = (num_records + hw - 1) / hw;
            std::vector<std::thread> gd_t;
            for (int ti = 0; ti < hw; ti++) {
                gd_t.emplace_back([&, ti]() {
                    uint64_t lo = (uint64_t)ti * rpt, hi = std::min(lo + rpt, num_records);
                    if (lo == 0) lo = 1;
                    auto& my = bnd_per_t[ti];
                    my.reserve(512);
                    for (uint64_t i = lo; i < hi; i++) {
                        const uint8_t* ra = h_sorted_buf + (i-1) * RECORD_SIZE;
                        const uint8_t* rb = h_sorted_buf + i * RECORD_SIZE;
                        bool diff = false;
                        for (int c = 0; c < sc_prefix; c++) {
                            if (ra[g_compact_map[c]] != rb[g_compact_map[c]]) { diff = true; break; }
                        }
                        if (diff) my.push_back(i);
                    }
                });
            }
            for (auto& th : gd_t) th.join();

            std::vector<uint64_t> all_bnd;
            all_bnd.push_back(0);
            for (int ti = 0; ti < hw; ti++)
                all_bnd.insert(all_bnd.end(), bnd_per_t[ti].begin(), bnd_per_t[ti].end());
            all_bnd.push_back(num_records);

            std::vector<std::pair<uint64_t, uint64_t>> groups;
            for (size_t i = 0; i + 1 < all_bnd.size(); i++) {
                uint64_t cnt = all_bnd[i+1] - all_bnd[i];
                if (cnt > 1) groups.push_back({all_bnd[i], cnt});
            }

            int active[KEY_SIZE]; int n_active = 0;
            for (int c = sc_prefix; c < g_compact_count; c++) {
                int pos = g_compact_map[c];
                if (pos >= 0 && pos < (int)KEY_SIZE) active[n_active++] = pos;
            }

            // Parallel per-group sort
            std::atomic<uint64_t> q{0};
            uint64_t ng = groups.size();
            std::vector<std::thread> fix_t;
            std::vector<std::vector<std::pair<std::vector<uint8_t>, uint32_t>>> tbufs(hw);
            for (int ti = 0; ti < hw; ti++) {
                fix_t.emplace_back([&, ti]() {
                    auto& tbuf = tbufs[ti];
                    while (true) {
                        uint64_t gi = q.fetch_add(1, std::memory_order_relaxed);
                        if (gi >= ng) break;
                        auto [start, count] = groups[gi];
                        tbuf.resize(count);
                        for (uint64_t j = 0; j < count; j++) {
                            tbuf[j].first.assign(
                                h_sorted_buf + (start+j)*RECORD_SIZE,
                                h_sorted_buf + (start+j)*RECORD_SIZE + KEY_SIZE);
                            tbuf[j].second = (uint32_t)j;
                        }
                        std::sort(tbuf.begin(), tbuf.end(),
                            [&](const auto& a, const auto& b) {
                                for (int k = 0; k < n_active; k++) {
                                    int p = active[k];
                                    if (a.first[p] != b.first[p])
                                        return a.first[p] < b.first[p];
                                }
                                return false;
                            });
                        // Reorder records in-place using a temp buffer
                        std::vector<uint8_t> tmp(count * RECORD_SIZE);
                        for (uint64_t j = 0; j < count; j++)
                            memcpy(tmp.data() + j * RECORD_SIZE,
                                   h_sorted_buf + (start + tbuf[j].second) * RECORD_SIZE, RECORD_SIZE);
                        memcpy(h_sorted_buf + start * RECORD_SIZE, tmp.data(), count * RECORD_SIZE);
                    }
                });
            }
            for (auto& th : fix_t) th.join();
            double fix_ms = ft.end_ms();
            printf("  Single-chunk fixup: %lu groups, %d active bytes, %.0f ms\n",
                   groups.size(), n_active, fix_ms);
            gpu_ms += fix_ms;
        } else {
            printf("  Single-chunk fixup: skipped (all %d varying bytes fit in %dB compact)\n",
                   g_compact_count, (int)COMPACT_KEY_SIZE);
        }
#endif
        r.total_ms = r.run_gen_ms = gpu_ms;
        r.num_runs = 1;
        r.pcie_h2d_gb = r.pcie_d2h_gb = total_bytes / 1e9;
        if (g_streaming_mode) {
            r.sorted_output = h_sorted_buf;
            r.sorted_output_size = total_bytes;
            r.sorted_output_is_mmap = h_sorted_buf_is_mmap;
        } else {
            r.sorted_output = nullptr; // sorted in-place in h_data
        }
        return r;
    }

    // ════════════════════════════════════════════════════════════════
    // SINGLE-PASS KEY-ONLY SORT
    //
    // Instead of: upload full records → GPU sort → download sorted records → merge
    // Do:         extract keys on CPU → upload 10% keys → GPU sort → download perm → CPU gather
    //
    // PCIe traffic: 8.4GB instead of 122.4GB for 60GB dataset (14.6× reduction!)
    // ════════════════════════════════════════════════════════════════

    uint64_t total_keys_bytes = num_records * KEY_SIZE;
    uint64_t total_perm_bytes = num_records * sizeof(uint32_t);

    // Pre-allocate host perm buffer (pinned for fast D2H)
    uint32_t* h_perm = nullptr;
    cudaMallocHost(&h_perm, total_perm_bytes);
    // Pre-allocate gather output buffer
    // Streaming mode: MAP_NORESERVE avoids committing all RAM upfront (pages
    // fault on demand, kernel can reclaim under pressure). Non-streaming uses
    // MAP_POPULATE for pre-faulted pages (lower gather latency).
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (g_streaming_mode) mmap_flags |= MAP_NORESERVE;
    else mmap_flags |= MAP_POPULATE;
    uint8_t* h_output = (uint8_t*)mmap(nullptr, total_bytes, PROT_READ|PROT_WRITE,
                                        mmap_flags, -1, 0);
    if (h_output == MAP_FAILED) h_output = nullptr;
    if (h_output) madvise(h_output, total_bytes, MADV_HUGEPAGE);
    bool h_output_is_mmap = (h_output != nullptr);
    if (!h_output) {
        h_output = (uint8_t*)malloc(total_bytes);
        if (h_output) madvise(h_output, total_bytes, MADV_HUGEPAGE);
    }

    WallTimer phase_timer;

    // ── Step 1+2: Upload keys to GPU ──
    // For small keys (KEY_SIZE ≤ 16): strided DMA extracts only keys
    // For large keys (KEY_SIZE > 16): contiguous upload of full records is faster
    //   because strided DMA has host-side read amplification
    printf("== Step 1+2: Key Upload ==\n");
    phase_timer.begin();

    // Free sort buffers and workspace to make room for key upload
    for (int i = 0; i < NBUFS; i++) { if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; } }
    sort_ws.free();
    if (d_key_buffer) { cudaFree(d_key_buffer); d_key_buffer = nullptr; }

    uint8_t* d_keys_10byte;
    uint8_t* h_keys = nullptr;

    // Variables used by both paths (prefix sort + LSD sort)
    uint32_t *d_perm_in = nullptr, *d_perm_out = nullptr;

    // Check if all keys fit in GPU memory (need keys + arena)
    size_t free_mem_now, dummy2;
    CUDA_CHECK(cudaMemGetInfo(&free_mem_now, &dummy2));
    size_t est_arena = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + 512*1024*1024;
    // PREFIX SORT experiment: disabled — fixup step (6-12s) makes it slower
    // than run-gen approach. The CPU fixup has to touch all data again,
    // negating the PCIe savings. Run-gen + K-way merge remains faster.
    // Keeping code for reference.
#if 0
    // For large keys: try PREFIX SORT (16B GPU sort + CPU fixup for ties).
    // Upload first 16 bytes via strided DMA, GPU sorts by 16B prefix (2 LSD passes),
    // CPU gathers into prefix-sorted order, then sorts within small contiguous groups.
    // PCIe = 16B×N up + 4B×N down vs 2×RECORD_SIZE×N for full-record round-trip.
    static constexpr int PREFIX_BYTES = 16;
    size_t prefix_total = num_records * PREFIX_BYTES;
    // Arena: 2×uint64 (sort keys) + 2×uint32 (perm) + CUB temp (~300MB for 600M records)
    size_t prefix_arena = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + 300*1024*1024;
    bool use_prefix_sort = (KEY_SIZE > 16 && prefix_total + prefix_arena < free_mem_now * 0.98);
    printf("  Prefix sort check: need %.1f GB, have %.1f GB (%.1f×0.95=%.1f) → %s\n",
           (prefix_total + prefix_arena)/1e9, free_mem_now/1e9,
           free_mem_now/1e9, free_mem_now*0.95/1e9,
           use_prefix_sort ? "YES" : "NO");

    if (use_prefix_sort) {
        printf("  Using PREFIX SORT: %dB prefix on GPU + CPU fixup for ties\n", PREFIX_BYTES);
        printf("  PCIe: %.1f GB H2D + %.1f GB D2H (vs %.1f GB full-record round-trip)\n",
               prefix_total/1e9, total_perm_bytes/1e9, 2*total_bytes/1e9);

        // Allocate prefix keys and sort workspace on GPU
        uint8_t* d_prefix_keys;
        CUDA_CHECK(cudaMalloc(&d_prefix_keys, prefix_total));
        CUDA_CHECK(cudaMalloc(&d_perm_in, num_records * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_perm_out, num_records * sizeof(uint32_t)));
        uint64_t* d_sort_keys, *d_sort_keys_alt;
        CUDA_CHECK(cudaMalloc(&d_sort_keys, num_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sort_keys_alt, num_records * sizeof(uint64_t)));

        // Step 1: Strided DMA — upload first PREFIX_BYTES of each record
        CUDA_CHECK(cudaMemcpy2D(d_prefix_keys, PREFIX_BYTES, h_data, RECORD_SIZE,
                                 PREFIX_BYTES, num_records, cudaMemcpyHostToDevice));
        double upload_ms = phase_timer.end_ms();
        r.pcie_h2d_gb = prefix_total / 1e9;
        printf("  Uploaded %.2f GB prefix keys in %.0f ms (%.2f GB/s effective)\n",
               prefix_total/1e9, upload_ms, total_bytes/(upload_ms*1e6));

        // Step 2: GPU LSD sort by 16-byte prefix (2 CUB passes)
        phase_timer.begin();
        int nthreads = 256;
        int nblks = (num_records + nthreads - 1) / nthreads;
        init_identity_kernel<<<nblks, nthreads>>>(d_perm_in, num_records);

        // CUB temp storage
        size_t cub_temp_bytes = 0;
        cub::DoubleBuffer<uint64_t> keys_buf(d_sort_keys, d_sort_keys_alt);
        cub::DoubleBuffer<uint32_t> idx_buf(d_perm_in, d_perm_out);

        cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
            keys_buf, idx_buf, (int)num_records, 0, 64);
        void* d_cub_temp;
        CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

        // LSD: sort by least significant 8 bytes first (bytes 8-15), then bytes 0-7
        uint32_t* perm_in = d_perm_in;
        uint32_t* perm_out = d_perm_out;

        for (int pass = 1; pass >= 0; pass--) {
            int byte_offset = pass * 8;
            // Extract 8 bytes from prefix buffer at PREFIX_BYTES stride
            thrust::counting_iterator<uint64_t> count_idx(0);
            thrust::device_ptr<uint64_t> sk_ptr(d_sort_keys);
            uint8_t* pk = d_prefix_keys;
            uint32_t* pi = perm_in;
            thrust::transform(count_idx, count_idx + num_records, sk_ptr,
                [pk, pi, byte_offset] __device__ (uint64_t i) {
                    uint32_t orig = pi[i];
                    const uint8_t* k = pk + (uint64_t)orig * PREFIX_BYTES + byte_offset;
                    uint64_t v = 0;
                    for (int b = 0; b < 8; b++) v = (v << 8) | k[b];
                    return v;
                });

            cub::DoubleBuffer<uint64_t> kb(d_sort_keys, d_sort_keys_alt);
            cub::DoubleBuffer<uint32_t> ib(perm_in, perm_out);
            size_t temp = cub_temp_bytes;
            cub::DeviceRadixSort::SortPairs(d_cub_temp, temp,
                kb, ib, (int)num_records, 0, 64);
            perm_in = ib.Current();
            perm_out = ib.Alternate();
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        double sort_ms = phase_timer.end_ms();
        printf("  GPU prefix sort (2 LSD passes): %.0f ms\n", sort_ms);

        // Step 3: Download sorted prefix keys + permutation
        phase_timer.begin();
        CUDA_CHECK(cudaMemcpy(h_perm, perm_in,
                               total_perm_bytes, cudaMemcpyDeviceToHost));
        // Download sorted prefix keys for group detection (need the 16B prefixes in sorted order)
        // Reconstruct by reading h_data[h_perm[i]]'s first 16 bytes on CPU (faster than GPU D2H)
        // Actually just download the sorted uint64 from the last CUB pass (bytes 0-7)
        uint8_t* h_sorted_prefixes = (uint8_t*)malloc(num_records * PREFIX_BYTES);
        // Read prefixes from h_data using permutation (sequential perm access, random h_data access)
        {
            int nt = std::max(1, (int)std::thread::hardware_concurrency());
            uint64_t per_t = (num_records + nt - 1) / nt;
            std::vector<std::thread> threads;
            for (int t = 0; t < nt; t++) {
                threads.emplace_back([&, t]() {
                    uint64_t lo = t * per_t, hi = std::min(lo + per_t, num_records);
                    for (uint64_t i = lo; i < hi; i++) {
                        memcpy(h_sorted_prefixes + i * PREFIX_BYTES,
                               h_data + (uint64_t)h_perm[i] * RECORD_SIZE, PREFIX_BYTES);
                    }
                });
            }
            for (auto& t : threads) t.join();
        }
        r.pcie_d2h_gb = total_perm_bytes / 1e9;
        double dl_ms = phase_timer.end_ms();
        printf("  Downloaded perm + extracted prefixes in %.0f ms\n", dl_ms);

        // Free GPU memory
        cudaFree(d_prefix_keys); cudaFree(d_perm_in); cudaFree(d_perm_out);
        cudaFree(d_sort_keys); cudaFree(d_sort_keys_alt); cudaFree(d_cub_temp);
        d_perm_in = d_perm_out = nullptr;

        // Step 4: CPU GATHER first (prefix-sorted order), THEN fixup on contiguous data
        // Gathering first puts each group's records contiguously in h_output,
        // so the fixup sort within each group has excellent L3 cache locality
        // (18.8MB per group fits in ~25MB L3).
        int hw_threads = std::max(1, (int)std::thread::hardware_concurrency());

        phase_timer.begin();
        printf("  CPU gather (prefix-sorted order)...\n");
        int PREFETCH_AHEAD = 256;
        uint64_t chunk_size = (num_records + hw_threads - 1) / hw_threads;
        std::vector<std::thread> threads;
        for (int t = 0; t < hw_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = t * chunk_size;
                uint64_t hi = std::min(lo + chunk_size, num_records);
                for (uint64_t i = lo; i < hi; i++) {
                    if (i + PREFETCH_AHEAD < hi)
                        __builtin_prefetch(h_data + (uint64_t)h_perm[i+PREFETCH_AHEAD] * RECORD_SIZE, 0);
                    memcpy(h_output + i * RECORD_SIZE,
                           h_data + (uint64_t)h_perm[i] * RECORD_SIZE, RECORD_SIZE);
                }
            });
        }
        for (auto& t : threads) t.join();
        double gather_ms = phase_timer.end_ms();
        printf("  Gathered %.2f GB in %.0f ms (%.2f GB/s)\n",
               total_bytes/1e9, gather_ms, total_bytes/(gather_ms*1e6));

        // Step 5: CPU fixup — sort within contiguous groups in h_output
        phase_timer.begin();
        printf("  CPU fixup: sorting within groups of equal 8B prefixes...\n");

        // Find group boundaries using sorted 16-byte prefixes
        std::vector<std::pair<uint64_t, uint64_t>> groups;
        uint64_t grp_start = 0;
        for (uint64_t i = 1; i <= num_records; i++) {
            if (i == num_records ||
                memcmp(h_sorted_prefixes + i * PREFIX_BYTES,
                       h_sorted_prefixes + grp_start * PREFIX_BYTES, PREFIX_BYTES) != 0) {
                groups.push_back({grp_start, i - grp_start});
                grp_start = i;
            }
        }
        printf("    %lu groups (avg %.0f records/group)\n",
               groups.size(), (double)num_records / groups.size());

        // Sort each group IN PLACE in h_output.
        // Groups avg 31 records → tiny, use simple insertion sort.
        uint64_t groups_per_thread = (groups.size() + hw_threads - 1) / hw_threads;
        threads.clear();
        for (int t = 0; t < hw_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t gs = t * groups_per_thread;
                uint64_t ge = std::min(gs + groups_per_thread, (uint64_t)groups.size());
                uint8_t tmp[RECORD_SIZE];
                for (uint64_t g = gs; g < ge; g++) {
                    auto [start, count] = groups[g];
                    if (count <= 1) continue;
                    // Insertion sort — optimal for small groups (avg 31)
                    uint8_t* base = h_output + start * RECORD_SIZE;
                    for (uint64_t i = 1; i < count; i++) {
                        uint8_t* cur = base + i * RECORD_SIZE;
                        if (key_compare(cur - RECORD_SIZE, cur, KEY_SIZE) <= 0) continue;
                        memcpy(tmp, cur, RECORD_SIZE);
                        uint64_t j = i;
                        while (j > 0 && key_compare(base + (j-1)*RECORD_SIZE, tmp, KEY_SIZE) > 0) {
                            memcpy(base + j*RECORD_SIZE, base + (j-1)*RECORD_SIZE, RECORD_SIZE);
                            j--;
                        }
                        memcpy(base + j*RECORD_SIZE, tmp, RECORD_SIZE);
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
        double fixup_ms = phase_timer.end_ms();
        printf("    Fixup: %.0f ms (%d threads)\n", fixup_ms, hw_threads);
        free(h_sorted_prefixes);

        r.run_gen_ms = upload_ms + sort_ms + dl_ms;
        r.merge_ms = fixup_ms + gather_ms;
        r.total_ms = r.run_gen_ms + r.merge_ms;
        r.num_runs = 1; r.merge_passes = 1;
        r.sorted_output = h_output;
        r.sorted_output_size = total_bytes;
        r.sorted_output_is_mmap = h_output_is_mmap;
        return r;
    }
#endif // PREFIX SORT disabled

    // Check: prefix merge buffers fit? Need: 8B pfx1 + 8B pfx2 + 4B perm = 20B per record
    size_t ovc_total = num_records * (sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t));
    bool use_ovc = (total_keys_bytes + est_arena > free_mem_now * 0.9) &&
                   (ovc_total + 2*buf_bytes + 512*1024*1024 < free_mem_now * 0.95);

    if (use_ovc) {
        printf("  Keys too large for single-pass (%.1f GB > %.1f GB)\n",
               total_keys_bytes/1e9, free_mem_now*0.9/1e9);
        printf("  Using OVC ARCHITECTURE: run-gen + GPU OVC merge + CPU gather\n");
        printf("  OVC buffers: %.1f GB (4B OVC + 8B prefix + 4B perm per record)\n", ovc_total/1e9);

        // Allocate prefix merge buffers first (persistent across run gen)
        CUDA_CHECK(cudaMalloc(&d_ovc_buffer, num_records * sizeof(uint64_t)));     // bytes 8-15
        CUDA_CHECK(cudaMalloc(&d_prefix_buffer, num_records * sizeof(uint64_t)));  // bytes 0-7
        CUDA_CHECK(cudaMalloc(&d_global_perm, num_records * sizeof(uint32_t)));
        ovc_buffer_records = num_records;
        ovc_run_offset = 0;

        // ── Native 32B OVC extension: try to allocate pfx3/pfx4 ──────────
        // With 4 × 8B prefix buffers the GPU merge can do 4-pass LSD on the
        // full 32B compact prefix instead of 2-pass on 16B. The compact prefix
        // is source-position-ordered, so correctness is preserved (compact-lex
        // ≤ full-key-lex on canonicals). Extraction reads the record once and
        // writes 4 uint64s instead of 2 — zero extra PCIe.
        //
        // Memory-conditional: only if total merge-peak footprint fits. At merge
        // time sort_ws is freed; peak = 4 × 8B (pfx1..4) + 4B (perm) + 8B (alt)
        // + 2 × 4B (perm_save + perm_alt) + CUB scratch ≈ 52B × N + ~1 GB.
        //
        //   SF50   (300 M): 15.6 GB + 1 GB = 16.6 GB ≤ 25 GB ✓
        //   SF100  (600 M): 31.2 GB + 1 GB = 32.2 GB > 25 GB ✗
        //
        // Gate: num_records * 60 + 1 GB < free_mem * 0.90. Override with
        // env OVC_32B=0 (force 16B) or OVC_32B=1 (attempt regardless; still
        // checks cudaMalloc return).
        bool try_32b = true;
        if (const char* e = getenv("OVC_32B")) {
            if (std::string(e) == "0") try_32b = false;
            // Any non-"0" value forces attempt.
        }
        if (try_32b) {
            size_t free_now_32b, total_32b;
            CUDA_CHECK(cudaMemGetInfo(&free_now_32b, &total_32b));
            size_t needed_32b = num_records * 52ULL + (size_t)(1ULL << 30); // +1 GB for CUB
            const char* force = getenv("OVC_32B");
            bool force_attempt = (force && std::string(force) != "0");
            if (!force_attempt && needed_32b > (size_t)(free_now_32b * 0.90)) {
                printf("  32B OVC extension: SKIPPED (would need %.1f GB, have %.1f GB usable)\n",
                       needed_32b / 1e9, free_now_32b * 0.90 / 1e9);
            } else {
                cudaError_t e3 = cudaMalloc(&d_pfx3_buffer, num_records * sizeof(uint64_t));
                cudaError_t e4 = (e3 == cudaSuccess) ?
                    cudaMalloc(&d_pfx4_buffer, num_records * sizeof(uint64_t))
                    : cudaErrorMemoryAllocation;
                if (e3 != cudaSuccess || e4 != cudaSuccess) {
                    if (d_pfx3_buffer) { cudaFree(d_pfx3_buffer); d_pfx3_buffer = nullptr; }
                    if (d_pfx4_buffer) { cudaFree(d_pfx4_buffer); d_pfx4_buffer = nullptr; }
                    printf("  32B OVC extension: cudaMalloc failed, falling back to 16B\n");
                } else {
                    printf("  32B OVC extension: ENABLED (pfx3+pfx4 allocated, %.1f GB extra)\n",
                           2.0 * num_records * 8 / 1e9);
                }
            }
        } else {
            printf("  32B OVC extension: DISABLED via OVC_32B=0\n");
        }

        // Memory strategy: try compact-key-only upload first (3.75× less PCIe).
        // If d_compact fits, we only need 1 CUB scratch buffer (not 3 full d_bufs).
        size_t ovc_used, ovc_dummy;
        CUDA_CHECK(cudaMemGetInfo(&ovc_used, &ovc_dummy));

#ifdef USE_COMPACT_KEY
        // Estimate chunk size for compact upload mode: only need sort_ws (no d_buf[0/1])
        // sort_ws per record: keys(8) + keys_alt(8) + indices(4) + indices_alt(4) + compact(32) = 56B
        // Plus 1 CUB scratch buffer
        if (!g_disable_compact) {
            size_t compact_budget = (size_t)(ovc_used * 0.92);
            // Query CUB for actual scratch needed (then pad generously)
            size_t cub_temp_query = 0;
            {
                cub::DoubleBuffer<uint64_t> k(nullptr, nullptr);
                cub::DoubleBuffer<uint32_t> v(nullptr, nullptr);
                // Use a large N estimate for the query
                cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_query,
                    k, v, (int)std::min(num_records, (uint64_t)200000000), 0, 64);
            }
            size_t cub_scratch_size = std::max(cub_temp_query * 2, (size_t)(128ULL * 1024 * 1024));
            uint64_t compact_buf_records = (compact_budget - cub_scratch_size) /
                (8 + 8 + 4 + 4 + COMPACT_KEY_SIZE);  // 56B per record
            // Cap to fit in h_pin staging (allocated at original buf_bytes in constructor)
            uint64_t orig_buf_bytes = ((gpu_budget / NBUFS) / RECORD_SIZE) * (uint64_t)RECORD_SIZE;
            uint64_t h_pin_cap = orig_buf_bytes / COMPACT_KEY_SIZE;
            if (compact_buf_records > h_pin_cap) compact_buf_records = h_pin_cap;

            sort_ws.allocate(compact_buf_records);
            if (sort_ws.d_compact) {
                CUDA_CHECK(cudaMalloc(&d_buf[2], cub_scratch_size));
                d_buf[0] = nullptr;
                d_buf[1] = nullptr;
                buf_records = compact_buf_records;
                buf_bytes = cub_scratch_size;  // CUB scratch (used as temp_bytes in sort)

                int total_chunks = (num_records + buf_records - 1) / buf_records;
                printf("  COMPACT UPLOAD: %.0fM rec/chunk, %d chunks, "
                       "PCIe %.1f GB (vs %.1f GB full), CUB scratch %.0f MB\n",
                       compact_buf_records/1e6, total_chunks,
                       (double)num_records * COMPACT_KEY_SIZE / 1e9,
                       (double)num_records * RECORD_SIZE / 1e9,
                       cub_scratch_size/1e6);
                goto run_generation;
            }
            sort_ws.free();
            printf("  (compact alloc failed, using full-record upload)\n");
        }
#endif

        {
            size_t ovc_budget = (size_t)(ovc_used * 0.90);
            uint64_t ovc_buf_records = (ovc_budget / NBUFS) / RECORD_SIZE;
            size_t ovc_buf_bytes = ovc_buf_records * RECORD_SIZE;
            printf("  Reduced triple-buffer: %d × %.2f GB (%.0f M records each)\n",
                   NBUFS, ovc_buf_bytes/1e9, ovc_buf_records/1e6);

            for (int i = 0; i < NBUFS; i++) {
                CUDA_CHECK(cudaMalloc(&d_buf[i], ovc_buf_bytes));
            }
            buf_records = ovc_buf_records;
            buf_bytes = ovc_buf_bytes;
            sort_ws.allocate(buf_records);
        }
run_generation:

        // Run generation with per-chunk CUB sort
        printf("\n== Phase 1: Run Generation (chunked LSD sort) ==\n");
        double rg_h2d = 0, rg_d2h = 0;
        double rg_ms;
        auto runs = generate_runs_pipelined(h_data, num_records, rg_ms, rg_h2d, rg_d2h);
        r.run_gen_ms = rg_ms;
        r.num_runs = runs.size();
        r.pcie_h2d_gb = rg_h2d / 1e9;
        printf("  %d runs in %.0f ms (%.2f GB/s)\n\n", r.num_runs, rg_ms, total_bytes/(rg_ms*1e6));

        // Phase 2: GPU Prefix Merge — CUB radix sort on (8B prefix, global_index)
        // The 8B prefix is an ABSOLUTE value (not relative like OVC), so CUB
        // radix sort gives correct ordering. LSD: sort by global_index first
        // (preserves run-internal order for ties), then by prefix.
        // This replaces the broken OVC merge-path approach.
        sort_ws.free();
        for (int i = 0; i < NBUFS; i++) { if (d_buf[i]) { cudaFree(d_buf[i]); d_buf[i] = nullptr; } }
        if (d_key_buffer) { cudaFree(d_key_buffer); d_key_buffer = nullptr; }

        // Two merge paths:
        //  - 2-pass: 16B prefix (pfx1+pfx2). Memory-parsimonious; used on SF100 where
        //    pfx3/pfx4 wouldn't fit.
        //  - 4-pass: 32B prefix (pfx1..pfx4). Used when pfx3/pfx4 were allocated
        //    during setup (SF50-class). ~150-300 ms extra merge time pays for
        //    itself when fixup is the bottleneck (SF50 ≈ 7-10 s fixup).
        bool use_32b_merge = (d_pfx3_buffer != nullptr && d_pfx4_buffer != nullptr);
        int effective_compact_prefix_bytes = use_32b_merge ? 32 : 16;

        if (use_32b_merge) {
            printf("== Phase 2: GPU 32B Prefix Merge (4 CUB LSD passes) ==\n");
        } else {
            printf("== Phase 2: GPU 16B Prefix Merge (2 CUB LSD passes) ==\n");
        }

        // We have: d_prefix_buffer (pfx1, bytes 0-7), d_ovc_buffer (pfx2, bytes 8-15),
        // optionally d_pfx3_buffer (bytes 16-23), d_pfx4_buffer (bytes 24-31),
        // d_global_perm (global record indices).
        //
        // LSD merge: sort from least significant 8B prefix to most significant.
        // CUB SortPairs carries only ONE satellite array; after each pass we gather
        // all other live arrays using the shuffle indices CUB produces when we pass
        // an identity array as the satellite.

        int nthreads2 = 256;
        int nblks2 = (num_records + nthreads2 - 1) / nthreads2;

        {
            size_t fr, tt;
            cudaMemGetInfo(&fr, &tt);
            printf("  GPU free before merge alloc: %.2f GB\n", fr/1e9);
        }

        // Allocate merge workspace. Same for both paths:
        // 1 extra 8B alt + 2 extra 4B slots (perm_save + perm_alt) + CUB scratch.
        uint64_t* d_pfx_alt;     // CUB key double-buffer alt (8B×N)
        uint32_t* d_perm_save;   // Backup of original perm before CUB overwrites it (4B×N)
        uint32_t* d_perm_alt;    // CUB val double-buffer / identity init (4B×N)
        CUDA_CHECK(cudaMalloc(&d_pfx_alt, num_records * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_perm_save, num_records * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_perm_alt, num_records * sizeof(uint32_t)));

        size_t cub_merge_temp = 0;
        {
            cub::DoubleBuffer<uint64_t> kb(nullptr,nullptr);
            cub::DoubleBuffer<uint32_t> vb(nullptr,nullptr);
            cub::DeviceRadixSort::SortPairs(nullptr, cub_merge_temp, kb, vb, (int)num_records, 0, 64);
        }
        void* d_cub_merge_temp;
        CUDA_CHECK(cudaMalloc(&d_cub_merge_temp, cub_merge_temp));

        {
            size_t fr, tt;
            cudaMemGetInfo(&fr, &tt);
            printf("  GPU free after merge alloc: %.2f GB\n", fr/1e9);
        }

        // Save original physical allocation pointers so cleanup is unambiguous.
        // Through rotations below, the class-member pointers can end up aliasing
        // the logical current-perm/current-pfx; freeing by original alloc avoids
        // double-frees regardless of CUB's internal pass parity.
        uint64_t* const alloc_p1 = d_prefix_buffer;
        uint64_t* const alloc_p2 = d_ovc_buffer;
        uint64_t* const alloc_p3 = d_pfx3_buffer;   // nullptr on 2-pass path
        uint64_t* const alloc_p4 = d_pfx4_buffer;   // nullptr on 2-pass path
        uint32_t* const alloc_perm = d_global_perm;

        // Save original perm — CUB will use d_global_perm as val double-buffer alt
        CUDA_CHECK(cudaMemcpy(d_perm_save, d_global_perm,
                               num_records * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

        // Init identity index for pass 1 satellite
        init_identity_kernel<<<nblks2, nthreads2>>>(d_perm_alt, num_records);

        WallTimer merge_timer;
        uint32_t h_has_ties = 0;
        uint32_t* final_perm = nullptr;     // holds final sorted perm after last pass
        double merge_ms = 0;

        if (use_32b_merge) {
            // ── 4-pass LSD: pfx4 → pfx3 → pfx2 → pfx1 ──
            // Logical pointers track which physical slot currently holds each array.
            // After each pass one src slot becomes stale (src already read, dst written
            // elsewhere), providing the free 8B alt for the next pass — so we need
            // only 1 extra 8B alloc beyond the 4 pfx buffers.
            uint64_t* cur_pfx1 = alloc_p1;
            uint64_t* cur_pfx2 = alloc_p2;
            uint64_t* cur_pfx3 = alloc_p3;
            uint64_t* cur_pfx4 = alloc_p4;
            uint64_t* free_pfx_alt = d_pfx_alt;
            uint32_t* cur_perm = alloc_perm;
            uint32_t* free_4B_a = d_perm_alt;  // initialized to identity above
            uint32_t* free_4B_b = d_perm_save; // src for perm gather in pass 1

            // ── PASS 1: Sort by pfx4 (LSB) ──
            merge_timer.begin();
            {
                cub::DoubleBuffer<uint64_t> kb1(cur_pfx4, free_pfx_alt);
                cub::DoubleBuffer<uint32_t> vb1(free_4B_a, cur_perm);
                size_t temp = cub_merge_temp;
                cub::DeviceRadixSort::SortPairs(d_cub_merge_temp, temp,
                    kb1, vb1, (int)num_records, 0, 64);

                uint64_t* sorted_p4 = kb1.Current();
                uint64_t* junk_8B  = kb1.Alternate();
                uint32_t* shuffle  = vb1.Current();
                uint32_t* junk_4B  = vb1.Alternate();

                // Gather pfx1 → junk_8B. cur_pfx1 slot becomes stale.
                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx1, shuffle, junk_8B, num_records);
                uint64_t* p1_next = junk_8B;
                uint64_t* slot_pfx1_stale = cur_pfx1;
                // Gather pfx2 → stale pfx1 slot. cur_pfx2 slot becomes stale.
                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx2, shuffle, slot_pfx1_stale, num_records);
                uint64_t* p2_next = slot_pfx1_stale;
                uint64_t* slot_pfx2_stale = cur_pfx2;
                // Gather pfx3 → stale pfx2 slot. cur_pfx3 slot becomes stale (our next free alt).
                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx3, shuffle, slot_pfx2_stale, num_records);
                uint64_t* p3_next = slot_pfx2_stale;
                uint64_t* next_free_alt = cur_pfx3;
                // Gather perm (from the saved copy) → junk_4B.
                gather_uint32_kernel<<<nblks2, nthreads2>>>(free_4B_b, shuffle, junk_4B, num_records);
                uint32_t* perm_next = junk_4B;

                cur_pfx4 = sorted_p4;
                cur_pfx1 = p1_next;
                cur_pfx2 = p2_next;
                cur_pfx3 = p3_next;
                free_pfx_alt = next_free_alt;
                cur_perm = perm_next;
                // shuffle slot becomes free after gathers; perm_save (free_4B_b) also free after last use
                free_4B_a = shuffle;
                free_4B_b = free_4B_b;  // still the same alloc, now stale = free
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            double pass1_ms = merge_timer.end_ms();
            printf("  Pass 1 (sort by pfx4 + gather): %.0f ms\n", pass1_ms);

            // ── PASS 2: Sort by pfx3 ──
            merge_timer.begin();
            init_identity_kernel<<<nblks2, nthreads2>>>(free_4B_a, num_records);
            {
                cub::DoubleBuffer<uint64_t> kb2(cur_pfx3, free_pfx_alt);
                cub::DoubleBuffer<uint32_t> vb2(free_4B_a, free_4B_b);
                size_t temp = cub_merge_temp;
                cub::DeviceRadixSort::SortPairs(d_cub_merge_temp, temp,
                    kb2, vb2, (int)num_records, 0, 64);

                uint64_t* sorted_p3 = kb2.Current();
                uint64_t* junk_8B  = kb2.Alternate();
                uint32_t* shuffle  = vb2.Current();
                uint32_t* junk_4B  = vb2.Alternate();

                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx1, shuffle, junk_8B, num_records);
                uint64_t* p1_next = junk_8B;
                uint64_t* slot1_stale = cur_pfx1;
                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx2, shuffle, slot1_stale, num_records);
                uint64_t* p2_next = slot1_stale;
                uint64_t* slot2_stale = cur_pfx2;
                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx4, shuffle, slot2_stale, num_records);
                uint64_t* p4_next = slot2_stale;
                uint64_t* next_free_alt = cur_pfx4;
                gather_uint32_kernel<<<nblks2, nthreads2>>>(cur_perm, shuffle, junk_4B, num_records);
                uint32_t* perm_next = junk_4B;
                uint32_t* prev_perm = cur_perm;

                cur_pfx3 = sorted_p3;
                cur_pfx1 = p1_next;
                cur_pfx2 = p2_next;
                cur_pfx4 = p4_next;
                free_pfx_alt = next_free_alt;
                cur_perm = perm_next;
                free_4B_a = shuffle;
                free_4B_b = prev_perm;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            double pass2_ms = merge_timer.end_ms();
            printf("  Pass 2 (sort by pfx3 + gather): %.0f ms\n", pass2_ms);

            // ── PASS 3: Sort by pfx2 ──
            merge_timer.begin();
            init_identity_kernel<<<nblks2, nthreads2>>>(free_4B_a, num_records);
            {
                cub::DoubleBuffer<uint64_t> kb3(cur_pfx2, free_pfx_alt);
                cub::DoubleBuffer<uint32_t> vb3(free_4B_a, free_4B_b);
                size_t temp = cub_merge_temp;
                cub::DeviceRadixSort::SortPairs(d_cub_merge_temp, temp,
                    kb3, vb3, (int)num_records, 0, 64);

                uint64_t* sorted_p2 = kb3.Current();
                uint64_t* junk_8B  = kb3.Alternate();
                uint32_t* shuffle  = vb3.Current();
                uint32_t* junk_4B  = vb3.Alternate();

                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx1, shuffle, junk_8B, num_records);
                uint64_t* p1_next = junk_8B;
                uint64_t* slot1_stale = cur_pfx1;
                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx3, shuffle, slot1_stale, num_records);
                uint64_t* p3_next = slot1_stale;
                uint64_t* slot3_stale = cur_pfx3;
                gather_uint64_kernel<<<nblks2, nthreads2>>>(cur_pfx4, shuffle, slot3_stale, num_records);
                uint64_t* p4_next = slot3_stale;
                uint64_t* next_free_alt = cur_pfx4;
                gather_uint32_kernel<<<nblks2, nthreads2>>>(cur_perm, shuffle, junk_4B, num_records);
                uint32_t* perm_next = junk_4B;
                uint32_t* prev_perm = cur_perm;

                cur_pfx2 = sorted_p2;
                cur_pfx1 = p1_next;
                cur_pfx3 = p3_next;
                cur_pfx4 = p4_next;
                free_pfx_alt = next_free_alt;
                cur_perm = perm_next;
                free_4B_a = shuffle;
                free_4B_b = prev_perm;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            double pass3_ms = merge_timer.end_ms();
            printf("  Pass 3 (sort by pfx2 + gather): %.0f ms\n", pass3_ms);

            // ── 32B tie check (between pass 3 and pass 4) ──
            // Array is stably sorted by (pfx2, pfx3, pfx4); pfx1 gathered to match.
            // An adjacent pair with all 4 prefixes equal means the 32 compact bytes
            // are identical → CPU fixup on remaining record bytes may be needed.
            {
                uint32_t* d_has_ties;
                CUDA_CHECK(cudaMalloc(&d_has_ties, sizeof(uint32_t)));
                CUDA_CHECK(cudaMemset(d_has_ties, 0, sizeof(uint32_t)));
                check_full_32B_ties_kernel<<<nblks2, nthreads2>>>(
                    cur_pfx1, cur_pfx2, cur_pfx3, cur_pfx4, num_records, d_has_ties);
                CUDA_CHECK(cudaMemcpy(&h_has_ties, d_has_ties, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cudaFree(d_has_ties);
            }
            printf("  GPU 32B tie check: %s\n",
                   h_has_ties ? "TIES FOUND — CPU fixup needed" : "NO TIES — fixup skipped");

            // ── PASS 4: Sort by pfx1 (MSB), carry perm directly ──
            merge_timer.begin();
            {
                cub::DoubleBuffer<uint64_t> kb4(cur_pfx1, free_pfx_alt);
                cub::DoubleBuffer<uint32_t> vb4(cur_perm, free_4B_a);
                size_t temp = cub_merge_temp;
                cub::DeviceRadixSort::SortPairs(d_cub_merge_temp, temp,
                    kb4, vb4, (int)num_records, 0, 64);
                final_perm = vb4.Current();
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            double pass4_ms = merge_timer.end_ms();
            printf("  Pass 4 (sort by pfx1): %.0f ms\n", pass4_ms);
            merge_ms = pass1_ms + pass2_ms + pass3_ms + pass4_ms;
            printf("  GPU 32B prefix merge total: %.0f ms\n", merge_ms);
        } else {
            // ── 2-pass LSD: pfx2 → pfx1 (existing 16B path) ──
            uint64_t* pfx1_gathered;
            uint32_t* perm_gathered;
            uint64_t* pass2_key_alt;
            uint32_t* pass2_val_alt;

            merge_timer.begin();
            {
                cub::DoubleBuffer<uint64_t> kb1(alloc_p2, d_pfx_alt);
                cub::DoubleBuffer<uint32_t> vb1(d_perm_alt, alloc_perm);
                size_t temp = cub_merge_temp;
                cub::DeviceRadixSort::SortPairs(d_cub_merge_temp, temp,
                    kb1, vb1, (int)num_records, 0, 64);
                CUDA_CHECK(cudaDeviceSynchronize());

                uint32_t* shuffle = vb1.Current();

                pfx1_gathered = kb1.Alternate();
                gather_uint64_kernel<<<nblks2, nthreads2>>>(
                    alloc_p1, shuffle, pfx1_gathered, num_records);

                perm_gathered = vb1.Alternate();
                gather_uint32_kernel<<<nblks2, nthreads2>>>(
                    d_perm_save, shuffle, perm_gathered, num_records);

                pass2_key_alt = kb1.Current();
                pass2_val_alt = vb1.Current();
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            double pass1_ms = merge_timer.end_ms();
            printf("  Pass 1 (sort by pfx2 + gather): %.0f ms\n", pass1_ms);

            {
                uint32_t* d_has_ties;
                CUDA_CHECK(cudaMalloc(&d_has_ties, sizeof(uint32_t)));
                CUDA_CHECK(cudaMemset(d_has_ties, 0, sizeof(uint32_t)));
                check_full_16B_ties_kernel<<<nblks2, nthreads2>>>(
                    pass2_key_alt, pfx1_gathered, num_records, d_has_ties);
                CUDA_CHECK(cudaMemcpy(&h_has_ties, d_has_ties, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                cudaFree(d_has_ties);
            }
            printf("  GPU 16B tie check: %s\n",
                   h_has_ties ? "TIES FOUND — CPU fixup needed" : "NO TIES — fixup skipped");

            merge_timer.begin();
            {
                cub::DoubleBuffer<uint64_t> kb2(pfx1_gathered, pass2_key_alt);
                cub::DoubleBuffer<uint32_t> vb2(perm_gathered, pass2_val_alt);
                size_t temp = cub_merge_temp;
                cub::DeviceRadixSort::SortPairs(d_cub_merge_temp, temp,
                    kb2, vb2, (int)num_records, 0, 64);
                final_perm = vb2.Current();
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            double pass2_ms = merge_timer.end_ms();
            printf("  Pass 2 (sort by pfx1): %.0f ms\n", pass2_ms);
            merge_ms = pass1_ms + pass2_ms;
            printf("  GPU 16B prefix merge total: %.0f ms\n", merge_ms);
        }

        // Download final permutation
        phase_timer.begin();
        CUDA_CHECK(cudaMemcpy(h_perm, final_perm, total_perm_bytes, cudaMemcpyDeviceToHost));
        double dl_ms = phase_timer.end_ms();
        r.pcie_d2h_gb = total_perm_bytes / 1e9;
        printf("  Downloaded perm in %.0f ms\n", dl_ms);

        // Free ALL GPU merge buffers by their ORIGINAL physical allocation pointers.
        // This avoids any aliasing hazard from CUB's internal buffer ping-pong.
        cudaFree(alloc_p1);
        cudaFree(alloc_p2);
        if (alloc_p3) cudaFree(alloc_p3);
        if (alloc_p4) cudaFree(alloc_p4);
        cudaFree(alloc_perm);
        cudaFree(d_pfx_alt);
        cudaFree(d_perm_alt);
        cudaFree(d_perm_save);
        cudaFree(d_cub_merge_temp);
        d_prefix_buffer = nullptr;
        d_ovc_buffer = nullptr;
        d_pfx3_buffer = nullptr;
        d_pfx4_buffer = nullptr;
        d_global_perm = nullptr;

        // CPU gather: apply permutation to original h_data
        // Uses non-temporal stores to avoid cache pollution on the write side,
        // freeing L3 cache for random reads from h_data.
        phase_timer.begin();
        printf("== Phase 3: CPU Gather ==\n");
        int hw_threads = std::max(1, (int)std::thread::hardware_concurrency());
        int PREFETCH_AHEAD = 512;
        uint64_t chunk_per_t = (num_records + hw_threads - 1) / hw_threads;
        std::vector<std::thread> threads;
        for (int t = 0; t < hw_threads; t++) {
            threads.emplace_back([&, t]() {
                uint64_t lo = t * chunk_per_t;
                uint64_t hi = std::min(lo + chunk_per_t, num_records);
                constexpr int WORDS = RECORD_SIZE / 8;  // 120/8 = 15
                for (uint64_t i = lo; i < hi; i++) {
                    if (i + PREFETCH_AHEAD < hi)
                        __builtin_prefetch(h_data + (uint64_t)h_perm[i+PREFETCH_AHEAD] * RECORD_SIZE, 0);
                    const int64_t* src = (const int64_t*)(h_data + (uint64_t)h_perm[i] * RECORD_SIZE);
                    int64_t* dst = (int64_t*)(h_output + i * RECORD_SIZE);
                    // Non-temporal streaming stores (bypass write cache)
                    for (int w = 0; w < WORDS; w++)
                        _mm_stream_si64((long long*)(dst + w), src[w]);
                }
                _mm_sfence();
            });
        }
        for (auto& t : threads) t.join();
        double gather_ms = phase_timer.end_ms();
        printf("  Gathered %.2f GB in %.0f ms (%.2f GB/s)\n",
               total_bytes/1e9, gather_ms, total_bytes/(gather_ms*1e6));

        // CPU fixup: only needed if GPU detected pfx1 ties (very rare with compact key).
        phase_timer.begin();
        double fixup_ms = 0;

#ifdef USE_COMPACT_KEY
        // Use runtime-detected compact map (filled by detect_compact_map at sort start)
        const int* h_compact_map = g_compact_map;
#endif

        // Determine effective prefix bytes the GPU sorted by.
        // With 4-pass merge (pfx1..pfx4) the GPU covers 32 compact bytes;
        // with 2-pass it covers 16. effective_compact_prefix_bytes was set
        // in Phase 2 based on whether d_pfx3/pfx4 were allocated.
        int effective_prefix_bytes;
#ifdef USE_COMPACT_KEY
        if (used_compact_prefix) {
            effective_prefix_bytes = std::min(effective_compact_prefix_bytes, g_compact_count);
        } else
#endif
        {
            effective_prefix_bytes = std::min(16, (int)KEY_SIZE);
        }

        // Skip fixup if: (a) no pfx1 ties detected on GPU, or (b) prefix covers full key
        bool need_fixup = h_has_ties;
#ifdef USE_COMPACT_KEY
        // If compact has ≤ effective_compact_prefix_bytes varying bytes total,
        // the GPU prefix IS the full sort key
        if (used_compact_prefix && g_compact_count <= effective_compact_prefix_bytes) need_fixup = false;
#endif
        if (KEY_SIZE <= 16) need_fixup = false;

        if (!need_fixup) {
            printf("== Phase 4: CPU Fixup (skipped — %s) ==\n",
                   !h_has_ties ? "no pfx1 ties" : "prefix covers full key");
        } else {
            printf("== Phase 4: CPU Prefix Fixup (%dB prefix, ties detected) ==\n", effective_prefix_bytes);
        }

        if (need_fixup) {
            int hw = std::max(1, (int)std::thread::hardware_concurrency());
            if (const char* e = getenv("FIXUP_THREADS")) { int v = atoi(e); if (v > 0) hw = v; }
            std::vector<std::pair<uint64_t, uint64_t>> groups;
            uint64_t gs = 0;

            auto grp_t0 = std::chrono::high_resolution_clock::now();
            // Parallel group detection: each thread finds prefix-change boundaries
            // in its chunk [lo, hi); serial stitch assembles groups of size > 1.
            // Prior serial scan was 4.8s on SF50 — the main Phase-4 bottleneck.
            std::vector<std::vector<uint64_t>> boundaries_per_thread(hw);
            {
                uint64_t rpt = (num_records + hw - 1) / hw;
                std::vector<std::thread> gd_threads;
                for (int t = 0; t < hw; t++) {
                    gd_threads.emplace_back([&, t]() {
                        uint64_t lo = (uint64_t)t * rpt;
                        uint64_t hi = std::min(lo + rpt, num_records);
                        if (lo == 0) lo = 1;   // boundary check needs [i-1, i]
                        auto& my_bnd = boundaries_per_thread[t];
                        // Rough estimate: 15k boundaries / 48 threads = ~350 per thread
                        my_bnd.reserve(1024);
                        for (uint64_t i = lo; i < hi; i++) {
                            const uint8_t* ra = h_output + (i-1) * RECORD_SIZE;
                            const uint8_t* rb = h_output + i * RECORD_SIZE;
                            bool boundary;
#ifdef USE_COMPACT_KEY
                            if (used_compact_prefix) {
                                boundary = false;
                                for (int c = 0; c < effective_prefix_bytes; c++) {
                                    if (ra[h_compact_map[c]] != rb[h_compact_map[c]]) { boundary = true; break; }
                                }
                            } else
#endif
                            {
                                boundary = (memcmp(ra, rb, effective_prefix_bytes) != 0);
                            }
                            if (boundary) my_bnd.push_back(i);
                        }
                    });
                }
                for (auto& t : gd_threads) t.join();
            }
            auto scan_t1 = std::chrono::high_resolution_clock::now();
            double scan_ms = std::chrono::duration_cast<std::chrono::microseconds>(scan_t1 - grp_t0).count() / 1e3;
            // Serial stitch: build groups list by concatenating boundaries.
            {
                std::vector<uint64_t> all_boundaries;
                all_boundaries.reserve(16384);
                all_boundaries.push_back(0);
                for (int t = 0; t < hw; t++) {
                    auto& bv = boundaries_per_thread[t];
                    all_boundaries.insert(all_boundaries.end(), bv.begin(), bv.end());
                }
                all_boundaries.push_back(num_records);
                for (size_t i = 0; i + 1 < all_boundaries.size(); i++) {
                    uint64_t count = all_boundaries[i+1] - all_boundaries[i];
                    if (count > 1) groups.push_back({all_boundaries[i], count});
                }
            }
            (void)gs;
            auto grp_t1 = std::chrono::high_resolution_clock::now();
            double grp_ms = std::chrono::duration_cast<std::chrono::microseconds>(grp_t1 - grp_t0).count() / 1e3;
            double stitch_ms = grp_ms - scan_ms;
            printf("  %lu groups with ties (avg %.0f records, scan %.0f + stitch %.0f ms)\n",
                   groups.size(),
                   groups.empty() ? 0.0 : (double)num_records / groups.size(),
                   scan_ms, stitch_ms);

            // Build list of "active" byte positions to compare during fixup.
            // In the compact path, invariant bytes are identical across all canonical
            // records (by the definition of "invariant" + verification that all records
            // conform). So within any tied group we only need to compare compact-map
            // positions NOT covered by the prefix — everything else is equal.
            // compact_map is position-ordered (ascending) ⇒ lex order preserved.
            int active_bytes[KEY_SIZE]; int n_active = 0;
#ifdef USE_COMPACT_KEY
            if (used_compact_prefix) {
                int start = std::min(effective_prefix_bytes, g_compact_count);
                for (int c = start; c < g_compact_count; c++) {
                    int pos = h_compact_map[c];
                    if (pos >= 0 && pos < (int)KEY_SIZE) active_bytes[n_active++] = pos;
                }
            } else
#endif
            {
                // Non-compact path: skip the first effective_prefix_bytes (they're
                // equal within a tied group by construction).
                for (int b = effective_prefix_bytes; b < (int)KEY_SIZE; b++)
                    active_bytes[n_active++] = b;
            }

            // Group-size distribution stats (to inform parallelism strategy).
            {
                uint64_t gmin = UINT64_MAX, gmax = 0, gsum = 0;
                uint64_t big_1k = 0, big_10k = 0, big_100k = 0, big_1m = 0, big_10m = 0;
                for (auto [s, c] : groups) {
                    if (c < gmin) gmin = c;
                    if (c > gmax) gmax = c;
                    gsum += c;
                    if (c >= 10000000) big_10m++;
                    else if (c >= 1000000) big_1m++;
                    else if (c >= 100000) big_100k++;
                    else if (c >= 10000) big_10k++;
                    else if (c >= 1000) big_1k++;
                }
                printf("  Groups stats: min=%lu max=%lu sum=%lu (distribution: 1k=%lu 10k=%lu 100k=%lu 1m=%lu 10m=%lu)\n",
                       gmin, gmax, gsum, big_1k, big_10k, big_100k, big_1m, big_10m);
            }

            // Global max-count for pre-allocating scratch (cheaper than per-thread
            // accounting now that work is distributed dynamically via atomic counter).
            uint32_t global_max_count = 0;
            for (auto [s, c] : groups)
                if (c > global_max_count) global_max_count = (uint32_t)c;

            // Per-thread phase timing for profiling.
            struct PhaseTimes { uint64_t pack_ns = 0, sort_ns = 0, reorder_ns = 0, skipped = 0; };
            std::vector<PhaseTimes> pt(hw);

            auto par_t0 = std::chrono::high_resolution_clock::now();
            // Atomic work-queue dispatch: each thread pulls a batch of groups at a
            // time. Eliminates load imbalance from fixed chunks (max group can be 7×
            // avg). Within each group we do: build 8B BE key from first ≤8 active
            // bytes, sort std::pair<key,orig_idx>, sub-sort runs of tied keys over
            // remaining active bytes, then reorder records.
            //
            // Fast path: when active_bytes form a contiguous run in the record (the
            // common case on compact-key TPC-H after the prefix has consumed all
            // sparse varying positions), we can do the 8B key via a single unaligned
            // 64-bit load + bswap, and the tail compare via memcmp — ~4-5× faster
            // than per-byte shift/load.
            std::atomic<uint64_t> next_group{0};
            const int kbytes = std::min(n_active, 8);
            const bool has_tail = (n_active > 8);
            bool ab_contig = (n_active > 0);
            for (int i = 1; i < n_active; i++) {
                if (active_bytes[i] != active_bytes[0] + i) { ab_contig = false; break; }
            }
            const int ab_base = ab_contig && n_active > 0 ? active_bytes[0] : 0;
            const int ab_tail_off = ab_base + kbytes;     // record byte offset for tail
            const int ab_tail_len = ab_contig ? (n_active - kbytes) : 0;
            if (ab_contig) {
                printf("  Fixup: active bytes are contiguous run [%d..%d] (8B key + %dB tail, fast path enabled)\n",
                       ab_base, ab_base + n_active - 1, ab_tail_len);
            }
            if (!groups.empty()) {
                std::vector<std::thread> fix_threads;
                for (int t = 0; t < hw; t++) {
                    fix_threads.emplace_back([&, t]() {
                        std::vector<std::pair<uint64_t, uint32_t>> keys(global_max_count);
                        std::vector<uint8_t> reorder_buf((size_t)global_max_count * RECORD_SIZE);
                        PhaseTimes& myp = pt[t];
                        // Batch size scales with total group count so every thread gets
                        // work even when #groups is small (e.g. adversarial inputs with
                        // ~1000 huge groups). 64 was fine for typical 3.15M-group SF50
                        // but capped parallelism at ~16 threads for low-count inputs.
                        const uint64_t batch = std::max<uint64_t>(1,
                            std::min<uint64_t>(64, (groups.size() + hw * 4 - 1) / (hw * 4)));
                        uint64_t gs_local;
                        while ((gs_local = next_group.fetch_add(batch, std::memory_order_relaxed)) < groups.size()) {
                            uint64_t ge_local = std::min(gs_local + batch, (uint64_t)groups.size());
                            for (uint64_t g = gs_local; g < ge_local; g++) {
                            auto [start, count] = groups[g];
                            uint8_t* base = h_output + start * RECORD_SIZE;

                            auto t0 = std::chrono::high_resolution_clock::now();
                            // Build 8-byte BE keys directly from record bytes.
                            if (ab_contig) {
                                // Fast path: one unaligned 8B load + bswap per record.
                                for (uint64_t j = 0; j < count; j++) {
                                    const uint8_t* src = base + j * RECORD_SIZE;
                                    uint64_t k;
                                    memcpy(&k, src + ab_base, 8);
                                    k = __builtin_bswap64(k);
                                    if (kbytes < 8) k &= (~(uint64_t)0) << ((8 - kbytes) * 8);
                                    keys[j].first = k;
                                    keys[j].second = (uint32_t)j;
                                }
                            } else {
                                // Scatter path: per-byte load + shift.
                                for (uint64_t j = 0; j < count; j++) {
                                    const uint8_t* src = base + j * RECORD_SIZE;
                                    uint64_t k = 0;
                                    for (int b = 0; b < kbytes; b++) {
                                        k = (k << 8) | (uint64_t)src[active_bytes[b]];
                                    }
                                    if (kbytes < 8) k <<= (size_t)(8 - kbytes) * 8;
                                    keys[j].first = k;
                                    keys[j].second = (uint32_t)j;
                                }
                            }
                            auto t1 = std::chrono::high_resolution_clock::now();
                            myp.pack_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

                            // Primary sort on 8B key (fast uint64 compare).
                            std::sort(keys.begin(), keys.begin() + count);

                            // Sub-sort: find runs of equal keys and break ties over
                            // remaining active bytes [kbytes..n_active). For data with
                            // high entropy in the first 8 active bytes, most runs are
                            // size 1 and this loop exits almost immediately.
                            if (has_tail) {
                                uint64_t i = 0;
                                while (i < count) {
                                    uint64_t j = i + 1;
                                    while (j < count && keys[j].first == keys[i].first) j++;
                                    if (j > i + 1) {
                                        if (ab_contig) {
                                            // Fast path: memcmp on contiguous tail bytes.
                                            std::sort(keys.begin() + i, keys.begin() + j,
                                                [&](const std::pair<uint64_t,uint32_t>& a,
                                                    const std::pair<uint64_t,uint32_t>& b) {
                                                    const uint8_t* ra = base + (uint64_t)a.second * RECORD_SIZE + ab_tail_off;
                                                    const uint8_t* rb = base + (uint64_t)b.second * RECORD_SIZE + ab_tail_off;
                                                    return memcmp(ra, rb, ab_tail_len) < 0;
                                                });
                                        } else {
                                            std::sort(keys.begin() + i, keys.begin() + j,
                                                [&](const std::pair<uint64_t,uint32_t>& a,
                                                    const std::pair<uint64_t,uint32_t>& b) {
                                                    const uint8_t* ra = base + (uint64_t)a.second * RECORD_SIZE;
                                                    const uint8_t* rb = base + (uint64_t)b.second * RECORD_SIZE;
                                                    for (int k = kbytes; k < n_active; k++) {
                                                        uint8_t ba = ra[active_bytes[k]];
                                                        uint8_t bb = rb[active_bytes[k]];
                                                        if (ba != bb) return ba < bb;
                                                    }
                                                    return false;
                                                });
                                        }
                                    }
                                    i = j;
                                }
                            }
                            auto t2 = std::chrono::high_resolution_clock::now();
                            myp.sort_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

                            bool already_sorted = true;
                            for (uint64_t j = 0; j < count; j++) {
                                if (keys[j].second != (uint32_t)j) { already_sorted = false; break; }
                            }
                            if (already_sorted) { myp.skipped++; continue; }
                            for (uint64_t j = 0; j < count; j++)
                                memcpy(reorder_buf.data() + j*RECORD_SIZE,
                                       base + (uint64_t)keys[j].second*RECORD_SIZE, RECORD_SIZE);
                            memcpy(base, reorder_buf.data(), count * RECORD_SIZE);
                            auto t3 = std::chrono::high_resolution_clock::now();
                            myp.reorder_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
                            }
                        }
                    });
                }
                for (auto& t : fix_threads) t.join();
            }
            auto par_t1 = std::chrono::high_resolution_clock::now();
            double par_ms = std::chrono::duration_cast<std::chrono::microseconds>(par_t1 - par_t0).count() / 1e3;
            fixup_ms = phase_timer.end_ms();
            uint64_t pk_ns = 0, so_ns = 0, ro_ns = 0, skip = 0;
            for (auto& p : pt) {
                pk_ns += p.pack_ns; so_ns += p.sort_ns; ro_ns += p.reorder_ns; skip += p.skipped;
            }
            double dn = 1e6 * (double)hw;
            printf("  Fixup: %.0f ms [group-detect %.0f + parallel %.0f] (per-thread: pack %.1f + sort %.1f + reorder %.1f ms, %lu skipped)\n",
                   fixup_ms, grp_ms, par_ms, pk_ns/dn, so_ns/dn, ro_ns/dn, skip);
        }

        r.merge_ms = merge_ms + dl_ms + gather_ms + fixup_ms;
        r.merge_passes = 2;
        r.sorted_output = h_output;
        r.sorted_output_size = total_bytes;
        r.sorted_output_is_mmap = h_output_is_mmap;
        r.total_ms = r.run_gen_ms + r.merge_ms;
        printf("  Total merge+gather+fixup: %.0f ms\n", r.merge_ms);
#ifdef USE_COMPACT_KEY
        // Hybrid correctness: the CPU extraction pass classified every record
        // as canonical (non-mapped bytes == V[]) or exception. If any exception
        // exists, the compact sort can't be trusted and we signal the caller to
        // retry with the full-key upload path.
        {
            size_t n_exc;
            {
                std::lock_guard<std::mutex> lk(g_exception_mtx);
                n_exc = g_exception_indices.size();
            }
            int v = g_violated_byte.load();
            if (n_exc == 0 && v == -1) {
                printf("[Correctness] Sample map verified against all %lu records — "
                       "compact sort is FULL-KEY-EQUIVALENT.\n", num_records);
                g_map_full_scan_verified = true;
            } else {
                fprintf(stderr,
                    "[Hybrid] %zu exception records found (first differing byte %d).\n"
                    "  Compact sort not safe. Discarding result and falling back to the\n"
                    "  full-key upload path for a guaranteed-correct sort.\n",
                    n_exc, v);
                r.sorted_output = nullptr;
                r.sorted_output_size = 0;
                r.needs_hybrid_retry = true;
            }
        }
#endif
        return r;
    }

    // Strided DMA: extract only KEY_SIZE bytes per record from host
    CUDA_CHECK(cudaMalloc(&d_keys_10byte, total_keys_bytes));
    CUDA_CHECK(cudaMemcpy2DAsync(
        d_keys_10byte, KEY_SIZE,
        h_data, RECORD_SIZE,
        KEY_SIZE, num_records,
        cudaMemcpyHostToDevice, streams[0]));
    r.pcie_h2d_gb = total_keys_bytes / 1e9;

    // Allocate arena for CUB sort: uint64 keys + uint32 perm + CUB temp
    size_t cub_temp_bytes = 0;
    {
        cub::DoubleBuffer<uint64_t> dk(nullptr,nullptr);
        cub::DoubleBuffer<uint32_t> dp(nullptr,nullptr);
        cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, dk, dp, (int)num_records, 0, 64);
    }
    size_t arena_sz = num_records * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + cub_temp_bytes;
    uint8_t* d_arena;
    CUDA_CHECK(cudaMalloc(&d_arena, arena_sz));

    uint64_t* d_sort_keys = (uint64_t*)d_arena;
    uint64_t* d_sort_keys_alt = d_sort_keys + num_records;
    d_perm_in = (uint32_t*)(d_sort_keys_alt + num_records);
    d_perm_out = d_perm_in + num_records;
    void* d_temp = (void*)(d_perm_out + num_records);

    double upload_ms = phase_timer.end_ms();
    printf("  Uploaded %.2f GB keys (strided, GPU-direct) in %.0f ms (%.2f GB/s effective)\n",
           total_keys_bytes/1e9, upload_ms, total_bytes/(upload_ms*1e6));

    // ── Step 3: GPU sort (extract uint64, init perm, CUB radix sort) ──
    printf("== Step 3: GPU CUB Radix Sort ==\n");
    phase_timer.begin();

    int nthreads = 256, nblks = (num_records + nthreads - 1) / nthreads;

    // ── LSD Multi-Pass Radix Sort for full KEY_SIZE correctness ──
    // Sort from least significant 8-byte chunk to most significant.
    // For KEY_SIZE=10: 2 passes (bytes 8-9, then bytes 0-7)
    // For KEY_SIZE=88: 11 passes (bytes 80-87, 72-79, ..., 0-7)
    // CUB radix sort is stable, so LSD ordering is correct.

    init_identity_kernel<<<nblks, nthreads, 0, streams[0]>>>(d_perm_in, num_records);

    int num_chunks = (KEY_SIZE + 7) / 8;  // ceil(KEY_SIZE / 8)
    printf("  LSD sort: %d passes for %d-byte key\n", num_chunks, KEY_SIZE);

    GpuTimer pass_timer;
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int byte_offset = chunk * 8;
        int chunk_bytes = std::min(8, KEY_SIZE - byte_offset);
        pass_timer.begin();

        // Extract uint64 from this chunk of the key (in permutation order)
        // Use a kernel that reads key[perm[i]][byte_offset:byte_offset+8]
        // For the FIRST pass (highest chunk idx), perm is identity → read directly
        if (chunk == num_chunks - 1 && chunk_bytes <= 2) {
            // Last chunk is ≤2 bytes — use uint16 sort (fewer radix passes)
            uint16_t* d_tie = reinterpret_cast<uint16_t*>(d_sort_keys);
            uint16_t* d_tie_alt = reinterpret_cast<uint16_t*>(d_sort_keys_alt);
            extract_tiebreaker_kernel<<<nblks, nthreads, 0, streams[0]>>>(
                d_keys_10byte, d_perm_in, d_tie, num_records, byte_offset);
            cub::DoubleBuffer<uint16_t> tie_buf(d_tie, d_tie_alt);
            cub::DoubleBuffer<uint32_t> perm_buf(d_perm_in, d_perm_out);
            size_t t = cub_temp_bytes;
            cub::DeviceRadixSort::SortPairs(d_temp, t,
                tie_buf, perm_buf, (int)num_records, 0, 16, streams[0]);
            d_perm_in = perm_buf.Current();
            d_perm_out = perm_buf.Alternate();
        } else {
            // Extract uint64 for this chunk, in current permutation order
            // Kernel: d_sort_keys[i] = big-endian uint64 from key[perm[i]][byte_offset:+8]
            extract_uint64_chunk_kernel<<<nblks, nthreads, 0, streams[0]>>>(
                d_keys_10byte, d_perm_in, d_sort_keys, num_records, byte_offset, chunk_bytes);

            cub::DoubleBuffer<uint64_t> keys_buf(d_sort_keys, d_sort_keys_alt);
            cub::DoubleBuffer<uint32_t> perm_buf(d_perm_in, d_perm_out);
            cub::DeviceRadixSort::SortPairs(d_temp, cub_temp_bytes,
                keys_buf, perm_buf, (int)num_records, 0, chunk_bytes * 8, streams[0]);
            d_perm_in = perm_buf.Current();
            d_perm_out = perm_buf.Alternate();
        }
        float pass_ms = pass_timer.end();
        printf("    Pass %d (bytes %d-%d): %.1f ms\n", num_chunks - chunk, byte_offset, byte_offset + chunk_bytes - 1, pass_ms);
    }
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    cudaFree(d_keys_10byte);

    double sort_ms = phase_timer.end_ms();
    printf("  Sorted %lu keys in %.0f ms\n", num_records, sort_ms);

    // ── Step 4: Download permutation ──
    printf("== Step 4: Download Permutation ==\n");
    phase_timer.begin();

    CUDA_CHECK(cudaMemcpy(h_perm, d_perm_in, total_perm_bytes, cudaMemcpyDeviceToHost));
    r.pcie_d2h_gb = total_perm_bytes / 1e9;

    cudaFree(d_arena);
    if (h_keys) cudaFreeHost(h_keys);

    double download_ms = phase_timer.end_ms();
    printf("  Downloaded %.2f GB perm in %.0f ms\n", total_perm_bytes/1e9, download_ms);

    // ── Step 5: CPU multi-threaded gather ──
    printf("== Step 5: CPU Gather ==\n");
    phase_timer.begin();

    {
        int num_threads = std::min(48, (int)std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        uint64_t chunk = (num_records + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; t++) {
            uint64_t start = (uint64_t)t * chunk;
            uint64_t end = std::min(start + chunk, num_records);
            if (start < end) {
                threads.emplace_back([=, &h_data, &h_output, &h_perm]() {
                    constexpr int BLOCK = 256;
                    const uint8_t* src_ptrs[BLOCK];
                    for (uint64_t base = start; base < end; base += BLOCK) {
                        int count = std::min((uint64_t)BLOCK, end - base);
                        for (int j = 0; j < count; j++) {
                            uint32_t idx = h_perm[base + j];
                            src_ptrs[j] = h_data + (uint64_t)idx * RECORD_SIZE;
                            __builtin_prefetch(src_ptrs[j], 0, 0);
                        }
                        for (int j = 0; j < count; j++) {
                            memcpy(h_output + (base + j) * RECORD_SIZE, src_ptrs[j], RECORD_SIZE);
                        }
                    }
                });
            }
        }
        for (auto& t : threads) t.join();
    }

    double gather_ms = phase_timer.end_ms();
    printf("  Gathered %.2f GB in %.0f ms (%.2f GB/s)\n",
           total_bytes/1e9, gather_ms, total_bytes/(gather_ms*1e6));

    cudaFreeHost(h_perm);

    r.run_gen_ms = upload_ms + sort_ms + download_ms;  // no separate extract step with cudaMemcpy2D
    r.merge_ms = gather_ms;
    r.merge_passes = 1;
    r.num_runs = 1;
    r.sorted_output = h_output;
    r.sorted_output_size = total_bytes;
    r.sorted_output_is_mmap = h_output_is_mmap;
    r.total_ms = r.run_gen_ms + r.merge_ms;
    // This path uses the FULL KEY_SIZE bytes on the GPU (no compaction), so
    // the sort is full-key-equivalent by construction — the compact-map
    // verification is not applicable here.
    printf("[Correctness] Full-key sort path (no compaction) — inherently correct.\n");

    printf("\n[ExternalSort] ═══════════════════════════════════════\n");
    printf("  DONE: %.0f ms (gen: %.0f + merge: %.0f)\n",
           r.total_ms, r.run_gen_ms, r.merge_ms);
    printf("  Throughput: %.2f GB/s\n", total_bytes/(r.total_ms*1e6));
    printf("  Runs: %d | Merge passes: %d\n", r.num_runs, r.merge_passes);
    printf("  PCIe: %.1f GB H2D + %.1f GB D2H = %.1f GB total (%.1fx amplification)\n",
           r.pcie_h2d_gb, r.pcie_d2h_gb,
           r.pcie_h2d_gb + r.pcie_d2h_gb,
           (r.pcie_h2d_gb + r.pcie_d2h_gb) / (total_bytes/1e9));
    printf("═══════════════════════════════════════════════════════\n");
    return r;
}

// ════════════════════════════════════════════════════════════════════

#ifdef EXTERNAL_SORT_MAIN

static void gen_data(uint8_t* d, uint64_t n) {
    srand(42);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t* r = d + i*RECORD_SIZE;
        for (int b = 0; b < KEY_SIZE; b++) r[b] = (uint8_t)(rand() & 0xFF);
        memset(r+KEY_SIZE, 0, VALUE_SIZE);
        memcpy(r+KEY_SIZE, &i, sizeof(uint64_t));
    }
}

// Generate random data to a file in streaming fashion (no full-dataset allocation).
// Writes in 256 MB chunks to avoid pinning the whole dataset in RAM.
static void gen_data_to_file(const char* path, uint64_t num_records) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); exit(1); }

    const size_t CHUNK = 256ULL * 1024 * 1024;  // 256 MB
    uint64_t recs_per_chunk = CHUNK / RECORD_SIZE;
    uint8_t* buf = (uint8_t*)malloc(CHUNK);
    if (!buf) { fprintf(stderr, "malloc failed for gen buffer\n"); exit(1); }

    srand(42);
    uint64_t written = 0;
    while (written < num_records) {
        uint64_t n = std::min(recs_per_chunk, num_records - written);
        for (uint64_t i = 0; i < n; i++) {
            uint8_t* r = buf + i * RECORD_SIZE;
            for (int b = 0; b < KEY_SIZE; b++) r[b] = (uint8_t)(rand() & 0xFF);
            memset(r + KEY_SIZE, 0, VALUE_SIZE);
            uint64_t idx = written + i;
            memcpy(r + KEY_SIZE, &idx, sizeof(uint64_t));
        }
        fwrite(buf, RECORD_SIZE, n, f);
        written += n;
        printf("\r  Generated %lu / %lu records (%.1f%%)", written, num_records,
               100.0 * written / num_records);
        fflush(stdout);
    }
    printf("\n");
    fclose(f);
    free(buf);
}

int main(int argc, char** argv) {
    double total_gb = 0.5;
    bool verify = true;
    const char* input_file = nullptr;
    const char* output_file = nullptr;
    int num_experiment_runs = 1;
    bool streaming = false;
    const char* gen_file = nullptr;  // --gen-to-file: generate data to file (streaming)
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--total-gb") && i+1<argc) total_gb = atof(argv[++i]);
        else if (!strcmp(argv[i],"--input") && i+1<argc) input_file = argv[++i];
        else if (!strcmp(argv[i],"--output") && i+1<argc) output_file = argv[++i];
        else if (!strcmp(argv[i],"--no-verify")) verify = false;
        else if (!strcmp(argv[i],"--runs") && i+1<argc) num_experiment_runs = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--streaming")) streaming = true;
        else if (!strcmp(argv[i],"--gen-to-file") && i+1<argc) gen_file = argv[++i];
        else { printf("Usage: %s [--total-gb N] [--input FILE] [--output FILE] [--no-verify] [--runs N] [--streaming] [--gen-to-file FILE]\n",argv[0]); return 0; }
    }

    // --gen-to-file: generate random data to a file, then sort from it
    if (gen_file && !input_file) {
        uint64_t gen_records = (uint64_t)(total_gb * 1e9) / RECORD_SIZE;
        printf("Generating %.2f GB random data to %s...\n", gen_records * RECORD_SIZE / 1e9, gen_file);
        WallTimer gt; gt.begin();
        gen_data_to_file(gen_file, gen_records);
        printf("  Generated in %.0f ms\n\n", gt.end_ms());
        input_file = gen_file;
        streaming = true;  // file-backed data uses streaming mode
    }

    // Determine data size from file or --total-gb
    uint64_t num_records, total_bytes;
    if (input_file) {
        FILE* f = fopen(input_file, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
        fseek(f, 0, SEEK_END);
        total_bytes = ftell(f);
        fclose(f);
        num_records = total_bytes / RECORD_SIZE;
        total_bytes = num_records * RECORD_SIZE; // round down
    } else {
        num_records = (uint64_t)(total_gb * 1e9) / RECORD_SIZE;
        total_bytes = num_records * RECORD_SIZE;
    }

    // Auto-enable streaming if dataset is large (>80% of available RAM)
    if (!streaming && input_file) {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        uint64_t total_ram = (uint64_t)pages * page_size;
        // Need ~2.5× data for pinned mode (h_data + h_output + staging)
        if (total_bytes * 2.5 > total_ram * 0.85) {
            printf("[AutoStream] Dataset %.1f GB would need ~%.1f GB RAM (have %.1f GB) — enabling streaming mode\n",
                   total_bytes/1e9, total_bytes*2.5/1e9, total_ram/1e9);
            streaming = true;
        }
    }

    printf("════════════════════════════════════════════════════\n");
    printf("  GPU External Merge Sort — %s Benchmark\n",
           streaming ? "Streaming" : "In-Memory");
    printf("════════════════════════════════════════════════════\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    printf("GPU: %s (%.1f GB HBM, %d SMs, %.0f GB/s BW)\n", props.name,
           props.totalGlobalMem/1e9, props.multiProcessorCount,
           2.0 * props.memoryClockRate * (props.memoryBusWidth/8) / 1e6);
    printf("Data: %.2f GB (%lu records × %d bytes)%s%s\n\n",
           total_bytes/1e9, num_records, RECORD_SIZE,
           input_file ? " (from file)" : " (random)",
           streaming ? " [STREAMING]" : "");

    uint8_t* h_data;
    bool h_data_is_mmap = false;
    cudaError_t alloc_err = cudaSuccess;

    if (streaming && input_file) {
        // Streaming mode: mmap the input file instead of loading into pinned memory.
        // This avoids the 2× RAM overhead (tmpfs file + pinned copy).
        // The sort pipeline reads h_data sequentially during compact extraction
        // and randomly during gather — both work fine with mmap + page cache.
        printf("Memory-mapping %.2f GB from %s (streaming, no pinned alloc)...\n",
               total_bytes/1e9, input_file);
        WallTimer gt; gt.begin();
        int fd = open(input_file, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "Cannot open %s\n", input_file); return 1; }
        h_data = (uint8_t*)mmap(nullptr, total_bytes, PROT_READ,
                                MAP_PRIVATE | MAP_NORESERVE, fd, 0);
        close(fd);
        if (h_data == MAP_FAILED) {
            fprintf(stderr, "mmap failed for %s\n", input_file);
            return 1;
        }
        h_data_is_mmap = true;
        // Advise the kernel about access patterns
        madvise(h_data, total_bytes, MADV_SEQUENTIAL);  // mostly sequential during extraction
        madvise(h_data, total_bytes, MADV_HUGEPAGE);
        printf("  Mapped in %.0f ms (pages will fault on demand)\n\n", gt.end_ms());
    } else {
        printf("Allocating %.2f GB pinned host memory...\n", total_bytes/1e9);
        alloc_err = cudaMallocHost(&h_data, total_bytes);
        if (alloc_err != cudaSuccess) {
            printf("  cudaMallocHost failed, falling back to malloc\n");
            h_data = (uint8_t*)malloc(total_bytes);
        }
        if (!h_data) { fprintf(stderr,"allocation failed\n"); return 1; }
        madvise(h_data, total_bytes, MADV_HUGEPAGE);

        if (input_file) {
            printf("Loading from %s...\n", input_file);
            WallTimer gt; gt.begin();
            FILE* f = fopen(input_file, "rb");
            size_t read = fread(h_data, 1, total_bytes, f);
            fclose(f);
            printf("  Loaded %.2f GB in %.0f ms (%.2f GB/s)\n\n",
                   read/1e9, gt.end_ms(), read/(gt.end_ms()*1e6));
        } else {
            printf("Generating random data...\n");
            WallTimer gt; gt.begin();
            gen_data(h_data, num_records);
            printf("  Generated in %.0f ms\n\n", gt.end_ms());
        }
    }

    // Set global streaming flag for sort() internals
    g_streaming_mode = streaming;
    if (streaming) {
        printf("[Streaming] Memory-efficient mode: output buffers use MAP_NORESERVE\n");
        printf("[Streaming] Peak RAM ~= 1× data size (vs ~2.5× in pinned mode)\n\n");
    }

    if (const char* e = getenv("ADV_ZERO_BYTES")) {
        int z = atoi(e);
        if (z > 0 && z <= KEY_SIZE) {
            printf("[ADVERSARIAL] Zeroing first %d bytes of each record (%lu records)...\n", z, num_records);
            WallTimer zt; zt.begin();
            #pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < num_records; i++)
                memset(h_data + i * RECORD_SIZE, 0, z);
            printf("[ADVERSARIAL] Done in %.0f ms\n\n", zt.end_ms());
        }
    }

    for (int run = 0; run < num_experiment_runs; run++) {
        if (num_experiment_runs > 1) {
            printf("\n══════════ Experiment run %d/%d ══════════\n", run+1, num_experiment_runs);
            // h_data is not mutated by sort() — sorted output goes to a separate h_output
            // buffer that's freed after verification. No reload needed.
        }

        // Hybrid correctness: try the compact path first. If it detects
        // exception records (non-mapped bytes that the sample missed), retry
        // with the full-key upload path which has no sampling assumption.
#ifdef USE_COMPACT_KEY
        g_force_no_compact = false;
        g_disable_compact = false;
        if (const char* e = getenv("DISABLE_COMPACT")) {
            if (std::string(e) != "0") {
                g_disable_compact = true;
                printf("[CONFIG] Compact key compression DISABLED via DISABLE_COMPACT\n");
            }
        }
#endif
        ExternalGpuSort::TimingResult result;
        {
            ExternalGpuSort sorter;
            result = sorter.sort(h_data, num_records);
        }
#ifdef USE_COMPACT_KEY
        if (result.needs_hybrid_retry) {
            printf("[Hybrid] Retrying with full-key upload path...\n");
            g_force_no_compact = true;
            ExternalGpuSort sorter;
            result = sorter.sort(h_data, num_records);
            g_force_no_compact = false;
        }
#endif

        const uint8_t* sorted = result.sorted_output ? result.sorted_output : h_data;

        if (verify) {
            printf("\nVerifying...\n");
            // Check 1: parallel sortedness — adjacent pairs in non-decreasing order.
            int hw = std::max(1u, std::thread::hardware_concurrency());
            std::atomic<uint64_t> first_bad{UINT64_MAX};
            uint64_t per_t = (num_records + hw - 1) / hw;
            std::vector<std::thread> sthreads;
            WallTimer st; st.begin();
            for (int t = 0; t < hw; t++) {
                sthreads.emplace_back([&, t]() {
                    uint64_t lo = (uint64_t)t * per_t;
                    uint64_t hi = std::min(lo + per_t, num_records);
                    if (lo == 0) lo = 1;
                    for (uint64_t i = lo; i < hi; i++) {
                        if (first_bad.load(std::memory_order_relaxed) != UINT64_MAX) break;
                        if (key_compare(sorted + (i-1)*RECORD_SIZE,
                                        sorted + i*RECORD_SIZE, KEY_SIZE) > 0) {
                            uint64_t cur = first_bad.load();
                            while (i < cur && !first_bad.compare_exchange_weak(cur, i)) {}
                            break;
                        }
                    }
                });
            }
            for (auto& t : sthreads) t.join();
            double sort_ms = st.end_ms();
            uint64_t fb = first_bad.load();
            if (fb == UINT64_MAX) {
                printf("  PASS sortedness: %lu records in non-decreasing order (%.0f ms)\n",
                       num_records, sort_ms);
            } else {
                printf("  FAIL sortedness: first violation at record %lu (%.0f ms)\n",
                       fb, sort_ms);
            }

            // Check 2: multiset preservation — sum of FNV-1a 64 hashes per record,
            // computed over BOTH the input buffer (h_data) and the sorted output.
            // Order-independent: equal iff output is a permutation of input.
            // Catches the "all-zero permutation" failure mode where every output
            // record is a copy of input[0] (which trivially passes sortedness).
            auto multiset_hash = [&](const uint8_t* data) {
                std::vector<std::atomic<uint64_t>> partials(hw);
                for (int t = 0; t < hw; t++) partials[t].store(0);
                std::vector<std::thread> hthreads;
                for (int t = 0; t < hw; t++) {
                    hthreads.emplace_back([&, t]() {
                        uint64_t lo = (uint64_t)t * per_t;
                        uint64_t hi = std::min(lo + per_t, num_records);
                        uint64_t local = 0;
                        for (uint64_t i = lo; i < hi; i++) {
                            uint64_t h = 0xcbf29ce484222325ULL;
                            const uint8_t* r = data + i * RECORD_SIZE;
                            for (int b = 0; b < (int)RECORD_SIZE; b++) {
                                h ^= r[b]; h *= 0x100000001b3ULL;
                            }
                            local += h;
                        }
                        partials[t].store(local);
                    });
                }
                for (auto& t : hthreads) t.join();
                uint64_t sum = 0;
                for (int t = 0; t < hw; t++) sum += partials[t].load();
                return sum;
            };
            WallTimer ht; ht.begin();
            uint64_t in_sum = multiset_hash(h_data);
            uint64_t out_sum = multiset_hash(sorted);
            double hash_ms = ht.end_ms();
            if (in_sum == out_sum) {
                printf("  PASS multiset:   output is a permutation of input "
                       "(hash=0x%016lx, %.0f ms)\n", in_sum, hash_ms);
            } else {
                printf("  FAIL multiset:   in=0x%016lx out=0x%016lx — output IS NOT "
                       "a permutation of input (records lost/duplicated/modified, %.0f ms)\n",
                       in_sum, out_sum, hash_ms);
            }
        }

        // Dump sorted output for independent external verification (tools/verify_sorted).
        // Only on the last experiment run to avoid repeated writes.
        if (output_file && run == num_experiment_runs - 1) {
            printf("Writing sorted output to %s...\n", output_file);
            FILE* of = fopen(output_file, "wb");
            if (!of) { fprintf(stderr, "  cannot open output file\n"); }
            else {
                WallTimer wt; wt.begin();
                size_t w = fwrite(sorted, 1, total_bytes, of);
                fclose(of);
                printf("  Wrote %.2f GB in %.0f ms\n", w/1e9, wt.end_ms());
            }
        }
        if (result.sorted_output) {
            if (result.sorted_output_is_mmap) munmap(result.sorted_output, result.sorted_output_size);
            else free(result.sorted_output);
        }

        printf("\nCSV,%s,%.2f,%lu,%d,%d,%.2f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f,%.1f\n",
               props.name, total_bytes/1e9, num_records,
               result.num_runs, result.merge_passes,
               result.run_gen_ms, result.merge_ms, result.total_ms,
               total_bytes/(result.total_ms*1e6),
               result.pcie_h2d_gb, result.pcie_d2h_gb,
               result.pcie_h2d_gb + result.pcie_d2h_gb,
               (result.pcie_h2d_gb + result.pcie_d2h_gb) / (total_bytes/1e9));
    }

    if (h_data_is_mmap) munmap(h_data, total_bytes);
    else if (alloc_err == cudaSuccess) cudaFreeHost(h_data);
    else free(h_data);
    return 0;
}
#endif
