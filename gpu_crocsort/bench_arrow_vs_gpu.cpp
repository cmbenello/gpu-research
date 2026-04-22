// Arrow CPU sort vs GPU sort — columnar benchmark (no DuckDB overhead)
// Shows raw sort performance on contiguous columnar data.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <algorithm>
#include <arpa/inet.h>  // htonl

#include <arrow/api.h>
#include <arrow/compute/api_vector.h>

// GPU sort C API
extern "C" {
    int gpu_sort_keys(const uint8_t* keys, uint32_t key_size, uint32_t key_stride,
                      uint64_t num_records, uint32_t* perm_out);
    int gpu_sort_and_gather(const uint8_t* keys, uint32_t key_size, uint32_t key_stride,
                            const uint8_t* payload, uint32_t payload_stride,
                            uint64_t num_records, uint8_t* sorted_payload);
    int gpu_sort_available(void);
    typedef struct {
        double total_ms, upload_ms, gpu_sort_ms, gather_ms, download_ms, fixup_ms;
        uint64_t num_fixup_groups, num_fixup_records;
    } GpuSortTiming;
    void gpu_sort_get_timing(GpuSortTiming* out);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    uint64_t N = 300000000;  // 300M
    if (argc > 1) N = strtoull(argv[1], nullptr, 10);

    int num_cols = 1;  // number of sort key columns
    if (argc > 2) num_cols = atoi(argv[2]);

    printf("=== Arrow CPU Sort vs GPU Sort ===\n");
    printf("Records: %luM, Sort columns: %d\n\n", N / 1000000, num_cols);

    // Generate random int32 data
    printf("Generating %luM random int32 values (%d columns)...\n", N / 1000000, num_cols);
    std::vector<std::vector<int32_t>> col_data(num_cols);
    std::mt19937 rng(42);
    for (int c = 0; c < num_cols; c++) {
        col_data[c].resize(N);
        for (uint64_t i = 0; i < N; i++) col_data[c][i] = rng();
    }

    // ── Arrow CPU sort ──
    printf("\n--- Arrow CPU Sort ---\n");
    {
        // Build Arrow arrays
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        arrow::compute::SortOptions sort_opts;

        for (int c = 0; c < num_cols; c++) {
            char name[16]; snprintf(name, sizeof(name), "c%d", c);
            fields.push_back(arrow::field(name, arrow::int32()));

            auto buf = arrow::Buffer::Wrap(col_data[c].data(), N * sizeof(int32_t));
            auto arr_data = arrow::ArrayData::Make(arrow::int32(), N, {nullptr, buf});
            arrays.push_back(arrow::MakeArray(arr_data));

            sort_opts.sort_keys.emplace_back(name, arrow::compute::SortOrder::Ascending);
        }

        auto schema = arrow::schema(fields);
        auto table = arrow::Table::Make(schema, arrays);

        // Warmup
        auto warm = arrow::compute::SortIndices(arrow::Datum(table), sort_opts);

        // Timed run
        double t0 = now_ms();
        auto result = arrow::compute::SortIndices(arrow::Datum(table), sort_opts);
        double t1 = now_ms();

        if (!result.ok()) {
            fprintf(stderr, "Arrow sort failed: %s\n", result.status().ToString().c_str());
            return 1;
        }
        printf("Arrow SortIndices: %.0f ms\n", t1 - t0);

        // Timed Take (apply permutation)
        double t2 = now_ms();
        auto take_result = arrow::compute::Take(arrow::Datum(table), arrow::Datum(*result));
        double t3 = now_ms();
        printf("Arrow Take:        %.0f ms\n", t3 - t2);
        printf("Arrow Total:       %.0f ms\n", (t1 - t0) + (t3 - t2));
    }

    // ── GPU sort ──
    printf("\n--- GPU Sort ---\n");
    if (!gpu_sort_available()) {
        printf("GPU not available\n");
        return 1;
    }
    {
        // Convert int32 to byte-comparable keys (big-endian with sign flip)
        uint32_t key_width = 4 * num_cols;
        std::vector<uint8_t> keys(N * key_width);

        double t_enc0 = now_ms();
        for (int c = 0; c < num_cols; c++) {
            for (uint64_t i = 0; i < N; i++) {
                uint32_t v = (uint32_t)col_data[c][i] ^ 0x80000000u;  // flip sign bit
                v = htonl(v);  // big-endian
                memcpy(&keys[i * key_width + c * 4], &v, 4);
            }
        }
        double t_enc1 = now_ms();
        printf("Key encoding:      %.0f ms\n", t_enc1 - t_enc0);

        std::vector<uint32_t> perm(N);

        // Warmup
        gpu_sort_keys(keys.data(), key_width, key_width, N, perm.data());

        // Timed run
        double t0 = now_ms();
        int rc = gpu_sort_keys(keys.data(), key_width, key_width, N, perm.data());
        double t1 = now_ms();

        if (rc != 0) {
            fprintf(stderr, "GPU sort failed (rc=%d)\n", rc);
            return 1;
        }

        GpuSortTiming timing;
        gpu_sort_get_timing(&timing);
        printf("GPU sort total:    %.0f ms (upload=%.0f sort=%.0f download=%.0f)\n",
               t1 - t0, timing.upload_ms, timing.gpu_sort_ms, timing.download_ms);

        // Timed Take equivalent (apply permutation to all columns)
        double t2 = now_ms();
        std::vector<int32_t> sorted_col(N);
        for (int c = 0; c < num_cols; c++) {
            for (uint64_t i = 0; i < N; i++) {
                sorted_col[i] = col_data[c][perm[i]];
            }
        }
        double t3 = now_ms();
        printf("CPU gather:        %.0f ms\n", t3 - t2);
        printf("GPU Total:         %.0f ms (encode + sort + gather)\n",
               (t_enc1 - t_enc0) + (t1 - t0) + (t3 - t2));
    }

    // ── GPU sort + GPU gather (full pipeline, no CPU gather) ──
    printf("\n--- GPU Sort + GPU Gather ---\n");
    {
        uint32_t key_width = 4 * num_cols;
        // Payload = all columns interleaved (each row = num_cols × int32)
        uint32_t payload_width = 4 * num_cols;
        std::vector<uint8_t> keys(N * key_width);
        std::vector<uint8_t> payload(N * payload_width);
        std::vector<uint8_t> sorted_payload(N * payload_width);

        double t_enc0 = now_ms();
        for (int c = 0; c < num_cols; c++) {
            for (uint64_t i = 0; i < N; i++) {
                uint32_t v = (uint32_t)col_data[c][i] ^ 0x80000000u;
                v = htonl(v);
                memcpy(&keys[i * key_width + c * 4], &v, 4);
                memcpy(&payload[i * payload_width + c * 4], &col_data[c][i], 4);
            }
        }
        double t_enc1 = now_ms();

        // Warmup
        gpu_sort_and_gather(keys.data(), key_width, key_width,
                            payload.data(), payload_width, N, sorted_payload.data());

        double t0 = now_ms();
        int rc = gpu_sort_and_gather(keys.data(), key_width, key_width,
                                     payload.data(), payload_width, N, sorted_payload.data());
        double t1 = now_ms();

        GpuSortTiming timing;
        gpu_sort_get_timing(&timing);
        printf("Encode:            %.0f ms\n", t_enc1 - t_enc0);
        printf("GPU full pipeline: %.0f ms (upload=%.0f sort=%.0f gather=%.0f download=%.0f)\n",
               t1 - t0, timing.upload_ms, timing.gpu_sort_ms, timing.gather_ms, timing.download_ms);
        printf("GPU+encode Total:  %.0f ms\n", (t_enc1 - t_enc0) + (t1 - t0));
    }

    // ── std::sort baseline ──
    printf("\n--- std::sort baseline ---\n");
    {
        std::vector<uint32_t> indices(N);
        for (uint64_t i = 0; i < N; i++) indices[i] = i;

        // Warmup with small subset
        auto cmp1 = [&](uint32_t a, uint32_t b) { return col_data[0][a] < col_data[0][b]; };

        double t0 = now_ms();
        if (num_cols == 1) {
            std::sort(indices.begin(), indices.end(), cmp1);
        } else {
            std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
                for (int c = 0; c < num_cols; c++) {
                    if (col_data[c][a] != col_data[c][b])
                        return col_data[c][a] < col_data[c][b];
                }
                return false;
            });
        }
        double t1 = now_ms();
        printf("std::sort:         %.0f ms\n", t1 - t0);
    }

    return 0;
}
