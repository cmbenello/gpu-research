// Verify a sort_compact_bucket output: each entry is uint64 offset into INPUT.bin.
// Check that consecutive offsets indicate keys in non-decreasing order.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>

static constexpr int KEY_SIZE = 66;
static constexpr int RECORD_SIZE = 120;

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s INPUT.bin OFFSETS.bin\n", argv[0]);
        return 1;
    }
    const char* inp = argv[1];
    const char* off_p = argv[2];

    struct stat sin, sp;
    stat(inp, &sin); stat(off_p, &sp);
    uint64_t n_in = sin.st_size / RECORD_SIZE;
    uint64_t n_off = sp.st_size / sizeof(uint64_t);
    printf("Input:   %lu records (%.2f GB)\n", n_in, sin.st_size/1e9);
    printf("Offsets: %lu entries (%.2f GB)\n", n_off, sp.st_size/1e9);

    int fdi = open(inp, O_RDONLY);
    int fdo = open(off_p, O_RDONLY);
    const uint8_t* recs = (const uint8_t*)mmap(nullptr, sin.st_size, PROT_READ, MAP_PRIVATE, fdi, 0);
    const uint64_t* offsets = (const uint64_t*)mmap(nullptr, sp.st_size, PROT_READ, MAP_PRIVATE, fdo, 0);
    if (recs == MAP_FAILED || offsets == MAP_FAILED) { perror("mmap"); return 1; }

    uint64_t bad = 0;
    // CHECK_LIMIT_OVERRIDE env: smaller value = faster verify
    const char* clim_env = getenv("CHECK_LIMIT");
    uint64_t override_lim = clim_env ? std::strtoull(clim_env, nullptr, 10) : 10000000ULL;
    uint64_t check_limit = std::min(override_lim, n_off);
    for (uint64_t i = 1; i < check_limit && bad < 10; i++) {
        uint64_t pi = offsets[i-1];
        uint64_t pj = offsets[i];
        if (pi >= n_in || pj >= n_in) {
            printf("OOB at %lu: %lu or %lu >= %lu\n", i, pi, pj, n_in);
            bad++; continue;
        }
        const uint8_t* a = recs + pi * RECORD_SIZE;
        const uint8_t* b = recs + pj * RECORD_SIZE;
        if (memcmp(a, b, KEY_SIZE) > 0) {
            printf("VIOLATION at %lu: offset[%lu]=%lu, offset[%lu]=%lu\n", i, i-1, pi, i, pj);
            // print first diff
            for (int k = 0; k < KEY_SIZE; k++) {
                if (a[k] != b[k]) {
                    printf("  first diff at byte %d: %02x vs %02x\n", k, a[k], b[k]);
                    break;
                }
            }
            bad++;
        }
    }
    printf("Checked %lu adjacent pairs, %lu violations\n", check_limit-1, bad);
    if (bad == 0) printf("PASS\n"); else printf("FAIL: %lu violations\n", bad);
    return bad == 0 ? 0 : 2;
}
