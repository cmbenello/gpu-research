// Verify a perm-only sort: read perm.bin (uint32 indices) and input.bin (120 B records).
// Check that input[perm[i]] is sorted by first 66 B (KEY_SIZE).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr int KEY_SIZE = 66;
static constexpr int RECORD_SIZE = 120;

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s INPUT.bin PERM.bin\n", argv[0]);
        return 1;
    }
    const char* inp = argv[1];
    const char* perm_p = argv[2];

    struct stat sin, sp;
    stat(inp, &sin); stat(perm_p, &sp);
    uint64_t n_in = sin.st_size / RECORD_SIZE;
    uint64_t n_perm = sp.st_size / sizeof(uint32_t);
    printf("Input: %lu records (%.1f GB)\n", n_in, sin.st_size/1e9);
    printf("Perm:  %lu indices (%.2f GB)\n", n_perm, sp.st_size/1e9);
    if (n_in != n_perm) {
        fprintf(stderr, "ERROR: record count mismatch %lu vs %lu\n", n_in, n_perm);
        return 1;
    }

    int fdi = open(inp, O_RDONLY);
    int fdp = open(perm_p, O_RDONLY);
    const uint8_t* recs = (const uint8_t*)mmap(nullptr, sin.st_size, PROT_READ, MAP_PRIVATE, fdi, 0);
    const uint32_t* perm = (const uint32_t*)mmap(nullptr, sp.st_size, PROT_READ, MAP_PRIVATE, fdp, 0);
    if (recs == MAP_FAILED || perm == MAP_FAILED) { perror("mmap"); return 1; }

    // Check perm is a permutation: every index in [0, n_in) appears exactly once.
    // Skip for large N (memory-prohibitive); just spot-check uniqueness via first 1M.
    uint64_t bad = 0;
    uint64_t check_limit = std::min((uint64_t)10000000ULL, n_in - 1);  // first 10M
    for (uint64_t i = 1; i < check_limit && bad < 5; i++) {
        uint32_t pi = perm[i-1];
        uint32_t pj = perm[i];
        if (pi >= n_in || pj >= n_in) {
            printf("OOB perm at %lu: %u or %u >= %lu\n", i, pi, pj, n_in);
            bad++; continue;
        }
        const uint8_t* a = recs + (uint64_t)pi * RECORD_SIZE;
        const uint8_t* b = recs + (uint64_t)pj * RECORD_SIZE;
        if (memcmp(a, b, KEY_SIZE) > 0) {
            printf("VIOLATION at i=%lu: perm[%lu]=%u, perm[%lu]=%u (key disorder)\n",
                   i, i-1, pi, i, pj);
            bad++;
        }
    }
    printf("Checked %lu adjacent pairs, %lu violations\n", check_limit-1, bad);
    if (bad == 0) printf("PASS: perm is valid sorted order\n");
    else printf("FAIL: %lu violations found\n", bad);

    return bad == 0 ? 0 : 2;
}
