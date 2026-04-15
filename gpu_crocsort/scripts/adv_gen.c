// Adversarial dataset generator for lineitem records (120B each, 66B key).
//
// Modes:
//   zero32   bytes 0..31 = 0 for all records (one giant tied group on 32B prefix)
//   zero8    bytes 0..7 = 0  (pfx1 useless; 32B groups still many)
//   pool1000 bytes 0..31 = template_i, i = record_index % 1000 (1000 coarse groups)
//   pool32   bytes 0..31 = template_i, i = record_index % 32   (32 huge groups)
//
// Usage: ./adv_gen <mode> <in_file> <out_file>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static const int RECORD = 120;
static const size_t CHUNK_RECORDS = 524288;  // 60MB chunks

static void fill_template(uint8_t* t, int idx) {
    // Deterministic 32B template derived from xorshift seeded with idx.
    uint64_t s = 0x9E3779B97F4A7C15ULL * (uint64_t)(idx + 1);
    for (int i = 0; i < 32; i += 8) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27; s *= 0x2545F4914F6CDD1DULL;
        memcpy(t + i, &s, 8);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <mode> <in> <out>\n", argv[0]); return 1; }
    const char* mode = argv[1];
    FILE* fin = fopen(argv[2], "rb");
    FILE* fout = fopen(argv[3], "wb");
    if (!fin || !fout) { perror("open"); return 1; }

    int pool = -1, zero_bytes = 0;
    if (!strcmp(mode, "zero32")) zero_bytes = 32;
    else if (!strcmp(mode, "zero8"))  zero_bytes = 8;
    else if (!strcmp(mode, "pool1000")) pool = 1000;
    else if (!strcmp(mode, "pool32"))   pool = 32;
    else { fprintf(stderr, "bad mode\n"); return 1; }

    uint8_t (*templates)[32] = NULL;
    if (pool > 0) {
        templates = calloc(pool, 32);
        for (int i = 0; i < pool; i++) fill_template(templates[i], i);
    }

    size_t buf_bytes = CHUNK_RECORDS * (size_t)RECORD;
    uint8_t* buf = malloc(buf_bytes);
    if (!buf) { perror("malloc"); return 1; }

    uint64_t rec_idx = 0;
    size_t total_read = 0;
    while (1) {
        size_t got = fread(buf, 1, buf_bytes, fin);
        if (got == 0) break;
        size_t nrec = got / RECORD;
        for (size_t r = 0; r < nrec; r++) {
            uint8_t* p = buf + r * RECORD;
            if (zero_bytes > 0) {
                memset(p, 0, zero_bytes);
            } else if (pool > 0) {
                memcpy(p, templates[rec_idx % pool], 32);
            }
            rec_idx++;
        }
        fwrite(buf, 1, got, fout);
        total_read += got;
        if ((total_read & 0xFFFFFFFFULL) == 0 || total_read % (1ULL << 30) == 0) {
            fprintf(stderr, "\r  %.1f GB written", total_read / 1e9);
            fflush(stderr);
        }
    }
    fprintf(stderr, "\n  %.1f GB total, %lu records\n", total_read / 1e9, rec_idx);
    free(buf);
    if (templates) free(templates);
    fclose(fin); fclose(fout);
    return 0;
}
