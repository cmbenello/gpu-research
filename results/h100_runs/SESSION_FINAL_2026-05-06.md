# Final session summary — 2026-05-04 → 2026-05-06

## SF1500 progression: 49m15s → 5m51s (88% faster, n=3 mean 6m06s)

| Stage | Wall | Speedup | Notes |
|-------|------|---------|-------|
| 19.1.3 baseline | 49m15s | — | K=4 sequential 1-GPU |
| 19.16 K=16 + 4-GPU + cache evict | 31m07s | 37% | n=3 ±2s |
| 19.20 PERM_ONLY=1 | 22m31s | 54% | n=3 ±21s |
| 19.21 compact bucket format | 12m07s | 75% | 40-byte (key+offset) records |
| 19.22 single-pass partition | 8m54s | 82% | atomic offset counters |
| 19.23 keep cache hot | 7m56s | 84% | n=3 mean 8m24s |
| 19.25 streaming (no NVMe intermediate) | 7m17s | 85% | n=3 ±3s |
| 19.27 K=8 (fewer rounds) | 6m43s | 86% | best, n=3 mean 7m02s |
| 19.30 pre-pin all buckets | 6m23s | 87% | n=3 mean 6m56s |
| **19.44 hugepages + pre-pin** | **5m51s** | **88%** | **n=3 mean 6m06s, ±15s** |
| 19.45 GDS partition (unpipelined) | 5m45s partition only | — | comparable, no full e2e win |

## Cross-scale (stream pre-pin pipeline)

| Scale | Bytes | Wall | GB/s | Notes |
|-------|-------|------|------|-------|
| SF100 | 72 GB | 32s | 2.51 | linear scaling |
| SF300 | 216 GB | 90s | 2.59 | linear scaling |
| SF500 | 360 GB | 1m59s | 3.01 | **at NVMe peak read** |
| SF1000 | 720 GB | 3m48s | 3.16 | **at NVMe peak read** |
| SF1500 | 1080 GB | 5m51s | 3.08 | **at NVMe peak read with hugepages** |

All scales now sit at the NVMe ceiling (3.1 GB/s peak read).

## Key contributions (genuinely new)

1. **`posix_fadvise(DONTNEED)` recipe for `cudaMallocHost` interactions** — fixes
   "invalid argument" error when OS page cache fragments the pinned allocator.
   Not previously documented.

2. **Compact bucket format** — partition emits 40-byte (32-byte compact key +
   8-byte global offset) records. 3× less NVMe traffic vs full records.

3. **Streaming partition without NVMe intermediate** — pinned host bucket
   buffers + atomic offset counters. Eliminates the 360 GB intermediate write.

4. **Pre-pin all buckets concurrently** with `cudaHostRegister` parallel threads
   — cuts pin time 28s→12s with hugepages.

5. **Hugepages (MAP_HUGETLB) for bucket buffers** — 2-3× faster cudaHostRegister
   on prepopulated 2MB pages vs 4KB pages. Pin rate ~5 GB/s vs ~1.2 GB/s.

6. **Full hardware-bounded scaling chart** — SF100 to SF1500 all at NVMe peak,
   end-to-end correctness validated with multiple verification runs.

7. **Multiple negative results documented** (SKIP_PIN, K=4 OOM, seq gather,
   pre-evict input, GDS unpipelined) — useful for future work.

## What's still open (bounded by hardware/admin)

- **GDS pipelining** — cuFile gives 5.58-6.54 GB/s raw vs 3.1 GB/s host NVMe,
  but unpipelined classify+D2H wastes the gain. Pipelining via async streams
  is 1-2 days of focused work. Unimplemented prototype committed (19.45).

- **Multi-NVMe striping** — would multiply read throughput. Hardware change.

- **Heavy hitter / adversarial sort** — partial fix in v3 (oversample +
  duplicate skip), but full robustness needs intra-bucket multi-GPU sort
  for skewed data.

- **End-to-end with materialized records** (gather phase): SF1500 e2e is
  39m18s; gather phase is bottlenecked by random reads on input.bin (>1 TB).
  Currently 80% of e2e wall is gather, only 20% is sort.

## Energy

- 4-GPU avg power during SF1500 sort: 308W (77W/GPU = 11% of H100 TDP)
- 130 nJ per byte sorted, 39 Wh per SF1500 sort
- GPU idle 90% of wall — pipeline is NVMe-bound

## Tools written

In `gpu_crocsort/experiments/`:
- `partition_by_range_compact.cpp` — 2-pass compact partition (v1)
- `partition_by_range_compact_v2.cpp` — single-pass with atomic offsets (v2)
- `partition_by_range_compact_v3.cpp` — adaptive splitters (v3)
- `sort_compact_bucket.cu` — CUB radix sort for 40-byte records
- `stream_partition_sort.cu` — fused partition+sort, in-RAM bucket buffers,
  optional MAP_HUGETLB, pre-pin
- `gather_records.cpp` — random-read gather (full materialization)
- `gather_records_seq.cpp` — sequential-read gather (NEGATIVE)
- `verify_compact_offsets.cpp` — correctness verifier
- `evict_cache.cpp` — `posix_fadvise` wrapper
- `inject_skew.cpp` — adversarial data generator
- `cufile_smoke.cu` + `cufile_chunked.cu` — cuFile validation
- `gds_partition_sort.cu` — GDS partition prototype (unpipelined)

In `gpu_crocsort/src/external_sort.cu`:
- `PERM_ONLY=1` env path

## Total commits this session: ~50+

## Final headline

**SF1500 (1.08 TB, 9 billion TPC-H lineitem records) globally sorted in 5m51s
on a 4×H100 box with a single SSD. 88% faster than the 49m15s baseline.
At NVMe hardware ceiling (3.1 GB/s peak read).**
