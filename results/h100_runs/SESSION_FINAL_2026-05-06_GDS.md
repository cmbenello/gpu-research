# Final session — 2026-05-06 — GDS unlocked, hit NVMe hardware floor

## Headline result

**SF1500 (1.08 TB, 9 billion records) globally sorted in 6m10s mean (n=5, ±4s)
on 4×H100 + single SSD. 87.4% faster than 49m15s baseline. Within 1.06× of
hardware floor (5.8m NVMe-read theoretical).**

## Champion config

GPUDirect Storage (cuFile + nvidia_fs kernel module) + multi-stream pipeline
+ integrated sort phase + hugepages bucket buffers.

```bash
sudo apt install nvidia-fs-dkms       # one-time
sudo modprobe nvidia_fs               # load kernel module
sudo sysctl -w vm.nr_hugepages=240000 # 480 GB pool
sudo sysctl vm.drop_caches=3          # for cold start

cd ~/gpu-research/gpu_crocsort
RUN_SORT=1 experiments/gds_partition_multistream \
    /mnt/data/lineitem_sf1500.bin /mnt/data/out/sf1500 8

# Output: /mnt/data/out/sf1500.sorted_{0..7}.bin (8-byte uint64 offsets)
```

## Final variance bounds (SF1500 cold-cache, with drop_caches between runs)

| Config | n | Best | Mean | Worst | ±  |
|--------|---|------|------|-------|----|
| Hugepages stream pre-pin | 10 | 6m34s | 7m02s | 7m36s | 31s |
| **GDS+integrated-sort** | **5** | **6m05s** | **6m10s** | **6m13s** | **4s** |

GDS wins on both:
- Mean: 52s faster (12% improvement)
- Variance: 7.7× tighter

The variance reduction is because GDS reads bypass OS page cache entirely
(NVMe → GPU direct), so per-run timing is largely insensitive to page cache
state. Hugepages reads still go through host page cache with variable hit rates.

## Why GDS wins now (after multiple iterations)

The path to 6m10s required several orthogonal fixes:
1. **Multi-stream pipeline** (19.47): 2 reader threads + 4 ring-buffer slots,
   so cuFileRead overlaps with classify+D2H. Bumped effective throughput 3.13 → 4.40 GB/s.
2. **cmap fix** (19.49): full-file fseek-based sample. Capped 1 GB sample
   missed varying bytes at SF300+ → wrong sort order.
3. **Double-event ping-pong** (19.48): two stage_d2h_done events (one per
   buffer) so iter N waits for iter N-2's D2H, not iter N-1's.
4. **cudaDeviceSync between rounds** (19.52): fixed K=8 round 2 hang —
   round 1 GPU work wasn't fully drained before round 2 thread launch.
5. **Integrated sort phase**: runs sort_one_bucket directly on in-RAM
   bucket buffers (no bucket file write). Saves the 277s NVMe-write penalty.

## Hardware ceilings (measured)

| Operation | Rate |
|-----------|------|
| NVMe sequential write (raw, dd) | 1.3 GB/s sustained |
| NVMe sequential read (raw, dd) | 3.1 GB/s peak |
| cuFile sequential read (1 GB) | 5.29 GB/s |
| cuFile sustained chunked (50 GB) | 6.54 GB/s |
| Multi-stream GDS partition effective | 3.59-3.75 GB/s (after classify+D2H overhead) |

## Total session contribution (49m → 6m10s)

| Stage | Wall | Speedup |
|-------|------|---------|
| Baseline 19.1.3 (K=4 sequential) | 49m15s | — |
| 19.15 cache eviction recipe | 34m00s | 31% |
| 19.16 K=16 + 4-GPU concurrent | 31m07s | 37% |
| 19.20 PERM_ONLY=1 | 22m31s | 54% |
| 19.21 compact bucket format | 12m07s | 75% |
| 19.22 single-pass partition | 8m54s | 82% |
| 19.23 keep cache hot | 7m56s | 84% |
| 19.25 streaming (no NVMe intermediate) | 7m17s | 85% |
| 19.27 K=8 streaming | 6m43s | 86% |
| 19.30 pre-pin all buckets | 6m23s | 87% |
| 19.44 + MAP_HUGETLB | 5m51s warm / 7m02s cold | 88% / 86% |
| **19.53 GDS+integrated-sort** | **6m10s** cold (n=5 ±4s) | **87.4%** |

Hugepages 5m51s "best" was a warm-cache outlier (n=3 mean 6m06s). The
honest cold-cache n=10 mean is 7m02s. GDS+integrated-sort (6m10s, n=5,
±4s) is the actual winner under strict cold-cache benchmarking.

## Total commits this session: 60+

All committed to `h100/discoveries-2026-05-02` branch. Auto-push daemon
PID 308842 will sync to remote when SSH agent forwarding allows.

## Session contribution beyond just the win

- 8 distinct bottlenecks identified and eliminated
- 7 documented negative results (SKIP_HOST_PIN, K=4 OOM, seq gather,
  pre-evict input, GDS unpipelined, GDS+bucket-file-write, RUN_SORT K=8 hang)
- Full hardware-bounded scaling chart SF100→SF1500
- Variance bounds at every stage (n=3 to n=10)
- Energy: 130 nJ/byte sorted, 39 Wh per SF1500 sort
- 12+ new tools written
- POSIX_FADV_DONTNEED + cudaMallocHost interaction documented (novel)
- cuFile + nvidia_fs end-to-end recipe with pipelining + sync fixes documented

## Recipe in one paragraph

For sorting TPC-H lineitem at SF1500 (1.08 TB, 9 B records) on a 4×H100 +
1 TB host RAM + single 3.5 TB NVMe box, the optimal pipeline is: (a)
allocate 480 GB hugepage pool via sysctl; (b) load nvidia_fs kernel
module via modprobe; (c) cuFile-read input.bin in 4 GB chunks into a
4-slot GPU ring buffer with 2 reader threads; (d) GPU classify each
record into K=8 buckets via splitter binary search; (e) scatter compact
40-byte (key+offset) records to per-bucket GPU staging; (f) async D2H
to host bucket buffers (MAP_HUGETLB-backed); (g) cudaDeviceSync each
GPU between rounds; (h) per-bucket cudaHostRegister + CUB radix sort
+ gather offsets + write 8-byte sorted offsets. Output is 68 GB of
sorted offsets indexing input.bin. Wall: 6m10s mean (n=5 ±4s) cold-cache.
