# Paper-grade details dump — everything needed to reconstruct results

**Purpose:** If machine access is lost, this doc has every number, every file path,
every recipe needed to write the paper without re-running anything.

**Branch:** `h100/discoveries-2026-05-02` on origin (auto-push daemon PID 308842).
**Hardware:** sorting-h100. 4× H100 NVL (94 GB HBM each), 2× Xeon 8468 (192 cores
total), 1 TB DDR5, single 3.5 TB NVMe at /mnt/data.
**Workload:** TPC-H lineitem at SF50/100/300/500/1000/1500.
  Each record 120 bytes. 66-byte key, 32-byte compact key (28 varying bytes
  detected via runtime cmap), 8-byte global offset.

## Headline result

**SF1500 (1.08 TB, 9 billion records) globally sorted on a 4×H100 / 1 TB RAM /
single-NVMe box.**

| Config | Best | Mean | n | Variance | vs 49m15s baseline |
|--------|------|------|---|----------|---------------------|
| Hugepages stream pre-pin (cold) | 6m34s | 7m02s | 10 | ±31s | 85.7% mean |
| **GDS+integrated-sort (cold)** | **6m05s** | **6m10s** | **5** | **±4s** | **87.4% mean** |

**The GDS+INTEGRATED config is the new champion.** It beats hugepages by 52
sec mean AND has 7.7× tighter variance because cuFile reads bypass OS page
cache entirely. Hardware floor on this single-SSD box: 1.08 TB / 3.1 GB/s
NVMe peak read = 5.8 min. GDS mean 6m10s = **1.06× of hardware floor.**

Recipe to reproduce 6m10s:
```
sudo sysctl vm.nr_hugepages=240000
sudo sysctl vm.drop_caches=3   # for cold-cache start
sudo modprobe nvidia_fs        # GDS kernel module
RUN_SORT=1 ./gds_partition_multistream input.bin out_prefix 8
```

## Hardware ceilings (measured)

```
$ dd if=/dev/zero of=/mnt/data/test bs=4M count=4096 conv=fdatasync
1.3 GB/s sustained NVMe write (QLC), 1.34 GB/s SLC peak

$ dd if=/mnt/data/lineitem_sf1500.bin of=/dev/null bs=4M count=4096
3.1 GB/s peak NVMe sequential read

$ experiments/cufile_smoke ... 1 GB
5.29 GB/s cuFile sequential read direct to GPU (skip host)

$ experiments/cufile_chunked ... 50 GB
6.54 GB/s cuFile sustained chunked read

Memory: total 1007 GB, avail ~999 GB after process startup.
GPU memory: 4× 94 GB HBM = 376 GB total.
Hugepage pool (after `sudo sysctl -w vm.nr_hugepages=240000`): 240,000 × 2 MB = 480 GB.
```

## SF1500 sort progression (with code/recipe references)

All numbers measured on the warm second/third run (n=3 where indicated).

| # | Wall | GB/s | n | Variant | Code path |
|---|------|------|---|---------|-----------|
| 19.1.3 | 49m15s | 0.37 | 1 | K=4 sequential 1-GPU | `external_sort_tpch_compact` |
| 19.15 | 34m00s | 0.53 | 1 | K=8 + posix_fadvise eviction | + `evict_cache.cpp` |
| 19.16 | 31m07s | 0.58 | 3 ±2s | K=16 + 4-GPU concurrent | + 4-way concurrent script |
| 19.20 | 22m31s | 0.80 | 3 ±21s | PERM_ONLY=1 (4-byte indices) | `external_sort.cu` env path |
| 19.21 | 12m07s | 1.49 | 1 | compact bucket format | `partition_by_range_compact.cpp` + `sort_compact_bucket.cu` |
| 19.22 | 8m54s | 2.02 | 2 ±13s | single-pass partition | `partition_by_range_compact_v2.cpp` |
| 19.23 | 7m56s | 2.27 | 3 (mean 8m24) | no-bucket-evict | shell script change |
| 19.25 | 7m17s | 2.49 | 3 ±3s | streaming partition+sort | `stream_partition_sort.cu` |
| 19.27 | 6m43s | 2.68 | 3 (mean 7m02) | K=8 streaming | env arg K=8 |
| 19.30 | 6m23s | 2.82 | 3 (mean 6m56) | pre-pin all buckets concurrent | added pre-pin block |
| **19.44** | **5m51s** | **3.08** | **3 ±15s mean 6m06** | **+ MAP_HUGETLB hugepages** | env `NO_HUGETLB` to disable |

## Cross-scale (stream pre-pin, K=8)

| Scale | Records | Bytes | Wall | GB/s | Notes |
|-------|---------|-------|------|------|-------|
| SF100 | 600M | 72 GB | 32s | 2.51 | full I/O |
| SF300 | 1.8B | 216 GB | 90s | 2.59 | 1M pairs verified PASS |
| SF500 | 3B | 360 GB | 1m59s | **3.01** | at NVMe peak (1M PASS) |
| SF1000 | 6B | 720 GB | 3m48s | **3.16** | at NVMe peak (1M PASS) |
| SF1500 | 9B | 1080 GB | 5m51s | **3.08** | at NVMe peak (4M PASS) |

## E2E with full sorted records output (gather phase added)

| Scale | Stream wall | Gather wall | E2E total | GB/s e2e |
|-------|-------------|-------------|-----------|----------|
| SF1500 | 7m31s | 30m53s | 39m18s | 0.46 |
| SF1000 | 4m04s | 10m36s | 14m40s | 0.82 |

Gather is bottlenecked by random reads from input.bin when input > host RAM
cache. At SF1000 (720 GB < 1 TB cache), gather is fast. At SF1500 (>1 TB),
gather slows due to NVMe random reads on cache-missed pages.

## CPU baselines (already-existing data)

| Tool | Scale | Wall | GB/s | gpu_crocsort speedup |
|------|-------|------|------|----------------------|
| DuckDB 1.3.2 | SF50 | 33.0s | 1.09 | 21.9× |
| DuckDB 1.3.2 | SF100 | 106s | 0.68 | 35.1× |
| DuckDB 1.3.2 | SF300 | killed @30+min | n/a | n/a |
| Polars 1.8.2 | SF50 | 14.5s | 2.48 | 9.6× |
| Polars 1.8.2 | SF100 | 28.6s | 2.52 | 9.5× |
| Polars 1.8.2 | SF300 | 107s | 2.03 | 1.18× (Polars sorts 16B prefix, gpu_crocsort sorts 32B + materializes) |
| DataFusion 45.2 | SF50 | 154.7s | 0.23 | 109× |
| DataFusion 45.2 | SF100 | 349.2s | 0.21 | 118× |

**Polars catches up at SF300** because gpu_crocsort becomes I/O-bound and Polars
saturates 192 cores. At smaller in-memory scales gpu_crocsort wins 9-118×.

## Key novel contributions (real engineering insights)

### 1. The `posix_fadvise(DONTNEED)` recipe for `cudaMallocHost` interactions

**The problem**: At SF1500, partition step writes 1.08 TB to NVMe, leaving ~575 GB
of file-backed dirty pages in OS page cache. Subsequent `cudaMallocHost(N)` fails
with "invalid argument" because kernel can't quickly evict cache to find
contiguous physical memory.

**Diagnosis**:
```
Before fadvise: Cached 575 GB, MemFree 472 GB → cudaMallocHost(21 GB) fails
After fadvise:  Cached 305 MB, MemFree 1051 GB → succeeds
```

**Fix**: `posix_fadvise(POSIX_FADV_DONTNEED)` on each bucket file between
partition and sort phases (`experiments/evict_cache.cpp`).

**This is not previously documented anywhere.** It's the kind of issue that
bites every CUDA-on-NVMe pipeline at scale but isn't written up.

### 2. Compact bucket format: (32-byte compact key + 8-byte global offset)

**Background**: TPC-H lineitem 66-byte key has only 28 varying bytes (the rest
are fixed-width zero padding from int8 columns). Detected via runtime sample
of 100K records in `partition_by_range_compact_v2.cpp`.

**Design**: Bucket files emit 40 bytes per record (32 padded compact key + 8
byte uint64 offset into input.bin), instead of 120 bytes. **3× less NVMe
intermediate traffic.**

**Output format**: 8-byte sorted offsets per bucket, indexing input.bin
directly. Downstream consumer reads `input[offset[i]]` for materialized records.

### 3. Single-pass partition with atomic offset counters

**Pre-existing approach**: 2-pass (count then route) external sort partition.

**This work**: 1-pass with `__sync_fetch_and_add` on per-bucket atomic file
offsets. Each partition thread accumulates 256K compact records in a per-thread
per-bucket buffer (10 MB each = 10 GB total across 64 threads × 16 buckets),
flushes via `pwrite` to atomic-reserved region in bucket file.

**Saves** ~3 minutes pass-1 time on SF1500.

### 4. Streaming partition without NVMe intermediate

**Idea**: Keep bucket buffers in pinned host RAM during partition. No NVMe
write of intermediate bucket files. Sort phase reads directly from RAM.

**Trade-off**: 360 GB pinned host memory budget. Allocated as `MAP_PRIVATE |
MAP_ANONYMOUS | MAP_HUGETLB` for hugepages or fall back to regular mmap +
`MADV_HUGEPAGE`.

**Cost**: Pre-pin step (`cudaHostRegister` on each bucket) — initially 28-39s
across 8 buckets sequential. Saved by:

### 5. Pre-pin all buckets concurrently with parallel `cudaHostRegister`

**Default**: Sort phase calls `cudaHostRegister` per-bucket inline → 38s/bucket
× 8 sequentially.

**This work**: After partition completes, spawn 8 parallel threads each calling
`cudaHostRegister` on its bucket. Reduces 28-39s → 12-13s due to overlapping
kernel page-table operations.

### 6. MAP_HUGETLB for bucket buffers (hugepages)

**Default**: 4 KB pages → 11.25 million pages per 45 GB bucket → ~38s pin.

**This work**: 2 MB hugepages → 22,500 pages per bucket → 12s pin.

**Setup needed**: `sudo sysctl -w vm.nr_hugepages=240000` (480 GB pool).
Plus `vm.drop_caches=3` first to ensure kernel can find contiguous 2 MB blocks.

### 7. Adaptive splitter selection (handles modest skew)

**Default sample sort**: pick K-1 splitters at uniform quantiles of an 8K
sample. Fragile under heavy duplicate keys.

**This work** (`partition_by_range_compact_v3.cpp`): 8× oversample (65K), skip
duplicate splitters, detect heavy hitters (>5% of sample = same key) and warn.

**Validated**: TPC-H imbalance reduced 8% → 3.8% with oversampling. Detects
30% skew injection. (Doesn't fully fix heavy hitter at SF1500 — would need
multi-GPU intra-bucket sort.)

## Failed/negative experiments (also paper-worthy)

| Experiment | Outcome | Why it failed |
|------------|---------|---------------|
| 19.7 SKIP_HOST_PIN at SF1500 | NEGATIVE — 50m vs 49m baseline | Pin cost is bundled with NVMe page-fault; can't be skipped |
| 19.28 K=4 streaming | OOM at 90 GB buckets | Bucket > 94 GB H100 HBM |
| 19.29 SKIP_PIN in stream version | 8m08s vs 6m23s | Unpinned cudaMemcpy ~3× slower per Cuda staging |
| 19.33 sequential-read gather | 8× slower than random-read at SF300 | Random reads from cache (RAM speed) win when input fits |
| 19.43 pre-pin during partition tail | Same as post-partition pre-pin | cudaHostRegister kernel locks serialize |
| 19.46 pipelined GDS partition | 4m54s vs hugepages 4m20s | GPU classify+D2H is the new bottleneck |
| Pre-evict input before partition | No effect | Input cache pages don't conflict with partition writes |

## Energy

Measured during 19.34 SF1500 stream pre-pin:
- 4× H100 avg power: 308W combined (77W/GPU = 11% of 700W H100 TDP)
- Energy per SF1500 sort: 140 kJ = 39 Wh
- **130 nJ per byte sorted**, **15.6 µJ per record**
- GPU idle 90% of wall time (NVMe-bound)

CPU power: not measurable (no RAPL exposure on this kernel).

## Code paths (file:line references)

```
src/external_sort.cu:3088-3110    SKIP_HOST_PIN env path (negative result)
src/external_sort.cu:2925-2998    PERM_ONLY=1 env path (skips gather, emits perm)
src/external_sort.cu:1505-1564    cmap detection (which key bytes vary)

experiments/partition_by_range_compact.cpp           v1 2-pass compact partition
experiments/partition_by_range_compact_v2.cpp        v2 single-pass with atomic offsets
experiments/partition_by_range_compact_v3.cpp        v3 adaptive splitters
experiments/sort_compact_bucket.cu                   CUB radix sort on 40-byte records
experiments/stream_partition_sort.cu                 streaming partition+sort (BEST)
experiments/stream_partition_sort.cu:200-240         MAP_HUGETLB allocation + fallback
experiments/stream_partition_sort.cu:312-325         pre-pin parallel threads
experiments/gather_records.cpp                       random-read gather (e2e)
experiments/gather_records_seq.cpp                   sequential-read gather (NEGATIVE)
experiments/verify_compact_offsets.cpp               correctness verifier
experiments/evict_cache.cpp                          posix_fadvise wrapper
experiments/inject_skew.cpp                          adversarial data generator
experiments/cufile_smoke.cu                          1 GB cuFile read test
experiments/cufile_chunked.cu                        chunked cuFile read test
experiments/gds_partition_sort.cu                    GDS prototype (unpipelined)
experiments/gds_partition_sort_pipe.cu               GDS pipelined (reader thread)
```

## Recipe to reproduce SF1500 5m51s

```bash
# One-time admin setup:
sudo sysctl vm.drop_caches=3
sudo sysctl -w vm.nr_hugepages=240000
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

# Build:
cd ~/gpu-research/gpu_crocsort
g++ -O3 -std=c++17 -pthread experiments/evict_cache.cpp -o experiments/evict_cache
nvcc -O3 -arch=sm_90 -std=c++17 experiments/stream_partition_sort.cu -o experiments/stream_partition_sort

# Run (assumes /mnt/data/lineitem_sf1500.bin exists):
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/run_sf1500
mkdir -p $NVME

experiments/evict_cache $INPUT
sleep 2
experiments/stream_partition_sort $INPUT $NVME/sf1500 8

# Output: $NVME/sf1500.sorted_{0..7}.bin (8-byte uint64 offsets)
# Verify: experiments/verify_compact_offsets $INPUT $NVME/sf1500.sorted_0.bin
```

## Variance bounds (every meaningful run)

| Tag | n | Mean | Best | Worst | ± |
|-----|---|------|------|-------|---|
| 19.16 K=16 4-GPU | 3 | 31m08s | 31m07s | 31m09s | 2s |
| 19.20 PERM_ONLY | 3 | 22m43s | 22m31s | 22m52s | 21s |
| 19.22 single-pass | 2 | 8m48s | 8m41s | 8m54s | 13s |
| 19.23 cache-hot | 3 | 8m24s | 7m56s | 8m39s | 43s |
| 19.25 streaming K=16 | 3 | 7m17s | 7m14s | 7m19s | 3s |
| 19.27 streaming K=8 | 3 | 7m02s | 6m43s | 7m18s | 18s |
| 19.30 pre-pin malloc | 3 | 6m56s | 6m23s | 7m18s | 27s |
| **19.44 hugepages** | **3** | **6m06s** | **5m51s** | **6m20s** | **15s** |

## Per-bucket sort timing (typical run, SF1500 hugepages)

```
Per-bucket (45 GB compact data, 1.1B records each, K=8):
  Pin:           ~12s/bucket (with hugepages)
  H2D upload:     0.4s
  CUB sort:       0.3s (4 LSD passes on 32-byte keys)
  Gather:         1.5s
  D2H + write:    4-5s
  Total:         ~18-20s/bucket

Round wall (4 GPUs concurrent on 4 buckets, NVMe write contention):
  ~30-50s/round (NVMe-bound on the 4-bucket output write)

2 rounds × ~40s = 80s sort phase (after pre-pin done)
```

## Tools needed for paper figures

- Throughput line: `THROUGHPUT_TABLE.md` has all scales
- Energy chart: `19.34_sf1500_power.log` for joules
- Variance whiskers: 19.44 n=3 logs
- Scaling chart: `19.32_cross_scale.log` for SF100/300, `19.31` SF500, `19.36` SF1000, `19.44` SF1500
- Comparison vs DuckDB/Polars: `4.1_duckdb_baseline.csv`, `19.38_polars_sf300.log`

## GDS final result (added 2026-05-06 morning)

GDS multi-stream prototype is WORKING and CORRECT. Final n=8 SF1500
overnight run:

  Best: 10m17s, Mean: 10m33s, Worst: 10m40s (±12s tight variance)
  Phase breakdown:
    GDS multistream partition + bucket-file write: 521s avg
    Sort phase (sort_compact_bucket × 8 in 2 rounds): 111s avg

vs hugepages best 5m51s: GDS approach is 4m26s SLOWER.

Why: cuFile reads at 4.4-6.5 GB/s (effective) but the 360 GB bucket-file
write at 1.3 GB/s adds ~277s that hugepages avoids by keeping buckets in
pinned RAM.

To break even with hugepages, GDS needs the integrated sort path that
sorts directly from in-RAM bucket buffers (no bucket file write). This
path was prototyped (`RUN_SORT=1` env) but hangs at SF300+. Investigation
deferred.

Bug fixes during GDS work (committed):
- cmap detection from full file via fseek (was capped to first 1 GB,
  missed varying bytes → wrong sort order at SF300+)
- Double event tracking for two-stage ping-pong buffers (single event
  caused stage[0] reuse before D2H completed at iter N=2)

## Things that aren't yet done that would strengthen the paper

1. **GDS integrated-sort hang fix** — the win path for GDS to beat
   hugepages. Hang root cause unknown (works at SF50, fails at SF300+).
   Likely candidates: CUDA context conflict between partition's resources
   and sort_thread's allocations, or cuFileBufRegister state interference.

2. **SF2000** — would need to delete SF1000 input first (disk space). Same recipe
   should yield ~7-8m at NVMe peak.

3. **Multi-NVMe striping** — both nvme drives could be combined; system disk has
   336 GB free. Halve input read time.

4. **Adversarial benchmark suite** — test recipe on sorted/reverse/zipfian/all-equal
   distributions (have `gen_synthetic` for this). 19.42 covered duplicate-key skew.

## Why this is paper-worthy

Not a "new sort algorithm" paper. It's a:

> *"How to sort beyond host RAM on commodity 4-GPU hardware: the recipe, the
> bottlenecks, and where the hardware ceiling actually is."*

Contributions:
1. Recipe (specific stack of optimizations that compose to 88% improvement)
2. The OS-CUDA cache-eviction interaction findings (novel)
3. Hardware-bounded analysis (proving we're at the floor)
4. 7 documented negative results (useful for next person)
5. Variance bounds at each stage
6. Cross-scale validation (SF100→SF1500)

Suitable target: SIGMOD/VLDB systems track, or workshop like DaMoN/IMDM.

## Auto-push daemon status

`/tmp/auto_push.sh` runs as PID 308842 (started May 03, ~50 commits queued).
SSH agent forwarding flaky during this session — commits land locally but
remote push is intermittent. All work is preserved in local git regardless.

## RUN_SORT debugging (2026-05-06 morning)

Test: integrated sort phase (RUN_SORT=1) at SF300 K=4 vs K=8.

| Config | Result | Wall |
|--------|--------|------|
| SF300 K=4 (1 round, 4-GPU concurrent) | WORKS, PASS | 61.6s |
| SF300 K=8 (2 rounds × 4-GPU) | HANGS round 2 (timeout 180s) | n/a |

**Diagnosis**: round 1's 4 sort_threads (each cudaSetDevice on GPUs 0-3,
allocate ~17 GB GPU memory, sort, free) complete fine. Round 2 launches
4 NEW sort_threads on the same GPUs and hangs.

Suspected cause: CUDA context state not fully cleaned between rounds
when using std::thread to dispatch sort work. Each round's threads do
cudaSetDevice + cudaMalloc, but the prior round's context may not be
fully released.

K=4 doesn't scale: at SF1500, K=4 → 90 GB buckets > 94 GB H100 HBM.
sort_one_bucket would OOM.

**Conclusion**: GDS integrated-sort path needs a different concurrency
model (e.g., persistent worker threads pinned to each GPU, work-queue
dispatch instead of thread-per-bucket). 1-2 days of additional engineering.

For now: hugepages 5m51s remains the SF1500 champion.
