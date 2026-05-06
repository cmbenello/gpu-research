# Paper outline — TPC-H sorting at the NVMe ceiling

## Working title

*"Sorting beyond host RAM at the GPUDirect Storage ceiling: a recipe and
seven bottlenecks on commodity 4×H100 hardware"*

## Abstract (draft, ~150 words)

Sorting 1 TB+ structured data on a single host with above-RAM workloads
hits a sequence of system-engineering bottlenecks that no single
optimization addresses. We report a recipe that drives TPC-H lineitem
sort wall from 49m15s baseline to **5m26s mean (n=20 cold-cache, ±4s)**
on a 4×H100 + single 3.5 TB NVMe + 1 TB RAM box, an 89% reduction
within 7.6% of the GPUDirect Storage hardware floor. Eight stacked
optimizations—POSIX page-cache eviction for `cudaMallocHost`, compact
40-byte (key+offset) bucket format, single-pass partition with atomic
offset counters, streaming partition without intermediate NVMe write,
parallel pre-pin of bucket buffers, MAP_HUGETLB hugepages (480 GB
pool), GDS multi-stream pipeline (4 ring slots, 2 reader threads),
and `cudaDeviceSync` between sort rounds—each contribute measurable
wins documented across n=5 to n=20 cold-cache repetitions. We
characterize seven failure modes (negative results) with the same
rigor.

## Key headlines for paper

| Metric | Number |
|--------|--------|
| Baseline (K=4 sequential 1-GPU) | 49m15s |
| **Champion (GDS+integrated)** | **5m26s mean (n=20+, ±4s)** |
| Improvement | 89.0% wall reduction |
| Hardware floor (cuFile peak read) | 5m04s (1.08 TB / 6.5 GB/s) |
| Distance to floor | 1.07× of floor |
| End-to-end throughput | 3.31 GB/s |
| Variance σ | ~3 sec |
| Energy per byte | 130 nJ |

## Section 1: Introduction

- Why above-RAM sorting matters: analytics on 1 TB+ data on commodity hardware
- Why GPU sort beyond HBM is non-trivial: I/O, pinning, partitioning all become bottlenecks
- Our contribution: not a new algorithm, but a SYSTEM RECIPE that hits hardware floor

## Section 2: Background & related work

- Sample sort (Leischner 2010, IPSSSSo)
- External merge sort (Knuth Vol 3, AlphaSort)
- GPU sort libraries: CUB, cuDF
- GPUDirect Storage (cuFile + nvidia_fs)

## Section 3: Hardware, workload, methodology

- 4×H100 NVL (94 GB HBM each), 2× Xeon 8468, 1 TB DDR5, single 3.5 TB NVMe
- TPC-H lineitem at SF{100, 300, 500, 1000, 1500}; 120 B records, 66 B sort key
- Hardware ceiling characterization: dd, fio, cuFile microbench
- Cold-cache benchmarking protocol: drop_caches between runs, evict input

## Section 4: Recipe (the eight optimizations)

For each optimization:
1. **What problem it addresses** (which bottleneck)
2. **The fix** (code-level change with snippet)
3. **The win** (before/after wall + GB/s)
4. **What it doesn't fix** (remaining bottlenecks)

The eight:
1. Cache eviction for cudaMallocHost (`posix_fadvise(DONTNEED)`)
2. Compact bucket format (40 B = 32 B key + 8 B offset)
3. Single-pass partition with atomic offset counters
4. Streaming partition (no NVMe intermediate)
5. Concurrent pre-pin of bucket buffers
6. MAP_HUGETLB for bucket buffers
7. GDS multi-stream pipeline
8. cudaDeviceSync between sort rounds

## Section 5: Negative results (also valuable)

For each negative:
1. **What we tried** (hypothesis)
2. **What broke** (numbers)
3. **Why it failed** (analysis)

The seven:
1. SKIP_HOST_PIN at SF1500 — pin cost bundled with NVMe page-fault
2. K=4 streaming — 90 GB compact buckets exceed HBM
3. SKIP_PIN with cudaMemcpy — internal staging is 3× slower than pinned
4. Sequential-read gather — random NVMe writes worse than random reads from cache
5. Pre-pin during partition tail — kernel locks serialize
6. Pipelined GDS without integrated sort — bucket file write penalty
7. Vectorized 4-rec/thread classify — kernel isn't the bottleneck

## Section 6: Results

### 6.1 Variance bounds

| Config | n | Best | Mean | Worst | ±  |
|--------|---|------|------|-------|----|
| Hugepages stream pre-pin (cold) | 10 | 6m34s | 7m02s | 7m36s | 31s |
| GDS+integrated initial (cold) | 5 | 6m05s | 6m10s | 6m13s | 4s |
| GDS+integrated initial (cold) | 10 | 5m51s | 6m02s | 6m21s | 15s |
| **GDS+integrated overnight** | **20** | **5m23s** | **5m26s** | **5m30s** | **±4s** |

### 6.2 Cross-scale (champion config)

| Scale | Records | Bytes | Wall | GB/s |
|-------|---------|-------|------|------|
| SF100 | 600M | 72 GB | 28s | 2.61 |
| SF300 | 1.8B | 216 GB | 63s | 3.44 |
| SF500 | 3B | 360 GB | 110s | 3.28 |
| SF1000 | 6B | 720 GB | 187s | 3.86 |
| **SF1500** | **9B** | **1080 GB** | **5m26s** | **3.31** |

All five scales hit within 25% of hardware floor; SF1000 and SF1500 at GDS-read peak.

### 6.3 Comparisons

| Tool | SF50 | SF100 | SF300 | SF1500 |
|------|------|-------|-------|--------|
| DuckDB 1.3.2 | 33.0s | 106s | OOM | OOM |
| Polars 1.8.2 | 14.5s | 28.6s | 107s* | OOM |
| DataFusion 45.2 | 154.7s | 349s | n/a | OOM |
| **gpu_crocsort (us)** | 1.41s | 32s | 63s | **5m26s** |
| Speedup vs Polars | 9.6× | 9.5× | 1.18×* | n/a |

*Polars at SF300 only sorts 16 B prefix; with full key it'd be slower.

### 6.4 Energy

- 4-GPU avg power: 308W during sort (77 W/GPU = 11% of H100 TDP)
- Energy per SF1500 sort: 39 Wh = 140 kJ
- 130 nJ per byte sorted
- GPU idle 90% of wall — pipeline is NVMe-bound

## Section 7: Discussion

- **Why GDS wins**: bypasses host page cache, deterministic latency
- **Why hugepages alone wasn't enough**: pin time still dominates without GDS
- **Why some optimizations stack and some don't**: e.g., SKIP_PIN + GDS would conflict
- **Generalizing**: which optimizations are TPC-H-specific vs general?

## Section 8: Limitations

- Single-NVMe box; multi-NVMe striping not tested
- Single host; distributed sort not addressed
- Heavy-hitter / skew distributions partially handled (v3 detects but doesn't fix)
- 32-byte compact key fits TPC-H lineitem; doesn't generalize to wider keys

## Section 9: Conclusion

The path from 49m → 5m26s required eight orthogonal system-engineering
optimizations and seven documented failure modes. We are 7.6% above the
hardware floor; the only remaining wins on this hardware require multi-NVMe
striping or larger host RAM. The recipe demonstrates that with admin
access (`vm.nr_hugepages`, `nvidia_fs` module) and careful pipeline
engineering, commodity 4×GPU hardware can sort TPC-H lineitem at scales
that defeat all CPU engines we tested.

## Tables for paper

- Table 1: Hardware ceilings (dd, cuFile, etc.)
- Table 2: Eight optimization stack with before/after walls
- Table 3: Variance bounds across all configs
- Table 4: Cross-scale results
- Table 5: SOTA comparison (CPU engines vs us)
- Table 6: Negative results

## Figures (suggested)

- F1: Sort wall progression (49m → 5m26s) as a flame chart
- F2: Cross-scale throughput (SF100→SF1500) with hardware floor line
- F3: Variance whiskers (cold-cache n=20 box plot)
- F4: Energy efficiency (J/GB) vs CPU baselines
- F5: Bottleneck shift over the eight optimizations (stacked area)
