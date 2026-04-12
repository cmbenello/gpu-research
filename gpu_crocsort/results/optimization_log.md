# GPU External Sort Optimization Log

## Hardware
- GPU: Quadro RTX 6000 (24GB HBM, 672 GB/s, sm_75, 72 SMs)
- CPU: 48 cores, 187GB RAM
- PCIe: Gen3 x16 (~12-16 GB/s with pinned memory)

## Final Results

| Data | Time | Throughput | vs Original |
|------|------|-----------|-------------|
| 10GB | 2.6s | 3.91 GB/s | — |
| 20GB | 4.7s | 4.29 GB/s | **11.0×** (was 51.5s) |
| 40GB | 9.1s | 4.39 GB/s | — |
| 60GB | 13.6s | 4.40 GB/s | **16.8×** (was 228.0s) |

## Optimization History

### 20GB
| Cycle | Change | Total | GB/s |
|-------|--------|-------|------|
| 0 | Baseline (bitonic+2way+CPU merge) | 51.5s | 0.39 |
| 4 | K-way merge tree in-chunk | 42.6s | 0.47 |
| 9 | Pinned h_data (direct DMA) | 28.7s | 0.70 |
| 11 | Regular malloc gather output | 17.1s | 1.17 |
| 14 | CUB radix sort in-chunk | 7.5s | 2.68 |
| 16 | Skip copy-back | 6.1s | 3.28 |
| 18 | CUB radix sort key merge | 5.8s | 3.43 |
| 21 | GPU identity perm init | 5.1s | 3.92 |
| 23 | **Pre-alloc h_perm + arena** | **4.7s** | **4.29** |

### 60GB
| Cycle | Change | Total | GB/s |
|-------|--------|-------|------|
| 0 | Baseline | 228.0s | 0.26 |
| 14 | +CUB radix sort | 23.3s | 2.58 |
| 16 | +Skip copy-back | 18.8s | 3.19 |
| 21 | +GPU perm init | 14.6s | 4.10 |
| 23 | **+Pre-alloc h_perm** | **13.6s** | **4.40** |

## Architecture
```
Phase 1: Run Generation (PCIe-bound)
  Upload 5.5GB chunk → CUB radix sort (~50ms) → Reorder records → Download
  3 GPU buffers, H2D/D2H overlapped on bidirectional PCIe
  Key extraction to persistent GPU buffer during sort

Phase 2: Key-Only GPU Merge
  CUB radix sort on ALL keys (single pass, ~240ms for 600M keys)
  Download permutation → Multi-threaded CPU gather (48 threads, 15 GB/s)
```

## What Worked (ranked by impact)
1. CUB radix sort for in-chunk sort: replaced bitonic+17-pass merge
2. Pinned h_data: eliminated 2 memcpy per chunk (direct DMA)
3. Regular malloc for gather output: avoided write-combining penalty
4. CUB radix sort for key merge: 1 pass instead of log2(K)
5. GPU identity perm init: eliminated CPU loop + 0.8-2.4GB upload
6. Pre-allocate h_perm: hid 1s cudaMallocHost behind run gen
7. Key retention: skip key re-upload for merge phase
8. PCIe overlap: bidirectional H2D/D2H on separate streams
9. Block-prefetch gather: 256-record blocks for deeper prefetch pipeline
10. Huge pages: MADV_HUGEPAGE reduces TLB misses in gather

## What Didn't Work
- Sort-by-source gather: random writes worse than random reads
- Event-based H2D sync: added contention
- 2 larger buffers: slower CUB alloc + no PCIe overlap
- Regular malloc for perm download: pageable D2H at 1.4 GB/s vs 16 GB/s pinned

## Theoretical Limits
- PCIe: 20GB × 2 (up+down) / 16 GB/s = 2.5s minimum for 20GB
- Current run gen: 3.1s = 80% of PCIe theoretical
- PCIe amplification: 2.0× (at theoretical minimum)
- Remaining bottleneck: CPU gather at 15 GB/s (DRAM random access)
