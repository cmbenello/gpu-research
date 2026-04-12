# GPU External Sort — Final Optimization Results

## Hardware
- GPU: Quadro RTX 6000 (24GB HBM, 672 GB/s, sm_75, 72 SMs)
- CPU: 2× Intel Xeon (24 cores × 2 HT = 48 logical, 2 NUMA nodes)
- RAM: 187GB DDR4
- PCIe: Gen3 x16 (~16 GB/s pinned DMA)

## Final Scaling Results

| Data | Time | Throughput | Runs | Merge Passes |
|------|------|-----------|------|-------------|
| 10GB | 2.5s | 3.92 GB/s | 2 | 1 |
| 20GB | 4.7s | 4.28 GB/s | 4 | 1 |
| 30GB | 6.7s | 4.45 GB/s | 6 | 1 |
| 40GB | 9.2s | 4.33 GB/s | 8 | 1 |
| 50GB | 11.3s | 4.44 GB/s | 10 | 1 |
| 60GB | 13.5s | 4.44 GB/s | 12 | 1 |

**Throughput: 4.3-4.4 GB/s, constant across all scales. PCIe amplification: 2.0×.**

## Improvement from Original

| Data | Original | Final | Speedup |
|------|----------|-------|---------|
| 20GB | 51.5s (0.39 GB/s) | 4.7s (4.28 GB/s) | **11.0×** |
| 60GB | 228.0s (0.26 GB/s) | 13.5s (4.44 GB/s) | **16.9×** |

## Architecture

```
Phase 1: Run Generation (PCIe-bound, ~63% of time)
  ┌─────────────────────────────────────────────────────┐
  │ For each 5.5GB chunk (pinned h_data → GPU → h_data):│
  │   H2D upload (0.34s) → CUB radix sort (0.05s)      │
  │   → Record reorder (0.03s) → D2H download (0.34s)  │
  │   H2D overlaps with prev D2H (bidirectional PCIe)   │
  │   Key extraction to persistent GPU buffer           │
  └─────────────────────────────────────────────────────┘

Phase 2: Key-Only Merge (CPU-gather-bound, ~37% of time)
  ┌─────────────────────────────────────────────────────┐
  │ 1. CUB radix sort ALL keys on GPU (single pass)    │
  │    600M uint64 keys sorted in ~240ms                │
  │ 2. Download permutation (2.4GB pinned D2H, 183ms)   │
  │ 3. Multi-threaded CPU gather (48 threads, 15 GB/s)  │
  │    Random read from sorted runs → sequential write  │
  │    Block-prefetch (256 records) + huge pages         │
  └─────────────────────────────────────────────────────┘
```

## All Optimizations Applied (24 cycles)

### Major wins (>10% improvement each):
1. **CUB radix sort** for in-chunk sort (cycle 14): replaced 17-pass bitonic+merge
2. **Pinned h_data** (cycle 9): eliminated 2 memcpy per chunk, direct DMA
3. **Regular malloc** for gather output (cycle 11): avoided write-combining penalty
4. **CUB radix sort** for key merge (cycle 18): 1 pass instead of log2(K)
5. **GPU identity perm init** (cycle 21): eliminated CPU loop + 0.8-2.4GB upload
6. **Pre-allocate h_perm** (cycle 23): hid 1s cudaMallocHost behind run gen
7. **Skip copy-back** (cycle 16): return sorted pointer instead of memcpy

### Minor wins (<5% each):
8. K-way merge tree (cycle 4): 6 HBM passes instead of 17 (later replaced by CUB)
9. PCIe H2D/D2H overlap (cycle 10): bidirectional PCIe
10. Pre-alloc K-way descriptors (cycle 7)
11. Block-prefetch gather (cycle 20): 256-record blocks
12. Huge pages MADV_HUGEPAGE (cycle 20): TLB miss reduction
13. Single merge arena allocation (cycle 22)
14. Pre-allocate gather output buffer (cycle 24)

### What didn't work:
- Sort-by-source gather: random writes worse than random reads
- Event-based H2D sync: contention
- 2 larger buffers: slower CUB alloc + no PCIe overlap
- Regular malloc for perm: pageable D2H at 1.4 vs 16 GB/s
- cudaMallocAsync: OOM (stream allocator couldn't reuse freed memory)

## Theoretical Limits
- PCIe minimum: 2 × data / 16 GB/s = 7.5s for 60GB (we achieve 8.5s = 88%)
- DRAM gather: 60GB / 40 GB/s sequential = 1.5s (we do 3.9s random = 2.6× overhead)
- Maximum possible: ~9.4s for 60GB (PCIe + gather, no overhead)
- Current: 13.5s = within 1.4× of theoretical maximum


## System Tuning (not in code)
- `echo always > /sys/kernel/mm/transparent_hugepage/enabled`
  - Enables 2MB transparent huge pages system-wide
  - Reduces TLB misses: gather went from 15.5 → 19.7 GB/s
  - 60GB: 13.5s → 12.7s (4.73 GB/s)

