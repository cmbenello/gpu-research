# GPU External Sort — Final Comprehensive Results

## System
- GPU: Quadro RTX 6000 (24GB GDDR6, 672 GB/s, sm_75)
- CPU: 2× Intel Xeon Silver 4116 (24C/48T), 192GB DDR4
- PCIe: Gen3 x16
- THP: enabled (`echo always > /sys/kernel/mm/transparent_hugepage/enabled`)

## Architecture: Single-Pass Key-Only Sort
```
1. cudaMemcpy2D (strided DMA: read full records, transfer only 10B keys)
2. LSD radix sort: CUB SortPairs on uint16 tiebreaker, then uint64 primary
3. Download permutation (D2H)
4. Multi-threaded CPU gather (48 threads, block-prefetch, huge pages)
```

## Random GenSort (100-byte records, verified correct)

| Data | Time | Throughput | vs Original (228s/60GB) |
|------|------|-----------|------------------------|
| 10GB | 1.7s | 6.06 GB/s | — |
| 20GB | 3.0s | 6.78 GB/s | **17.5×** |
| 40GB | 5.2s | 7.66 GB/s | — |
| 60GB | 7.5s | 7.97 GB/s | **30.3×** |

## TPC-H Lineitem (verified correct, full orderkey+linenumber ordering)

| Dataset | Size | Records | Time | Throughput |
|---------|------|---------|------|-----------|
| SF10 | 6 GB | 60M | 1.1s | 5.47 GB/s |
| SF50 | 30 GB | 300M | 3.7s | 8.02 GB/s |

## Competitive Context

| System | Dataset | Hardware | Throughput |
|--------|---------|----------|-----------|
| **GPU CrocSort (ours)** | **60GB GenSort** | **RTX 6000** | **8.0 GB/s** |
| CrocSort CPU (VLDB 2026) | 200GB GenSort | 2×Xeon, 192GB, NVMe | ~0.4 GB/s |
| DuckDB 1.4 | in-memory | CPU only | ~0.5-2 GB/s |
| Stehle-Jacobsen (SIGMOD 2017) | 64GB KV | GPU | ~1 GB/s |
| CUB DeviceRadixSort | in-HBM | A100 | 29 GKey/s |

## Optimization Journey (40 cycles)
- Original: 228s → Final: 7.5s = **30.3× speedup**
- Key innovations: CUB radix sort, single-pass key-only architecture,
  strided DMA, pinned memory, event-only GPU sync, mmap+MAP_POPULATE
- GPU utilization: 3% of wall time (0.5s out of 7.5s)
- PCIe amplification: 0.14× (sublinear!)
