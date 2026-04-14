# E2E Experiment Results (2026-04-14)

Machine: Quadro RTX 6000 (24GB, Turing sm_75), 2× Xeon Silver 4116, 192GB RAM, 2 NUMA.

All times exclude data loading/generation. Median of 2-3 runs shown (page cache warms after run 1).

## TPC-H lineitem (88B key → 66B effective → 26B compact, 120B records)

| Size  | Records | Path            | Run Gen | Merge | Gather  | Total   | Throughput |
|-------|---------|-----------------|---------|-------|---------|---------|------------|
| SF10  | 60M     | Single-pass     | 0.98s   | —     | 0.21s   | 1.19s   | 6.0 GB/s   |
| SF50  | 300M    | OVC + compact   | 1.29s   | 9ms   | 1.30s   | 2.68s   | 13.4 GB/s  |
| SF100 | 600M    | OVC + compact   | 1.97s   | 9ms   | 0.91-1.82s | 3.26-4.0s | 18-22 GB/s |
| SF100 | 600M    | OVC non-compact | 5.94s   | 9ms   | 1.72-2.05s | 7.85-8.18s | 8.8-9.2 GB/s |

## GenSort (10B key, 100B records)

| Size  | Records | Path         | Run Gen | Gather | Total  | Throughput | Paper  |
|-------|---------|--------------|---------|--------|--------|------------|--------|
| 10GB  | 100M    | Single-pass  | 0.88s   | 0.33s  | 1.20s  | 8.3 GB/s   | 1.6s   |
| 30GB  | 300M    | Single-pass  | 1.99s   | 1.00s  | 3.00s  | 10.0 GB/s  | 4.2s   |
| 60GB  | 600M    | Single-pass  | 3.63s   | 0.93s  | 4.56s  | 13.2 GB/s  | 7.9s   |

## Key Findings

- **TPC-H SF100 compact: 3.26s best** (vs 13.2s baseline; 4x speedup today from 8.3s → 3.26s)
- **GenSort 60GB: 4.56s** (vs paper's 7.9s; 1.7x speedup from CPU gather improvements)
- **PCIe amplification**: compact TPC-H = 0.3x (19GB keys vs 72GB data); GenSort = 0.14x
- **Compact upload win**: 2.1-2.5x vs non-compact on TPC-H (3.26s vs 7.85s for SF100)

## Optimizations Applied Today

1. **2-pass CUB LSD merge for 16B prefix** — 9ms GPU merge, GPU tie detection (atomicOr)
2. **CPU-side compact key extraction** — upload 32B compact vs 120B records (3.75x less PCIe)
3. **Skip d_buf[0]/d_buf[1] in compact mode** — frees 7.4GB GPU, enables 170M rec/chunk (4 vs 19)
4. **Non-temporal stores in gather** — `_mm_stream_si64` × 15 per record, bypasses write cache
5. **CPU extract / GPU sort pipeline overlap** — pre-warms TLB, helps gather 
6. **Multi-run mode** — `--runs N` without reload (h_data is read-only in sort)

## Bottleneck Analysis

For TPC-H SF100 @ 3.26s:
- Run gen: 1.97s (60%) — PCIe H2D limited (~12 GB/s Gen3 x16)
- Perm download: 0.19s (6%)
- CPU gather: 0.91-1.82s (28-56%) — DRAM random-access, 40-80 GB/s with warm caches
- GPU merge: 9ms (<1%)
