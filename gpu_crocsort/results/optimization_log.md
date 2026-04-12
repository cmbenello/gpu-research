# GPU External Sort Optimization Log

## Hardware
- GPU: Quadro RTX 6000 (24GB HBM, 672 GB/s, sm_75, 72 SMs)
- CPU: 48 cores, 187GB RAM
- PCIe: Gen3 x16 (~12 GB/s per direction)

## 20GB Results
| Cycle | Change | Run Gen | Merge | Total | GB/s |
|-------|--------|---------|-------|-------|------|
| 0 | Baseline (bitonic+2way+CPU merge) | 37.1s | 14.5s | 51.5s | 0.39 |
| 4 | K-way merge tree in-chunk | 27.9s | 14.7s | 42.6s | 0.47 |
| 9 | Pinned h_data (direct DMA) | 13.9s | 14.8s | 28.7s | 0.70 |
| 10 | Bidirectional PCIe overlap | 12.8s | 14.5s | 27.3s | 0.73 |
| 11 | Regular malloc for gather | 12.9s | 4.2s | 17.1s | 1.17 |
| 14 | **CUB radix sort** | **3.1s** | **4.4s** | **7.5s** | **2.68** |

## 60GB Results
| Cycle | Change | Total | GB/s |
|-------|--------|-------|------|
| 0 | Baseline | 228.0s | 0.26 |
| 11 | +pinned+overlap+gather | 52.4s | 1.15 |
| 14 | **+CUB radix sort** | **23.3s** | **2.58** |

## Current Bottleneck (20GB, 7.5s total)
- Run gen: 3.1s (42%) — PCIe-limited, GPU sort ~50ms
- Merge: 4.4s (58%) — GPU key merge 2.0s + gather 1.9s + copy 0.7s
