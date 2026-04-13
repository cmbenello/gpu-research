# GPU External Sort: Hardware Analysis & Projections

## Current System: Quadro RTX 6000 (Turing, sm_75)

### Hardware Specs
| Parameter | Value |
|-----------|-------|
| GPU | Quadro RTX 6000 |
| Architecture | Turing (sm_75) |
| HBM Capacity | 24 GB GDDR6 |
| HBM Bandwidth | 672 GB/s |
| SMs | 72 |
| PCIe | Gen3 x16 (~15.75 GB/s per direction) |
| Host RAM | 187 GB DDR4 |
| CPU | 2× Xeon (24C/48T) |

### Measured Performance (60GB, 600M × 100B records)
| Phase | Time | Throughput | Bottleneck |
|-------|------|-----------|------------|
| Run Generation | 6.28s | 9.55 GB/s | PCIe bandwidth |
| Key Merge (CUB) | 0.46s | — | GPU compute |
| Perm Download | 0.20s | 12.0 GB/s | PCIe D2H |
| CPU Gather | 3.06s | 19.6 GB/s | DRAM random access |
| **Total** | **10.16s** | **5.91 GB/s** | |

### Bottleneck Breakdown
```
Run gen: 6.28s (62%)  — PCIe Gen3 x16 bidirectional
  120 GB total PCIe traffic / 6.28s = 19.1 GB/s (aggregate bidirectional)
  Theoretical max: 31.5 GB/s bidirectional → 61% utilization
  Gap: CUB sort (50ms × 12 chunks) + reorder (30ms × 12) = 0.96s overhead

Merge: 3.88s (38%)  — DRAM random access
  CUB key sort:     0.24s  (600M uint64 at ~2.5 GKey/s)
  Perm init+extract: 0.03s  (GPU kernels, trivial)
  Perm download:     0.20s  (2.4GB D2H at 12 GB/s)
  CPU gather:        3.06s  (600M × 100B random reads at 19.6 GB/s)
  Overhead:          0.35s  (key buffer free, output malloc)
```

## Projections: How Different GPUs Change Things

### Key Insight from CrocSort Paper (Eq. 5)
The single-step feasibility condition `T_gen × T_merge ≤ Eρ²M² / (D×P)` tells us:
- Larger M (HBM) → fewer runs → shallower merge → less total I/O
- Our system already achieves single-pass merge (CUB re-sort), so M only affects run gen chunk size

### NVIDIA A100 (Ampere, sm_80)

| Parameter | RTX 6000 | A100 80GB | Change |
|-----------|----------|-----------|--------|
| HBM | 24 GB GDDR6 | 80 GB HBM2e | 3.3× |
| HBM BW | 672 GB/s | 2039 GB/s | 3.0× |
| SMs | 72 | 108 | 1.5× |
| PCIe | Gen3 x16 | Gen4 x16 | 2.0× |
| Max Shared Mem/SM | 64 KB | 164 KB | 2.6× |

**Projected 60GB performance:**

Phase 1 (Run Gen):
- PCIe Gen4: ~32 GB/s per direction vs ~16 GB/s → 2× faster transfers
- 80GB HBM: 60GB fits in ONE chunk → NO streaming needed!
- Upload 60GB at 32 GB/s = 1.88s, CUB sort = 0.05s, download = 1.88s
- **Projected run gen: ~3.8s** (vs 6.28s = 1.7× faster)

Phase 2 (Merge):
- If 60GB fits in HBM: merge happens entirely on GPU, no CPU gather!
- CUB sort 600M keys at ~6 GKey/s (3× faster than RTX 6000): ~0.1s
- Record reorder in HBM: 60GB × 2 (read+write) at 2039 GB/s = 0.06s
- **Projected merge: ~0.16s** (vs 3.88s = 24× faster!)

**Projected total: ~4.0s** (vs 10.1s = 2.5× faster, 15 GB/s)

But wait — if 60GB fits in 80GB HBM, we don't need external sort at all!
The external sort only matters for data > HBM. For A100:
- External sort needed for D > 80GB
- 200GB dataset: 200/26.7GB chunks = 8 runs, PCIe Gen4 round-trip = 12.5s
- Merge: CUB key sort 0.1s + GPU reorder 0.2s + perm download 0.4s + gather 6s
- **200GB projected: ~19s** (10.5 GB/s)

### NVIDIA H100 (Hopper, sm_90)

| Parameter | RTX 6000 | H100 SXM | Change |
|-----------|----------|----------|--------|
| HBM | 24 GB | 80 GB HBM3 | 3.3× |
| HBM BW | 672 GB/s | 3350 GB/s | 5.0× |
| SMs | 72 | 132 | 1.8× |
| PCIe | Gen3 x16 | Gen5 x16 | 4.0× |
| NVLink | — | 900 GB/s | new |
| TMA | — | Tensor Memory Accelerator | new |

**Projected 60GB performance:**

Phase 1 (Run Gen):
- 60GB fits in 80GB HBM → single chunk, no streaming
- H2D at 64 GB/s (Gen5) = 0.94s
- CUB sort at ~20 GKey/s = 0.003s (negligible)
- D2H at 64 GB/s = 0.94s
- **Projected run gen: ~1.9s** (vs 6.28s = 3.3× faster)

Phase 2: Same as A100 — all in HBM, ~0.16s

**Projected total: ~2.1s** (vs 10.1s = 4.8× faster, 28.6 GB/s)

For truly external data (>80GB):
- 200GB: 200/26.7GB = 8 chunks
- PCIe Gen5: 200GB × 2 / 64 GB/s = 6.25s run gen
- Merge: ~7s (CPU gather still DRAM-limited)
- **200GB projected: ~13s** (15.4 GB/s)

### Multi-GPU (NVLink)

With 4× H100 via NVLink (900 GB/s per link):
- 320GB total HBM → 200GB fits in aggregate HBM
- Each GPU sorts 50GB locally using CUB
- NVLink all-to-all exchange for merge: 200GB / 900 GB/s = 0.22s
- **200GB projected: ~2s** (100 GB/s!)

### Summary of Projections

| GPU | 60GB | 200GB | Key Advantage |
|-----|------|-------|---------------|
| RTX 6000 (current) | 10.1s | ~33s* | — |
| A100 80GB | ~4.0s | ~19s | 80GB HBM, PCIe Gen4, 3× HBM BW |
| H100 80GB | ~2.1s | ~13s | PCIe Gen5, 5× HBM BW, TMA |
| 4× H100 NVLink | ~1.0s | ~2s | 320GB aggregate HBM, 900 GB/s links |

*RTX 6000 200GB estimated from scaling trend

## What Would Change in the Code

### For A100/H100 (same code, better hardware):
1. **Larger chunks**: 80GB HBM → ~26GB per buffer (3 buffers) → fewer runs
2. **Faster CUB sort**: Higher SM count + HBM BW → negligible sort time
3. **Wider shared memory**: 164KB/SM (A100) → larger K-way merge partitions
4. **PCIe Gen4/5**: 2-4× faster transfers → run gen proportionally faster

### Code changes needed:
- `ARCH=sm_80` or `sm_90` in Makefile (already parameterized)
- No algorithmic changes — CUB, events, pinned memory all portable
- Could increase `RECORDS_PER_BLOCK` with more shared memory
- H100 `cp.async` and TMA could replace explicit memcpy for prefetch

### For Multi-GPU:
- NCCL AllToAll for inter-GPU data exchange
- Range-partitioned merge: each GPU merges its key range independently
- Need: GPU-to-GPU direct communication (NVLink)
- Significant new code for multi-GPU coordination
