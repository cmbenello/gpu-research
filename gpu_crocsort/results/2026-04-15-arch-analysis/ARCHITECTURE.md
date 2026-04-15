# Architectural Portability: How the Sort Would Move On Other Hardware

Date:   2026-04-15
Baseline: Quadro RTX 6000 (Turing, sm_75, 24 GB HBM, 672 GB/s GPU BW, PCIe 3.0 x16 ~12 GB/s D2H)
Host:   48-core (24C/48T), DDR4-2933 ~20 GB/s sustained per-socket, 128 GB

Current SF50 warm median: **6.21 s** (8B key fixup baseline, committed on `exp/fixup-fast-comparator`).
Current SF100 warm median: **7.82 s** (memory-gated 16B path).

This document walks through the SIX phases of the pipeline and asks "what changes on a different machine?" for each.

## Phase map (SF50 warm, current hardware)

| Phase | ms | % of warm total | Dominant resource |
|:------|---:|---:|:------|
| Run generation (2 × CUB LSD) | 2200 | 35 % | GPU compute + HBM |
| GPU 4-pass 32B merge        | 610 | 10 % | GPU compute + HBM |
| D2H permutation download    | 115 | 2 %  | PCIe  |
| CPU gather (scatter)        | 1700 | 27 % | **Host DRAM BW** |
| CPU fixup group-detect      | 540 | 9 %  | Host DRAM BW + invariance of `h_output` layout |
| CPU fixup parallel sort     | 1100 | 18 % | Host compute + L1/L2 |

Total warm ≈ 6.21 s. Anything that doesn't land in at least one of those boxes is a second-order cost.

## H100 / A100 — "drop it onto a much bigger GPU"

Relevant deltas vs RTX 6000:
- **HBM BW**: H100 SXM = 3350 GB/s (5× us). A100 = 1550 GB/s (2.3×).
- **HBM capacity**: 80 GB on both → the **SF100 memory gate flips**. 32B path becomes viable at SF100 (we need ~32 GB total, fits easily in 80 GB).
- **PCIe**: Gen4 x16 (H100) → 26 GB/s D2H (2×). Or NVLink-C2C (GH200) → 900 GB/s (90× us). Or PCIe Gen5 on H100 PCIe card → 52 GB/s (4× us).
- **SM count / FP32 throughput**: 3–5× more compute (not load-bearing — we are memory-bound in every GPU phase).

Predicted impact, phase by phase:

### Run generation (2200 ms → ~700 ms on H100, ~1100 ms on A100)
CUB LSD on compact keys is HBM-bandwidth bound. 5× HBM BW → ~5× faster ceiling, minus scheduler/launch overhead. Actual expected speedup: 3–3.5× (so 600–700 ms on H100). Chunking disappears on SF50 because all 300 M records' keys fit in 24 GB with room to spare — single run-gen, no multi-chunk merge.

### 4-pass 32B merge (610 ms → ~180 ms on H100)
Also bandwidth bound. Same 3–4× speedup. A more interesting question: is 4-pass still the right choice? With H100's capacity we could allocate all 4 prefixes + permutation + ping-pong buffers without memory pressure. But more importantly, **we could sort a 32B struct directly** instead of doing 4 × 8B LSD passes. One CUB sort over 32B keys eliminates the 3 gathers between passes. Ceiling ~150 ms; 2.5 × cheaper than the 4-pass approach.

### D2H download (115 ms → ~60 ms on PCIe 4, ~20 ms on PCIe 5, **~2 ms on NVLink-C2C**)
Linearly proportional to bus BW. On GH200 this phase **disappears into noise** — `cudaMemcpy` a 1.2 GB permutation across C2C takes ≈ 1.3 ms.

### CPU gather (1700 ms → unchanged on dual-socket, **~350 ms on Grace**)
This is scatter-read of 36 GB from host DRAM into an output buffer. Rate on RTX-6000 rig = 21 GB/s sustained, matching single-socket DDR4-2933. **This is the single most architecture-sensitive phase**:
- **Dual-socket EPYC / Sapphire Rapids (8-channel DDR5)**: 80 GB/s → 450 ms (4× faster).
- **Grace (LPDDR5X, 1000 GB/s advertised, ~500 GB/s sustained to one cluster)**: ~72 ms theoretical; more likely 300–400 ms after cache-thrash and NUMA. This is the phase where Grace wins hardest.
- **On GH200 specifically**: you'd skip the gather entirely because the GPU can materialize the sorted output *in host memory* via NVLink-C2C writes during Pass 4. Phase 3 goes to zero. That's a ~1.7 s win alone.

### Fixup group-detect (540 ms → 250 ms Sapphire Rapids, ~150 ms Grace, **0 ms if moved to GPU**)
Currently 48 threads scanning 32 scattered compact-byte positions per adjacent pair. Bound by host DRAM read BW for the record buffer and by cache-line efficiency.
- 2× DRAM BW = roughly 2× speedup (linear).
- **Moving the scan to GPU**: I implemented and tested this today — it produces the boundary bitmap in 45 ms GPU compute, but the 300 MB D2H download costs ~25 ms on PCIe 3, and Pass 4 grows by +58 ms because we need to gather pfx2/3/4 into final order (4-pass LSD leaves them in pre-pass-4 order). Net savings on RTX 6000 with PCIe 3: ~175 ms (masked by gather variance). **On H100 PCIe 4 + NVLink-C2C on GH200**: download is ~2 ms, pass 4 gathers are ~15 ms, and the 540 ms scan disappears entirely. Big win on those architectures.

### Fixup parallel sort (1100 ms → scales with cores × DRAM)
Two sub-phases: pack (~550 ms) and sort+reorder (~550 ms).
- **Pack** is contiguous active-byte memcpy+bswap. Bandwidth bound at ~7 GB/s × 48 threads ÷ N-threads-worth-DRAM. Doubles on DDR5. On AVX-512 machines, pack could fuse into a single `vpermb` + `vmovdqu64` loop — but the bottleneck is DRAM BW not compute, so the speedup is shallow.
- **Sort**: 3.15 M × std::sort on ~95 records. L1-resident inner loop. Scales with single-core speed and branch predictor quality, **not** with DRAM. Zen 4 / Zen 5 with a richer µop cache would be 20–30 % faster per thread. AVX-512 radix would win but the 8 B key already makes this cache-resident work.

## Grace Hopper (GH200) — specifically, the killer case

GH200 is the one architecture where our design **breaks open**. Three phases either collapse or vanish:

| Phase | RTX 6000 | GH200 | Why |
|:------|---------:|------:|:---|
| Run gen | 2200 ms | ~600 ms | HBM3 is 5× BW |
| GPU merge | 610 ms | ~150 ms | Same |
| D2H perm | 115 ms | ~2 ms | C2C 900 GB/s instead of PCIe 12 GB/s |
| CPU gather | 1700 ms | **0 ms** | GPU writes output directly into Grace unified memory during Pass 4 |
| Group-detect | 540 ms | ~0 ms | Done on GPU, bitmap read is direct |
| Fixup sort | 1100 ms | ~450 ms | LPDDR5X pack ~3× faster + more cores |
| **Total** | **6210 ms** | **~1200 ms** | **5.2 × faster** |

The architecturally-cleanest thing about GH200 is that the "gather" phase is an artifact of PCIe. On a coherent interconnect the sorted records never need to move: Pass 4 emits them in place.

## Single-socket 96-core Zen 5 with H100 PCIe 5

Contrasting scenario: fast CPU, fast GPU, but still PCIe-separated. I would expect:

| Phase | RTX 6000 | 96C Zen 5 + H100 PCIe 5 | Why |
|:------|---------:|------:|:---|
| Run gen | 2200 ms | 500 ms | HBM3 4× + single-chunk |
| GPU merge | 610 ms | 150 ms | HBM BW |
| D2H perm | 115 ms | 15 ms | PCIe 5 is ~52 GB/s practical |
| CPU gather | 1700 ms | **450 ms** | DDR5-5600 12-channel ~150 GB/s ÷ 4 (scatter efficiency) |
| Group-detect | 540 ms | 100 ms | BW + core count |
| Fixup sort | 1100 ms | 350 ms | 96 cores × pack+sort parallelism, DDR5 pack |
| **Total** | **6210 ms** | **~1560 ms** | **4× faster** |

Even without NVLink-C2C, the combination of 96-core DDR5 host + H100 PCIe 5 brings us to ~1.6 s. The gather phase remains the single biggest item.

## What the current design would NOT benefit from

Some architectures don't help much:

### Dual-CPU with high core count but DDR4
E.g. an old 2P Xeon 80-core with DDR4-2933. Adding more cores doesn't help fixup because the pack phase is DRAM-BW bound, not core-bound. You'd see +20 % at most, not 2×.

### GPUs with more SMs but same HBM BW
E.g. Quadro RTX 8000 (same Turing arch, 48 GB, same 672 GB/s). More capacity but same BW. SF100 would gain (32B path unlocked) but SF50 wouldn't change.

### PCIe 4 without CPU or GPU upgrades
Only helps the 115 ms D2H phase. 60 ms saved. Not load-bearing.

## What SIMD / AVX-512 changes in the inner loop

Current fast path in fixup packs 8B key via `__builtin_bswap64` + 21B tail via `memcmp`. This is already well vectorized by modern compilers; AVX-512 adds specific wins in two places:

1. **Pack loop (`pack_key_contig_tail`)**. 48 threads doing stride-29-byte reads from records. AVX-512 `vpbroadcastb` + mask gather would saturate a single thread's LLC read BW at ~15 GB/s (vs ~7 GB/s scalar). But total DRAM BW remains the cap, so 2× single-thread gain translates to ~1.2× aggregate. Marginal.

2. **Tail memcmp on 21B**. AVX-512 masked vpcmpeq over 32B + tzcnt on the result would cost 2 instructions vs ~6 in a glibc memcmp path. Aggregate savings ~50 ms across 3.15 M calls. Small.

AVX-512 wouldn't change **architecture** of the sort — it's just constant-factor cheaper compute on a phase that's already cheap.

## Translation rule (for the paper)

A sort like this generalizes across GPU architectures with a simple scaling rule:

```
T ≈ B / G_bw_hbm + N×K / C_pcie + N×R / H_bw_dram + N × f(key_complexity) / (cores × L1_bw)
     └─────┬────┘   └────┬────┘   └─────┬────┘   └──────────┬──────────┘
         run_gen+merge   D2H       gather+scan     fixup inner
```

where B = bytes of compact keys resident on GPU, K = perm size, R = record size, N = records, f grows roughly linearly in key compares per record. On the RTX 6000 rig we're saturating the second and third terms (PCIe + DRAM) — so any architecture that loosens either gets a near-linear speedup; architectures that scale cores or SM count alone see very little.

The GH200 case is qualitatively different because it **eliminates** the second and third terms: unified memory means the gather and the D2H collapse into the same work as the GPU merge's final write.

## Concrete next-step recommendations by target architecture

- **RTX 6000 / Turing with PCIe 3**: next lever is compact active-bytes pack with pre-fetch (likely +100 ms) and GPU-boundary-bitmap with pinned-buffer reuse (would settle the variance).
- **H100 PCIe**: re-enable the 32B path on SF100 (memory gate will pass) and move boundary detection to GPU by default. Estimated SF100 ~2 s, SF50 ~1.5 s.
- **GH200 Grace Hopper**: pipe Pass 4 output directly to host memory across C2C. Collapse gather. Estimated SF50 ~1.2 s.

The pipeline's "shape" is correct for all of these; only the gather phase needs an architecture-specific rewrite when C2C is available.
