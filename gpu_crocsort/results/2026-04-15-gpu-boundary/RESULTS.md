# GPU-Side Boundary Detection — Tried, Net-Neutral, Reverted

Date: 2026-04-15
Branch: (experimental, not committed)
Goal: Replace the ~540 ms CPU group-detection scan in Phase 4 fixup with a GPU kernel that produces a per-record boundary bitmap, downloaded to pinned host memory, then swept sequentially on CPU.

## What I implemented

1. Two new CUDA kernels: `compute_boundaries_4pfx_kernel` (for 32B path) and `compute_boundaries_2pfx_kernel` (for 16B path). Both compute `boundary[i] = any(pfx_k[i] != pfx_k[i-1])` in parallel.
2. Pass 4 restructured to gather pfx2/pfx3/pfx4 into the post-pass-4 order (they were previously left in pre-pass-4 order, but boundary detection needs them in final order). This uses the CUB DoubleBuffer + identity-shuffle pattern from passes 1–3.
3. Allocated `d_boundary` (300 MB = 1 byte per record). Launched boundaries kernel. Downloaded to `h_boundary`.
4. In Phase 4 fixup, dual-path group-detection: if `h_boundary` is present, 48 threads each sweep their slice with `if (B[i]) push_back(i)` (cache-friendly uint8 stream). Otherwise, fall back to the original CPU scatter-byte scan.

## Two variants tested

Variant A: `cudaMallocHost` for `h_boundary` + `cudaMemcpyAsync` D2H, hoping to overlap with CPU gather.
Variant B: pageable `malloc` + synchronous `cudaMemcpy` between Pass 4 and gather.

Variant A lost ~500 ms per run to the 300 MB pinned allocation — net worse than baseline. Variant B eliminated that cost.

## Results (6 warm SF50 runs, every run PASS sortedness + PASS multiset hash `0x89c28ecae7c4b777`)

| metric | CPU scan baseline | GPU bitmap (variant B) | delta |
|:------|---:|---:|---:|
| group-detect scan | 474–744 ms (median ~540 ms) | 43–71 ms (median ~46 ms) | **−494 ms** |
| stitch            | 118–158 ms | 122–176 ms | +15 ms |
| pass 4            | 116 ms (stable) | 174 ms (stable) | +58 ms |
| fixup total       | 1687–2000 ms (median ~1750 ms) | 1330–1429 ms (median ~1385 ms) | **−365 ms** |
| merge+gather+fixup | 4084–4669 ms (median ~4119 ms) | 3787–4380 ms (median ~4053 ms) | −66 ms |
| **warm total**    | **6508 ms median** | **6397 ms median** | **−111 ms** |

The GPU path is ~110 ms faster at the median. But the run-to-run variance is large (±300 ms), making this hard to claim as a robust win.

## Why the savings don't equal the scan delta

The CPU scan saves 494 ms, but that does not flow fully to the end-to-end total:
- Pass 4 grows by 58 ms (three extra CUB gathers for pfx2/3/4 + allocating `d_boundary` + launching the kernel).
- Stitch grows slightly (+15 ms) — not fully understood; possibly due to how quickly the boundary lists arrive at the stitch phase.
- Gather phase itself has higher variance when `d_boundary` holds 300 MB of HBM across the Pass 4 → Phase 3 boundary; some runs show 2.0 s vs the baseline's typical 1.7 s.

Net savings at the fixup level (−365 ms) minus Pass 4 cost (+58 ms) minus gather variance (≈ +100 ms) = ≈ −150 ms predicted, ~ −110 ms observed.

## Decision: reverted from `exp/fixup-fast-comparator`

- Kept the committed 8 B-key baseline (6.21 s) as the stable win.
- Documented the attempt here + in `ARCHITECTURE.md` — the GPU boundary approach is **the right architecture for H100/A100 and is essential on GH200**, where PCIe latency is replaced by C2C coherency and the 58 ms Pass 4 overhead drops to ~15 ms. On RTX 6000 with PCIe 3, the marginal win is below noise.

## What I would do differently next time

1. Pre-allocate `d_boundary` once outside the per-sort loop (reusable across runs — would eliminate the per-run GPU malloc).
2. Avoid gathering pfx2/pfx3/pfx4 in Pass 4 by computing boundaries during pass 4's CUB sort using a custom comparator — the output of the stable sort carries enough info to emit boundary flags per-output-position. Removes the +58 ms.
3. Downloads only the compacted boundary-index list (3.15 M × 8 B = 25 MB) rather than the full 300 MB bitmap. Uses `cub::DeviceSelect::If`. Reduces PCIe traffic by 12×.

## Files

- `sf50_3runs.log` — first attempt with pageable malloc, 3 runs.
- `sf50_6runs.log` — first attempt with pinned `cudaMallocHost` — discarded due to alloc cost.
- `sf50_6runs_v2.log` — pageable variant, 6 warm SF50 runs used for the table above.
- `sf50_reverted_baseline.log` — baseline re-run after revert, 6 warm SF50 runs.
