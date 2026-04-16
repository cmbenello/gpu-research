# Compression Isolation: Compact Key Upload ON vs OFF

Date: 2026-04-16
Binary: `external_sort_tpch_compact` @ `exp/fixup-fast-comparator` + `DISABLE_COMPACT` env var
Hardware: Quadro RTX 6000 (24 GB HBM, PCIe 3 x16 ~12 GB/s), 48-core DDR4-2933 host.

## What this measures

The compact-key upload path detects invariant byte positions across all records,
then uploads only the varying bytes (packed into a 32B buffer) instead of full
120B records. This reduces PCIe traffic by 3.75× on TPC-H SF50 (61/66 key bytes
vary → 32B compact prefix uploaded vs 120B full record).

This experiment isolates that contribution by toggling `DISABLE_COMPACT=1`, which
forces the full-record triple-buffer upload path on the same binary + same data.

## SF50 — 6 warm runs each, all verified PASS sortedness + PASS multiset

### End-to-end

| Config | Chunks | PCIe H2D | Warm median (ms) | Min (ms) | Max (ms) |
|:-------|-------:|---------:|------------------:|---------:|---------:|
| Compact ON  | 2 (170M/chunk) | 9.6 GB  | **6 414** | 6 299 | 6 647 |
| Compact OFF | 9 (36M/chunk)  | 36.0 GB | **7 477** | 7 367 | 8 052 |
| **Delta** | | | **+1 063 ms (+16.6%)** | | |

### Per-phase breakdown (medians)

| Phase | Compact ON | Compact OFF | Delta | % | Dominant resource |
|:------|----------:|-----------:|------:|--:|:------------------|
| Run generation (H2D + GPU sort) | 2 171 | 3 183 | **+1 012** | +46.6% | **PCIe BW + GPU HBM** |
| GPU 32B prefix merge | 606 | 627 | +21 | +3.5% | GPU HBM |
| D2H permutation | 93 | 93 | 0 | 0% | PCIe |
| CPU gather | 1 722 | 1 771 | +49 | +2.8% | Host DRAM BW |
| CPU fixup | 1 804 | 1 808 | +4 | +0.2% | Host compute + L1/L2 |
| **Total** | **6 414** | **7 477** | **+1 063** | **+16.6%** | |

### What drives the 1.06 s delta

1. **PCIe traffic**: compact uploads 9.6 GB (32B × 300M); full uploads 36.0 GB
   (120B × 300M). At PCIe 3 sustained ~12 GB/s, the raw transfer delta is 2.2 s.
   Pipeline overlap (H2D ‖ GPU sort) hides ~1.2 s of that → net exposed ~1.0 s.

2. **GPU sort passes**: compact does 2 chunks × 4-pass LSD = 8 CUB radix sorts.
   Full does 9 chunks × 9-pass LSD (66B key) = 81 CUB sorts. HBM-bound, so
   per-pass cost is low, but the sheer pass count adds ~200 ms.

3. **Merge topology**: 9-way merge (compact OFF) vs 2-way (compact ON). The 9-way
   uses 3 merge passes; the 2-way uses 4 (32B prefix, 4-pass LSD). Net: 627 vs
   606 ms — surprisingly close because both are HBM-bound.

4. **Gather + fixup are identical**: same data layout, same 3.15M tie groups (avg
   95 records). Compact only affects the upload path, not the CPU-side pipeline.
   Minor difference: compact OFF skips 100.6K singleton groups vs 84.6K — the
   per-chunk full-key sort gives slightly better discrimination at merge time.

## Architectural implication

Compact saves 16.6% on RTX 6000 with PCIe 3. On faster interconnects:

| Architecture | PCIe BW | Expected compact savings |
|:-------------|--------:|------------------------:|
| RTX 6000 (PCIe 3 x16) | ~12 GB/s | **16.6% (measured)** |
| H100 (PCIe 5 x16) | ~52 GB/s | ~5-8% (PCIe mostly hidden) |
| GH200 (NVLink C2C) | ~900 GB/s | ~1-2% (upload is noise) |

As PCIe BW grows, compact's upload savings shrink because the transfer is hidden
behind GPU sort. The GPU-side benefit (4-pass vs 9-pass LSD) persists but is
worth only ~200 ms — a constant that doesn't scale with data size.

## Files

- `sf50_compact_on.log` — 6 warm runs, compact enabled (default).
- `sf50_compact_off.log` — 6 warm runs, compact disabled via `DISABLE_COMPACT=1`.
