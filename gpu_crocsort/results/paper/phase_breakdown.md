# Phase Breakdown — GPU CrocSort on RTX 6000

Hardware: Quadro RTX 6000 (24 GB, PCIe 3.0), 48-core DDR4-2933.
Binary: `external_sort_tpch_compact` @ `exp/fixup-fast-comparator` (8e403d7).
All times: warm median (ms). Records: 120B (66B key + 54B value).

## Multi-chunk datasets (OVC merge path)

| Phase                | SF50 (36 GB) | SF100 (72 GB) | Random 400M (48 GB) |
|:---------------------|-------------:|--------------:|--------------------:|
| Records              |        300M  |         600M  |              400M   |
| Chunks               |           2  |            4  |               10    |
| Vary / 66            |       61/66  |        27/66  |            66/66    |
| Compact key          |     32B pfx  |      32B pfx  |         full 66B    |
|                      |              |               |                     |
| **Phase 1: Run Gen** |    **2,200** |    **3,750**  |          **4,170**  |
| **Phase 2: GPU Merge** |    **610** |      **833**  |            **566**  |
|   _Merge passes_     |     4 (32B)  |      2 (16B)  |          2 (16B)    |
| **D2H Perm**         |       **94** |      **187**  |            **122**  |
| **Phase 3: Gather**  |    **1,710** |    **3,650**  |          **2,670**  |
| **Phase 4: Fixup**   |    **1,760** |        **0**  |              **0**  |
|   _Group-detect_     |          630 |            —  |                —    |
|   _Parallel sort_    |        1,110 |            —  |                —    |
|   _Groups_           |      3.15M   |            —  |                —    |
|   _Avg group size_   |           95 |            —  |                —    |
|                      |              |               |                     |
| **Total**            |    **6,370** |    **8,020**  |          **7,530**  |
| **GB/s**             |     **5.65** |     **8.98**  |           **6.38**  |

## Single-chunk datasets (in-memory LSD path)

| Phase                | SF10 (7.2 GB) | Taxi 12mo (4.6 GB) | Random 60M (7.2 GB) |
|:---------------------|----------:|----------:|----------:|
| Records              |     60M   |     38M   |     60M   |
| Vary / 66            |   61/66   |   60/66   |   66/66   |
|                      |           |           |           |
| **H2D Upload**       |    ~15    |     —     |    ~15    |
| **GPU LSD Sort**     |  ~1,300   |     —     |  ~1,300   |
| **D2H Perm**         |    ~21    |     —     |    ~21    |
| **CPU Gather**       |   ~435    |     —     |   ~440    |
| **Fixup**            |      0    |     —     |      0    |
|                      |           |           |           |
| **Total**            | **1,754** | **1,393** | **1,765** |
| **GB/s**             |  **4.10** |  **3.30** |  **4.08** |

_Note: Taxi 12mo and Taxi 1mo use the single-chunk path where all phases are fused
into a single timer ("gen"). Detailed sub-phase breakdown is not emitted by the
binary for the single-pass path — it reports only total time. SF10 and Random 60M
use the multi-pass LSD path which reports per-step times._

## Key observations

1. **SF50 is fixup-dominated**: 1,760 / 6,370 = 28% of total time is CPU fixup.
   The 61/66 varying bytes overflow 32B prefix → 3.15M tie groups need CPU resolution.

2. **SF100 has zero fixup**: 27/66 varying bytes fit entirely in 32B compact prefix.
   Bottleneck shifts to gather (3,650 ms = 46% of total) — pure DRAM bandwidth.

3. **Random 400M has zero fixup**: 66/66 vary but 16B prefix has NO TIES because
   uniform random 16B prefixes are unique across 400M records (collision prob ≈ 10⁻¹⁹).

4. **Run Gen scales sub-linearly with data size** (amortized GPU pipeline):
   - SF50 (36 GB): 2,200 ms → 16.4 GB/s
   - SF100 (72 GB): 3,750 ms → 19.2 GB/s
   - Random 400M (48 GB): 4,170 ms → 11.5 GB/s (full 66B key, no compact)

5. **GPU Merge is cheap**: 600-800 ms regardless of data size — it operates on
   compact OVC+prefix records, not full records.

6. **Gather is DRAM-BW bound**: ~18-21 GB/s across all datasets. Scales linearly
   with data size (1,710 ms for 36 GB, 3,650 ms for 72 GB).

## Sources

- SF50: `results/2026-04-16-compression-isolation/sf50_compact_on.log` (6 runs, median run)
- SF100: `results/2026-04-15-scaling/sf100_6runs.log` + `results/2026-04-16-adversarial-sf100/sf100_baseline.log`
- Random 400M: `results/overnight_2026-04-16/random_400M.log` (3 runs, run 2)
- SF10: `results/paper/sf10_phase.log` (3 runs, fresh on current binary)
- Overnight CSV medians: `results/overnight_2026-04-16/master.log`
