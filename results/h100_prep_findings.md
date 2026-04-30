# H100 prep — findings from lincoln + roscoe (2026-04-30)

While the H100 is in transit, ran prep experiments on the cs.uchicago boxes to:
1. Validate the bitpack patch on real-world (NYC Taxi) data
2. Pin down variance with 5-warm-run baselines at the working scales
3. Sanity-check the GPU vs CPU crossover for synthetic data
4. End-to-end-test the new fast TPC-H generator

## TL;DR

- **The bitpack PCIe win depends on data structure.** Structured data (TPC-H lineitem) benefits 25%; uniform random doesn't compress at all; small datasets that fit in a single chunk get the codec but no wall-time gain.
- **Variance is small (~3-5%).** All 5-run sets stayed within a tight band, so the headline numbers are reliable.
- **The fast TPC-H generator is byte-identical to the reference** and roughly 9× faster end-to-end at SF1 (3s tpchgen-cli + 10s encode vs 123s for the slow path).

## SF20 5x variance — RTX 2080

| Config | best | median | worst | H2D PCIe | Chunks |
|---|---|---|---|---|---|
| baseline | 3.29 s | 3.36 s | 3.44 s | 3.84 GB | 3 |
| **bitpack** | **3.18 s** | **3.26 s** | 3.31 s | **2.88 GB** | **2** |

PCIe drops 25% reliably; wall-time gain is small (~3%) because gather still dominates this scale.

## SF50 5x variance — P5000

| Config | best | median | worst | H2D PCIe | Chunks |
|---|---|---|---|---|---|
| baseline | 8.96 s | 9.28 s | 9.34 s | 9.60 GB | 3 |
| **bitpack** | **8.87 s** | 9.23 s | **9.43 s** | **7.20 GB** | **2** |

Same 25% PCIe drop as SF20. Wall-time tied to gather phase, which compression doesn't shrink at this scale.

## Where bitpack does NOT help

### Uniform random 100M (12 GB) — RTX 2080

| Config | best | median | H2D PCIe |
|---|---|---|---|
| baseline | 3.64 s | 3.65 s | 7.20 GB |
| bitpack | 3.59 s | 3.65 s | **7.20 GB (no change)** |

Random byte values fill all 256 buckets per byte position, so the compact-key + bitpack pipeline finds no compressible structure. **Lesson: the codec wins are about real data structure, not about the algorithm.**

### NYC Taxi 6mo (2.34 GB) — both GPUs

| GPU | Config | best | H2D PCIe | Chunks |
|---|---|---|---|---|
| RTX 2080 | baseline | 715 ms | 1.29 GB | 1 |
| RTX 2080 | bitpack | 705 ms | 1.29 GB | 1 |
| P5000 | baseline | 601 ms | 2.34 GB | 1 |
| P5000 | bitpack | 601 ms | 2.34 GB | 1 |

NYC Taxi key has ~158 compressible bits → 24 B padded (same as TPC-H), but the dataset fits in a single GPU chunk so PCIe isn't the bottleneck. The codec runs invisibly. **Lesson: compression matters when PCIe matters; PCIe matters when the data crosses chunk boundaries.**

## GPU vs CPU crossover (synthetic, P5000, --verify on)

| N (records) | Wall time | Throughput |
|---|---|---|
| 100 K | 139 ms | 0.07 GB/s |
| 1 M | 309 ms | 0.32 GB/s |
| 10 M | 3.5 s | 0.29 GB/s |
| 30 M | 10.4 s | 0.29 GB/s |
| 100 M | (still running) | — |

Sub-linear scaling at small N is dominated by setup and verification cost. GPU sort efficiency only matters at hundreds of millions of records. (Confirms the prior crossover-around-5K-rows finding from earlier work.)

## Fast TPC-H generator validated

`h100/gen_tpch_fast.py` now produces **byte-identical output** to `gpu_crocsort/gen_tpch_normalized.py` after fixing three encoding bugs (date/decimal biases, l_quantity scale).

Performance comparison at SF1:

| Generator | Time |
|---|---|
| gen_tpch_normalized.py (DuckDB dbgen + Python encode) | 123 s |
| **gen_tpch_fast.py (tpchgen-cli + Python encode)** | **13 s** (3 s gen + 10 s encode) |

Speedup is roughly 9× at SF1. At SF100 the encode step (Python loop) becomes the bottleneck and the speedup is closer to 2×, but that's still ~15 minutes saved per dataset.

## Implications for the H100 week

1. **Tier 0 sanity numbers should be very fast:** SF10 in <0.5 s, SF50 in <1 s, SF100 in <2 s. If they're not, something's wrong with the build (probably arch).
2. **Tier 1 envelope (SF300/500/1000) is where bitpack should finally show wall-time wins.** The H100's 80 GB HBM means more data per chunk, but PCIe 5.0 is also 2.6× faster than PCIe 3.0 — so PCIe stays the bottleneck for the very-large workloads.
3. **The merge-arena work is queued at Tier 3.2.** It's the only path to SF1000+ on hardware below 80 GB, but on the H100 itself the existing 22 GB merge fits easily.
4. **gen_tpch_fast.py saves ~15 min per dataset gen on big SFs** — meaningful when generating SF300/500/1000.

All numbers committed to `research/overnight-runs-cs-uchicago` with reproducer scripts in `h100/prep_lincoln.sh` + `h100/prep_roscoe.sh`.
