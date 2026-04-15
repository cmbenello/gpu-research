# Experiment Plan for the Paper

Date: 2026-04-15
Context: SF50 6.21 s / SF100 7.82 s on RTX 6000 + 48C DDR4. All core optimizations are in. Remaining wall-clock gains on this hardware are <200 ms and below noise. A paper needs a clear story, calibrated comparisons, and ablation evidence, not tighter micro-opts.

## What the paper should claim

A hybrid GPU-CPU external sort for large-record tabular data (TPC-H lineitem, 120 B records) that:
1. **Compacts keys on the upload path** — PCIe bytes drop 4× vs full records.
2. **Sorts prefix on GPU** — 16 B default, adaptive 32 B when memory permits.
3. **Resolves ties on CPU via parallel group-detect + work-queue dispatch + cache-resident per-group sort on an 8 B condensed key** (this is the contribution that converted SF50 from slower than SF100 to faster).
4. **Is correctness-by-construction**: position-ordered compact map + multiset hash verifier + sortedness check on every run.

Secondary claims:
- Single-GPU+single-socket system throughput comparable to or better than established distributed engines (DuckDB, ClickHouse) on tabular-sort benchmarks.
- A portability model that predicts performance on H100 / A100 / GH200 using three bandwidth parameters.

## The experiments I would run

Grouped by what they contribute to the paper.

### A. Scaling curve (core result table)

Goal: "how does the sort scale with data size" — this is table 1 in any sort paper.

- **SF1 (0.72 GB), SF10 (7.2 GB), SF30 (21.6 GB), SF50 (36 GB), SF100 (72 GB)**, 6 warm runs each, median + min + max reported.
- Fit power law on (size → ms). Expected slope ≈ 1.0 (linear in bytes) because we're BW-bound, not compute-bound.
- Deliverable: `results/paper/scaling_curve.tsv` + a reproducible `scripts/scaling_curve.sh`.

Cost: ~2 h wall. Most of it is SF100 × 6 warm runs.

### B. Comparison against baselines

Goal: "who are we faster than and by how much?" — this is figure 2.

Each baseline run on the same machine, same input file, same verification:
- **DuckDB** — `COPY FROM … ORDER BY` or `SELECT … ORDER BY … LIMIT`. Already noted at 4.7×/4.4×/~26× slower than us in memory.
- **ClickHouse (embedded)** — `ORDER BY` on a MergeTree table.
- **Apache Spark (local mode)** — `df.orderBy`. Expected much slower; included for completeness.
- **pandas / polars** — for SF1-10 only (they OOM at SF50). Polars is the interesting one — their sort is GPU-less but fast.
- **sort -k / GNU coreutils** — textbook CPU external sort, for the "naive baseline" column.
- **cudf's sort** — GPU-only; the reason we invented our hybrid is because cudf runs out of memory at SF50.

For each, report warm median total wall-clock. Bonus: break their time into load / sort / output phases if their telemetry exposes it.

Cost: ~1 day to set up and run all. DuckDB and Polars are ~30 min. ClickHouse setup is the main time sink.

### C. Ablation study

Goal: "which of your optimizations actually mattered?" — this is figure 3.

Run SF50 six warm runs with each optimization disabled individually:
1. Baseline (full key upload, no compact) — expected ~30 s (old).
2. + compact key upload (no 16 B prefix merge) — not directly runnable since the prefix merge is required after compact upload; skip.
3. + 16 B prefix merge (no 32 B extension path, no fixup optimizations, single-threaded group-detect) — the "ab92985" baseline ~12.4 s.
4. + parallel group-detect (serial sort) — expected ~9 s based on 74c5749.
5. + parallel group-detect + work-queue + pre-allocated scratch — 6.56 s (our `fixup-parallel-group-detect` point).
6. + 8 B uint64 key + contiguous fast path — 6.21 s (current committed).
7. + 32 B adaptive path (auto-enabled on SF50) — same 6.21 s, but adversarial-robustness improves. Discuss in text.

These are mostly toggleable via env vars that already exist (`FIXUP_THREADS`, `OVC_32B`) or by checking out specific commits. Write a driver script that iterates commits and runs them.

Cost: ~3 h, mostly SF50 × 6 × 6 configs.

### D. Correctness robustness — the adversarial story

Goal: "your algorithm is sound even on weird inputs" — this is the correctness section (§4).

Datasets to build:
1. **TPC-H SF50 default** — already in tree.
2. **TPC-H with adversarial re-shuffle** — scramble the byte layout of lineitem to place all varying bytes at positions ≥ 32. Goal: demonstrate 16 B prefix gives ~100 % ties, 32 B still doesn't resolve, so fixup handles the whole dataset in one group. This is the worst case I hypothesized in `ADVERSARIAL_ANALYSIS.md`.
3. **All-same key** (every record has key = 0). Fixup must sort the whole record buffer by tail bytes (none exist in our setup, but active_bytes would be empty).
4. **Random uniform keys** (GenSort-style 10-byte keys). We already have numbers (3.87–4.30 s for 30 GB, see LOG.tsv lines 10-12). Include the distribution.
5. **Skewed keys** (Zipfian). Generate a synthetic 36 GB dataset with 80 % of records sharing ~10 unique prefixes. Tests group-size distribution.

For each, run 6 warm iterations, include multiset hash + sortedness check, and report median ms.

Cost: adversarial dataset gen ~1 h, running ~3 h.

### E. Architectural portability — one hardware swap if budget permits

Goal: "your cost model is predictive, not post-hoc" — this is §6.

The ARCHITECTURE.md predictions:
- H100 + PCIe 5 + 96C Zen 5: SF50 ~1.56 s (from 6.21).
- GH200: SF50 ~1.2 s.

If we have cloud access to even one of these, we should run the exact same binary (recompile with appropriate sm_XX). The value is in the comparison: "predicted X, measured Y, delta is …". Even getting one data point makes the portability claim real.

Fallback if hardware unavailable: run on a second, different machine already in the lab — e.g. if there's a 64-core DDR5 box or an A100 system. Any point other than the RTX 6000 data improves the curve.

Cost: ~4 h of cloud GPU time if we go H100, or 1 day setup on a local second box.

### F. Micro-profiling for §5 (implementation details)

Goal: "where does the time go at each scale?" — this is table 3 / figure 4.

For SF10, SF30, SF50, SF100 emit a phase breakdown CSV:
- run_gen_ms, merge_ms, d2h_ms, gather_ms, group_detect_ms, fixup_parallel_ms, pack_ns_per_record, sort_ns_per_record, reorder_ns_per_record.

We already print most of these to stderr; just need to parse and collect into a single table. Write `scripts/collect_phase_breakdowns.py`.

Cost: 30 min to write the script, all runs already done for A.

### G. GPU-side boundary detection (finish-what-I-started)

Goal: "on architectures where the bandwidth asymmetry flips, the optimization that was net-neutral becomes a clear win" — this is a late-paper demonstration of the portability model.

Ship the GPU boundary code (reverted today) on a separate branch `exp/gpu-boundary`. Keep disabled by default on RTX 6000, enable via `GPU_BOUNDARY=1`. On an H100 rerun, show that with PCIe 5 the net win appears and becomes an ablation point E-proof.

Cost: 2 h to re-apply, 1 h to test + wrap in env flag.

## Proposed order

The paper needs A + B + C to even have a story. D is the correctness section. E is the best single thing we could add to make the paper land.

Sequence if I had a week:
- Day 1: A (scaling curve) + C (ablation) — these share infrastructure.
- Day 2: B (competitor setup + runs) — the setup is the long pole.
- Day 3: D (adversarial dataset gen + runs).
- Day 4: E if hardware available, else F (deep profile) and G.
- Day 5: paper writing, figure generation.

## What I would NOT do more of

1. **Micro-optimizing RTX 6000 further.** Any ~100 ms win is under-the-noise and not publishable.
2. **Repeatedly re-running the same SF50.** Median after 6 runs is the answer; more runs don't help without a methodology change (e.g. drop page cache, throttle off).
3. **New algorithm variants without a correctness story.** The entropy-reorder regret from Apr-15 AM is a warning: any change that "might break sort" costs a week of retraction. Always keep multiset + sortedness checks on.

## Deliverables (for the user to decide on)

If you want me to proceed without further input, I'd start at **A + C** — they share infrastructure, build the baseline table the paper needs, and take ~3 h combined. Deliverable is `results/paper/scaling_and_ablation.md` + TSV + reproducible scripts.

Alternatively, **D (adversarial datasets)** is the most scientifically interesting single experiment because it tests a hypothesis we articulated but haven't measured — would take about half a day.

Or **B (baselines)** if you want the comparison story locked in first — half a day to a full day.

Pick one and I'll build the infrastructure and run it.
