# More experiment ideas

Extending `overnight_experiments.md` with experiments motivated by `related_accelerators.md`. Grouped by theme.

Date: 2026-04-29

## Theme 1: compression actually extends the sort envelope

This is the most compelling story for the PI talk. The current envelope on each GPU is:
- RTX 6000 (24 GB): SF100
- P5000 (16 GB): SF50
- RTX 2080 (8 GB): SF20

If compression cuts key bytes by 2x, we should be able to push each one scale factor up.

### Experiments
1. **Envelope extension on RTX 2080.** Run SF40 with FOR-compressed 16B compact keys (vs current 32B compact). Question: does it now fit?
2. **Envelope extension on P5000.** Run SF100 with the same compression. Question: does the OVC path now succeed?
3. **PCIe time vs scale, with and without compression.** Plot upload time as a function of SF for both compact-only and compact+FOR. Should be a straight line with FOR shifted down.

## Theme 2: real-world data compressibility

The compact-key scan currently saves 2-8x on TPC-H. What about messier datasets?

### Experiments
4. **NYC Taxi compressibility.** Run a3_codec_ratios on taxi 1mo/6mo/12mo. Report bytes-per-key and codec winners per column.
5. **Synthetic skew sweep.** Generate 100M keys with controlled compressibility (1x, 2x, 4x, 8x, 16x via Zipf distributions). Sort each, plot end-to-end time as a function of compression ratio.
6. **Dictionary-friendly columns.** TPC-H's `l_returnflag` is 4 values, `l_linestatus` is 2 values. Test pure dictionary encoding on these columns alone — should hit 100x ratio.

## Theme 3: where does the GPU actually start winning?

Current crossover is around 5K rows. With compression that shifts because PCIe time falls.

### Experiments
7. **Crossover sweep.** Sort 1K, 10K, 100K, 1M, 10M, 100M random int32 keys. Compare GPU-only vs Polars vs CPU std::sort. Find the new crossover with FOR-compressed upload.
8. **Per-key cost decomposition.** At 1B keys, what fraction of wall time is encode vs upload vs sort vs gather? With FOR, that shifts.

## Theme 4: alternatives to K-way CPU merge

The K-way CPU merge is the part of the pipeline most hostile to GPUs. Let's measure its actual cost and try to remove it.

### Experiments
9. **Profile the merge.** Run `perf stat` on K-way merge for K=2, 4, 8, 16. Report branch mispredict rate, L1/L2/L3 miss rate, cycles per merged row.
10. **GPU sample-sort prototype.** Implement a minimal GPU sample-sort: sample 1024 splitters, partition input on GPU, radix-sort each partition. Compare wall time vs current GPU-sort + CPU-merge.
11. **Throughput as a function of K.** For a fixed dataset, force varying K (chunk size) and plot end-to-end time. Find optimal K.

## Theme 5: hardware envelope by key width

Sort throughput scales with key bytes (radix passes). Quantify this.

### Experiments
12. **Key-width sweep.** Sort 100M records with key sizes: 8B, 16B, 32B, 66B (TPC-H compact), 88B (TPC-H full). Plot wall time vs key bytes.
13. **Compact vs full key TPC-H at all scales.** Already in the queue (compact ON/OFF), but ensure consistent reporting.

## Theme 6: CPU baselines (apples to apples)

The DuckDB and Polars numbers in our slides are from the gtx box. Re-measure on roscoe and lincoln so we can quote a single coherent comparison per machine.

### Experiments
14. **DuckDB on the same data.** Run `SELECT * FROM lineitem ORDER BY ...` against duckdb on roscoe and lincoln. Same TPC-H scale factors.
15. **Polars on the same data.** Same as above but Polars.
16. **CPU std::sort on raw 88B keys.** Lower-bound baseline: just call std::sort on the keys directly, single threaded and parallel.

## Suggested priority for next overnight queue

If you want to pick five for the next run:

1. **Theme 1, exp 1+2** — envelope extension on both small GPUs with FOR. Headline result.
2. **Theme 4, exp 9** — perf stat on K-way merge. Quantifies the cost we want to eliminate.
3. **Theme 2, exp 4** — NYC taxi compressibility. Shows compression works on real data.
4. **Theme 3, exp 7** — crossover sweep with FOR. Updated crossover number for the talk.
5. **Theme 5, exp 12** — key-width sweep. Cleanest "scaling" plot for the paper.

## What needs to be built first

- FOR encoder/decoder (CUDA + Python) — required for theme 1, 3, 5
- DuckDB ORDER BY harness — required for theme 6 (probably 30 min of scripting)
- Sample-sort prototype — required for theme 4 exp 10 (1-2 days of CUDA work)
- Synthetic data generator with controlled compressibility — required for theme 2 exp 5

The FOR codec is the highest-leverage thing to build. Everything in theme 1 and most of theme 3 needs it.
