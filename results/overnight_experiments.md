# Overnight experiment plan

Experiments to run unattended, motivated by the related work survey in `related_accelerators.md`. Each one is scoped to be runnable end-to-end without intervention and produces a concrete number you can drop into the next progress report.

Date: 2026-04-21

## Setup checklist (run once before kicking off the overnight queue)

- Confirm gtx machine is free for the night, no other GPU jobs queued
- TPC-H SF10, SF50, SF100 already generated under `data/tpch/`
- GenSort 60GB already generated
- Build `gpu_crocsort` with `-O3 -DNDEBUG` and verify it passes `test_gpu_sort.py`
- Free up at least 100 GB scratch on the data drive
- Pin clocks: `nvidia-smi -pm 1; nvidia-smi --lock-gpu-clocks=1395`

## Group A: compressibility profiling (cheap, run first)

These do not modify the sort path, just measure what is in the data. Run first because they tell you which downstream experiments are worth the time.

### A1. Per-column byte entropy
For every TPC-H lineitem sort key column at SF10/50/100:
- Compute Shannon entropy per byte position
- Plot byte-position histogram
- Report bytes-per-key after a perfect order-preserving encoder
- Output: `results/overnight/a1_byte_entropy.csv`

Expected runtime: 10 minutes.

### A2. Compact-key scan effectiveness baseline
For each TPC-H scale factor, log:
- Bytes per key before compact-key scan
- Bytes per key after compact-key scan
- Ratio
- Time spent in scan

This gives the baseline that any compression must beat.

Expected runtime: 5 minutes.

### A3. Codec compression ratios (no GPU work)
On the same key columns, measure compression ratio (no decode) for:
- FOR (frame of reference)
- FOR + bit-packing
- Dictionary encoding (full alphabet)
- Mostly-order-preserving dictionary (top-N then escape)
- HOPE (if there is a usable reference impl, otherwise skip)

Output: `results/overnight/a3_codec_ratios.csv`. Columns: codec, key_column, raw_bytes, compressed_bytes, ratio.

Expected runtime: 30 minutes.

## Group B: GPU codec microbenchmarks

Implement minimal CUDA kernels for the most promising codecs from A3. Measure decode and direct-sort throughput in isolation.

### B1. FOR decode throughput on GPU
- Implement `for_decode_kernel` in CUDA
- Measure GB/s of decoded keys at sizes 100M, 300M, 1B
- Compare to PCIe upload bandwidth (12 GB/s on test rig)

Output: `results/overnight/b1_for_decode.csv`.

Expected runtime: 30 minutes.

### B2. Bit-pack decode throughput on GPU
Same as B1 for bit-packed integer streams (8/16/24/32-bit packed widths).

Output: `results/overnight/b2_bitpack_decode.csv`.

Expected runtime: 30 minutes.

### B3. Direct-sort on FOR-encoded keys
Verify radix sort works without decoding when keys are FOR-encoded (after subtracting min, the codes are byte-comparable).
- Run CUB radix sort on raw int32 keys vs FOR-encoded int24/int16/int8 keys
- Compare wall time and correctness

This is the experiment that proves operating on compressed data is viable.

Output: `results/overnight/b3_direct_sort.csv`.

Expected runtime: 45 minutes.

## Group C: end-to-end sort with compression

These rerun the existing TPC-H benchmarks with compression added at the compact-key-scan stage.

### C1. TPC-H ORDER BY at SF10/50/100 with FOR keys
- Modify compact-key-scan to emit FOR-encoded varying bytes
- Sort directly on the encoded bytes
- Compare end-to-end wall time vs current CrocSort baseline
- Verify results match (diff against baseline output)

Output: `results/overnight/c1_tpch_for.csv`. Columns: scale, baseline_s, with_for_s, speedup, pcie_bytes_baseline, pcie_bytes_with_for.

Expected runtime: 90 minutes (3 scales x 2 configs x 5 runs).

### C2. Compressibility-vs-speedup sweep on synthetic data
- Generate 100M-row synthetic int32 datasets with controlled compressibility (1x, 2x, 4x, 8x, 16x via low-cardinality columns)
- Run CrocSort with and without FOR for each
- Plot speedup as a function of compression ratio

This tells you the crossover where compression starts being worth its overhead.

Output: `results/overnight/c2_synthetic_sweep.csv`.

Expected runtime: 45 minutes.

## Group D: K-way merge replacement

Standalone evaluation of the merge phase without changing the rest of the pipeline yet.

### D1. K-way merge profiling
Instrument the existing CPU K-way merge with `perf stat`:
- Branch misprediction rate
- L1/L2/L3 miss rate
- Cycles per merged row
- Compare K=2, 4, 8, 16

This quantifies the cost we are trying to eliminate.

Output: `results/overnight/d1_merge_profile.csv`.

Expected runtime: 30 minutes.

### D2. GPU sample-sort proof of concept
- Implement a minimal GPU sample-sort: random sample 1024 elements, sort them on GPU, use as splitters, partition full input on GPU, radix-sort each partition independently
- Run on TPC-H SF50 keys (fits in 24 GB GPU memory)
- Compare wall time vs current GPU-sort + CPU-merge pipeline

This is the headline experiment for the next progress update. If it works, the K-way merge phase goes away entirely.

Output: `results/overnight/d2_sample_sort.csv`.

Expected runtime: 2 hours (most of this is engineering, not running, so probably do this manually rather than overnight).

## Group E: out-of-core regime

Test compression where it should help most: when PCIe is the bottleneck.

### E1. TPC-H SF100 with FOR + compact-key-scan
Full external-sort pipeline with compression at every stage that touches PCIe:
- Compact-key-scan (already there)
- FOR encoding on the varying bytes
- Sort on encoded keys
- Decode only when materializing final output

Compare against current SF100 baseline (8.02s).

Output: `results/overnight/e1_sf100_compressed.csv`.

Expected runtime: 90 minutes (multiple runs for variance).

### E2. Chunk-size sweep with compression
With compression enabled, the optimal chunk size changes (smaller compressed keys mean more keys per GPU memory budget).
- Run SF100 with chunk sizes: 50M, 100M, 200M, 400M rows
- Plot wall time vs chunk size
- Pick optimal for both baseline and compressed configs

Output: `results/overnight/e2_chunk_sweep.csv`.

Expected runtime: 2 hours.

## Suggested overnight queue (in order)

1. A1, A2, A3 (run sequentially, all CPU-side, ~45 min total)
2. B1, B2 (microbenchmarks, ~1h total)
3. B3 (direct sort on FOR, ~45 min)
4. C1 (TPC-H with FOR, 90 min)
5. C2 (synthetic sweep, 45 min)
6. E1 (SF100 compressed, 90 min)
7. E2 (chunk sweep, 2h)
8. D1 (merge profiling, 30 min)

Total: about 8 hours. Fits in a single overnight run.

D2 (sample sort) is engineering-heavy and should be done interactively, not in the overnight queue.

## What to look at in the morning

Three numbers that decide the next two weeks of work:

1. **A3 compression ratio.** If FOR + bit-pack gets you below 4 bytes per key on TPC-H, the rest of this is worth doing. If not, pivot to dictionary or HOPE.
2. **C1 SF50 speedup.** If the FOR variant beats current CrocSort by more than 1.3x at SF50, the compression direction is real.
3. **D1 merge cost.** If branch mispredict cost in the merge phase is more than 30% of merge cycles, sample sort is the obvious next move and you can justify the engineering time for D2.

If any of these come out negative, the related_work directions to fall back on are:
- Mostly-order-preserving dictionary (Liu 2019) for higher ratios at the cost of giving up strict order
- FSST for string-heavy keys
- The FPGA sorter pattern (IEICE 2017) of packing successive blocks rather than per-key encoding
