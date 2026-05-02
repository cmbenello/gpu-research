# H100 research queue

Prioritized experiment list for the H100 week. The autoresearch loop reads this file every iteration, picks the highest-priority unticked experiment, runs it, ticks it off, and writes findings to `results/h100_runs/<id>.md`.

Status legend:
- `[ ]` queued
- `[~]` in progress
- `[x]` done (link to writeup)
- `[!]` failed (link to error)

The loop is allowed to **add new experiments** at the bottom when findings warrant.

**Pacing for the H100 week:** Tiers 0-3 are the must-haves (sanity, scale envelope, compression validation, architecture). Tiers 4-7 are the publication baselines and writeup. Tiers 8-14 are research depth — the loop should reach them only after the must-haves are done. If a Tier 8+ experiment takes more than 2 hours of engineering work to set up, the loop should **mark it `[!]` with reason "out of session scope"** and move on rather than block.

## Tier 0 — Sanity (run first, fast)

- [x] **0.1 baseline_smoke** — 1M synthetic + SF10 PASS after fixing two real bugs (non-compact build, compact fast-path silent FAIL). 352 ms, 20.4 GB/s effective on SF10 compact. → [`results/h100_runs/0.1_baseline_smoke.md`](../results/h100_runs/0.1_baseline_smoke.md)
- [x] **0.2 sf50_baseline** — TPC-H SF50, 5 warm runs each, baseline (12.7 GB/s warm median) + USE_BITPACK (12.8 GB/s). Bitpack ≈ no-op at SF50; the prepack cost cancels the LSD-pass savings. → [`results/h100_runs/0.2_sf50_baseline.md`](../results/h100_runs/0.2_sf50_baseline.md)
- [x] **0.3 sf100_baseline** — TPC-H SF100, 5 warm runs each. Baseline best 4.50 s / 16.0 GB/s; bitpack best 4.78 s. **Slower than RTX 6000 baseline (3.74 s)** because slow-path uploads full 66B keys, not 32B compact keys — root cause to fix before claiming any H100 win. → [`results/h100_runs/0.3_sf100_baseline.md`](../results/h100_runs/0.3_sf100_baseline.md)

## Tier 1 — Scale envelope (push the H100's 80 GB)

- [x] **1.1 generate_sf300** — Generated SF300 lineitem (216.00 GB) in 1447 s (24.1 min). Encode at 0.16 GB/s is the bottleneck, peak RSS 565 GB. → [`results/h100_runs/1.1_generate_sf300.md`](../results/h100_runs/1.1_generate_sf300.md)
- [x] **1.2 sf300_baseline** — SF300 baseline best 10.10 s / 21.4 GB/s warm; bitpack best 8.94 s / 24.2 GB/s warm (+13%). RTX 6000 OOMs at this scale. **First clean H100 win.** OVC architecture engages here, compact upload PCIe = 50.4 GB (5× compression). → [`results/h100_runs/1.2_sf300_baseline.md`](../results/h100_runs/1.2_sf300_baseline.md)
- [x] **1.3 generate_sf500** — Generated 360 GB in 2623 s (43.7 min), peak RSS 686 GB. → [`results/h100_runs/1.3_generate_sf500.md`](../results/h100_runs/1.3_generate_sf500.md). **WARNING:** the same code path projects to peak RSS ~1.2 TB at SF1000 — won't fit on this box; gate 1.5/1.6 on chunked-encoder fix (1.1.1).
- [ ] **1.4 sf500_baseline** — sort SF500, baseline + USE_BITPACK.
- [ ] **1.5 generate_sf1000** — Generate SF1000 (~720 GB), if disk has space.
- [ ] **1.6 sf1000_baseline** — sort SF1000.
- [ ] **1.7 envelope_chart** — plot wall-time and GB/s vs scale on H100.
- [ ] **1.8 generate_sf1500** — Generate SF1500 (~1.08 TB lineitem). 3.3 TB free on /mnt/data, so this fits even with parallel intermediates. Watch df during gen.
- [ ] **1.9 sf1500_baseline** — sort SF1500 single-GPU. Almost certainly out-of-HBM; tests the streaming/host-staging path under pressure.
- [ ] **1.10 generate_sf2000** — Generate SF2000 (~1.44 TB lineitem). Right at ~50% of disk; abort if df < 25% free mid-gen.
- [ ] **1.11 sf2000_baseline** — sort SF2000. The single-GPU upper bound for this box; if it doesn't fit, that's the multi-GPU motivation in print.
- [ ] **1.12 host_ram_staging_sf1000** — exploit the 1 TB host RAM: pre-load SF1000 (~720 GB) entirely into pinned host memory, then sort GPU-staged out of RAM. Compare to NVMe-staged numbers from 1.6. Should isolate "PCIe + sort" from "NVMe read".
- [ ] **1.13 host_ram_staging_sf1500** — same trick at SF1500 (~1.08 TB) — slightly exceeds 1 TB RAM after overhead, so probably falls back to mixed RAM+NVMe. Useful data point.

## Tier 2 — Compression validation at scale

- [ ] **2.1 codec_ratios_sf100** — run a3_codec_ratios on SF100 to confirm ratios are scale-invariant.
- [ ] **2.2 bitpack_pcie_sweep** — measure H→D PCIe bytes baseline vs USE_BITPACK at SF50/100/300/500/1000.
- [ ] **2.3 compounding_chart** — sort time vs key width chart on H100 (HBM3 bandwidth changes the curve).
- [ ] **2.4 dictionary_codec** — implement and measure dictionary encoding for low-cardinality columns. Drop in slot for `pack()` style.

## Tier 3 — Architecture changes

- [ ] **3.1 sample_sort_prototype** — implement minimal GPU sample-sort (sample + partition + per-bucket radix). Compare e2e to current K-way merge on SF100.
- [ ] **3.2 chunked_merge_phase** — chunk the global merge phase so SF1000 fits without 22 GB atomic alloc. Required for ultra-large.
- [ ] **3.3 record_compression** — experiment: bit-pack the satellite (full record) data, not just the key. Cuts gather memory traffic.
- [ ] **3.4 gpu_gather_pcie5** — try GPU-side gather (upload full records, gather on device, download sorted). Should finally win on PCIe 5.

## Tier 4 — Cross-validation against published baselines

- [ ] **4.1 duckdb_at_scale** — DuckDB ORDER BY at SF100, SF300 if it completes. Compare wall-time.
- [ ] **4.2 polars_at_scale** — Polars sort at SF100, SF300.
- [ ] **4.3 cudf_baseline** — cuDF sort_values on the same data. RAPIDS comparison.
- [ ] **4.4 mendsort_position** — verify our GB/s position relative to MendSort (JouleSort 2023).
- [ ] **4.5 gensort_100gb** — official GenSort 100 GB benchmark, head-to-head with published numbers.

## Tier 5 — Real-world data

- [ ] **5.1 nyctaxi_12mo** — full year of NYC Yellow Taxi data, sort + DuckDB baseline.
- [ ] **5.2 btc_blockchain** — Bitcoin transaction sort by block,index. (Subject to data availability.)
- [ ] **5.3 wide_records** — synthesise 256 B records and sort. Tests gather more aggressively.

## Tier 6 — Deep profiling

- [ ] **6.1 nsys_trace_sf100** — Nsight Systems trace of the SF100 sort, save to `results/h100_runs/profiles/`.
- [ ] **6.2 hbm_bandwidth_saturation** — synthetic test that hits HBM3 ceiling. Sets the upper bound on what sort throughput can ever be on H100.
- [ ] **6.3 phase_breakdown_sweep** — phase-level breakdown (encode/upload/sort/merge/gather) for SF50/100/300/500/1000. Plot how the bottleneck moves with scale.

## Tier 7 — Paper / report material

- [ ] **7.1 final_charts** — regenerate the PI-progress chart pack with H100 numbers.
- [ ] **7.2 weekly_writeup** — `results/weekly_progress_h100.md` with all findings, formatted for the PI.
- [ ] **7.3 paper_intro_outline** — outline of the VLDB / DaMoN submission based on the H100 numbers.

## Tier 8 — Codec breadth (when does which compressor win?)

- [ ] **8.1 codec_matrix_tpch** — for SF50 + SF100, measure per-column ratio for every codec we support: FOR, FOR+bitpack, dictionary (full alphabet), dictionary (top-256 + escape), RLE, plain. Save as a heatmap. Reveals which codec wins per column type.
- [ ] **8.2 dict_encoding_path** — implement and integrate a CPU-side dictionary encoder for the low-cardinality columns (l_returnflag, l_linestatus). Compare end-to-end vs FOR+bitpack alone.
- [ ] **8.3 rle_for_sorted_prefix** — when input is partially pre-sorted, run-length-encode the long-run prefix bytes. Measure on `time-series-sorted` data.
- [ ] **8.4 codec_per_column_choice** — automatic per-column codec selection based on the entropy scan (Tier 8.1 results). Pick the best codec per column, then pack.
- [ ] **8.5 hope_style_codec** — try a HOPE-style (Zhang SIGMOD 2020) order-preserving dictionary on the string columns. Compare to fixed-width padded UTF-8.

## Tier 9 — Workload diversity (the data itself, not the codec)

- [ ] **9.1 skewed_zipf** — synthesise 100M records where one sort key follows Zipf(α=1.5). Sort + measure. Tests how compact-key + bitpack handle high-skew distributions.
- [ ] **9.2 already_sorted** — run the sort on input that's already in sorted order. Tests the best case (radix sort still does N work but PCIe is just a copy).
- [ ] **9.3 reverse_sorted** — input in reverse order. Worst case for naive insertion sort, no different for radix.
- [ ] **9.4 nearly_sorted** — 99% sorted with 1% random shuffle. Common real-world pattern.
- [ ] **9.5 wide_records** — synthesise 256 B records (vs TPC-H 120 B). Tests gather phase scaling with record width.
- [ ] **9.6 string_heavy** — 100M records where the sort key is a 64 B variable-length string. Tests the string-padding path and exposes if string sort hits a different bottleneck.

## Tier 10 — Cross-library baselines

- [ ] **10.1 cudf_sort_values** — RAPIDS cuDF `sort_values()` on the same data. Direct GPU-to-GPU comparison. If cuDF wins, we have a problem; if we win, that's the paper.
- [ ] **10.2 thrust_sort** — `thrust::sort` direct call (the "naive" GPU sort). Establishes the ceiling for what an off-the-shelf algorithm gets.
- [ ] **10.3 cub_devicesort_keys_only** — straight CUB `DeviceRadixSort::SortKeys` on byte-comparable keys. Removes any framework overhead. The pure-codec story comes through here.
- [ ] **10.4 nccl_singlegpu_sanity** — NCCL allreduce sanity (sets up the multi-GPU baseline if H100 is dual).

## Tier 11 — Adversarial / robustness

- [ ] **11.1 with_nulls** — TPC-H with synthetic NULL values in numeric columns. Verify our encoding handles them.
- [ ] **11.2 stability_test** — sort with intentional duplicate keys (e.g. take SF10 and replicate keys 10×). Measure if the sort is stable (preserves original record order for equal keys).
- [ ] **11.3 mixed_asc_desc** — composite key with some columns ascending, some descending. Tests the encode path that requires bit-flipping for descending order.
- [ ] **11.4 oom_recovery** — intentionally request a sort that won't fit. Verify the OOM path is graceful (not a crash, not a kernel hang).
- [ ] **11.5 power_throttle_check** — sort SF100 in a tight loop for 30 minutes. Measure GPU clock + temperature drift. If throttling is real, document it.

## Tier 12 — System integration

- [ ] **12.1 postgres_create_index** — write a Postgres `CREATE INDEX` extension that calls `gpu_crocsort` for `tuplesort_performsort()`. Measure CREATE INDEX wall-time on SF50 vs Postgres native.
- [ ] **12.2 sqlite_vacuum_index** — same but for SQLite.
- [ ] **12.3 duckdb_external_operator** — retry the DuckDB custom operator now that we have compression: maybe the smaller key compensates for the column→row serialization overhead this time.
- [ ] **12.4 sort_merge_join** — implement sort-merge join on TPC-H using our sort. Compare to DuckDB's hash-join for queries Q3, Q9.
- [ ] **12.5 spark_sort** — if PySpark is installable, sort SF50 in Spark single-node. Mostly to have a coordinator-style baseline.

## Tier 13 — Algorithm extensions

- [ ] **13.1 top_k_selection** — implement top-K (where K = 1M out of 100M). Don't need full sort. Compare to current sort + truncate.
- [ ] **13.2 sample_sort_with_learned_splitters** — instead of random sample for splitters, learn them from a tiny model. Reduces sample variance.
- [ ] **13.3 sort_aggregate_fusion** — sort + group-by aggregate as a single pass. Skips materializing the full sorted output.
- [ ] **13.4 streaming_sort** — process input larger than disk by streaming runs through GPU + writing back to NVMe.

## Tier 14 — Energy / efficiency

- [ ] **14.1 joule_per_sorted_gb** — measure power draw during SF100 sort with `nvidia-smi --query-gpu=power.draw`. Compute J/GB. Compare to MendSort's published numbers.
- [ ] **14.2 hbm3_vs_pcie5_balance** — at what record count does HBM3 bandwidth, not PCIe5, become the bottleneck on H100?
- [ ] **14.3 compression_pareto** — pareto curve of compression ratio vs encode CPU cost. Helps choose the right codec for a given workload.

## Tier 15 — Multi-GPU + scale-out

This box: **4× H100 NVL (94 GB each), all-to-all NV6 NVLink** (verified via `nvidia-smi topo -m` 2026-05-02).
GPU0/1 share NUMA node 0 (CPU 0-10 even); GPU2/3 share NUMA node 1.
Total aggregate HBM: 376 GB — SF500 (~360 GB) fits entirely in aggregate HBM; SF1000 needs streaming.

- [x] **15.1 detect_multigpu** — confirmed: 4 GPUs, all NV6 (6-link NVLink) between every pair → see results/h100_runs/15.1_topology_2026-05-02.md (recorded inline in this commit).
- [ ] **15.2 nvlink_bandwidth** — build CUDA samples' `p2pBandwidthLatencyTest` (or roll a small p2p memcpy bench). Measure pair-wise NVLink b/w. Expected: ~300 GB/s per direction with NV6 (50 GB/s × 6).
- [ ] **15.3 partition_then_sort_2gpu** — partition input by sample splitters, send half to GPU0 / half to GPU1, sort independently, merge. SF100 across 2 GPUs (same-NUMA pair).
- [ ] **15.4 partition_then_sort_4gpu** — same as 15.3 but 4-way across all 4 GPUs. SF100 + SF300 + SF500. SF500 case is the headline because it fits entirely in 4×94 GB aggregate HBM.
- [ ] **15.5 dataparallel_meta_merge_sf1000** — shard SF1000 into 4 chunks (~SF250 each), sort each chunk on its own GPU concurrently, then meta-merge sorted runs on host (or via NVLink-streamed merger). The "effective SF1000 with 4× peak HBM" plan.
- [ ] **15.6 nccl_all_to_all_sort** — proper distributed sort: each GPU starts with 1/4 of input, NCCL all-to-all to redistribute by global splitter, sort locally. SF300 + SF500.
- [ ] **15.7 multigpu_scaling_curve** — sort throughput vs GPU count for 1, 2, 4 GPUs (SF100 + SF300 + SF500). Plot scaling efficiency. Strong-scaling at fixed problem; weak-scaling at fixed per-GPU problem.
- [ ] **15.8 numa_aware_pair** — compare 2-GPU runs that pair across NUMA (GPU0+GPU2) vs same-NUMA (GPU0+GPU1). Quantifies host-side NUMA cost when GPUs talk through host.
- [ ] **15.9 sf2000_4gpu** — push the 4-GPU + host-RAM-staging combo to SF2000 (~1.44 TB). Combines 1.10 + 15.5 — likely the largest single-machine sort number we'll have on this box.

## Tier 16 — Sort in real systems (the "why anyone cares" experiments)

- [ ] **16.1 sort_groupby_aggregate** — implement `SELECT col, SUM(x) FROM t GROUP BY col ORDER BY col` end-to-end with our sort. Measure wall time vs DuckDB's hash aggregate + sort.
- [ ] **16.2 sort_window_lag** — `LAG(price) OVER (PARTITION BY ticker ORDER BY ts)` requires a sorted partition. Time-series workload.
- [ ] **16.3 sort_percentile** — sort + index for arbitrary percentile lookup. Useful for monitoring/alerting use cases.
- [ ] **16.4 sort_dedup** — sort + adjacent-equal detection for distinct count. Compare to hash-based DISTINCT.
- [ ] **16.5 sort_for_compression** — sorting data first then compressing it (Parquet, ORC). Measure compression ratio improvement post-sort. Quantifies sort-as-preprocessing value.
- [ ] **16.6 join_throughput** — sort-merge join Q3 + Q9 of TPC-H end-to-end. Compare to DuckDB hash-join.

## Tier 17 — Theoretical / roofline model

- [ ] **17.1 roofline_h100** — given (HBM3 = 3.35 TB/s, PCIe5 = 32 GB/s, FP32 = 67 TFLOPS), compute the roofline ceiling for radix sort. Compare measured throughput.
- [ ] **17.2 fit_scaling_law** — fit `time = α·N·log(N) + β·N` (encode + sort) and `time = γ·bytes_pcie / 12` (PCIe-bound) to measured numbers across SF10–SF1000. Predicts SF10000.
- [ ] **17.3 codec_compression_model** — given (column entropy, range, n_distinct), predict which codec wins. Closed-form vs measured Tier 8.1 heatmap.
- [ ] **17.4 amdahl_for_compression** — Amdahl-style: if PCIe is X% of wall time, max speedup from compressing PCIe to 0 is 1/(1-X). Validates how much compression CAN ever help.

## Tier 18 — Numeric edge cases + correctness

- [ ] **18.1 nan_handling** — synthetic float dataset with NaN values. Verify they sort to a consistent position (top or bottom), don't crash.
- [ ] **18.2 negative_zero** — float dataset with -0.0 and +0.0 mixed. Verify they're treated as equal under our encoding (or document if they're not).
- [ ] **18.3 infinity_handling** — float dataset with +∞, -∞. Verify byte encoding maintains order.
- [ ] **18.4 unicode_strings** — non-ASCII strings (UTF-8 multi-byte). Verify the byte-comparable encoding still gives correct lex order.
- [ ] **18.5 date_overflow** — dates around year 2038 (uint32 epoch overflow), year 9999 (TPC-H date max). Confirm encoding doesn't wrap.
- [ ] **18.6 huge_keys** — key size larger than typical: 256 B, 1024 B keys. Tests the radix-pass-count scaling.

## Tier 19 — Reproducibility + artifact

- [ ] **19.1 docker_image** — Dockerfile that builds the entire stack. `docker run sortbench gpu_crocsort --input ...` should Just Work.
- [ ] **19.2 paper_repro_script** — single shell script that regenerates every figure in the paper with one command.
- [ ] **19.3 dataset_download** — `download_datasets.sh` that fetches NYC Taxi, BTC blockchain, etc. with checksums.
- [ ] **19.4 cite_check** — list of citations the paper needs (Merrill+Grimshaw, Stehle+Jacobsen, Nicholson DaMoN, Zhang HOPE, Liu MOPD, Boncz FSST, ...). Verified against actual experiments.
- [ ] **19.5 hardware_inventory** — script that records exact GPU model, driver, CUDA version, PCIe gen/lanes, CPU, RAM, OS, kernel. One-shot machine fingerprint for reproducibility.

## Tier 20 — Stretch / "if everything else is done"

- [ ] **20.1 learned_codec** — fit a tiny per-column NN to learn an order-preserving compression. Probably worse than FOR but interesting.
- [ ] **20.2 sort_as_service** — wrap as a gRPC service. Measure tail latency (p50/p99/p999) over a workload of small + large sorts.
- [ ] **20.3 streaming_top_k** — given an unbounded stream, maintain the top K elements in real time using GPU.
- [ ] **20.4 sort_explanation** — for a given sort result, output a "compression report" explaining what each codec did and why. Aids debugging + paper figures.

---

## Hardware profile (2026-05-02, sorting-h100)

- **GPUs:** 4× NVIDIA H100 NVL, 95830 MiB HBM each (aggregate 376 GB), driver 550.163.01, CUDA 12.4
- **Interconnect:** NV6 between every GPU pair (full all-to-all NVLink). GPU0+GPU1 share NUMA node 0; GPU2+GPU3 share NUMA node 1.
- **CPU:** 192 cores
- **Host RAM:** 1024 GiB (≈1014 GiB MemAvailable at boot)
- **Boot disk:** 419 GB at `/` (do NOT generate data here)
- **Data disk:** 3.5 TB ext4 at `/mnt/data` (3.3 TB free at start). All gen + intermediate writes go here.
- **CUDA arch:** sm_90 (Hopper)

## Running notes (loop appends to this section)

### 2026-05-02 — 0.1 surfaced two bugs

- **Build:** `external-sort-tpch` and `external-sort` had a 2-error compile failure (COMPACT_KEY_SIZE / runtime_compact_size unconditionally referenced). Fixed in 1944409.
- **Correctness:** `external_sort_tpch_compact` had a silent FAIL on the data-fits-in-GPU fast path (NULL cmap → sorted garbage). Fixed in dc42168.
- **Implication:** any prior SF10/SF20/SF30 compact numbers in `results/` that went through the fast path are suspect. Need to re-baseline.

### Proposed follow-up items (from 0.1)

- **0.1.1 verify_in_bootstrap** — add `--runs 1` to setup.sh's compact smoke step so future bootstraps catch silent verifier FAILs.
- **0.1.2 fastpath_slowpath_boundary** — pick a size right at `buf_records` (~179 M on H100) and run `--runs 3` to confirm fast and slow paths produce identical output.
- **0.1.3 rebaseline_pre_dc42168** — re-run any prior compact-path SF10/SF20/SF30 numbers; results before dc42168 used incorrect compact keys.

