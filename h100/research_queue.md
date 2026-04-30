# H100 research queue

Prioritized experiment list for the H100 week. The autoresearch loop reads this file every iteration, picks the highest-priority unticked experiment, runs it, ticks it off, and writes findings to `results/h100_runs/<id>.md`.

Status legend:
- `[ ]` queued
- `[~]` in progress
- `[x]` done (link to writeup)
- `[!]` failed (link to error)

The loop is allowed to **add new experiments** at the bottom when findings warrant.

## Tier 0 — Sanity (run first, fast)

- [ ] **0.1 baseline_smoke** — 1M synthetic + SF10 with verify (already in setup, just record numbers).
- [ ] **0.2 sf50_baseline** — TPC-H SF50, 5 warm runs, baseline + USE_BITPACK. Confirms the patch works on H100.
- [ ] **0.3 sf100_baseline** — TPC-H SF100, 5 warm runs, baseline + USE_BITPACK. The headline number — should beat the RTX 6000's 3.74 s by a wide margin.

## Tier 1 — Scale envelope (push the H100's 80 GB)

- [ ] **1.1 generate_sf300** — Generate SF300 lineitem (~216 GB) on local NVMe.
- [ ] **1.2 sf300_baseline** — sort SF300, baseline + USE_BITPACK, 3 runs each.
- [ ] **1.3 generate_sf500** — Generate SF500 (~360 GB) if disk allows.
- [ ] **1.4 sf500_baseline** — sort SF500, baseline + USE_BITPACK.
- [ ] **1.5 generate_sf1000** — Generate SF1000 (~720 GB), if disk has space.
- [ ] **1.6 sf1000_baseline** — sort SF1000.
- [ ] **1.7 envelope_chart** — plot wall-time and GB/s vs scale on H100.

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

---

## Running notes (loop appends to this section)
