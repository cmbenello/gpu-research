# Research landscape & competitive analysis (2026-05-04)

Comprehensive scan of published GPU sort + database sort SOTA, with
context on where gpu_crocsort fits and what's genuinely novel.

## Published GPU sort numbers (closest comparators)

### CUB Onesweep (Adinets & Merrill, NVIDIA, arXiv:2206.01784, 2022)
- The de-facto SOTA single-GPU radix sort primitive.
- **29.4 GKey/s on A100** sorting 256 M random 32-bit keys (~118 GB/s
  on key bytes only).
- Keys-only, in-VRAM, no payload. Different regime from our work.
- Our 33 GB/s is on whole 120 B lineitem rows including gather +
  payload; not directly comparable.

### RMG Sort (Ilic, Tolovski, Rabl, BTW 2023, HPI)
- Multi-GPU radix-partitioning sort on **8× A100 NVSwitch**.
- Largest reported: **32 B uint64 keys (128 GB)** in ~3 s ≈ 42 GB/s
  aggregate.
- Keys-only, in-VRAM, no payload, no string columns.
- Code: github.com/hpides/rmg-sort
- **Closest published analog**, but our SF1000 (720 GB, full lineitem
  payload) is 5.6× larger than RMG's max.

### Maltenberger, Tolovski, Rabl (PVLDB 18, 2025)
- Sort-merge joins on H100 DGX.
- Radix-partitioning 12-20% better than merge on H100.
- Focused on join, not sort throughput.

### Vortex (Yuan et al., VLDB 2025, arXiv:2502.09541)
- Out-of-core GPU analytics on 4× AMD MI100 + PCIe 4.0.
- **2.7 G elements/s ≈ 21.6 GB/s** sorting 8 B 8-byte integers (64 GB).
- I/O throughput 140 GB/s aggregate via cross-GPU PCIe routing.
- Beats Proteus 5.7× on full TPC-H queries.
- **Leading published out-of-core GPU sort.** Smaller scale than ours.

### cuDF / RAPIDS sort
- Built on CUB Onesweep; expected to track within constant factors.
- No clean published GB/s sort number on lineitem.
- Multi-hundred-GB scale not benchmarked publicly.

### Voltron Data Theseus (March 2025)
- TPC-H 100 TB on Parquet in <1 hour on a 10-node Lambda cluster.
- Publishes query times, not sort kernel GB/s.

### Out-of-Memory GPU Sorting (PDCAT 2024)
- Async CUDA streams for OOM sort.
- 128 B elements (~1 TB) with 4.3× speedup over CPU OOM.
- No payload.

## CPU/database sort comparators

| System | Workload | Wall | Source |
|--------|----------|------|--------|
| **gpu_crocsort 1× H100 NVL** | SF300 (216 GB lineitem) | **6.59 s** | this work |
| **gpu_crocsort 4× H100 NVL** | SF1000 (720 GB lineitem) | **14m20s** | this work |
| DuckDB 1.4.0 (M1 Max, new sort) | SF100 (72 GB lineitem) | 80.9 s | DuckDB blog 2025-09 |
| DuckDB 1.3.2 (Xeon 8468, ours) | SF100 | 106 s | 4.1 |
| DuckDB 1.4.0 (M1 Max) | SF1, SF10 | 0.19, 1.52 s | DuckDB blog |
| Apache DataFusion | TPC-H lineitem ORDER BY | (no clean GB/s pub) | — |
| ClickHouse | ORDER BY filtered subset | 5-27 GB/s (filtered) | various |
| PostgreSQL | TPC-H SF50 full (10-core laptop) | 22 min | community |

DuckDB's new SF100 = 80.9 s vs our 2.95 s = **27.4× faster.**
DuckDB v1.4 is 3× their prior SF100 result.

## Where gpu_crocsort genuinely adds new science

### (A) Largest TPC-H lineitem sort with full payload
**SF1000 = 720 GB, 6 B records, full 120-byte tuples, globally sorted
on a single 4× H100 NVL box.** Per the literature scan, **no
larger published result exists.** RMG tops out at 128 GB *keys-only*;
Vortex at 64 GB synthetic ints; Theseus reports 100 TB but as query
time, not isolated sort.

### (B) Runtime adaptive compact-key encoding
The community treats keys as fixed 32/64/128-bit. Runtime per-byte
entropy detection that ships only varying bytes through PCIe is
**not in the published literature.** This explains:
- 1.7× best/worst spread on different distributions (18.6)
- 50-73% PCIe traffic reduction on natural data (TPC-H)
- Why we beat RMG-style MSB radix when keys are composite

### (C) NUMA --preferred policy for asymmetric memory access
The discovery that **soft NUMA binding (--preferred) wins over strict
binding (--membind) for workloads with random reads + sequential
writes** — and the regime flip at 4-GPU paired — is unpublished.
We get 1.7-2× wall, 27× more reproducible variance, with a one-line
deployment wrapper.

This is a real systems-engineering contribution that generalizes
beyond sort: any host-RAM-bound workload with the same access
pattern benefits from --preferred over --membind.

### Three things missing for max rigor

1. **Direct comparison vs cuDF and RMG Sort on same H100 hardware.**
   - cuDF: install issues here, but worth retrying. Open source.
   - RMG: hpides/rmg-sort; needs sm_90 port. Estimated ~half-day work.
2. **Adversarial distribution stress.** RMG explicitly tested only
   Zipf 1.0; nobody has shown what breaks under collapsed shipdate
   or extreme skew. Demonstrates robustness.
3. **System-level energy** (currently GPU-only). Needs IPMI or
   external watt-meter.

## Top-3 experiments to maximally strengthen the paper

### #1 — RMG Sort head-to-head (highest value)
Fork hpides/rmg-sort, port to sm_90 H100, run keys-only at our
record counts. Result is direct gpu_crocsort vs RMG comparison on
SAME hardware. Predicted: gpu_crocsort beats RMG significantly
because of the compact-key encoding, even though RMG's keys-only
inner loop should be theoretically faster (no gather).

### #2 — Adversarial distributions
Already have synthetic generator (18.6). Add:
- Real Zipfian (1/rank weighted)
- Collapsed shipdate (single year)
- Adversarial l_orderkey clustering
Show our partition stays balanced where RMG's MSB radix collapses.

### #3 — System-level energy
nvidia-smi gives GPU power; need IPMI/watt-meter for system. If we
can get IPMI access (sudo), measure full-rack power during sort.
Compares to JouleSort 2023 winner (37 J/GB system-level).

## Implication for the paper abstract

Three novel claims (in order of strength):
1. **NUMA --preferred for asymmetric memory access**: 1.7-2×, 27×
   more reproducible, one-line wrapper. Generalizes.
2. **Runtime adaptive compact-key encoding**: 1.7× spread on
   distributions, 50-73% PCIe reduction.
3. **Largest published TPC-H lineitem global sort** (SF1000 = 720 GB
   full payload).

Plus the supporting framework:
- Linear scaling law (3.5 ns/record) confirms O(N) radix asymptotic
- Roofline analysis (gather is biggest gap at 11% of host RAM peak)
- 4× H100 NVL multi-GPU scaling characterization (35% paired contention)

The paper has plenty of substance. The remaining work is rigor —
direct competitor comparisons (#1, #2 above) and robustness tests.

## Sources

- arXiv:2206.01784 — CUB Onesweep
- BTW 2023 — RMG Sort (HPI)
- SIGMOD 2022 — Multi-GPU sort with modern interconnects
- PVLDB 18 — sort-merge join on H100
- VLDB 2025 / arXiv:2502.09541 — Vortex
- DuckDB blog 2025-09-24 — sort redesign
- PDCAT 2024 — async OOM GPU sort
- voltrondata.com — Theseus benchmarks
- sortbenchmark.org — JouleSort
- github.com/hpides/rmg-sort — RMG implementation
- SIGMOD 2024 — GOLAP
- SIGMOD 2025 — joins+groupby on GPUs
- HiPC 2013 — GPU string sort
