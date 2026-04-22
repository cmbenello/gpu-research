# Related work from other accelerators (inspiration only)

Survey of work on FPGAs, ASICs, and irregular-workload accelerators that informs CrocSort design choices. We are not building on FPGA, but their design history is useful as inspiration for the GPU-side compression and merge phases.

Date: 2026-04-21

## Why look at other accelerators

The two parts of CrocSort that GPUs handle worst are:
1. PCIe-bound key transfer (12 GB/s on PCIe 3.0, ~50 GB/s on PCIe 5.0)
2. K-way merge (branch-heavy, irregular memory access, kills warp efficiency)

Both have been studied for years on FPGAs and on irregular-workload accelerators. Even if we never touch that hardware, the codecs and merge-network designs they use are directly portable to GPU.

## FPGA sort accelerators

The FPGA sort literature is the closest prior art, because FPGAs hit the same bandwidth wall and chose compression as the answer.

- Cost-Effective and High-Throughput Merge Network (SIGARCH 2016) builds dedicated merge networks that run as fixed-dataflow pipelines. No branch divergence because the network is synthesized at compile time. Useful as a model for what a sample-sort partitioner could look like in CUDA.
- A High Performance FPGA-Based Sorting Accelerator with Data Compression (IEICE 2017) packs successive compressible 512-bit data blocks into a single 512-bit block before writing to external memory. Halves bandwidth in the right cases. Same idea CrocSort wants on PCIe.
- An Adaptable High-Throughput FPGA Merge Sorter for Database Analytics (FPL 2020) targets database workloads directly. Reports 49x over an A53 core for sort, 27x for query analytics.
- FPGA-Accelerated Samplesort for Large Data Sets (FPGA 2020) takes the sample-sort approach we want to evaluate as a K-way-merge replacement.
- FPGA-Accelerated Compression of Integer Vectors (DaMoN 2020) bit-packing at PCIe line rate. Confirms cheap codecs are viable on streaming data.

Takeaway: FPGAs solved the bandwidth problem with order-preserving compression and dedicated merge networks. We can borrow the codec choices directly.

## Microsoft Project Catapult

Datacenter-scale FPGA deployment, 40x on Bing decision trees, FPGAs sit on the PCIe bus and between server and TOR switch. Doesn't focus on sort but proves the deployment model works at hyperscale. Relevant only as a long-term framing for who would care about a sort accelerator.

## ASIC / functional-unit augmentation

- Fast Radix (IEEE 2015) and RadixBoost (IEEE 2015) add a special radix-sort instruction to CPU/GPU microarchitecture. Tightly coupled to existing pipelines, not standalone accelerators. Useful as motivation that custom sort hardware is being built.

## Andrew Chien's UpDown (UChicago)

UpDown is purpose-built for irregular workloads: fine-grained dynamic parallelism, event-driven scheduling, software-controlled threading, no caches. Designed for sparse / graph computation, but the workload pattern (data-dependent control flow, irregular memory access, frequent fine-grained synchronization) matches K-way merge almost exactly.

Reference paper: When merging and branch predictors collide. Shows branch mispredicts cause up to 5x slowdown on merge. This is the core argument for offloading merge from CPU to a deterministic-dataflow accelerator.

If we ever extend this work to a heterogeneous design, UpDown is the natural target for the merge phase. For now, the inspiration is just: aggressively avoid the merge phase entirely (sample sort).

## Order-preserving compression literature

CPU columnar databases solved this for analytics two decades ago. None of it has been ported to GPU radix sort, which is the gap CrocSort can fill.

- HOPE: High-speed Order-Preserving Encoder (Zhang, SIGMOD 2020, CMU PDL). Fast dictionary-based encoder, byte-comparable output, designed for in-memory search trees. Almost directly portable.
- FSST: Fast Random Access String Compression (Boncz, VLDB 2020, CWI). Used in DuckDB. Variable-length string codes with random access.
- Mostly Order Preserving Dictionaries (Liu, ICDE 2019, UChicago). Relaxes strict order preservation for big compression gains. Same group ecosystem as Chien.
- Frame of Reference (FOR) and bit-packing. The workhorse codes from MonetDB and Vectorwise. Trivially byte-comparable for fixed-width unsigned domains after subtracting min.
- Integrating compression and execution in column-oriented database systems (Abadi, SIGMOD 2006). Foundational paper on operating directly on compressed data.

Takeaway: there is a rich design space of order-preserving codecs. CrocSort can systematically evaluate which ones work best for GPU radix sort.

## Recent GPU + compression work

- The Effectiveness of Compression for GPU-Accelerated Queries on Out-of-Memory Datasets (Nicholson, Chasialis, Boffa, Ailamaki, DaMoN 2025). Shows lightweight compression schemes substantially widen the bandwidth envelope for GPU queries on out-of-memory datasets. Doesn't focus on sort specifically. CrocSort fills that gap.

## Summary of design ideas to steal

| Idea | Source | CrocSort use |
|---|---|---|
| Order-preserving dictionary | HOPE 2020 | encode string keys for direct radix sort |
| FOR + bit-pack | Vectorwise / FPGA work | cheapest codec to evaluate first |
| Compress before transfer | IEICE 2017 FPGA sorter | apply at compact-key-scan stage |
| Sample sort partitioner | FPGA Samplesort 2020 | replace K-way merge entirely |
| Mostly-ordered dict | Liu 2019 | relax order constraint when compression ratio matters more |
| Operate on compressed data | Abadi 2006 | radix-sort compressed keys without decode |
