# Production Deployment Guide for gpu_crocsort on H100 NVL

**Last updated:** 2026-05-04
**Hardware:** sorting-h100 (4× H100 NVL, dual Xeon 8468, 1 TB host RAM)
**Status:** All numbers verified across 2026-05-03/04 sessions

This is the deployment recipe for getting the best sort throughput on
this hardware. All recommendations are backed by experiments in
`results/h100_runs/`.

## TL;DR

```bash
# Always wrap your sort with numactl. The wrap depends on GPU count:

# Single GPU (best for SF50/SF100/SF300):
numactl --cpunodebind=0 --preferred=0 \
    ./external_sort_tpch_compact \
    --input lineitem.bin --runs 5

# 4 GPU partition mode (15.4 / 17.3.2.3 — each GPU sorts 1/4 by position):
for i in 0 1 2 3; do
    NODE=$(( i / 2 ))
    CUDA_VISIBLE_DEVICES=$i numactl --cpunodebind=$NODE --membind=$NODE \
        ./external_sort_tpch_compact ... &
done
wait

# 4 GPU distributed (15.5.3 / 17.3.2.7.6 — globally sorted via partition):
# Pair A (GPUs 0+1, node 0):
NO_MAP_POPULATE=1 numactl --cpunodebind=0 --preferred=0 ./sort ... &
# Pair B (GPUs 2+3, node 1) after Pair A finishes:
NO_MAP_POPULATE=1 numactl --cpunodebind=1 --preferred=1 ./sort ... &
```

## Why two different policies (`--preferred` vs `--membind`)?

This is the most important rule:

| Configuration | Policy | Reason |
|---------------|--------|--------|
| 1 GPU | `--preferred=N` | One node has free DDR5 channels to spill writes to (2× write bw) |
| 4 GPU partition | `--membind=N` per pair | Both nodes already saturated; --membind's determinism wins |
| 4 GPU distributed | `--preferred=N` per pair | Pairs are sequenced not concurrent, so --preferred's safety > determinism |

See `17.3.2.7.4_numa_policy_comparison.md` and
`17.3.2.7.5_4gpu_preferred_vs_membind.md` for the full analysis.

## Best-tested numbers

### Single GPU (1× H100 NVL, numactl --preferred=0)

| Scale | Records | Bytes | Wall | Throughput | Notes |
|-------|---------|-------|------|------------|-------|
| SF10  | 60 M | 7.2 GB | 0.35 s | 20.4 GB/s | fast path (no Phase 3) |
| SF50  | 300 M | 36 GB | 1.41 s | 25.5 GB/s | numactl + post-0.3.1 |
| SF100 | 600 M | 72 GB | 2.95 s | 24.4 GB/s | first to beat RTX 6000 |
| **SF300** | **1.8 B** | **216 GB** | **6.59 s** | **32.7 GB/s** | **highest single-GPU** |
| SF500 | 3 B | 360 GB | OOM | — | exceeds 94 GB HBM |

Rationale: SF300 hits highest throughput because OVC architecture
amortizes setup overhead; smaller workloads spend proportionally more
time in extract + setup.

### 4× H100 partition (each GPU sorts 1/4, NOT globally sorted)

| Scale | Bytes | Wall | Aggregate GB/s | Method |
|-------|-------|------|------------------|--------|
| SF500 | 360 GB | 5.48 s | **65.7** | --membind per pair (17.3.2.3) |

### 4× H100 distributed (sample-partition + paired sort, globally sorted)

| Scale | Bytes | Wall | End-to-end GB/s | Method |
|-------|-------|------|------------------|--------|
| SF500 | 360 GB | 7m10s | 0.84 | --preferred per pair (17.3.2.3.3) |
| **SF1000** | **720 GB** | **20m08s** | **0.60** | --preferred per pair (17.3.2.7.6) |

Distributed sort produces 4 sorted bucket files; the global sort is
implicit in iterating them in disjoint key-range order. **First time
SF1000 has been globally sorted on this hardware** (15.5.3 + 17.3.2.7.6).

## Bottleneck location by scale

| Scale | Dominant bottleneck | % of wall | Optimization path |
|-------|---------------------|-----------|--------------------|
| SF10 | GPU compute (CUB radix) | 70%+ | fast path; no further wins |
| SF50/SF100 | CPU compact extract + gather | 60% | numactl helps gather (50%) |
| SF300 | CPU gather, then GPU OVC merge | 50% / 30% | numactl --preferred 2× DDR5 wr |
| SF500/SF1000 distributed | NVMe partition | 39% / 39% | needs faster storage |

## Why NUMA matters so much for this workload

The sort has 3 distinct memory access patterns:

1. **Compact extract** (Phase 1): sequential read of input + sequential
   write of compact keys. ~25% efficient on host RAM.
2. **PCIe upload** (Phase 1): chunked H2D. 28-55 GB/s realized.
   NUMA-independent (DMA bypasses CPU).
3. **CPU gather** (Phase 3): random read of 120 B records using
   permutation + sequential write. **This is where numactl wins.**

The gather phase reads ~70 GB random + writes ~70 GB sequential per
SF100. Random reads benefit from local-node placement (low latency);
sequential writes benefit from spreading across nodes (high bandwidth).
Hence `--preferred` is the right policy: forces input to local node
while letting output spill to other node.

Reference: `17.3_numa_gather.md` (cold-cache 3.9× win),
`17.3.2.1_runs5_validation.md` (1.84× warm),
`17.3.2.7_preferred_vs_membind.md` (12.9% --preferred over --membind),
`17.3.2.7.5_4gpu_preferred_vs_membind.md` (--membind wins multi-GPU).

## Variance: numactl is far more reproducible

| Policy | StdDev / Median (CV) |
|--------|----------------------|
| Default | 13.8% (run-to-run uncertainty huge) |
| `--membind=0` | 0.2% (nearly deterministic) |
| `--preferred=0` | 0.5% (nearly deterministic) |

The variance reduction alone is paper-worthy: under default scheduling,
the kernel makes non-deterministic NUMA placement decisions; under
numactl, the wall is reproducible to ~1%.

## Single-GPU regime: peak per-GPU throughput

Best per-record cost: SF300 at **32.7 GB/s** = 4.3 microjoules/record
(sort energy from `14.1_joule_per_gb.md`). The H100 NVL hits ~50% of
its HBM3 peak (2191 GB/s effective from 6.2) on the GPU LSD radix
sort phase, which is canonical for a "good radix sort." The gather
phase is at ~22% of host DDR5 peak (after numactl).

## Multi-GPU regime: scaling efficiency

| Config | Aggregate GB/s | per-GPU | Scaling efficiency |
|--------|-----------------|---------|---------------------|
| 1 GPU SF300 | 32.7 | 32.7 | 1.00× (baseline) |
| 2 GPU one-per-node SF300/2 | 50.8 | 25.4 | 1.83× / 92% |
| 4 GPU two-per-node SF500/4 | 65.7 | 16.4 | 2.37× / 59% |

Pairing 2 GPUs per node loses 33% to memory channel contention. For
maximum per-GPU efficiency, run 2-per-box (one per node). For
maximum aggregate, run 4 with the contention penalty.

## Reproducible run

To reproduce all numbers in this guide:

```bash
cd ~/gpu-research/gpu_crocsort

# Build
source ~/gpu-research/h100/env.sh
make external-sort-tpch-compact

# SF50/SF100/SF300 single-GPU best
for sf in 50 100 300; do
    CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        ./external_sort_tpch_compact \
        --input /mnt/data/lineitem_sf$sf.bin --runs 5
done

# SF500 4-GPU partition (15.4 method)
bash /tmp/run_17.3.2.3_4gpu_numa.sh

# SF500 distributed (globally sorted)
bash /tmp/run_17.3.2.3.3_15.5.3_numactl.sh

# SF1000 distributed (globally sorted)
bash /tmp/run_17.3.2.7.6_sf1000.sh
```

## Hardware caveats (this box)

- 2 NUMA nodes, 96 cores each (192 logical, 96 physical per node).
- 1 TB host RAM split 515 GB / 516 GB across nodes.
- 4× H100 NVL: GPU 0/1 → node 0 PCIe; GPU 2/3 → node 1 PCIe.
- 3.5 TB NVMe (single device, no RAID) at /mnt/data — 6.2 GB/s
  sequential read peak, 3 GB/s sustained partition rate.
- 504 GB tmpfs at /dev/shm.

The NUMA-node-2-GPUs-each topology forces the `--preferred` (1-GPU)
vs `--membind` (4-GPU) trade-off. On a 1-GPU-per-node box (DGX H100 has 8
GPUs / 1 NUMA node), the trade-off would differ.

## Recommended reading order

1. `THROUGHPUT_TABLE.md` — current best numbers
2. `17.3_numa_gather.md` — the NUMA discovery
3. `17.3.2.1_runs5_validation.md` — variance bounds
4. `17.3.2.7_preferred_vs_membind.md` — why --preferred wins
5. `17.3.2.7.4_numa_policy_comparison.md` — full policy table
6. `17.3.2.7.5_4gpu_preferred_vs_membind.md` — multi-GPU regime flip
7. `17.3.2.7.6_sf1000_distributed.md` — SF1000 with NUMA wrap

## What's NOT in this guide (out of scope)

- libnuma-coded internal NUMA affinity (would need libnuma-dev install
  and code changes; the wrapper approach was sufficient).
- Disabling `kernel.numa_balancing` (needs sudo; would tighten default
  variance but doesn't affect numactl-wrapped runs).
- Multi-node distributed sort (single-node only on this hardware).
- Faster storage (NVMe-fabric, RAID0) — would address the 39% partition
  bottleneck.
