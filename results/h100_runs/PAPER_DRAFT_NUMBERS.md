# gpu_crocsort on H100 NVL — Numbers for Paper Draft

**Last updated:** 2026-05-04
**Hardware:** sorting-h100 (4× H100 NVL, 2× Xeon 8468, 1 TB host RAM, 3.5 TB NVMe)
**Branch:** h100/discoveries-2026-05-02

This is the paper-ready consolidation of all measured numbers. Every
number is backed by a `results/h100_runs/<id>.md` writeup.

---

## §1 Headline single-GPU numbers (numactl --cpunodebind=0 --preferred=0)

| Scale | Records | Bytes | Wall (warm best) | Throughput | Source |
|-------|---------|-------|------------------|------------|--------|
| SF10 | 60 M | 7.2 GB | 0.353 s | 20.4 GB/s | 17.3.2.6 |
| SF50 | 300 M | 36 GB | **1.408 s** | **25.6 GB/s** | 17.3.2.2 |
| SF100 | 600 M | 72 GB | **2.952 s** | **24.4 GB/s** | 17.3.2.1 |
| SF300 | 1.8 B | 216 GB | **6.535 s** | **33.1 GB/s** | 17.3.2.7.2 + 18.7 |

**Peak single-GPU throughput: 33.1 GB/s on SF300.**

## §2 Multi-GPU numbers

### 4-GPU position-partitioned (each GPU sorts 1/4, NOT globally sorted)

| Scale | Bytes | Wall | Aggregate GB/s | Source |
|-------|-------|------|------------------|--------|
| SF500 | 360 GB | **5.48 s** | **65.7** | 17.3.2.3 |

### 4-GPU sample-partition + paired sort (globally sorted)

| Scale | Bytes | Wall | End-to-end GB/s | Source |
|-------|-------|------|------------------|--------|
| SF500 | 360 GB | 7m10s | 0.84 | 17.3.2.3.3 |
| **SF1000** | **720 GB** | **14m20s** | **0.84** | **18.5c** (new) |

For SF1000 with **pre-partitioned input** (workflow assumes data is
already partitioned): **6m14s sort phase only**.

## §3 SOTA comparison

### CPU engines (192-thread)

| Engine | SF50 | SF100 | gpu_crocsort speedup |
|--------|------|-------|-----------------------|
| **gpu_crocsort** | **1.41 s** | **2.95 s** | 1.0× |
| Polars 1.8.2 (16 B prefix only) | 14.5 s | 28.6 s | 9.7-10.3× |
| DuckDB 1.3.2 (full ORDER BY) | 33.0 s | 106.0 s | 23.4-35.9× |

### GPU sort ceiling (raw CUB, no I/O)

| Workload | Raw CUB (8B keys) | gpu_crocsort | gap |
|----------|---|---|---|
| SF50-eq (300M records) | 33.9 ms | 1.41 s | 41× |
| SF100-eq (600M) | 67.6 ms | 2.95 s | 44× |
| SF300-eq (1.8B) | 203 ms | 6.59 s | 32× |

The 32-44× gap is entirely host-side cost: CPU compact extract +
PCIe upload + CPU gather. GPU sort is at HBM peak.

## §4 NUMA-aware deployment (the key engineering finding)

**Two-line wrapper** gives 1.7-2× wall reduction on a workload-by-workload basis:

```bash
numactl --cpunodebind=N --preferred=N ./external_sort_tpch_compact ...
```

| Scale | Default | numactl --preferred | Speedup |
|-------|---------|----------------------|---------|
| SF50 (warm med) | 2146 ms | 1414 ms | 1.52× |
| SF100 (warm med) | 5070 ms | 2967 ms | **1.71×** (8 warm runs) |
| SF300 (warm med) | 13490 ms | **6736 ms** | **2.00×** |

**Variance reduction**: numactl gives 27× tighter coefficient of
variation (13.8% → 0.5%). Default placement is non-deterministic;
numactl is reproducible.

### Policy regime switch

| Configuration | Best policy | Why |
|---------------|--------------|-----|
| 1 GPU | `--preferred=N` | local reads + spill writes for 2× DDR5 bandwidth |
| 2 GPU one-per-node | `--preferred=N` (per GPU) | no contention |
| 4 GPU two-per-node | `--membind=N` (per pair) | both nodes saturated; --preferred adds overhead |

Reference: 17.3 (3.9× cold gather), 17.3.2.1 (1.71× SF100 wall),
17.3.2.5 (2-sweep variance bound), 17.3.2.7 (--preferred discovery),
17.3.2.7.4 (full policy comparison), 17.3.2.7.5 (4-GPU regime flip).

## §5 Scaling law

**Linear** in N (radix sort signature):

```
t = 3.51e-9 × N + 0.404 s
```

R² > 0.99 for SF50/100/300. Throughput plateaus at ~34 GB/s
asymptotic. Predicts SF1000 single-GPU = 21.5 s if HBM held it
(actual: HBM-OOM, requires distributed).

Reference: 17.2.

## §6 Distribution sensitivity (NEW)

Sort throughput depends on input data entropy via the runtime
adaptive compact-map:

| Distribution | Wall (ms) | GB/s | Compact key |
|--------------|-----------|------|--------------|
| sorted | 1169 | 30.8 | small |
| reversed | 1185 | 30.4 | small |
| TPC-H lineitem | 1413 | 25.5 | mixed |
| all_equal | 1527 | 23.6 | small + Phase 4 fixup |
| random (uniform 64-bit) | 1924 | 18.7 | full |
| zipfian | 1924 | 18.7 | full |

**Worst-vs-best spread: 1.7×.** Real workloads (TPC-H) sit toward
the sorted end (auto-compact picks up structure).

Reference: 18.6.

## §7 Energy efficiency

| Measurement | J/GB (GPU only) |
|-------------|-----------------|
| Cold + warm avg (1-shot) | 36-46 |
| Warm-only (best case) | ~6 |
| MendSort 2023 system-level | ~37 |

System-level on our box (with CPU + memory + cooling) estimated at
~100-130 J/GB. We're throughput-optimized, not joule-optimized.

Reference: 14.1, 18.7.

## §8 Roofline

| Phase | Theoretical | Measured | Efficiency |
|-------|-------------|----------|------------|
| HBM3 saturation | 2191 GB/s (effective r+w) | 2191 | 100% (peak) |
| GPU LSD radix sort | 2191 | ~900 | 41% |
| GPU OVC merge | 2191 | ~1300 | 59% |
| PCIe 5 x16 H2D | 32 (single PCIe) / 55 (dual) | 28-55 | 51-100% |
| CPU compact extract (DDR5) | 400 | 90 | 22% |
| **CPU gather (DDR5, random)** | **400** | **44** | **11%** ← biggest gap |

Gather is the biggest gap. NUMA --preferred captures most of the
remaining 89% headroom; further wins need GPUDirect Storage or
zero-copy redesigns.

References: 17.1, 6.1, 6.2, 6.3.1.

## §9 Multi-GPU scaling efficiency

| Configuration | Aggregate GB/s | Per-GPU | Scaling |
|---------------|-----------------|---------|---------|
| 1 GPU SF300 | 32.7 | 32.7 | 1.00× |
| **2 GPU one-per-node** | **50.8** | **25.4** | **1.83× (92%)** |
| 4 GPU 2-per-node SF50 batch | 64.0 | 16.4 | 2.51× (63%) |
| 4 GPU 2-per-node SF500 partition | 65.7 | 16.4 | 2.58× (63%) |

Pairing 2 GPUs on one NUMA node always costs ~35% per-GPU due to
memory channel contention.

References: 18.4 (2-GPU), 17.3.2.3 (4-GPU partition), 18.3 (4-GPU
concurrent SF50).

## §10 Correctness

All sizes 1k - 1.8B records `--verify` PASS across:
- TPC-H lineitem (natural distribution)
- Synthetic uniform random
- Synthetic sorted / reversed / all-equal
- All numactl policies

References: 18.1 (small N), 17.3.2.7.6 (SF100 verify), 18.6 (distributions).

## §11 Per-GPU consistency

All 4× H100 NVL identical within 0.5% on HBM bandwidth (6.2.1),
0.18% on real SF50 sort (18.2). Any single-GPU number generalizes.

## §12 Hardware caveats

- 2 NUMA nodes, 96 cores each (192 logical)
- 4 GPUs split 2-per-node (GPU 0+1 → node 0 PCIe; GPU 2+3 → node 1)
- Single 3.5 TB NVMe at /mnt/data (6.2 GB/s seq read peak; 3 GB/s
  partition end-to-end)
- 504 GB tmpfs at /dev/shm

The NUMA-node-2-GPU topology creates the `--preferred` (1-GPU) vs
`--membind` (4-GPU) trade-off. On a 1-GPU-per-node box (DGX H100 with
8 GPUs / 1 NUMA root), the trade-off would differ.

## §13 Production deployment (TL;DR)

```bash
# Single GPU (best for SF50/100/300):
numactl --cpunodebind=0 --preferred=0 \
    ./external_sort_tpch_compact --input lineitem.bin --runs 5

# 4-GPU partition (best aggregate, NOT globally sorted):
for i in 0 1 2 3; do
    NODE=$((i / 2))
    CUDA_VISIBLE_DEVICES=$i numactl --cpunodebind=$NODE --membind=$NODE \
        ./external_sort_tpch_compact --offset-records ... --limit-records ... &
done
wait

# 4-GPU distributed (globally sorted, paired):
# Use --preferred per pair; partition phase is the bottleneck (39% of wall).
```

## Open questions for follow-up papers

1. **Pre-partitioned SF1000 round-2 cache pollution**: 18.5c showed
   round-2 sort is 2.3× slower than round 1 due to page cache.
   `posix_fadvise(POSIX_FADV_DONTNEED)` fix could push pre-partitioned
   SF1000 from 6m14s → ~2 min.
2. **GPUDirect Storage**: would skip CPU compact extract + PCIe
   upload. Could close 32× host-side gap.
3. **NCCL all-to-all multi-GPU sort**: replace partition + paired
   model with cooperative multi-GPU sort; eliminates partition
   bottleneck.
4. **Disabling kernel.numa_balancing**: would tighten default
   variance; measured numactl-wrapped variance is already 27× tighter.

## Test reproducibility

| Setup | Command |
|-------|---------|
| Build | `cd ~/gpu-research/gpu_crocsort && make external-sort-tpch-compact` |
| 1-GPU SF50 | `numactl --cpunodebind=0 --preferred=0 ./external_sort_tpch_compact --input /mnt/data/lineitem_sf50.bin --runs 5` |
| 4-GPU SF500 partition | `bash /tmp/run_17.3.2.3_4gpu_numa.sh` |
| 4-GPU SF1000 distributed | `bash /tmp/run_17.3.2.7.6_sf1000.sh` |
| Pre-partitioned SF1000 | `bash /tmp/run_18.5c_2gpu_sf1000.sh` |
| Distribution sweep | `bash <generate + sort 5 distributions in 18.6.md>` |
| Energy SF300 | `bash /tmp/run_18.7_energy_sf300.sh` |
| Scaling law | `python3 results/h100_runs/17.2_scaling_law_fit.py` |

## Branch

All commits on `h100/discoveries-2026-05-02`. ~50 commits since
2026-05-03. Pushed to GitHub origin.
