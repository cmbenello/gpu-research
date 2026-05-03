# Session 2026-05-04 morning — NUMA + roofline characterization

**Dates active:** 2026-05-03 19:30 → 2026-05-04 0X:XX
**Branch:** h100/discoveries-2026-05-02
**Theme:** Set roofline ceilings, then identify and fix the gather bottleneck.

## Headline win of the session

**A two-line drop-in change — wrapping the sort binary with
`numactl --cpunodebind=0 --membind=0` — gives 1.84× SF100 wall
speedup (5480 ms → 2980 ms median across 4 warm runs).** 24.2 GB/s
sustained throughput, the highest single-H100 number we've measured
so far. (See `17.3.2.1_runs5_validation.md`.)

## What was learned

### 1. HBM3 saturation ceiling: 2191 GB/s (effective r+w) — 6.2 + 6.2.1

The H100 NVL HBM3 ceiling for streaming kernels is **2191 GB/s
read+write** (65% of the 3.35 TB/s spec sheet). Read-only is
**1521 GB/s**. All four GPUs in this box give the same numbers
within 0.5%. So when CUB radix sort hits ~50% of this, it's already
a "good radix sort" — chasing GPU optimizations is low-ROI.

### 2. Roofline: gather is the biggest gap — 17.1.1

| Phase | Theoretical | Measured | Efficiency |
|-------|-------------|----------|------------|
| GPU LSD radix sort | 2191 GB/s | 900 | 41% |
| GPU OVC merge | 2191 GB/s | 1300 | 59% |
| PCIe upload | 32 GB/s | 19 | 60% |
| CPU compact extract | 400 GB/s | 90 | 22% |
| **CPU gather** | 400 GB/s | **44** | **11%** |

CPU gather has by far the largest gap to the host RAM ceiling.
Everything we did from 17.2 onward chases that gap.

### 3. Gather thread scaling is non-monotonic — 17.2

| Working set | Optimal threads | GB/s peak |
|-------------|-----------------|-----------|
| 12 GB | 96 | 65 |
| 72 GB | 32 | 41 |
| 216 GB | 128 | 56 |

At 72 GB the working set busts L3, so 32 is the sweet spot. At
216 GB everything is DRAM-bound regardless, and many threads in
flight queue more memory requests. At 12 GB the working set fits in
L3, and more threads help.

Production-code change: GATHER_THREADS env var unified across all 3
gather code paths (lines 1403, 1922, 2930), default 64 retained.

### 4. NUMA bind-node-0 gives 3.9× cold gather speedup — 17.3

Cold-cache random gather of 72 GB at 32 threads:
| Policy | GB/s |
|--------|------|
| default (kernel auto) | 5.09 |
| `numactl --cpunodebind=0 --membind=0` | **19.84** ← 3.9× |
| `numactl --interleave=all` | 14.27 |

The kernel default policy interleaves pages across both nodes; half
the random reads go cross-socket and pay UPI latency. Single-node
binding makes everything local.

### 5. Real-sort speedup with numactl wrap — 17.3.2 + 17.3.2.1

SF100 production sort, 5 warm runs each, median:
| Config | wall | throughput |
|--------|------|------------|
| default | 5480 ms | 13.1 GB/s |
| numactl bind-0 | **2980 ms** | **24.2 GB/s** |

Default cold (run 1) takes 45680 ms with first-touch NUMA pathology.
numactl cold takes 3097 ms. The 14.7× cold ratio is much larger than
the 1.84× warm ratio because cold runs hit the worst-case page-faulting
behavior.

## Open follow-ups (not landed this session)

- **17.3.2.2** — re-baseline SF50 + SF300 + SF1000 under numactl
  (in flight at session end)
- **17.3.2.3** — multi-GPU NUMA spread (GPU 0/1 → node 0,
  GPU 2/3 → node 1). Should give same ~1.8× per GPU.
- **17.3.2.4** — implement libnuma binding inside the sort binary
  (no wrapper requirement).

## Key commits this session

- `b87080d` h100/6.2.1: HBM sweep across 4 GPUs x 5 sizes — uniform 2191 GB/s peak
- `a3e36e1` h100/17.1.1: roofline efficiency charts
- `06af4af` h100/17.3: NUMA-bound gather is 3.9x faster than default
- `3a8e172` h100/17.3.2: numactl wrap gives 21% SF100 wall speedup
- `0XX...` h100/17.3.2.1: --runs 5 confirms numactl 1.84x SF100

## What's NOT changed in production code

The default 64-thread gather behavior is preserved. The session's
takeaway is that the user/operator should:

```bash
numactl --cpunodebind=0 --membind=0 ./external_sort_tpch_compact ...
```

This is documented in `17.3.2_numactl_wrap.md` as the deployment
recommendation.

## Gather scaling is not predicted by gather_bench

Gather_bench at 72 GB random shows 32 threads as peak (41 GB/s).
Production SF100 at 64 threads got 36 GB/s (close to bench's 36 GB/s
at 64). But production single-warm wall variance is too high (1300
to 2900 ms) to call a default change confidently. Hence env override
without default change.

## Hardware reminder

- 2× Xeon Platinum 8468, 96 cores each (192 logical CPUs)
- 1 TB host RAM (515 GB on each NUMA node)
- 4× H100 NVL, 94 GB HBM3 each, NV6 NVLink full all-to-all
- GPU 0/1 → NUMA node 0 PCIe; GPU 2/3 → NUMA node 1 PCIe
- 3.5 TB NVMe at /mnt/data

## How to use these findings

If you're running the sort on this box, **always wrap with numactl**.
The 21-46% wall reduction is "free." If you're writing the paper,
report numbers under numactl as the headline (it's the right
configuration; not numactl is leaving 1.5-2× on the table).
