# SF1500 hardware boundary analysis — where the 7m19s comes from and what's left

**Date:** 2026-05-05
**Best result:** 7m19s (streaming, n=1) / 7m56s (compact v3, n=3 best)

## NVMe ceiling on this single SSD

Measured by `dd`:
- Sequential write: 1.3 GB/s peak (SLC), ~1.14 GB/s sustained (QLC)
- Sequential read: 3.1 GB/s peak

## Theoretical floors at SF1500 (1.08 TB)

| Pipeline                          | Min NVMe traffic       | Theoretical wall |
|-----------------------------------|------------------------|------------------|
| Read input only (no output)       | 1.08 TB read           | **5.8 min**      |
| Read input + tiny output (offsets) | 1.08 TB read + 68 GB write | **5.9 min**  |
| Read input + full output          | 1.08 TB read + 1.08 TB write | **15.8 min** (read & write don't overlap fully on single SSD) |

## Where 19.25 streaming (7m19s, offsets-only) sits

```
cmap detect:                  6.8s
unpinned mmap alloc:          0s
partition (1.08 TB read + 360 GB write to RAM):  4m48s  ← NVMe-read-bound (3.76 GB/s effective)
sort phase (4-GPU concurrent × 4 rounds):        2m11s  ← pinning + sort + output write
TOTAL:                                            7m19s
```

**Gap to floor: 7m19s vs 5.9m floor = 1.4 min (24%).**

The gap consists of:
- Sort phase pin-on-demand: ~30s × 4 rounds shared across GPUs
- 68 GB output write at 1.14 GB/s = 60s (overlapped partially)
- Per-GPU CUDA context init + memory allocation overhead

## Why we can't easily break the floor

Going below 5.8 min requires **less than 1.08 TB of NVMe traffic per run.**
That means:
- (a) **Reading input from RAM cache** — input.bin is 1.08 TB, host RAM is 1 TB. Doesn't fit.
- (b) **Reading less than full input** — would require a projected/columnar layout, which is a different workload.
- (c) **Multiple NVMe drives** — not available on this box.
- (d) **GPUDirect Storage (cuFile + nvidia_fs)** — library installed but kernel module not loaded.

## Where the e2e bottleneck moves to

For the **full-records output e2e** (39m18s in 19.26):
- Stream phase: 7m31s (NVMe-read-bound on input)
- Gather phase: 30m53s ← **new bottleneck**

The gather is bottlenecked by **cache miss on input.bin during random reads**. With
1.08 TB input.bin and 1 TB cache, ~8% of input reads spill to NVMe random reads at
~50-100K IOPS. With 9B random reads of 120-byte records, this is the new ceiling.

## What's actually next (in priority order)

1. **Lock variance on 7m19s** — n=2 in flight; need n=3.
2. **Better gather** — sequential input read + scatter to random output positions.
   Random NVMe writes might be faster than random reads. Could halve gather to ~15m.
3. **GPUDirect Storage** — would skip host pin entirely, ~30s saved on sort phase.
   Requires admin to load nvidia_fs module.
4. **Multi-NVMe** — would multiply NVMe ceiling. Hardware change.

For research purposes, we're at the hardware boundary. The 7m19s number is the
right headline for "external sort with offsets output" on this single-SSD config.

## Cost of the gather bottleneck

The 39m18s e2e is dominated by gather (30m53s = 80% of wall). The stream phase
(7m31s) does the heavy work. For workloads that DON'T need a materialized
sorted-records file (e.g., a database engine's external sort operator that
feeds aggregations), the 7m19s offsets-only output is the right primitive.

Any application that needs a file of sorted records pays the gather penalty
regardless of GPU sort speed — it's purely an NVMe I/O cost. This is why
modern columnar engines avoid materializing sorted rows when possible.
