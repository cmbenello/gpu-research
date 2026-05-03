# Tier A bottleneck-attack results — 2026-05-03

This documents what happened when I tried to attack the bottlenecks
identified in `BOTTLENECKS.md`. Three Tier A items planned:

| Item | Predicted | Result |
|------|-----------|--------|
| NUMA-pin gather threads | ~1.5× gather speedup | **NEGATIVE — 3-4× slower** |
| Conditional MAP_POPULATE | n/a (fix for unintended regression) | **+9× gather throughput restored** (5 GB/s → 45 GB/s warm best at SF300) |
| 0.3.1 slow-path compact upload | ~2× SF50/100 wall | **DEFERRED — code path more interleaved than scoped, needs more time** |

Plus an unrelated critical regression fix (segfault) that surfaced in the process.

## 1. Critical regression fix — 1.6.1 segfaulted SF300 Phase 1

When I tried to retest SF300 after my pinning change, the binary
segfaulted *before* my code ran. Bisected to commit 9fc99f8 (1.6.1
lazy h_pin) — that commit's lazy `cudaMallocHost(&h_pin[i], buf_bytes)`
used the member `buf_bytes`, but `buf_bytes` gets overwritten to the
CUB scratch size (~128 MB) at line 2140 in the OVC compact-upload
setup. So h_pin was being allocated with 128 MB, then the extract
loop wrote past its end → SIGSEGV.

Fix in commit 4741407: compute h_pin alloc size from the same
formula the constructor used:

```c
const uint64_t h_pin_alloc_bytes =
    ((gpu_budget / NBUFS) / RECORD_SIZE) * (uint64_t)RECORD_SIZE;
```

Verified SF300 PASS afterwards.

**This is important**: every SF300+ sort that ran after 1.6.1 would have
crashed. The 1.12 / 1.12.1 results were partial because of this — they
hit a different OOM before reaching the segfault. So the queue-state on
this branch *until 4741407 landed* misrepresented what worked.

## 2. NUMA-pin gather threads — negative result

### Hypothesis

Pin each gather thread to a fixed core. Combined with first-touch page
allocation, output pages would land on the NUMA node where the writing
thread runs → all writes NUMA-local → faster.

### Tried it two ways

**v1 — single-core pinning** (`pin_thread_to_core(t)` for thread `t`):
- SF300 warm gather: 17.6 s (vs 5.2 s unpinned baseline) — **3.4× slower**

**v2 — node-level pinning** (`pin_thread_to_numa_node(t & 1)` →
all even cores or all odd cores, scheduler picks within node):
- SF300 warm gather: 16.5 s — **3.2× slower**

### Why it didn't work

The 216 GB pinned input buffer sits on whichever NUMA node the
*loader* thread ran on (NUMA 0 by default). When I forced gather
threads to NUMA 1, *every read* by those threads hit remote memory at
~2× the latency. The unpinned default lets the OS dynamically balance
threads against where buffers actually live.

To make pinning win, the input AND output buffers would need to be
explicitly partitioned across nodes (e.g., first 108 GB of input on
NUMA 0, last 108 GB on NUMA 1). That's a much bigger refactor — filed
as part of Tier B work.

### What we left in the code

A comment block in `external_sort.cu:36-45` documenting the negative
result so future contributors don't re-try the same approach. No
runtime code change.

## 3. MAP_POPULATE — accidentally re-broken by 1.12.1, restored conditionally

### Discovery

After the segfault fix, SF300 *still* showed gather at 17 s (3× slower
than the pre-1.12.1 baseline of 5.2 s). Bisected to commit cb76cd5
(1.12.1, "drop MAP_POPULATE"). The original MAP_POPULATE was
allocating output pages up-front by the *main* thread on a single NUMA
node; without it, the 192 gather threads each first-touched their own
output range, scattering pages across both NUMA nodes and causing
remote-memory traffic on every cache-line write.

### Fix in commit 76e1e8d

Conditional `MAP_POPULATE` based on host-RAM headroom:

```c
bool use_populate = (input_bytes + output_bytes + 64 GB headroom
                     < host_ram_bytes);
int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS
                 | (use_populate ? MAP_POPULATE : 0);
```

For SF300: input 216 GB + output 216 GB + 64 GB headroom = 496 GB <
1024 GB host RAM → MAP_POPULATE enabled.
For SF1000: 720 + 720 + 64 = 1504 GB > 1024 GB → MAP_POPULATE skipped
(falls back to 1.12.1 lazy behavior).

### Measured result at SF300 warm best

```
                                  gather_ms  gather_GB/s  total_ms
pre-1.6.1 (original 1.2 result):       5232        41.3       10101
after 1.12.1 (no MAP_POPULATE):       17208        12.6       34197
after 76e1e8d (conditional):           4760        45.4       23373
```

Gather phase **9× faster** after the conditional fix vs the broken
state, and slightly faster than the original baseline (45.4 vs 41.3 GB/s).

But the *total* wall time is still 2.3× worse than the original
baseline (23.4 s vs 10.1 s). The increase is in the run-gen phase
(now 17.9 s vs original 4.2 s). Hypothesis: MAP_POPULATE allocates
216 GB output up-front, which competes with the input file's page
cache. On run 2, the input has to be partly re-read from NVMe.

This is a real residual issue. Two paths forward:
- **Reuse output buffer across runs** instead of fresh mmap per run.
  Saves the up-front allocation cost per iteration.
- **Two-pass gather** would eliminate the need for the random-pattern
  reads that benefit from MAP_POPULATE in the first place.

Both are Tier B work.

## 4. 0.3.1 (slow-path compact upload) — deferred

After spending the session on the NUMA / MAP_POPULATE investigation,
0.3.1 is more risky than the 1-2 hour scope I originally estimated.
The slow-path code at line 2536 shares structure with the OVC
fall-through (which OOMs at SF500), so changes here have to be
careful about both paths. Better as a dedicated session.

## Net wins from this session

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| SF300 single-GPU works at all | broken (segfault since 1.6.1) | PASS | ✓ unblocked |
| SF300 warm gather | 17.2 s (broken state) | 4.8 s | **3.6× faster** |
| SF300 warm wall total | 34.2 s (broken state) | 23.4 s | **1.5× faster** |
| SF300 warm wall vs original | 10.1 s | 23.4 s | -2.3× regression vs original baseline |

Current state is *better than the broken state* but *worse than the
original*. The "missing" 13 seconds is in run-gen, suspected to be
input page-cache thrash from MAP_POPULATE. Tier B "two-pass gather"
+ "reuse output buffer" should close the gap and likely improve on
the original.

## Plan revision for Tier B

After this exercise the priority list shifts a bit. Original "Tier B
two-pass gather" stays the biggest single lever. Add to it:

1. **Two-pass gather** (was already #1 in Tier B) — sort perm by
   source location → sequential reads → expected 2-3× gather speedup
   on top of current 4.8 s. Predicted: gather → ~2 s.
2. **Reuse output buffer across `--runs` iterations** — eliminates the
   per-run MAP_POPULATE cost. Predicted: run-gen → 4-5 s instead of 18 s.
3. **NUMA-partitioned buffers** (only if 1+2 don't get us back to <10 s
   wall) — explicit `mbind` on the input and output to split across
   nodes, then re-enable thread pinning.

After Tier B, Tier C (multi-GPU) becomes the headline story for SF500+
where single-GPU is fundamentally constrained by the 94 GB HBM ceiling
regardless of any host-side optimization.

## Commits this session (all on h100/discoveries-2026-05-02, awaiting push)

```
4741407 external_sort: fix 1.6.1 regression that segfaulted SF300 OVC path
76e1e8d external_sort: conditional MAP_POPULATE + document negative NUMA-pin result
```

Plus the queue-tracking commits and BOTTLENECKS.md from earlier in the session.
