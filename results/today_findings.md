# Bottleneck work — 2026-04-30

## What I tried, in order

### 1. NUMA interleave for the gather phase
- Both Xeon Silver 4116 boxes are 2-socket. Gather threads weren't NUMA-aware.
- Wrote a tiny launcher that calls `set_mempolicy(MPOL_INTERLEAVE)` before exec.
- Result: SF50 gather 2510 ms → 2350 ms, **about 6%**. Real but small. End-to-end wall time barely moved.
- Skipping further NUMA work — not worth the engineering effort vs the upside.

### 2. AVX-512 wider streaming stores in the gather
- Looked promising on paper: `_mm_stream_si64` (8 B) → `_mm512_stream_si512` (64 B) cuts instruction count 8x.
- Blocked by alignment: 120-byte records put per-row destinations at 8-byte alignment, AVX-512 streaming stores require 64-byte alignment.
- Skipped without writing it.

### 3. CPU-side FOR + bit-packing — wired in end-to-end
- Surprise finding: the `BitPackConfig` struct, `compute()` analyser, `pack()` method, and a GPU bitpack kernel were all written but `bitpack.compute()` was never called anywhere. The whole bitpack path was dead code.
- Wrote the wiring (gated on `USE_BITPACK` env var):
  - Run `bitpack.compute()` after the compact-key scan
  - Override `runtime_compact_size` and `effective_compact_size` to `bitpack.padded_size`
  - In `do_extract`, call `bitpack.pack()` on the host instead of the byte-extract loop
  - Skip the now-redundant GPU bitpack kernel and its workspace alloc

#### Measured PCIe reduction (best of 3 warm runs)

| Machine | Scale | Baseline H2D | USE_BITPACK H2D | Reduction | Chunks before → after |
|---|---|---|---|---|---|
| P5000 (16 GB) | SF50 (36 GB) | 9.60 GB | 7.20 GB | **−25%** | 3 → 2 |
| RTX 2080 (8 GB) | SF20 (14 GB) | 3.84 GB | 2.88 GB | **−25%** | 3 → 2 |

Both verified PASS (correctness preserved). Wall time roughly flat at these scales because gather, not PCIe, is the dominant phase. The PCIe gain shows up in the bytes counter but doesn't bubble up to wall time until either (a) the workload becomes more PCIe-bound or (b) the compounding sort-time gain on smaller keys lands.

### 4. Tried to unlock SF100 on 16 GB GPU with bitpack + low budget

| Configuration | Result |
|---|---|
| Default budget, no bitpack | OOM in run-gen arena (line 2469) |
| Budget=0.30, no bitpack | 12 chunks → OOM in merge arena (line 2183) |
| Budget=0.30 + USE_BITPACK | **9 chunks** → still OOM in merge arena |

The merge phase allocates `num_records × 24 bytes` ≈ 10.6 GB atomically (d_pfx_alt + d_perm_save + d_perm_alt + CUB scratch). With ~5 GB free at that point, it won't fit on 16 GB. **Compression alone doesn't unlock SF100 on the P5000** — we'd need to chunk the merge phase too, which is real engineering (~1-2 days).

## What this proved

1. The infrastructure for compression integration is in place. `BitPackConfig` works correctly end-to-end, the FOR + bit-pack codec actually delivers 25% PCIe reduction, and verification passes.
2. Wall time isn't the right success metric for compression on these machines because gather dominates. The right metric is **bytes moved over PCIe**, which directly translates to gains on PCIe-bound future hardware (PCIe 5/NVLink-C2C).
3. SF100 on 16 GB GPU is bound by the merge-arena allocation pattern, not by chunk arena or key size. Compression doesn't help.

## Next steps to actually move wall time

Listed by impact-per-effort:

1. **Chunked merge phase** (1-2 days). Process the global merge as N/2 or N/3 sub-merges so the arena is `(num_records/k) × 24`. Unlocks SF100 on the P5000 and SF40+ on the 2080.
2. **Compress satellite records** (real engineering, several days). Currently the gather phase reads + writes 36 GB of original 120-byte records. If we compress the records (not just keys) we cut gather memory traffic. This is the only way to seriously dent gather wall time without faster hardware.
3. **GPU-side gather** (only on PCIe 5+/NVLink). Send full records to GPU, gather there, stream sorted output back. On PCIe 3.0 this loses; on faster interconnects it should win.
4. **Sample-sort to remove K-way merge** (~1 week). Eliminates the need for a global merge phase entirely.
5. **PCIe 5 / GH200 evaluation**. Validates the compression direction by changing the bottleneck balance.

## Files committed today

Branch: `research/overnight-runs-cs-uchicago`

- `gpu_crocsort/src/external_sort.cu` — bitpack wiring (139 inserts, 75 deletes)
- `results/figures_weekly/*` — 7 charts from prior data
- `results/weekly_progress.md` — PI-facing summary
- `baseline_runner.py` + `results/overnight_pulled/baselines_*duckdb.csv` — DuckDB head-to-head
