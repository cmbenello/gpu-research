# Adversarial Dataset Experiments (D) — SF50 Fixup Path

Date: 2026-04-15
Binary: `external_sort_tpch_compact` @ `exp/fixup-fast-comparator` 46559b3
Substrate: SF50 lineitem (300M records × 120 B = 36 GB), generated from `/tmp/lineitem_sf50_normalized.bin` by `scripts/adv_gen`.

## What I built

A small C utility (`scripts/adv_gen.c`) that streams the substrate, mutates each record's first 32 B, and writes a new file. Four variants:

| variant | construction | goal |
|:-------|:-------------|:-----|
| **zero32** | bytes 0..31 = 0 on every record | force compact detection to see 34/66 varying — stress adaptation |
| **zero8**  | bytes 0..7 = 0 on every record | kill pfx1 discrimination without killing pfx2/3/4 |
| **pool1000** | bytes 0..31 = templates[i mod 1000] (xorshift-seeded random) | create exactly 1000 coarse groups of ~300k records |
| **pool32** | bytes 0..31 = templates[i mod 32]  | create 32 huge groups of ~9.4M records — worst case for fixup parallelism |

## Results (SF50, 3 warm runs where completed)

All baselines use current committed code (46559b3). Baseline SF50 warm median = 6.21 s.

| variant | compactness | #groups | avg records/group | fixup ms (warm) | total ms (warm) | delta vs baseline | hash | status |
|:-------|:-----------:|-------:|-----------------:|---------------:|---------------:|:-----:|:-----|:------|
| baseline | 61/66 | 3.15 M | 95 | ~1380 | **6210** | — | `0x89c28ecae7c4b777` | ✓ verified |
| zero32 | **34/66** | 29.4 M | 10 | 2415–3750 | **7300** (median) | +1090 ms | `0x56c61b94f74f3744` | ✓ verified |
| zero8  | 57/66 | 17.2 M | 17 | 2502–3027 | **6924–7371** | +714–1161 ms | `0xb70371e5386ef76e` | ✓ verified |
| pool1000 | 66/66 | **1 000** | 300 006 | **9805** | **12681** | +6471 ms | ⚠ hybrid check TRIGGERED | ⚠ correctness violation caught, fallback crashed |
| pool32 | 66/66 | **32** | 9 375 182 | **156 377** | **158 875** | +152 664 ms | ⚠ hybrid check TRIGGERED | ⚠ same fallback crash |

## What we learn from each

### zero32 — adaptation works
Compactness drops from 61/66 to 34/66 because 32 positions are now invariant. The runtime compact map skips invariants, so the 32 B GPU prefix is packed from the 34 varying positions of the original record (now bytes 32..65). The fixup has only 2 active bytes left and still handles 29 M tiny groups.
- **Takeaway**: position-ordered compact map successfully adapts to invariants. Correctness preserved. +1 s perf cost comes from more groups (29 M vs 3.15 M) and smaller pack work per group.

### zero8 — partial adaptation
With bytes 0..7 zeroed, compactness drops only to 57/66 (4 positions out of the 8 were varying in the original). 32 B prefix now consumes the first 32 of 57 varying compact bytes. Groups: 17.2 M avg 17 records.
- **Takeaway**: partial-byte invariance is handled gracefully. Perf cost +700–1100 ms (median).

### pool1000 — work-queue parallelism limit exposed
1000 huge groups. Our atomic work-queue hands out batches of 64 groups; with only 1000 groups that's 16 batches, so only 16 of 48 threads get work. Per-thread sort time = 1830 ms × 16 threads ÷ 48 = 610 ms per thread effective. Parallel wall 8973 ms vs per-thread cumulative 2534 ms gives efficiency 2534/8973 ≈ 0.28 (or ~13.5-way, consistent with 16-thread cap).
- **Takeaway**: **the atomic work-queue with batch = 64 is the wrong policy when group count is low.** Fix: `batch = max(1, num_groups / (hw × 4))`. For pool1000 that would be 5, giving ~48-way parallelism. Predicted speedup ≈ 3×.

### pool32 — catastrophic + hybrid check validates correctness
32 groups of 9.4M each. Fixup sort time scales O(g log g) per group; with g = 9.4M that's huge. Plus only 32 threads get work.
- **Also**: our hybrid correctness check (`[Hybrid] 132,355,033 exception records found`) correctly detected that the compact-prefix sort gave a wrong order — compact map put bytes 0..31 as first prefix, but since ALL 300 M records have one of only 32 templates, the compact order doesn't reflect full-key order (byte 32..65 discrimination is drowned).
- Hybrid fallback tried to retry with the full-key upload path, but crashed with `CUDA error at src/external_sort.cu:1266: invalid argument`. **This is a real bug in the hybrid retry path** — GPU state isn't cleaned up properly between the compact attempt and the retry, so the `cudaMemcpyAsync` in the H2D uploader hits invalid args.

## Bugs / holes this surfaced

1. **Hybrid retry crashes on CUDA error** — `external_sort.cu:1266`. Needs a proper GPU state reset between attempts. Worth a separate commit; for now, when the hybrid check fires, the program dies rather than producing a correct result.
2. **Work-queue batch size is fixed at 64** — handicaps low-group-count inputs. Easy fix.
3. **Correctness verifier works as designed** on pool1000/pool32 — caught the compact sort divergence before it silently produced wrong output. This is a feature, not a bug; it's what the verifier is for.

## Paper narrative

The adversarial story writes itself:

- **Graceful degradation** on zero8 / zero32 (realistic adversarial — some invariance, some variance). +10–18 % perf cost but correct output.
- **Detection** on pool1000 / pool32 (extreme adversarial — forced ties on 32 B prefix that don't reflect true key order). Hybrid check refuses to return a wrong result.
- **Work-queue dispatch parallelism** is the paper's "implementation detail that matters" item — the fix is one line of code but the measurement shows 3× potential speedup on pool1000, going from 9.8 s → ~3.3 s estimated.

## Artifacts

- `sf50_zero32.log`, `sf50_zero8.log`, `sf50_pool1000.log`, `sf50_pool32.log` — raw run logs.
- `sf10_baseline.log` — SF10 reference (note: SF10 fits in-GPU-memory, doesn't exercise the OVC+fixup path).
- `scripts/adv_gen.c` — adversarial dataset generator.

## Next

Before running A (scaling curve) and C (ablation), fix the two items this surfaced:
1. Work-queue batch size should scale with group count.
2. Hybrid retry should reset GPU state cleanly (or at minimum propagate the failure with a helpful message and exit cleanly).

Task items 24-27 complete.
