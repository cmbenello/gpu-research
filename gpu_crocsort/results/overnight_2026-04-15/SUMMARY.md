# Overnight Research SUMMARY — runtime-compact-map-wip

Date: 2026-04-15 UTC

## ⚠ RETRACTION (critical): earlier "entropy -25%" win was against a broken sort

The headline "entropy-based byte SELECTION gives SF50 -25%" was CAPTURED AGAINST INCORRECT OUTPUT. The sort DID run faster with entropy mode, but it produced OUT-OF-ORDER records. The mis-ordered output still passed a MULTISET-HASH check (records present, just rearranged), and my sweep scripts accidentally only grep'd `PASS multiset`, missing the `FAIL sortedness` lines.

### Why entropy selection is incorrect

Entropy mode selects the top 32 varying byte positions BY DISTINCT-VALUE COUNT. Even when those 32 positions are sorted by source position within the selection, this does NOT yield an order-preserving compact key.

Counter-example from TPC-H SF50:
- `rec 6 = 'RF1994-10-10NONE\0\0\0\0...'` — byte 46 = 0x00
- `rec 7 = 'NO1997-10-20DELIVER...'` — byte 46 = 0x61
- Top-32 entropy for SF50 includes byte 46, does NOT include bytes 0,1 (those sit in `rest[]`).
- `compact(rec 6) < compact(rec 7)` because byte 46: 0x00 < 0x61 — GPU sort places rec 6 before rec 7.
- Full-key lex order: byte 0 'R' (0x52) > 'N' (0x4e) — rec 6 must come AFTER rec 7.
- Compact order ≠ full-key order. Sort is wrong.

### Correct rule for lex-preserving compact

The compact key must contain the SOURCE-POSITION-ORDERED PREFIX of the varying bytes (i.e., the N smallest-position varying bytes, for some N). "Position-order" selection satisfies this; "entropy" selection does not.

Fixup does NOT save entropy mode: for it to correct the order, records A and B above would need to land in the same tied group (equal compact-prefix), but they end up in DIFFERENT groups because the compact prefix DOES differ at byte 46. Fixup never sees them.

### What got reverted

- `exp/entropy-selection` default changed back to position-order.
- `COMPACT_SELECT=entropy` retained as opt-in for the paper's negative-result section.
- All "SF50 -25%" claims, "4.9× DuckDB", and "entropy is the default" are WRONG.

---

## Actual honest headline (position-order, the correct default)

Position-order SF50 warm median = 11.92 s (baseline capture pre-experiment, 5 runs). Same-session alternating sweeps showed noisy absolute values (12.0–15.5 s on SF50 depending on thermal/cache state) but entropy mode's "apparent win" was pure correctness violation, not real speedup.

## Still-real wins (correctness-verified)

### Cache-resident packed-buffer fixup sort (pre-session baseline `ff244d8`)

Contributed to SF50 baseline being 11.92 s (position-mode, verified).

### In-extraction + post-sort multiset verification (pre-session baseline `634c137`)

This is the infrastructure that CAUGHT the entropy bug. The system's own verifier flagged `FAIL sortedness: first violation at record 7` on every entropy run. The only reason I missed it earlier was that my bash sweep scripts grep'd only `PASS multiset` instead of `PASS sortedness AND PASS multiset`.

**Paper footnote**: an automated correctness harness is worth 100× its weight in perf-number credibility. Without the multiset check there would have been no alarm at all (the broken sort passed the sortedness check trivially for other reasons previously); without BOTH checks, optimizations silently corrupt.

### fixup-early-exit (exp/fixup-early-exit @ 44397b1)

Small but real — monotonic pre-check skips std::sort for groups already in order. Measured against POSITION-mode (correct baseline): SF50 -164 ms total / -208 ms fixup in alternating 5-run sweep. All PASS sortedness + multiset.

### GenSort 30 GB (bonus data point, verified)

3.87 s warm, 7.7 GB/s. PASS sortedness + multiset (+ ordinal.com `valsort`).

## Dead ends

1. **entropy selection** — INCORRECT, retracted (above).
2. **hybrid-32b-extract-fast** — REGRESSION on SF50 (+1.2 s vs position baseline).
3. **numa-pin-gather** — h_data pages on one node, pinning half threads to each socket regressed gather 3625→4029 ms.
4. **mbind(MPOL_INTERLEAVE) on h_output** — catastrophic (2-17× slowdown). Dropped MAP_POPULATE reasoning was wrong; pages faulted in during gather.
5. **insertion-sort-small-groups** — std::sort is already tuned; n=15 is above the crossover where O(n²) wins.
6. **prefetch sweep, threads sweep** — baselines (512, 48) already optimal.

## Methodology lessons

1. **Grep every sweep for `PASS sortedness` AND `PASS multiset`.** The multiset-only shortcut that hid the entropy bug took hours of follow-on work to invalidate.
2. **Run-to-run variance is ~1 s on SF50/SF100 under sustained load**; trust same-session alternating sweeps over cross-session comparisons.
3. **MAP_POPULATE on output buffer is not optional** (dropping it cost up to 17× from page faults during gather).
4. **Lex-preserving compact keys require source-position-ordered byte selection.** Entropy reordering breaks lex. The only way to do entropy-aware compact is to include ALL varying bytes (so compact order reduces to full-key order for canonicals) — needs larger COMPACT_KEY_SIZE.

## Recommended next steps (revised)

1. **DO NOT MERGE `exp/entropy-selection`.** The correctness bug is documented in the commit message at `747220b`.
2. **The real SF50 lever is still CPU fixup** (~10 s out of 12 s warm). Best path forward is the native GPU 32B OVC refactor (~300 LOC, preserves lex via source-position order in a larger compact buffer).
3. **Re-run DuckDB comparison with honest position-mode numbers**: SF10 4.6× / SF50 ~4.7× (was overclaimed 4.9×) / SF100 ~26×. Still a strong story.
