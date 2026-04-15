# Why SF50 is slower than SF100 — and why current behavior is adversarial-vulnerable

## The paradox
SF100 (600 M records, 72 GB) sorts in **7.98 s**.
SF50  (300 M records, 36 GB) sorts in **11.9 s** — slower, despite half the data.

## Root cause

The GPU compact-key merge sorts by the **first 16 bytes** of a compact key. The compact key is built from runtime-detected varying byte positions, in source-position order. For every record pair, the GPU assigns order based on those 16 bytes; any pair tied on all 16 needs a CPU fixup that re-sorts by the full 66-byte record key.

For TPC-H lineitem the sample detects which byte positions actually vary:

- **SF100**: 27 varying bytes. First 16 of those (positions `0,1,4,5,8,9,12,13,19,20,21,29,37,44,45,50`) span record bytes 0–50 and happen to include byte 50 (l_orderkey territory → high-entropy). Almost every adjacent pair differs at the GPU level → CPU fixup **skipped entirely**.
- **SF50**: 61 varying bytes. First 16 by source position are `0,1,5,7,8,10,11,12,13,14,15,16,17,18,19,20` — clustered at positions 0–20 (low-entropy date prefix). 290 M of 300 M records end up tied on that prefix → 10+ s of CPU fixup.

It's not that SF50 is harder to sort. It's that SF50's **first 16 varying bytes happen to not discriminate** while SF100's do. Pure luck of byte layout.

## The adversarial case

A dataset where the first 16 varying byte positions are entirely constant or near-constant would force the ENTIRE input into one tied group. CPU fixup would then = std::sort on N records = O(N log N) × full-key-compare. For N=600 M at 50 ns/compare, that's ~8.7 B compares × 50 ns = 435 s serial, ~9 s with 48 threads (memory-bandwidth-bound).

The current sort is **correctness-correct on any input** (position-order selection guarantees compact-lex = full-key-lex for canonicals), but its **performance varies 10×** based on byte layout. That's not robust.

## Why entropy-order doesn't fix it (retracted from earlier in the session)

Picking the 16 highest-entropy varying bytes for the GPU prefix sounds intuitive but produces wrong output. If two records differ at byte B (not in the selected high-entropy set) AND byte P > B (in the set), compact decides at P while full-key decides at B. They disagree. Fixup can't save it because the records land in DIFFERENT tied groups.

Counter-example: rec 6 = `RF1994-10-10NONE\0...`, rec 7 = `NO1997-10-20DELIVER...`. Byte 46 (selected) puts rec 6 first. Byte 0 (not selected) says rec 7 first. Sort is wrong.

## What WOULD fix it

### Option A (recommended): native GPU 32B OVC refactor
- Extract pfx3 + pfx4 (compact bytes 16–31) during run-gen, in the same kernel that extracts pfx1/pfx2. Zero extra CPU work, ~8 GB extra GPU memory on SF50 (fits; SF100 wouldn't need it).
- Extend GPU 16B merge to 4-pass LSD on pfx1..pfx4 = 32B.
- Preserves correctness (source-position order, just more bytes in prefix).
- SF50 prefix now covers record bytes 0–36, including more potential discriminators. Not bulletproof against truly adversarial data, but catches realistic cases like SF50 where the discriminating bytes are at positions 20–36.
- Effort: ~200 lines careful refactor of run_generation.cu + OVC merge path.

### Option B (adversarial-proof, slower): sample-sort partition before merge
- Sample N records, build bucket boundaries based on their actual values.
- Route each record to a bucket by FULL-KEY comparison (CPU cost proportional to N × depth).
- Sort buckets independently on GPU.
- No tied-group-on-compact problem — each bucket is genuinely small and its records definitely discriminate on the bucket's byte range.
- Effort: ~400 lines new code; significantly different architecture.

### Option C (honest workaround): fail gracefully on adversarial data
- Detect tied-group density post-GPU-merge (we already have h_has_ties).
- If a single group contains > X% of records, abort and retry with a different strategy (e.g., CPU parallel radix sort).
- Predictable: for adversarial data, pay a reliable CPU sort cost instead of unknown fixup time.
- Effort: ~100 lines. Least elegant but quickest.

## Why we can't just use single-pass full-key sort

Quadro RTX 6000 has 25.4 GB HBM. SF50's full 66-byte keys = 19.8 GB. Plus CUB scratch (~8 GB) + perm + triple buffers = 35+ GB. OOMs immediately. Tested and confirmed on `exp/force-single-pass`.

This path works for SF10 only (4 GB keys fit). For SF50 and larger, we MUST compact or partition. Which means we're stuck with the prefix-discrimination problem on this hardware.

## Current honest numbers on this branch

| Scale | Method | Time | Robustness |
|------:|--------|-----:|------------|
| SF10  | full-key strided DMA | 1.74 s | predictable (PCIe-bound) |
| SF50  | OVC + compact (16B prefix) | 11.9 s | **vulnerable — first 16 bytes happen to not discriminate** |
| SF100 | OVC + compact (16B prefix) | 7.98 s | lucky — first 16 bytes discriminate |
| GenSort 10 GB | full-key strided DMA | 1.63 s | predictable |
| GenSort 30 GB | full-key strided DMA | 3.87 s | predictable |

All verified PASS sortedness + multiset.
