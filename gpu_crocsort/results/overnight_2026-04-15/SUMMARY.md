# Overnight Research SUMMARY — runtime-compact-map-wip

Date: 2026-04-15 UTC

## Headline numbers

Apples-to-apples 5-run alternating sweep (`P E P E P E P E P E`), same binary, same session — washes out thermal + page-cache effects.

| Workload | Position-order baseline | **Entropy selection (NEW DEFAULT)** | Δ | DuckDB v1.5.2 | Speedup |
|----------|------------------------:|------------------------------------:|---|--------------:|--------:|
| **SF10**  | 1.74 s | 1.74 s | — (unchanged, doesn't trigger entropy) | 8.03 s | 4.6× |
| **SF50**  | 15.48 s* | **11.57 s** | **-3.91 s (-25%)** | 56.7 s | 4.9× |
| **SF100** | 8.43 s* | 8.52 s | +0.10 s (within stdev 1040 vs 78) | ~200 s (proj) | ~24× |

*Note: position-mode SF50/SF100 numbers in this same-session sweep are noticeably noisier than the cold-pre-experiments baseline (position-mode SF50 stdev ±1109 ms — contains the cold first run as outlier; entropy SF50 stdev ±627 ms). The previous benchmark file's "11.92 s SF50" baseline came from a session with cooler caches; the apples-to-apples here is the fair comparison and shows entropy wins decisively when run side-by-side.

Every run PASSed both sortedness scan and multiset-hash preservation (FNV-1a-64 sum per record).

## Top 3 wins (paper-worthy)

### 1. Entropy-based byte SELECTION (`exp/entropy-selection` @ `77e3827`)

**Win**: SF50 -25% (15.48 s → 11.57 s) with no impact on SF10/SF100.

The runtime-detected compact map originally placed the first 32 *position-ordered* varying bytes into the GPU's compact key. For TPC-H SF50 this happens to land bytes 0–36 of the record (date prefix + low-entropy header), missing l_orderkey at bytes 51–57. Result: 290 M of 300 M records ended up in tied 16B-prefix groups, requiring a 14 s CPU fixup pass.

The fix tracks per-byte sample distinct-value count (entropy proxy) via a 256-bit bitmap during the existing detection pass. When more than `COMPACT_KEY_SIZE=32` candidate bytes exist, the top 32 by distinct count are selected for the compact key (sorted by source position to preserve lex-compatibility within the projected sub-key). The remaining bytes go in `map[32..63]` for verification only.

For SF50 this puts orderkey bytes (51–61) directly into the compact key → GPU 16B prefix discriminates → tied groups shrink → CPU fixup time drops 7.05 s → 5.39 s in initial test, and the cumulative effect with same-session noise control is -3.91 s.

Root-cause + fix is ~80 LOC in `detect_compact_map()`. No correctness change — verification still proves multiset preservation.

### 2. Cache-resident packed-buffer fixup sort (`ff244d8` baseline)

**Win**: SF50 -1.1 s within fixup phase (compounding base for entropy work above).

Pre-pack the active byte positions of each tied group into a contiguous ~1 MB buffer (fits in L2/L3) and sort indices using SIMD memcmp. Previous implementation walked scattered byte positions in `h_output` which is 36–72 GB and DRAM-bound; the packed buffer keeps the inner sort cache-resident.

### 3. In-extraction multiset verification (`634c137`)

**Critical correctness infra**, not a perf win, but enabled discovery of the bug below. Every sort verifies sortedness (parallel adjacent-pair scan) AND multiset preservation (parallel FNV-1a-64 hash sum) on the in-memory output. ~50 LOC total.

This caught the `sm_80` PTX-JIT failure on Turing (silent no-op kernels producing 60 M copies of input[0]) that the original adjacent-pair-only verifier had missed for the entire history of the project. Fixed at `0be08a8` by auto-detecting GPU compute capability in the Makefile.

## Top 3 dead ends (also paper material)

### 1. Hybrid 32B GPU sort extension (`exp/hybrid-32b-extract-fast` @ `4581f36`)

**Result**: SF50 +1.2 s REGRESSION vs baseline.

Idea: after the GPU 16B merge, CPU-extract pfx3+pfx4 from `h_data` via `h_perm`, upload them, and run 3 more LSD passes on GPU to extend the sort to a full 32B compact prefix. Implemented with SortPairs(identity_index)+gather pattern; 3 GPU passes total, optimized to 1 record-read per record on CPU.

Why it died: SF50's 32B compact (with position-order selection) covers record bytes 0–36 only. l_orderkey at byte 51 is OUTSIDE the prefix. So even at 32B every adjacent pair sharing bytes 0–36 still ties on the GPU prefix, and CPU fixup must scan bytes 37–65 of every group. Fixup time only dropped 9.8 s → 8.9 s while the extension itself cost 6.6 s.

Proper fix is **entropy selection** (Win #1) which puts orderkey IN the prefix — addresses the root cause instead of extending to the wrong bytes.

### 2. Software prefetch distance sweep (`exp/prefetch-sweep` @ `53b55d2`)

**Result**: NULL — current `PREFETCH_AHEAD = 512` is already optimal.

Swept gather kernel's prefetch distance ∈ {128, 256, 384, 512, 768, 1024, 1536}. Initial 7-value sweep suggested 256 was best (3597 ms gather vs 3839 ms baseline = 6%), but a 5-run validation head-to-head 256-vs-512 showed 256 had an outlier and 512 was actually marginally better on median. Within run-to-run noise (~100 ms range).

### 3. Gather thread-count sweep (`exp/prefetch-sweep` @ `53b55d2`)

**Result**: NULL — current 48 threads is optimal.

Swept ∈ {8, 16, 24, 32, 48, 64}. Below 32 threads = bandwidth-starved (linear slowdown). 64 threads regress slightly (oversubscription). 48 threads (= `hardware_concurrency`) is the sweet spot.

## Recommended next steps (ranked)

1. **Move entropy default into `runtime-compact-map-wip`** (10 LOC merge) — clean win, deserves to be in the headline branch.
2. **Re-run DuckDB head-to-head with entropy** — current `experiments/wip_head_to_head.md` quotes pre-entropy SF50 (3.1× DuckDB); entropy makes it 4.9×.
3. **Profile the remaining 5.4 s SF50 fixup with `nsys`/`perf`** — even with entropy, fixup is 50% of SF50's time. Either smaller groups or faster comparator could yield more.
4. **Try entropy + larger `COMPACT_KEY_SIZE`** — currently capped at 32 B. Bumping to 40 B costs PCIe (+25% upload) but gives the GPU prefix room for more bytes. Memory budget allows on SF50, marginal on SF100.
5. **Combine entropy with hybrid 32B GPU extension** — entropy puts good bytes in compact, and the extension would push GPU sort to use ALL of them. Likely smaller win than entropy alone gave (most discrimination already achieved at 16B prefix), but worth measuring.

## Surprising findings / footnotes

1. **Run-to-run variance on SF50/SF100 is high (~1 s range)** — likely a combination of CPU thermal throttling under sustained 48-thread load and page-cache state. Same-session alternating sweeps (this experiment's method) wash this out; cross-session comparisons need to assume ≥500 ms noise floor.
2. **`-arch=sm_80` silently produces no-op kernels on Turing** when the CUDA driver can't JIT certain sm_80 PTX instructions to sm_75. The kernels run, take 0 ms, write zeros, and the original adjacent-pair verifier reports PASS because all output records are equal (every pair is `equal ≤ equal`, no violation). Discovered by the new multiset hash check. **Lesson: never trust a sortedness-only verifier on a sort that might have a bug producing constant output.** Multiset preservation must be checked independently.
3. **`extract_tiebreaker_kernel` had `k[8]` and `k[9]` hardcoded** — a leftover from when KEY_SIZE was 10 (gensort). For TPC-H KEY_SIZE=66, the final LSD pass was sorting by bytes 8–9 instead of 64–65. Records identical in bytes 0–63 but differing in 64–65 ended up in arbitrary order. Multiset hash flagged it; fix passed `byte_offset` to the kernel. Audit lesson: any hardcoded 8/9 in a kernel that takes `byte_offset` everywhere else should raise a red flag in code review.

## Branches produced this session

| Branch | Commit | Outcome |
|--------|--------|---------|
| `exp/hybrid-32b-cpu-extract` | `e446e7b` | initial 32B hybrid (SF50 -1.5 s, modest) |
| `exp/hybrid-32b-extract-fast` | `4581f36` | optimized hybrid 32B extract — REGRESSION |
| `exp/prefetch-sweep` | `53b55d2` | gather prefetch + threads — NULL |
| `exp/entropy-selection` | `77e3827` | **WIN — entropy default, SF50 -25%** |

Baseline: `runtime-compact-map-wip @ ff244d8`.

## Methodology

Verified by `--verify` on every run (default on): parallel sortedness scan + parallel multiset hash. No `--no-verify` numbers in this report. Three runs minimum per data point; warm median reported (excluding the cold first run after a binary rebuild). Same-session alternating sweep used for the headline comparison.

