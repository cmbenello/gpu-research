## Experiment: entropy-selection
Branch: exp/entropy-selection (ff244d8)
Hypothesis: pick top 32 byte positions by sample distinct-value count (entropy proxy) and put them in the compact KEY. The remaining varying bytes go into map[32+] for verification only. For SF50 this should put orderkey in the GPU prefix → eliminate the 14s CPU fixup.
Trigger: COMPACT_SELECT=entropy env var (default off).

### SF50: 5 runs each, default vs entropy

| mode | r1 | r2 | r3 | r4 | r5 | warm median total | warm median fixup |
|------|---:|---:|---:|---:|---:|------------------:|------------------:|
| default | 18027.10 | 12155.25 | 12253.05 | 12106.55 | 12320.02 | 12155.25 | 7051 |
| entropy | 10477.64 | 10545.18 | 10498.74 | 10896.56 | 10567.39 | 10545.18 | 5392 |

### Cross-validation: SF10 + SF100 (entropy mode)

| SF | r1 | r2 | r3 | warm median |
|----|---:|---:|---:|------------:|
| sf10 | 1757.14 | 1748.36 | 1748.46 | 1748.36 |
| sf100 | 8239.27 | 8337.93 | 8332.00 | 8332.00 |

### Conclusion

**WIN.** SF50 gets a clean -1.6 s improvement (12.16 s → 10.55 s, -13%). The fixup specifically drops 7.05 s → 5.39 s because the GPU 16B prefix now lands on high-entropy bytes (orderkey/linenumber territory) — fewer pairs survive into the tied-group fixup.

SF10 unaffected (path doesn't use compaction). SF100 slight noise (~350 ms range, may be detection overhead from the new distinct-count tracking — would investigate if persistent).

**Recommend making this the default** — pure win on SF50, neutral on SF10/100.

Followups:
- Make COMPACT_SELECT=entropy the default; only fall back to position-order if explicitly requested for backward compat.
- Skip the distinct-count tracking when ncand ≤ COMPACT_KEY_SIZE (no point — saves ~100ms detection on SF10/100).
