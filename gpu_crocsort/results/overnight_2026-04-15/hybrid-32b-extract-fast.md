## Experiment: hybrid-32b-extract-fast
Branch: exp/hybrid-32b-extract-fast (built from exp/hybrid-32b-cpu-extract)
Hypothesis: original hybrid 32B had 2.5s+1.7s of CPU extract overhead. By extracting all 4 prefixes in one pass and using GPU gather (instead of CPU re-extract for pfx3), overhead drops by ~2-3s. Combined with smaller fixup groups (32B prefix vs 16B), should make SF50 faster than baseline.

### SF50 5-run results (verified, all PASS sortedness + multiset)

| run | total ms | extension ms | fixup ms |
|----:|---------:|-------------:|---------:|
| 1 (cold) | 17502 | 12436 | 12958 |
| 2 | 13332 | 6598 | 9076 |
| 3 | 13109 | 6618 | 8905 |
| 4 | 13135 | 6666 | 8906 |
| 5 | 13064 | 6599 | 8812 |

Warm median total: 13109 ms.

### Baseline comparison

| | baseline (no extension) | optimized hybrid 32B | delta |
|---|---:|---:|---:|
| SF50 warm median total | 11924 ms | 13109 ms | **+1185 ms (-9.9% — REGRESSION)** |
| SF100 warm median | 7981 ms | 8110 ms (extension skipped, no ties) | within noise |
| SF10 warm median | 1755 ms | 1773 ms (full-key path, no extension) | within noise |

### Root cause of regression

The 32B compact prefix on SF50 covers record bytes 0-36 of the 66-byte key. l_orderkey lives at bytes 51-57 — NOT in the compact prefix. So even after the GPU sorts to 32B, every adjacent record pair sharing bytes 0-36 still ties on the GPU prefix, and CPU fixup must compare bytes 37-65 of every group. Fixup time only drops 9.8s → 8.9s (saves ~10%) because the discriminating bytes are outside the 32B window.

Meanwhile the extension itself costs 6.6s (extract 2.1s + upload 0.7s + GPU work 0.4s + ... + segment scan 0.15s). Net: -1.2s on SF50.

### Conclusion

**NEGATIVE RESULT.** Confirms that simply extending the GPU sort prefix is not the answer for datasets where the discriminating bytes fall outside the source-order-first-32 window. The right optimization is entropy-based byte SELECTION (Task 8): pick which 32 of N varying bytes go in the compact map, prioritizing high-variance bytes. For TPC-H lineitem this should select orderkey/linenumber bytes, putting them in the compact map → GPU prefix discriminates correctly → fixup minimal.

