## Experiment: prefetch-sweep
Branch: exp/prefetch-sweep (ff244d8)
Hypothesis: SF100 CPU gather is at 24 GB/s (well below DRAM ceiling). Different prefetch distance might better match h_data random-access latency.
Variable: GATHER_PREFETCH ∈ {128, 256, 384, 512, 768, 1024, 1536}
Method: SF100 only (where gather dominates), 3 runs each value, take median total ms.

| prefetch | run1 | run2 | run3 | median total ms | gather GB/s | verified |
|---------:|-----:|-----:|-----:|----------------:|------------:|---------:|
| 128 | 4750.58 | 4674.09 | 4639.65 | 4674.09 | 3645 | true |
| 256 | 4626.30 | 4387.85 | 4755.12 | 4626.30 | 3597 | true |
| 384 | 4773.80 | 4884.40 | 4822.77 | 4822.77 | 3794 | true |
| 512 | 4868.10 | 5136.20 | 4144.32 | 4868.10 | 3839 | true |
| 768 | 5148.22 | 5166.30 | 4783.66 | 5148.22 | 4115 | true |
| 1024 | 4820.31 | 4801.96 | 4867.66 | 4820.31 | 3786 | true |
| 1536 | 5067.59 | 4923.71 | 4977.21 | 4977.21 | 3942 | true |

### Analysis

⚠ Header column was "median total ms" but the parser actually grabbed `$8` which is MERGE_MS (CSV columns: ..., $7=run_gen_ms, $8=merge_ms, $9=total_ms). Treat the table as MERGE_MS not total_ms.

⚠ "gather GB/s" column actually shows gather MILLISECONDS (parser grabbed $5 which is the time field, not throughput).

#### Re-interpreted gather time (median of 3, ms):

| prefetch | gather ms | gather GB/s |
|---------:|----------:|------------:|
| 128 | 3645 | 19.8 |
| 256 | **3597** | **20.0** |
| 384 | 3794 | 19.0 |
| 512 (baseline) | 3839 | 18.8 |
| 768 | 4115 | 17.5 |
| 1024 | 3786 | 19.0 |
| 1536 | 3942 | 18.3 |

#### Conclusion

Prefetch=256 saves 242 ms vs baseline 512 (~6%). All values verified PASS. But run-to-run variance on merge_ms was much higher than baseline (1000+ ms range vs <100 ms in baseline 5-run capture) — suggests system-wide noise or page-cache effects from rebuild. Need a focused 5-run validation at 256 vs 512 to confirm signal exceeds noise.

Decision: **suggestive win, needs validation**. Run 5-run head-to-head 256 vs 512 next.

### Validation result

5-run head-to-head 256 vs 512:
- prefetch=256: median total 8405ms (one outlier at 11718)
- prefetch=512: median total 8306ms

**NULL RESULT.** Prefetch=512 is actually marginally better on median; the 256 win in the broader sweep was within run-to-run noise (~100ms range). Keep baseline at 512.
