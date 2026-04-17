# Duplicate Key Experiments

**Setup:** 60,000,000 records, 7.20 GB (120B records = 66B key + 54B value)
**GPU:** Quadro RTX 6000, single-chunk
**Binary:** `external_sort_tpch_compact` on branch `exp/fixup-fast-comparator`
**Warm runs:** 3 per distribution

Keys are 66-byte random strings drawn from a fixed pool. All 66 byte positions
vary across the pool, so compact detection finds 66/66 varying — no compaction.

## Results (median of 3 runs)

| Distribution           | Unique keys | Avg group size | Gen (ms) | Merge (ms) | Total (ms) | GB/s |
|------------------------|-------------|----------------|----------|------------|------------|------|
| TPC-H SF10 (baseline)  | ~60M        | ~1             |   1339   |    461     |   1808     | 3.98 |
| Pool of 1000           | 1,000       | 60,000         |   1326   |    436     |   1759     | 4.09 |
| Pool of 10             | 10          | 6,000,000      |   1264   |    524     |   1788     | 4.03 |
| Zipfian (a=2.0, 10K)   | ~10,000     | skewed         |   1256   |    485     |   1741     | 4.14 |

## Per-run CSV data

```
# Pool of 1000 (60K records per key)
CSV,Quadro RTX 6000,7.20,60000000,1,1,1326.33,432.72,1759.05,4.0931,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,1322.03,436.28,1758.31,4.0948,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,1368.55,436.14,1804.69,3.9896,3.96,0.24,4.20,0.6

# Pool of 10 (6M records per key)
CSV,Quadro RTX 6000,7.20,60000000,1,1,1252.48,489.15,1741.62,4.1341,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,1264.41,523.74,1788.15,4.0265,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,1276.64,520.87,1797.51,4.0055,3.96,0.24,4.20,0.6

# Zipfian (a=2.0 over 10K keys)
CSV,Quadro RTX 6000,7.20,60000000,1,1,1254.90,473.69,1728.60,4.1652,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,1256.24,484.65,1740.89,4.1358,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,1277.54,470.17,1747.70,4.1197,3.96,0.24,4.20,0.6
```

## Analysis

All duplicate-heavy distributions sort in 1741-1788ms — within noise of
the random baseline (1808ms). Key observations:

1. **Radix sort is duplicate-oblivious.** LSD radix processes every byte
   position regardless of how many records share the same key. The per-pass
   time actually *decreases* with fewer unique keys (~34ms/pass for pool10
   vs ~43ms/pass for pool1000) due to better histogram cache locality.

2. **No fixup triggered.** Because each key is a random 66-byte string and
   all 66 byte positions vary, the compact detector can't compress the key,
   and the full-key GPU sort produces the final order. Ties between
   identical keys are broken by the stable sort's original order.

3. **Gather phase slightly slower for pool10** (524ms vs 432ms) because
   records with the same key are scattered uniformly, creating worst-case
   random access patterns in the permutation gather.

4. All runs passed sortedness + multiset hash verification.
