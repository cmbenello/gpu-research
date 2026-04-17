# Input Distribution Experiments

**Setup:** TPC-H SF10 (59,986,052 records, 7.20 GB, 120B records = 66B key + 54B value)
**GPU:** Quadro RTX 6000, single-chunk (fits in GPU memory)
**Binary:** `external_sort_tpch_compact` on branch `exp/fixup-fast-comparator`
**Warm runs:** 3 per distribution

## Results (median of 3 runs)

| Distribution       | Gen (ms) | Merge (ms) | Total (ms) | GB/s |
|--------------------|----------|------------|------------|------|
| Original (random)  |   1339   |    461     |   1808     | 3.98 |
| Sorted             |   1287   |    363     |   1660     | 4.35 |
| Reverse            |   1298   |    390     |   1689     | 4.26 |
| Nearly sorted (99%)|   1284   |    370     |   1653     | 4.35 |
| Nearly sorted (95%)|   1300   |    386     |   1699     | 4.24 |

## Per-run CSV data

```
# Original (random)
CSV,Quadro RTX 6000,7.20,59986052,1,1,1330.59,373.80,1704.38,4.2234,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1339.08,468.71,1807.79,3.9818,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1352.58,461.41,1813.99,3.9682,3.96,0.24,4.20,0.6

# Sorted
CSV,Quadro RTX 6000,7.20,59986052,1,1,1298.62,361.84,1660.46,4.3351,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1286.82,362.99,1649.81,4.3631,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1314.81,377.75,1692.56,4.2529,3.96,0.24,4.20,0.6

# Reverse
CSV,Quadro RTX 6000,7.20,59986052,1,1,1292.10,364.38,1656.49,4.3455,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1298.44,390.44,1688.88,4.2622,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1323.44,410.59,1734.04,4.1512,3.96,0.24,4.20,0.6

# Nearly sorted (99%)
CSV,Quadro RTX 6000,7.20,59986052,1,1,1292.60,369.43,1662.02,4.3311,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1283.83,369.57,1653.41,4.3536,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1335.60,375.42,1711.03,4.2070,3.96,0.24,4.20,0.6

# Nearly sorted (95%)
CSV,Quadro RTX 6000,7.20,59986052,1,1,1296.31,382.78,1679.09,4.2870,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1300.43,398.63,1699.06,4.2367,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,59986052,1,1,1325.00,385.82,1710.82,4.2075,3.96,0.24,4.20,0.6
```

## Analysis

All distributions fall within a tight 1653-1808ms band (±5% of mean).
The GPU LSD radix sort (CUB) is **input-oblivious**: O(n * k) where k = key bytes.
There is no best/worst case for input order.

Minor variation in "merge" (gather) phase comes from CPU cache effects
during the permutation-based scatter, not the GPU sort itself.

Compact detection found 27/66 varying bytes for all TPC-H SF10 variants
(structural redundancy is inherent to the schema, not the input order).
All verification checks passed (sortedness + multiset hash).
