# Fixup Thread Scaling: T=1 to 48

**Binary:** `external_sort_tpch_compact` (compact key, KEY_SIZE=66, VALUE_SIZE=54)
**Data:** TPC-H SF50 lineitem, 300M records, 36 GB (61/66 varying bytes)
**GPU:** Quadro RTX 6000 (24 GB HBM, PCIe 3.0 x16)
**CPU:** 2x Xeon Gold 6248 (40 cores / 80 threads), 192 GB DDR4
**Warm runs:** 3 per thread count, median reported

## Results

| Threads | Fixup (ms) | Group-detect (ms) | Parallel sort (ms) | Total (ms) | Speedup vs T=1 |
|---------|-----------|-------------------|---------------------|------------|----------------|
|    1    |  28,317   |      7,615        |       20,690        |   32,882   |     1.0x       |
|    2    |  15,656   |      4,580        |       11,063        |   20,361   |     1.6x       |
|    4    |   9,210   |      3,310        |        5,823        |   13,472   |     2.4x       |
|    8    |   4,011   |      1,183        |        2,816        |    8,963   |     3.7x       |
|   16    |   2,856   |      1,173        |        1,666        |    7,575   |     4.3x       |
|   24    |   2,477   |      1,147        |        1,287        |    7,199   |     4.6x       |
|   32    |   2,233   |      1,128        |        1,088        |    6,949   |     4.7x       |
|   48    |   1,905   |        812        |        1,081        |    6,585   |     5.0x       |

Non-fixup baseline (constant): gen ~2,200ms + GPU merge ~600ms + gather ~1,700ms ≈ 4,500ms

## Per-run CSV data

```
# T=1
CSV,Quadro RTX 6000,36.00,300005811,2,2,2144.86,30737.23,32882.09,1.0948,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2279.76,30500.26,32780.02,1.0983,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2270.13,31104.08,33374.22,1.0787,9.60,1.20,10.80,0.3

# T=2
CSV,Quadro RTX 6000,36.00,300005811,2,2,2189.60,18270.72,20460.32,1.7595,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2291.84,17779.11,20070.95,1.7937,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2241.83,18118.93,20360.76,1.7681,9.60,1.20,10.80,0.3

# T=4
CSV,Quadro RTX 6000,36.00,300005811,2,2,2158.91,11711.98,13870.88,2.5954,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2155.95,11303.90,13459.85,2.6747,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2137.01,11335.05,13472.06,2.6722,9.60,1.20,10.80,0.3

# T=8
CSV,Quadro RTX 6000,36.00,300005811,2,2,2181.59,6874.75,9056.34,3.9752,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2205.62,6757.27,8962.89,4.0166,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2171.77,6769.46,8941.23,4.0264,9.60,1.20,10.80,0.3

# T=16
CSV,Quadro RTX 6000,36.00,300005811,2,2,2234.44,5459.83,7694.27,4.6789,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2243.35,5331.78,7575.14,4.7525,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2124.32,4899.85,7024.17,5.1253,9.60,1.20,10.80,0.3

# T=24
CSV,Quadro RTX 6000,36.00,300005811,2,2,2231.03,4967.74,7198.77,5.0010,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2211.96,5035.31,7247.26,4.9675,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2135.31,4857.03,6992.34,5.1486,9.60,1.20,10.80,0.3

# T=32
CSV,Quadro RTX 6000,36.00,300005811,2,2,2187.61,4705.19,6892.80,5.2229,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2221.22,4728.18,6949.40,5.1804,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2248.18,4743.89,6992.06,5.1488,9.60,1.20,10.80,0.3

# T=48
CSV,Quadro RTX 6000,36.00,300005811,2,2,2207.73,4377.54,6585.27,5.4669,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2251.74,4511.77,6763.51,5.3228,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2106.08,4322.63,6428.71,5.6000,9.60,1.20,10.80,0.3
```

## Analysis

### Near-linear scaling from 1 to 8 threads

Fixup time drops from 28.3s (T=1) to 4.0s (T=8), a 7.1x speedup
with 8 threads — 89% parallel efficiency. The fixup workload
(3.15M independent groups of ~95 records each) maps perfectly
to thread-level parallelism with the atomic work-queue dispatch.

### Diminishing returns past 8 threads

| Threads | Fixup (ms) | Marginal gain |
|---------|-----------|--------------|
|   8→16  | 4011→2856 | 1.4x         |
|  16→24  | 2856→2477 | 1.2x         |
|  24→32  | 2477→2233 | 1.1x         |
|  32→48  | 2233→1905 | 1.2x         |

Two factors limit scaling past 8 threads:

1. **Group-detect scan bottleneck:** The sequential scan over 300M
   records to identify tie-group boundaries is memory-bandwidth-bound.
   At T=8 it's already 1.2s — close to the DDR4 streaming rate for
   a 1.2 GB scan (300M × 4B OVC). More threads can't help a
   memory-bound scan.

2. **Per-group sort is small:** Average group size is 95 records.
   At T=48, each thread processes groups so quickly that atomic
   work-queue contention and cache-line bouncing become significant.

### Fixup is the dominant cost at T=1, minor at T=48

| Threads | Fixup / Total | Non-fixup / Total |
|---------|--------------|-------------------|
|    1    |    86%       |       14%         |
|    8    |    45%       |       55%         |
|   32    |    32%       |       68%         |
|   48    |    29%       |       71%         |

At T=48, fixup drops to 1.9s out of 6.6s total — the GPU phases
(gen + merge = 2.8s) and CPU gather (1.7s) dominate. Further fixup
optimization yields diminishing total improvement.

### Practical recommendation

T=8 captures most of the benefit (4.0s fixup, 9.0s total) with
minimal thread contention. T=32 (hardware_concurrency default)
gives 2.2s fixup at 6.9s total — near the 6.4s floor at T=48.
The 8→32 range is the sweet spot for this workload.
