# Scaling Curve: 1M to 300M Records

**Binary:** `external_sort_tpch66` (no compact, KEY_SIZE=66, VALUE_SIZE=54)
**GPU:** Quadro RTX 6000 (24 GB HBM, PCIe 3.0 x16)
**Data:** Random 120B records (66B key + 54B value), all 66 key bytes varying
**Warm runs:** 3 per size, median reported

## Results

| Records | Data (GB) | Median (ms) | GB/s | Chunks | Gen (ms) | Merge (ms) |
|---------|-----------|-------------|------|--------|----------|------------|
|   1M    |   0.12    |      20     | 6.07 |    1   |      20  |      0     |
|   5M    |   0.60    |      98     | 6.11 |    1   |      98  |      0     |
|  10M    |   1.20    |     196     | 6.12 |    1   |     196  |      0     |
|  30M    |   3.60    |     588     | 6.13 |    1   |     588  |      0     |
|  60M    |   7.20    |   1,188     | 6.06 |    1   |     970  |    218     |
| 120M    |  14.40    |   2,082     | 6.92 |    1   |   1,621  |    461     |
| 200M    |  24.00    |   3,105     | 7.73 |    1   |   2,644  |    461     |
| 300M*   |  36.00    |   4,169     | 8.64 |    7   |   2,970  |  1,200     |

*300M = TPC-H SF50 (structured keys, 61/66 varying), multi-chunk (7 runs).

## Per-run CSV data

```
# 1M (0.12 GB)
CSV,Quadro RTX 6000,0.12,1000000,1,0,19.95,0.00,19.95,6.0162,0.12,0.12,0.24,2.0
CSV,Quadro RTX 6000,0.12,1000000,1,0,19.75,0.00,19.75,6.0744,0.12,0.12,0.24,2.0
CSV,Quadro RTX 6000,0.12,1000000,1,0,19.75,0.00,19.75,6.0745,0.12,0.12,0.24,2.0

# 5M (0.60 GB)
CSV,Quadro RTX 6000,0.60,5000000,1,0,98.26,0.00,98.26,6.1065,0.60,0.60,1.20,2.0
CSV,Quadro RTX 6000,0.60,5000000,1,0,98.18,0.00,98.18,6.1111,0.60,0.60,1.20,2.0
CSV,Quadro RTX 6000,0.60,5000000,1,0,98.21,0.00,98.21,6.1096,0.60,0.60,1.20,2.0

# 10M (1.20 GB)
CSV,Quadro RTX 6000,1.20,10000000,1,0,196.14,0.00,196.14,6.1182,1.20,1.20,2.40,2.0
CSV,Quadro RTX 6000,1.20,10000000,1,0,196.06,0.00,196.06,6.1207,1.20,1.20,2.40,2.0
CSV,Quadro RTX 6000,1.20,10000000,1,0,196.08,0.00,196.08,6.1201,1.20,1.20,2.40,2.0

# 30M (3.60 GB)
CSV,Quadro RTX 6000,3.60,30000000,1,0,587.73,0.00,587.73,6.1253,3.60,3.60,7.20,2.0
CSV,Quadro RTX 6000,3.60,30000000,1,0,587.71,0.00,587.71,6.1255,3.60,3.60,7.20,2.0
CSV,Quadro RTX 6000,3.60,30000000,1,0,587.58,0.00,587.58,6.1268,3.60,3.60,7.20,2.0

# 60M (7.20 GB)
CSV,Quadro RTX 6000,7.20,60000000,1,1,972.75,215.17,1187.92,6.0610,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,969.83,220.64,1190.48,6.0480,3.96,0.24,4.20,0.6
CSV,Quadro RTX 6000,7.20,60000000,1,1,966.24,218.79,1185.03,6.0758,3.96,0.24,4.20,0.6

# 120M (14.40 GB)
CSV,Quadro RTX 6000,14.40,120000000,1,1,1624.46,458.30,2082.77,6.9139,7.92,0.48,8.40,0.6
CSV,Quadro RTX 6000,14.40,120000000,1,1,1620.84,460.67,2081.51,6.9181,7.92,0.48,8.40,0.6
CSV,Quadro RTX 6000,14.40,120000000,1,1,1601.42,471.01,2072.42,6.9484,7.92,0.48,8.40,0.6

# 200M (24.00 GB)
CSV,Quadro RTX 6000,24.00,200000000,1,1,2690.49,431.29,3121.79,7.6879,13.20,0.80,14.00,0.6
CSV,Quadro RTX 6000,24.00,200000000,1,1,2621.61,483.85,3105.46,7.7283,13.20,0.80,14.00,0.6
CSV,Quadro RTX 6000,24.00,200000000,1,1,2643.90,417.46,3061.36,7.8397,13.20,0.80,14.00,0.6

# 300M (36.00 GB) — SF50 no-compact, 7 chunks
CSV,Quadro RTX 6000,36.00,300005811,7,2,2971.70,1181.46,4153.16,8.6683,36.00,1.20,37.20,1.0
CSV,Quadro RTX 6000,36.00,300005811,7,2,2969.22,1199.58,4168.80,8.6357,36.00,1.20,37.20,1.0
CSV,Quadro RTX 6000,36.00,300005811,7,2,2969.87,1209.89,4179.76,8.6131,36.00,1.20,37.20,1.0
```

## Analysis

### Near-perfect linear scaling (1M-60M)

From 1M to 60M records, throughput holds steady at ~6.1 GB/s.
Time scales linearly: 10x more data → ~10x more time.
The system is throughput-bound by PCIe 3.0 bandwidth (~15.8 GB/s
unidirectional) and GPU sort rate.

### Throughput increases at larger sizes (120M-300M)

Throughput rises from 6.1 to 8.6 GB/s as data grows:
- **120M (14.4 GB):** 6.9 GB/s — GPU sort and PCIe overlap better
- **200M (24.0 GB):** 7.7 GB/s — GPU is fully utilized
- **300M (36.0 GB):** 8.6 GB/s — triple-buffered streaming hides PCIe

This is because at small sizes, the GPU sort completes before the
full PCIe round-trip. At larger sizes, GPU compute and PCIe transfers
overlap via CUDA streams, improving utilization.

### Architecture transition at ~4 GB

Below ~4 GB (30M records): single-pass GPU sort with full-record
round-trip (H2D records → GPU sort → D2H records). PCIe amplification
= 2.0x (upload + download full records).

Above ~4 GB: key-value separation kicks in. Only keys uploaded to GPU
(66B of 120B), permutation downloaded (4B/record), CPU applies
permutation gather. PCIe amplification drops to 0.6x.

### Multi-chunk transition at ~25 GB

At 200M (24 GB), data still fits in GPU memory (single chunk).
At 300M (36 GB), the system automatically splits into 7 chunks
with triple-buffered streaming: while GPU sorts chunk N, the
CPU gathers chunk N-1 and uploads chunk N+1. The pipeline achieves
higher throughput (8.6 GB/s) than single-chunk because GPU is
never idle.
