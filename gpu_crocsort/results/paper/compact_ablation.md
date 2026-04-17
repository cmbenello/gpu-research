# Compact Key Ablation: SF50

**Dataset:** TPC-H SF50, 300,005,811 records, 36.00 GB (120B records = 66B key + 54B value)
**GPU:** Quadro RTX 6000 (24 GB HBM, PCIe 3.0 x16)
**Warm runs:** 3 per configuration

## Result: Non-compact is 1.6x faster

| Metric                  | Compact              | No-compact           |
|-------------------------|----------------------|----------------------|
| **Total (median)**      | **6,651 ms**         | **4,169 ms**         |
| Run generation          | 2,214 ms (2 chunks)  | 2,970 ms (7 chunks)  |
| GPU merge               | 607 ms (32B, 4 pass) | 5 ms (16B, 2 pass)   |
| D2H permutation         | 93 ms                | 94 ms                |
| CPU gather              | 2,004 ms (18 GB/s)   | 1,101 ms (33 GB/s)   |
| CPU fixup               | 1,734 ms (3.15M grps)| 0 ms (no ties)       |
| PCIe H2D                | 9.6 GB               | 36.0 GB              |
| Chunks                  | 2                    | 7                    |
| Merge prefix            | 32B (compact)        | 16B (full key)       |
| Ties after merge        | 3,153,346 groups     | 0                    |

## Per-run CSV data

```
# Compact (external_sort_tpch_compact)
CSV,Quadro RTX 6000,36.00,300005811,2,2,2158.99,4621.82,6780.81,5.3092,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2213.65,4437.68,6651.33,5.4126,9.60,1.20,10.80,0.3
CSV,Quadro RTX 6000,36.00,300005811,2,2,2232.90,4325.55,6558.45,5.4892,9.60,1.20,10.80,0.3

# No-compact (external_sort_tpch66)
CSV,Quadro RTX 6000,36.00,300005811,7,2,2971.70,1181.46,4153.16,8.6683,36.00,1.20,37.20,1.0
CSV,Quadro RTX 6000,36.00,300005811,7,2,2969.22,1199.58,4168.80,8.6357,36.00,1.20,37.20,1.0
CSV,Quadro RTX 6000,36.00,300005811,7,2,2969.87,1209.89,4179.76,8.6131,36.00,1.20,37.20,1.0
```

## Why compact loses on SF50

The compact optimization reduces PCIe volume (9.6 vs 36 GB) but introduces
three costs that exceed the savings:

### 1. Fixup cost: +1,734 ms
SF50 has 61/66 varying bytes. Only 32 fit in the compact prefix buffer.
The remaining 29 positions create 3.15M tie groups (avg 95 records each),
requiring a full CPU fixup phase: group detection (613ms) + parallel
per-group sort (1,105ms).

### 2. GPU merge cost: +602 ms
Compact uses a 32B prefix merge (4 CUB LSD passes at ~155ms each).
No-compact uses 16B prefix merge (2 passes), and the first 16 bytes
of the full 66-byte key are discriminating enough to resolve ALL ties
(0 tie groups). This is because the full key preserves the natural
byte ordering where high-order bytes are most significant.

### 3. Gather locality: +903 ms
No-compact produces 7 smaller runs (~43M records each), which have
better cache behavior during the permutation-based gather (33 GB/s).
Compact produces 2 large runs (~150M records each), leading to more
cache thrashing in the scatter (18 GB/s).

### PCIe savings insufficient
Compact saves 26.4 GB of PCIe transfers, but triple-buffered run
generation hides most of this behind GPU sort. The run-gen phase is
only 756ms slower without compact (2,970 vs 2,214 ms), while the
three costs above total 3,239 ms.

## Implications for the paper

Compact key detection is a tradeoff, not a universal win. It benefits
workloads where:
- Varying bytes ≤ 32 (no fixup — e.g., SF100 with 27 varying)
- PCIe bandwidth is the bottleneck (older Gen 3 with many chunks)
- CPU cores are limited (fixup is CPU-bound)

On SF50 with 61 varying bytes, the non-compact path is superior because
the full-key LSD sort produces a globally correct order with zero
post-processing.

## SF100 comparison: compact wins 5.1x

| Metric              | Compact (SF100)     | No-compact (SF100)    |
|---------------------|---------------------|-----------------------|
| **Total (median)**  | **7,982 ms**        | **39,538 ms**         |
| Chunks              | 7                   | 19                    |
| Varying bytes       | 27/66               | 66/66 (full key)      |
| Ties after merge    | 0                   | 0                     |
| PCIe H2D            | ~18 GB              | 72 GB                 |

SF100 has only 27/66 varying bytes → all fit in 32B compact prefix →
zero fixup. Compact is 5.1x faster because:
1. **Fewer chunks (7 vs 19):** Compact keys are smaller, more fit per chunk
2. **Less PCIe (18 vs 72 GB):** 27B compact vs 66B full key upload
3. **Faster run-gen:** CUB sorts 32B keys in 4 passes vs 66B in 9 passes

### No-compact SF100 per-run CSV data
```
CSV,Quadro RTX 6000,72.00,600037902,19,2,35344.32,3374.69,38719.02,1.8597,72.00,2.40,74.40,1.0
CSV,Quadro RTX 6000,72.00,600037902,19,2,36582.41,2956.08,39538.49,1.8211,72.00,2.40,74.40,1.0
CSV,Quadro RTX 6000,72.00,600037902,19,2,35744.59,4966.89,40711.48,1.7687,72.00,2.40,74.40,1.0
```

## Summary: compact tradeoff depends on varying-byte count

| Dataset | Varying bytes | Compact (ms) | No-compact (ms) | Winner     |
|---------|--------------|-------------|-----------------|------------|
| SF50    | 61/66        | 6,651       | 4,169           | No-compact (1.6x) |
| SF100   | 27/66        | 7,982       | 39,538          | Compact (5.0x)    |

The crossover point is ~32 varying bytes (the compact prefix width).
Below 32, compact wins because it sorts the complete discriminating
prefix with no fixup. Above 32, the CPU fixup cost and reduced GPU
merge efficiency overwhelm the PCIe savings.
