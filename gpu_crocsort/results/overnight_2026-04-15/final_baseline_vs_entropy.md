## Final apples-to-apples benchmark: position vs entropy
Branch: exp/entropy-selection (77e3827)
Same binary, same machine, alternating modes to wash out thermal/cache effects.

Method: 5 runs each mode, alternating (P E P E P E P E P E).

| SF | mode | r1 | r2 | r3 | r4 | r5 | warm median | warm stdev |
|---:|------|---:|---:|---:|---:|---:|------------:|-----------:|
| sf10 | position | 1749.10 | 1761.08 | 1732.70 | 1745.97 | 1741.05 | 1741.05 | 10 |
| sf10 | entropy | 1763.35 | 1761.50 | 1734.98 | 1740.70 | 1755.34 | 1740.70 | 11 |
| sf50 | position | 14017.70 | 14534.40 | 15478.96 | 17593.35 | 16011.97 | 15478.96 | 1109 |
| sf50 | entropy | 12220.07 | 12981.24 | 11566.02 | 11373.21 | 12168.95 | 11566.02 | 627 |
| sf100 | position | 8388.18 | 8088.81 | 10771.97 | 8829.66 | 8425.25 | 8425.25 | 1040 |
| sf100 | entropy | 8485.56 | 8568.97 | 8397.29 | 8602.69 | 8522.85 | 8522.85 | 78 |
