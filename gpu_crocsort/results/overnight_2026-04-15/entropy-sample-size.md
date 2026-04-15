## Experiment: entropy-sample-size
Branch: exp/entropy-selection (77e3827)
Hypothesis: 1M sample might miss true entropy ranking. Larger sample → better selection → lower fixup.

| sample | r1 | r2 | r3 | warm median | detect ms |
|-------:|---:|---:|---:|------------:|----------:|
| 1000000 | 13611.06 | 11296.73 | 11154.86 | 11154.86 | 35 |
| 5000000 | 11205.86 | 11448.23 | 11474.50 | 11448.23 | 94 |
| 10000000 | 11620.35 | 11626.04 | 11652.18 | 11626.04 | 185 |
| 60000000 | 11362.37 | 11490.64 | 11592.43 | 11490.64 | 968 |
