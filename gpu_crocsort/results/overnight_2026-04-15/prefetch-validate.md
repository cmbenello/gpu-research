## Experiment: prefetch-validate
Branch: exp/prefetch-sweep (ff244d8)
Hypothesis: prefetch=256 saves ~240ms vs 512 on SF100. 5 runs each, head-to-head.

| prefetch | run1 | run2 | run3 | run4 | run5 | min total | median total | min gather | median gather |
|---------:|-----:|-----:|-----:|-----:|-----:|----------:|-------------:|-----------:|--------------:|
| 256 | 8405.23 | 8453.45 | 8298.23 | 11718.99 | 8242.20 | 8242.20 | 8405.23 | 3545 | 3712 |
| 512 | 8306.25 | 8412.11 | 8297.05 | 8321.11 | 8288.08 | 8288.08 | 8306.25 | 3619 | 3627 |
