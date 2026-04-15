## Experiment: threads-sweep
Branch: exp/prefetch-sweep (ff244d8)
Hypothesis: 48 threads (default = hardware_concurrency) might oversubscribe vs N physical cores. SF100 gather is bandwidth-bound — fewer threads with more sequential work might pull less random L3 churn.
Variable: GATHER_THREADS ∈ {8, 16, 24, 32, 48, 64}
SF100 only, 3 runs each

| threads | run1 | run2 | run3 | median total | median gather ms | verified |
|--------:|-----:|-----:|-----:|-------------:|-----------------:|---------:|
| 8 | 14976.81 | 14850.22 | 13705.12 | 14850.22 | 10161 | true |
| 16 | 9944.08 | 9668.78 | 9670.74 | 9670.74 | 4974 | true |
| 24 | 9318.92 | 9406.91 | 9240.68 | 9318.92 | 4611 | true |
| 32 | 8409.65 | 8611.82 | 8676.97 | 8611.82 | 3948 | true |
| 48 | 8348.42 | 8277.27 | 8377.01 | 8348.42 | 3679 | true |
| 64 | 8387.42 | 8469.28 | 8353.12 | 8387.42 | 3719 | true |

### Conclusion

Current 48-thread default is optimal. 64 threads regress slightly (oversubscription). Below 32 threads, gather is bandwidth-starved.

**NULL RESULT.** Confirms current config is correct.
