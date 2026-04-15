## Experiment: insertion-sort-small-groups (NULL)

Branch: exp/insertion-sort-small-groups (deleted — reverted on exp/entropy-selection)

### Hypothesis
nsys showed SF50 has 19.5M tied groups of ~15 records each. std::sort on 15 records has high per-call overhead (introsort setup, recursion). Insertion sort on n ≤ 32 should be ~10× faster per group.

### Result
SF50 fixup 5.4s → 7.8-8.2s (REGRESSION ~40%).

### Why wrong

Back-of-envelope cost analysis per 15-record group:
- std::sort: ~58 compares (O(n log n)) + ~100 ns function call overhead = 58×20 + 100 = **1260 ns**
- insertion: ~225 compares (n² / 2 for already-shuffled input) + ~10 ns overhead = 225×20 + 10 = **4510 ns**

Insertion sort is SLOWER for n=15 because the O(n²) compare count dominates the function call overhead. The crossover is around n ≈ 3–5 for typical introsort implementations. My hypothesis that "std::sort overhead dominates small groups" was wrong — libstdc++ introsort is highly tuned and its branch structure is lighter than I assumed.

**Lesson**: always dimensional-analyze the compare count before assuming a lower-overhead algorithm wins.
