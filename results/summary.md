# CrocSort GPU Research — Final Summary

## Top 10 Ideas Overall (by score)

### 1. End-to-End Unified GPU CrocSort Architecture (10/10)
- **CrocSort ref**: `src/sort/ovc/sorter.rs:ExternalSorterWithOVC`
- **GPU primitive**: All — cooperative_groups, cp.async, TMA, SM clusters, L2 persistence, __shfl_sync, DMA
- Combines all individual techniques into a complete GPU sort system: OVC in registers, sparse indexes in L2, replacement selection in SM clusters, byte-balanced partitioning, zero-copy merge with DMA. The full CrocSort pipeline on GPU.
- **Experiment**: End-to-end comparison vs CUB/Thrust/ModernGPU on GenSort/lineitem/HeavyKey at 80-640GB.

### 2. Warp-Level OVC: 32-bit OVC in Registers for Comparison Shortcut (9/10)
- **CrocSort ref**: `src/ovc/offset_value_coding_32.rs:OVCU32` (line 100-148)
- **GPU primitive**: 32-bit register comparison, __shfl_sync for OVC propagation
- OVCU32's packed 32-bit format maps perfectly to GPU registers. Comparison becomes a single PTX integer compare instruction. On high-redundancy data (lineitem), >90% of comparisons resolve via OVC without any HBM key reads.
- **Experiment**: Merge throughput with OVC vs full key comparison, varying prefix redundancy.

### 3. Register-File OVC Storage for Zero-Latency Comparison (9/10)
- **CrocSort ref**: `src/ovc/tree_of_losers_ovc.rs:LoserTreeOVC`
- **GPU primitive**: Register file (255 registers/thread), indexed register access
- For K<=16 merge, the entire OVC loser tree fits in 16 registers. Every comparison is a register-register operation (1 cycle) vs 20+ cycles for shared memory.
- **Experiment**: Register-OVC vs smem tree for K=4,8,16,32, measuring cycles per comparison.

### 4. OVC-Enhanced Merge Eliminating Key Memory Reads (9/10)
- **CrocSort ref**: `src/sort/ovc/merge.rs:ZeroCopyMergeWithOVC` (line 79)
- **GPU primitive**: cp.async for selective key prefetch on OVC tie-break only
- Keys are 32-bit OVCs in registers. Only on tie-break (~10% of cases on redundant data) does the kernel fetch the full key from HBM. Transforms merge from memory-bound to compute-bound.
- **Experiment**: HBM read traffic per comparison with OVC-gated vs always-read merge.

### 5. Zero-Copy Merge Value Streaming via DMA Engine (9/10)
- **CrocSort ref**: `src/sort/ovc/merge.rs:ZeroCopyMergeWithOVC::merge_into` (line 163)
- **GPU primitive**: cudaMemcpyAsync (DMA copy engine), concurrent copy + compute streams
- Decouples key comparison (SM) from value transfer (DMA engine). SM processes OVC tournament while copy engine moves winner values. Matches CrocSort's CPU pattern where I/O and comparison overlap.
- **Experiment**: Merge throughput with SM-only vs SM+DMA on varying key:value ratios.

### 6. Cluster-Level Replacement Selection on H100 SM Clusters (9/10)
- **CrocSort ref**: `src/replacement_selection/tol_mm_ovc.rs:ReplacementSelectionOVCMM`
- **GPU primitive**: H100 SM clusters (4 SMs), distributed shared memory
- 4-SM cluster provides 912KB workspace (4x single SM), enabling ~4x longer runs. Loser tree in SM0's smem, record buffers distributed across cluster. Cross-SM access at ~5 cycle latency.
- **Experiment**: Run length and E factor with 1-SM vs 4-SM cluster on H100.

### 7. SM-Partitioned T_gen/T_merge Decoupling (8/10)
- **CrocSort ref**: `src/sort/core/engine.rs:SorterCore::run_generation_internal` (line 292)
- **GPU primitive**: cooperative_groups, shared memory workspace
- Maps Eq.5 to GPU: partition SMs between run generation (compute-heavy) and merge (bandwidth-heavy). The feasibility equation determines optimal SM allocation ratio.
- **Experiment**: SM split ratio sweep on A100 with GenSort 80GB.

### 8. 5% L2 Residency Control for Sparse Index Pages (8/10)
- **CrocSort ref**: `src/sort/core/engine.rs:INDEX_BUDGET_PCT=5`; `src/sort/core/sparse_index.rs`
- **GPU primitive**: cudaAccessPolicyWindow (L2 persistence API)
- Pin 5% of L2 (~2MB) for sparse index pages. Merge boundary searches hit L2 instead of HBM, reducing latency ~10x.
- **Experiment**: Boundary search latency with/without L2 persistence.

### 9. Multi-GPU NVLink Sort with M/D Transfer Property (8/10)
- **CrocSort ref**: `src/sort/core/engine.rs:SorterCore`; cloud paper transfer property
- **GPU primitive**: NCCL AllToAll, NVLink
- M/D ratio is invariant under GPU scaling (both M and D scale linearly with N GPUs). Optimal per-GPU config transfers directly to multi-GPU.
- **Experiment**: Config transfer validation across 1-8 GPUs.

### 10. Shared-Memory Sparse Index for Intra-SM Boundary Search (8/10)
- **CrocSort ref**: `src/sort/core/sparse_index.rs:SparseIndex`; `docs/merge_partitioning_with_sparse_indexes.md`
- **GPU primitive**: Shared memory, CUB BlockLoad
- Load sparse index pages into shared memory for ~60ns boundary lookup (vs ~400ns from HBM). A 228KB SM holds ~11,400 index entries.
- **Experiment**: Boundary search latency from HBM vs L2 vs smem.

---

## Recommended Starting Points

### 1. Warp-Level OVC Merge (cycles 4+5) — Start Here
**Why**: Highest individual impact with clearest implementation path. OVCU32's 32-bit packed format maps 1:1 to GPU registers. The merge inner loop becomes ~3 PTX instructions per comparison instead of memcmp. The selective fetch pattern (only read keys on OVC tie) has dramatic impact on high-redundancy data. CrocSort's CPU experiments already validate the ~50% reduction on lineitem — the GPU version should show even larger gains because key reads hit HBM (100ns) not L1 (1ns).

**Implementation path**: (1) Port OVCU32 encoding/decoding to CUDA. (2) Build register-resident K=8 loser tree. (3) Add selective cp.async key fetch on tie-break. (4) Benchmark against CUB merge on lineitem.

### 2. Zero-Copy Merge with DMA Value Streaming (cycle 12) — High Value/Effort Ratio
**Why**: CrocSort's ZeroCopyMerge design (keys in tree, values streamed) is perfectly suited for GPU's split SM + DMA architecture. SM utilization should jump from ~40% to ~80% when values are offloaded. The implementation mostly requires correct CUDA stream orchestration rather than complex kernel code.

**Implementation path**: (1) Split key and value data into separate HBM regions. (2) Key-only merge kernel on compute stream. (3) Value transfer queue consumed by DMA on copy stream. (4) Verify output correctness.

### 3. Shared-Memory Sparse Index + Byte-Balanced Partitioning (cycles 7+2) — Enables Scalability
**Why**: Without byte-balanced partitioning, GPU merge has the same HeavyKey problem CrocSort solves on CPU. The sparse index + double binary search is architecturally critical and unique to CrocSort. L2 persistence for sparse indexes is a straightforward CUDA API call. This component is necessary for any serious GPU external sort implementation.

**Implementation path**: (1) Port SparseIndex page format to GPU. (2) Configure L2 persistence via cudaStreamAttribute. (3) Implement double binary search kernel matching CrocSort's select_boundary_by_size. (4) Test on HeavyKey with 256x payload skew.

---

## Coverage Map

| CrocSort Mechanism | Ideas | Avg Score | Top Score |
|---|---|---|---|
| **OVC encoding/comparison** | 15 ideas (cycles 4-6) | 7.3 | 9 |
| **Analytical model (Eq.5, T*, M/D)** | 10 ideas (cycles 1-3) | 6.9 | 8 |
| **Sparse indexes & boundary search** | 12 ideas (cycles 7-9) | 6.6 | 8 |
| **Memory management (allocator, extents)** | 8 ideas (cycles 10-11) | 5.9 | 8 |
| **Zero-copy merge & I/O** | 5 ideas (cycle 12) | 6.8 | 9 |
| **Multi-GPU & cross-cutting** | 10 ideas (cycles 13-15) | 7.1 | 10 |
| **Replacement selection** | 5 ideas (across cycles) | 7.6 | 9 |

**Most covered**: OVC encoding (15 ideas) — natural GPU fit due to register-width alignment.
**Least covered**: Memory management internals (8 ideas, lower avg score) — CrocSort's best-fit allocator is highly CPU-specific; GPU adaptation is valuable but less novel.

---

## Open Questions

1. **OVC hit rate on GPU workloads**: CrocSort shows negligible OVC benefit on GenSort but ~50% on lineitem. What's the OVC hit rate distribution for GPU-typical workloads (ML training data shuffling, graph edge sorting, scientific simulation output)? This determines whether the OVC register optimization is broadly applicable or niche.

2. **SM cluster distributed shared memory latency**: H100 documentation claims ~5 cycle latency for cross-SM dsmem access, but real-world contention under replacement selection's random access pattern may be higher. Need microbenchmark of dsmem latency under concurrent read/write from all 4 SMs.

3. **L2 persistence interference**: Pinning 5% of L2 for sparse indexes reduces effective L2 for run data during merge. At what point does the sparse index benefit (faster boundary search) get offset by increased L2 misses on run data? CrocSort's 5% budget may need adjustment for GPU where L2 pressure is higher.

4. **Register-resident loser tree maximum fanin**: For K>16, register pressure forces spill to local memory (HBM). What is the actual performance crossover point where shared-memory tree beats register tree? This determines the optimal merge fanin and number of merge levels.

5. **Expansion factor E on GPU**: Does replacement selection in shared memory achieve E=2 on random data as it does on CPU? GPU's warp-synchronous execution and limited shared memory may introduce artifacts (e.g., warp divergence during tree operations reducing effective throughput, causing smaller E). Need empirical validation before the analytical model can be trusted for GPU configuration.
