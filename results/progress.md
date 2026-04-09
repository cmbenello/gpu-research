# CrocSort GPU Research — Progress Log

## Cycle 1 — 2026-04-09T00:01
Ideas:
1. SM-Partitioned T_gen/T_merge Decoupling (8/10)
2. M/D Ratio Governs GPU Sort Performance Surface (7/10)
3. Huffman Merge Schedule as Warp-Level Persistent Merge Tree (8/10)
4. Two-Sided Memory Threshold as L1/L2 Crossover (6/10)
5. Phase Asymmetry as Kernel Occupancy Balancing (5/10)
Top idea: SM-Partitioned T_gen/T_merge Decoupling (score 8/10)
CUDA sketch: cycle1_sm_partitioned_decoupling.cu

## Cycle 2 — 2026-04-09T00:10
Ideas:
1. Symmetric Boundary T* as GPU Multi-SM Run Pooling (8/10)
2. Storage Latency Sensitivity on Multi-GPU NVLink (7/10)
3. Budget Split L* for Dynamic Shared Memory Carveout (8/10)
4. 5% L2 Residency Control for Sparse Index Pages (8/10)
5. Expansion Factor E=2 Under GPU Register Pressure (5/10)
Top idea: 5% L2 Residency Control for Sparse Index Pages (score 8/10)
CUDA sketch: cycle2_l2_sparse_index_residency.cu

## Cycle 3 — 2026-04-09T00:20
Ideas:
1. Run Count Prediction via GPU-Adapted Eq.5 with HBM Bandwidth (7/10)
2. Presorted Input Detection via GPU Streaming Histogram (7/10)
3. Imbalance Factor r as GPU Thread Block Load Balancing (6/10)
4. Read Amplification Tracking as GPU Counter Metric (6/10)
5. GPU Merge Cost Model with L_avg Predicting Kernel Launches (8/10)
Top idea: GPU Merge Cost Model with L_avg (score 8/10)
CUDA sketch: cycle3_huffman_merge_graph.cu

## Cycle 4 — 2026-04-09T00:30
Ideas:
1. Warp-Level OVC: 32-bit OVC in Registers (9/10)
2. OVC Duplicate Shortcut as Warp-Level Fast Path (8/10)
3. OVC-Aware HBM Prefix Truncation for Merge I/O (7/10)
4. Register-File OVC Storage for Zero-Latency Comparison (9/10)
5. OVC derive_ovc_from as Parallel Warp Operation (8/10)
Top idea: Warp-Level OVC: 32-bit OVC in Registers (score 9/10)
CUDA sketch: cycle4_warp_ovc_merge.cu

## Cycle 5 — 2026-04-09T00:40
Ideas:
1. OVC-Enhanced Merge Eliminating Key Memory Reads (9/10)
2. OVC Encoding During Run Generation via Warp Scan (7/10)
3. OVC Flag Bits as Predication Mask (7/10)
4. OVC-Guided Prefetch Distance (6/10)
5. OVC Tournament Tree in Tensor Cores (7/10)
Top idea: OVC-Enhanced Merge Eliminating Key Reads (score 9/10)
CUDA sketch: cycle5_ovc_selective_fetch_merge.cu

## Cycle 6 — 2026-04-09T00:50
Ideas:
1. OVC-Compressed Run Format Reducing HBM Footprint (7/10)
2. Batched OVC Recomputation for Multi-Run Merge (6/10)
3. OVC64 vs OVC32 Tradeoff on GPU Register Width (6/10)
4. OVC-Aware Merge Path Partitioning (7/10)
5. OVC as Compressed Sort Key for Radix Sort Hybrid (7/10)
Top idea: OVC-Compressed Run Format (score 7/10)
CUDA sketch: (none — score below threshold for full sketch)

## Cycle 7 — 2026-04-09T01:00
Ideas:
1. Shared-Memory Sparse Index for Boundary Search (8/10)
2. GPU Sparse Index Construction During Run Gen (7/10)
3. Byte-Balanced Partitioning as GPU Kernel (8/10)
4. Sparse Index Page Recycling via GPU Memory Pool (6/10)
5. Sparse Index Dynamic Stride via Online Statistics (6/10)
Top idea: Shared-Memory Sparse Index for Boundary Search (score 8/10)
CUDA sketch: cycle7_smem_sparse_index_search.cu

## Cycle 8 — 2026-04-09T01:10
Ideas:
1. GPU Sparse Index as Texture Memory (7/10)
2. Multi-Run Boundary Search via Cooperative Thread Array (7/10)
3. Stall-Aware Byte-Balanced Search with Candidate Expansion (7/10)
4. File Offset Dual Role for Cache-Aligned I/O (6/10)
5. Sparse Index Bootstrap from GPU Initial Fill (5/10)
Top idea: Sparse Index as Texture Memory (score 7/10)
CUDA sketch: (none — score below threshold)

## Cycle 9 — 2026-04-09T01:20
Ideas:
1. MultiSparseIndexes as GPU Segmented Array (7/10)
2. Hierarchical Sparse Index: Coarse SMEM + Fine L2 (8/10)
3. CrocSort Run ID as GPU Stream ID (6/10)
4. Sampling as Warp-Level Conditional Write (5/10)
5. Boundary Quality Metric: Thread Time Imbalance (5/10)
Top idea: Hierarchical Sparse Index (score 8/10)
CUDA sketch: (deepened only)

## Cycle 10 — 2026-04-09T01:30
Ideas:
1. GPU Best-Fit Allocator in Shared Memory (8/10)
2. cp.async Double-Buffering for Replacement Selection (8/10)
3. Tree-of-Losers Bank-Conflict-Free Layout (7/10)
4. GPU AllocHandle Encoding Matching CrocSort (5/10)
5. Late Fence Slots as Warp-Level Bitmask (6/10)
Top idea: GPU Best-Fit Allocator in Shared Memory (score 8/10)
CUDA sketch: cycle10_smem_bestfit_allocator.cu

## Cycle 11 — 2026-04-09T01:40
Ideas:
1. GPU Extent Model with HBM Huge Pages (5/10)
2. Free-List Indexing via __clz (5/10)
3. Run Sink as GPU Async Writer (6/10)
4. Coalescing Block Merge via Warp Ops (5/10)
5. GPU Sort Discard Mode (4/10)
Top idea: Run Sink as GPU Async Writer (score 6/10)
CUDA sketch: (none — score below threshold)

## Cycle 12 — 2026-04-09T01:50
Ideas:
1. GPU AlignedWriter as Coalesced Burst Writer (6/10)
2. GPU AlignedReader as Prefetch-Ahead Reader (7/10)
3. Zero-Copy Merge Value Streaming via DMA (9/10)
4. IoStatsTracker as GPU Counter Aggregator (4/10)
5. GPU Direct I/O for External Sort Beyond HBM (7/10)
Top idea: Zero-Copy Merge Value Streaming via DMA (score 9/10)
CUDA sketch: cycle12_zerocopy_merge_dma.cu

## Cycle 13 — 2026-04-09T02:00
Ideas:
1. Multi-GPU NVLink Sort with M/D Transfer Property (8/10)
2. Skew Handling on SIMD Lanes via OVC Load Balancing (7/10)
3. Cloud Latency Findings on GPU-PCIe-NVMe Hierarchy (7/10)
4. GenSort vs Lineitem Optimal Surface on GPU (6/10)
5. CrocSort Thread Pool as GPU Persistent Threads (7/10)
Top idea: Multi-GPU NVLink Sort with Transfer Property (score 8/10)
CUDA sketch: cycle13_multi_gpu_nvlink_sort.cu

## Cycle 14 — 2026-04-09T02:10
Ideas:
1. TMA-Accelerated Run Generation on H100 (8/10)
2. Cluster-Level Replacement Selection on H100 SM Clusters (9/10)
3. H100 Async Barrier for Phase Transition (6/10)
4. FP8 OVC Encoding for Tensor Core Merge (6/10)
5. Distributed Loser Tree Across SM Cluster (8/10)
Top idea: Cluster-Level Replacement Selection on H100 (score 9/10)
CUDA sketch: cycle14_sm_cluster_replacement_selection.cu

## Cycle 15 — 2026-04-09T02:20
Ideas:
1. End-to-End Unified GPU CrocSort Architecture (10/10)
2. Adaptive GPU Sort Selection via CrocSort Cost Model (7/10)
3. GPU CrocSort as Database Sort Operator (7/10)
4. GPU CrocSort Profiling Framework (5/10)
5. Streaming GPU CrocSort for Continuous Ingestion (7/10)
Top idea: End-to-End Unified GPU CrocSort Architecture (score 10/10)
CUDA sketch: cycle15_unified_gpu_crocsort.cu
