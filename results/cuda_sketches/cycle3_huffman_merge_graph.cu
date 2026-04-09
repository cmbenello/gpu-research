// CrocSort GPU Transfer: Huffman Merge Schedule as CUDA Graph
// Maps from: src/sort/core/engine.rs:multi_merge_with_hooks (multi-level merge)
//            Total I/O = 2D*(1 + L_avg); Huffman schedule minimizes L_avg
// Concept: Pre-compute optimal merge schedule on host, capture as cudaGraph,
//          execute entire multi-level merge without per-level kernel launch overhead.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <queue>
#include <algorithm>

// ── Host-side: Huffman merge schedule computation ──────────────────

struct MergeNode {
    int level;              // Merge tree level (0 = leaf = input run)
    uint64_t bytes;         // Total bytes at this node
    int fanin;              // Number of inputs to merge at this node
    std::vector<int> inputs; // Input node indices
    int output_run_id;      // Output run ID for this merge
};

// Compute Huffman-optimal merge schedule
// Maps from: multi_merge_with_hooks recursive fanin logic
// Minimizes L_avg = sum(level_i * bytes_at_level_i) / total_bytes
struct HuffmanSchedule {
    std::vector<MergeNode> nodes;
    int total_levels;
    double L_avg;           // Average merge depth per byte

    // Build Huffman schedule from run sizes
    // Similar to how multi_merge_with_hooks decides merge passes
    void build(const std::vector<uint64_t>& run_sizes, int max_fanin) {
        // Priority queue: merge smallest runs first (Huffman principle)
        auto cmp = [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) {
            return a.first > b.first; // min-heap by bytes
        };
        std::priority_queue<std::pair<uint64_t, int>,
                           std::vector<std::pair<uint64_t, int>>,
                           decltype(cmp)> pq(cmp);

        // Initialize leaves
        for (int i = 0; i < (int)run_sizes.size(); i++) {
            nodes.push_back({0, run_sizes[i], 0, {}, i});
            pq.push({run_sizes[i], i});
        }

        // Build merge tree bottom-up
        total_levels = 0;
        while (pq.size() > 1) {
            int fanin = std::min(max_fanin, (int)pq.size());
            MergeNode merge_node;
            merge_node.fanin = fanin;
            merge_node.bytes = 0;
            merge_node.output_run_id = (int)nodes.size();

            for (int i = 0; i < fanin; i++) {
                auto [bytes, idx] = pq.top();
                pq.pop();
                merge_node.inputs.push_back(idx);
                merge_node.bytes += bytes;
                merge_node.level = std::max(merge_node.level, nodes[idx].level + 1);
            }

            total_levels = std::max(total_levels, merge_node.level);
            int node_idx = (int)nodes.size();
            nodes.push_back(merge_node);
            pq.push({merge_node.bytes, node_idx});
        }

        // Compute L_avg
        uint64_t total_bytes = 0;
        double weighted_depth = 0;
        for (int i = 0; i < (int)run_sizes.size(); i++) {
            // Find depth of leaf i in merge tree
            int depth = compute_leaf_depth(i);
            weighted_depth += depth * (double)run_sizes[i];
            total_bytes += run_sizes[i];
        }
        L_avg = total_bytes > 0 ? weighted_depth / total_bytes : 0;
    }

    int compute_leaf_depth(int leaf_idx) const {
        // Trace from leaf to root
        int depth = 0;
        int current = leaf_idx;
        for (int i = (int)nodes.size() - 1; i >= 0; i--) {
            for (int inp : nodes[i].inputs) {
                if (inp == current) {
                    depth++;
                    current = i;
                    break;
                }
            }
        }
        return depth;
    }
};

// ── Device-side: K-way merge kernel ────────────────────────────────

// Loser tree node for merge — maps from tree_of_losers.rs:Node
struct __align__(8) MergeTreeNode {
    uint32_t key_prefix;    // First 4 bytes of key
    uint16_t source_idx;    // Input run index
    uint16_t flags;         // Sentinel flags
};

// Single merge pass kernel — called once per level in the schedule
// Maps from: ZeroCopyMergeWithOVC::merge_into (merge.rs:163)
__global__ void merge_pass_kernel(
    const uint8_t** __restrict__ input_runs,    // Input run pointers
    const uint64_t* __restrict__ input_sizes,   // Input run sizes
    uint8_t* __restrict__ output_run,           // Output run buffer
    uint64_t* __restrict__ output_size,         // Output size
    const int fanin,                            // Number of inputs
    const int record_size,
    const int num_partitions                    // Thread blocks = merge partitions
) {
    extern __shared__ uint8_t smem[];

    const int partition_id = blockIdx.x;
    const int tid = threadIdx.x;

    // Shared memory layout:
    // [0..tree_bytes): Loser tree
    // [tree_bytes..tree_bytes+fanin*buf_size): Per-source read buffers
    MergeTreeNode* tree = reinterpret_cast<MergeTreeNode*>(smem);
    const int tree_bytes = fanin * sizeof(MergeTreeNode);
    uint8_t* read_buffers = smem + tree_bytes;
    const int buf_size_per_source = 4096; // 4KB per source read buffer

    // Each partition merges a range of the output
    // Maps from: merge_once_with_hooks boundary computation
    uint64_t total_records = 0;
    for (int i = 0; i < fanin; i++) {
        total_records += input_sizes[i] / record_size;
    }
    uint64_t records_per_partition = (total_records + num_partitions - 1) / num_partitions;
    uint64_t my_start = partition_id * records_per_partition;
    uint64_t my_end = min(my_start + records_per_partition, total_records);

    // TODO: Use byte-balanced partitioning (cycle 7 idea) instead of count-balanced
    // This requires sparse index boundary search

    // Initialize loser tree with first record from each source
    // Maps from: LoserTree::reset_from_iter (tree_of_losers.rs:46)
    if (tid < fanin) {
        // Load first record's key prefix from each input run
        if (input_sizes[tid] > 0) {
            uint32_t key_prefix = *reinterpret_cast<const uint32_t*>(input_runs[tid]);
            tree[tid].key_prefix = key_prefix;
            tree[tid].source_idx = tid;
            tree[tid].flags = 0;
        } else {
            tree[tid].key_prefix = 0xFFFFFFFF; // LateFence
            tree[tid].source_idx = tid;
            tree[tid].flags = 4;
        }
    }
    __syncthreads();

    // Build tournament tree
    // TODO: parallel tree build

    // Main merge loop
    // Maps from: ZeroCopyMergeWithOVC::merge_into loop (merge.rs:164-209)
    uint64_t output_offset = my_start * record_size;
    uint64_t per_source_offset[64]; // TODO: dynamic for large fanin
    for (int i = 0; i < fanin; i++) per_source_offset[i] = 0;

    for (uint64_t i = my_start; i < my_end; i++) {
        if (tid == 0) {
            // Extract winner
            int winner_src = tree[0].source_idx;

            // Copy winner record to output
            // Maps from: ZeroCopyMergeWithOVC — value streaming
            // In full implementation, use DMA copy engine (cycle 12 idea)
            const uint8_t* src = input_runs[winner_src] + per_source_offset[winner_src];
            uint8_t* dst = output_run + output_offset;
            // TODO: use cp.async for async copy
            memcpy(dst, src, record_size);

            per_source_offset[winner_src] += record_size;
            output_offset += record_size;

            // Advance: load next record from winner's source
            if (per_source_offset[winner_src] < input_sizes[winner_src]) {
                uint32_t new_key = *reinterpret_cast<const uint32_t*>(
                    input_runs[winner_src] + per_source_offset[winner_src]);
                tree[0].key_prefix = new_key;

                // Replay tournament (push)
                // Maps from: LoserTree::push (tree_of_losers.rs:98)
                int slot = (fanin + winner_src) / 2;
                while (slot >= 1) {
                    if (tree[0].key_prefix > tree[slot].key_prefix) {
                        MergeTreeNode tmp = tree[0];
                        tree[0] = tree[slot];
                        tree[slot] = tmp;
                    }
                    slot /= 2;
                }
            } else {
                // Source exhausted — mark with LateFence
                tree[0].key_prefix = 0xFFFFFFFF;
                tree[0].flags = 4;
                // Replay to find new winner
                int slot = (fanin + winner_src) / 2;
                while (slot >= 1) {
                    if (tree[0].key_prefix > tree[slot].key_prefix) {
                        MergeTreeNode tmp = tree[0];
                        tree[0] = tree[slot];
                        tree[slot] = tmp;
                    }
                    slot /= 2;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0 && partition_id == 0) {
        *output_size = total_records * record_size;
    }
}

// ── Host-side: Capture merge schedule as CUDA Graph ────────────────

// cudaGraph_t build_merge_graph(
//     const HuffmanSchedule& schedule,
//     uint8_t** d_run_buffers,
//     int record_size,
//     int num_partitions_per_merge
// ) {
//     cudaGraph_t graph;
//     cudaGraphCreate(&graph, 0);
//
//     // For each level in the Huffman schedule, add kernel node to graph
//     // Level dependencies ensure correct execution order
//     std::vector<cudaGraphNode_t> level_nodes;
//
//     for (int level = 1; level <= schedule.total_levels; level++) {
//         for (const auto& node : schedule.nodes) {
//             if (node.level == level && node.fanin > 0) {
//                 cudaKernelNodeParams params = {};
//                 params.func = (void*)merge_pass_kernel;
//                 params.gridDim = dim3(num_partitions_per_merge);
//                 params.blockDim = dim3(256);
//                 params.sharedMemBytes = node.fanin * sizeof(MergeTreeNode) +
//                                         node.fanin * 4096;
//                 // Set kernel arguments...
//
//                 cudaGraphNode_t kernel_node;
//                 // Add dependencies from input nodes at level-1
//                 cudaGraphAddKernelNode(&kernel_node, graph,
//                     /* dependencies */, /* num_deps */,
//                     &params);
//                 level_nodes.push_back(kernel_node);
//             }
//         }
//     }
//
//     return graph;
// }
//
// // Launch:
// cudaGraphExec_t exec;
// cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
// cudaGraphLaunch(exec, stream);
// cudaStreamSynchronize(stream);
// printf("Huffman L_avg = %.3f, total HBM traffic = %.1f GB\n",
//        schedule.L_avg, 2.0 * total_data_bytes * (1 + schedule.L_avg) / 1e9);
