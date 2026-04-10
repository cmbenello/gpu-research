#pragma once
#include <cstdint>
#include "ovc.cuh"
#include "record.cuh"

// ============================================================================
// GPU Loser Tree with OVC — port of CrocSort's LoserTreeOVC
// Source: crocsort_repo/src/ovc/tree_of_losers_ovc.rs
//
// Layout: nodes[0] = winner, nodes[1..capacity] = losers
// Leaf for source i = capacity + i (conceptual, not stored)
// Parent of node j  = j / 2
//
// Each node stores: (ovc: uint32_t, source_idx: int)
// OVC comparison: single uint32_t compare gives correct ordering
// On tie: fall back to full key comparison from global memory
// ============================================================================

// Maximum merge fan-in supported
static constexpr int MAX_MERGE_K = 16;

// Loser tree state — stored in registers or shared memory
struct LoserTreeState {
    uint32_t ovc[MAX_MERGE_K];    // OVC per tree node
    int      src[MAX_MERGE_K];    // Source index per tree node
    int      capacity;            // Actual fan-in (power of 2)
};

// ── Build initial tournament ───────────────────────────────────────
// Maps from: LoserTreeOVC::reset_from_iter (tree_of_losers_ovc.rs)
// Pass each initial value through the tree from its leaf to root.

__device__ inline void loser_tree_init(
    LoserTreeState& tree,
    const uint32_t* initial_ovcs,    // K initial OVC values
    int K                            // Fan-in (must be power of 2)
) {
    tree.capacity = K;

    // Initialize all nodes to early fence
    for (int i = 0; i < K; i++) {
        tree.ovc[i] = OVC_EARLY_FENCE;
        tree.src[i] = -1;
    }

    // Insert each initial value via pass()
    for (int i = 0; i < K; i++) {
        // Candidate starts with the initial value
        uint32_t cand_ovc = initial_ovcs[i];
        int cand_src = i;

        // Climb from leaf to root
        int slot = (K + i) / 2;
        while (slot >= 1) {
            if (cand_ovc > tree.ovc[slot] ||
                (cand_ovc == tree.ovc[slot] && cand_src > tree.src[slot])) {
                // Candidate loses — swap
                uint32_t tmp_ovc = cand_ovc;
                int tmp_src = cand_src;
                cand_ovc = tree.ovc[slot];
                cand_src = tree.src[slot];
                tree.ovc[slot] = tmp_ovc;
                tree.src[slot] = tmp_src;
            }
            slot /= 2;
        }

        // Winner reaches root (node 0)
        tree.ovc[0] = cand_ovc;
        tree.src[0] = cand_src;
    }
}

// ── Push new value, return old winner ──────────────────────────────
// Maps from: LoserTreeOVC::push (tree_of_losers_ovc.rs)
// Replaces current winner with new_ovc from source new_src.
// Replays tournament. Returns old winner's (ovc, source).

__device__ inline void loser_tree_push(
    LoserTreeState& tree,
    uint32_t new_ovc,
    int new_src,
    uint32_t& out_winner_ovc,
    int& out_winner_src
) {
    // Save old winner
    out_winner_ovc = tree.ovc[0];
    out_winner_src = tree.src[0];

    // Start candidate at the new value
    uint32_t cand_ovc = new_ovc;
    int cand_src = new_src;

    // Climb from leaf of the OLD winner's source to root
    int slot = (tree.capacity + out_winner_src) / 2;

    while (slot >= 1) {
        // Compare: loser stays at node, winner moves up
        if (cand_ovc > tree.ovc[slot] ||
            (cand_ovc == tree.ovc[slot] && cand_src > tree.src[slot])) {
            // Candidate loses
            uint32_t tmp_ovc = cand_ovc;
            int tmp_src = cand_src;
            cand_ovc = tree.ovc[slot];
            cand_src = tree.src[slot];
            tree.ovc[slot] = tmp_ovc;
            tree.src[slot] = tmp_src;
        }
        slot /= 2;
    }

    // New winner at root
    tree.ovc[0] = cand_ovc;
    tree.src[0] = cand_src;
}

// Simplified push that doesn't return old winner (used internally)
__device__ inline void loser_tree_push_no_return(
    LoserTreeState& tree,
    uint32_t new_ovc,
    int new_src
) {
    uint32_t dummy_ovc;
    int dummy_src;
    loser_tree_push(tree, new_ovc, new_src, dummy_ovc, dummy_src);
}

// ── Peek at current winner ─────────────────────────────────────────
__device__ inline bool loser_tree_peek(
    const LoserTreeState& tree,
    uint32_t& out_ovc,
    int& out_src
) {
    if (ovc_is_late_fence(tree.ovc[0])) return false;
    out_ovc = tree.ovc[0];
    out_src = tree.src[0];
    return true;
}

// ── Mark current winner's source as exhausted ──────────────────────
// Maps from: LoserTreeOVC::mark_current_exhausted
__device__ inline bool loser_tree_mark_exhausted(
    LoserTreeState& tree,
    uint32_t& out_ovc,
    int& out_src
) {
    if (ovc_is_late_fence(tree.ovc[0])) return false;
    int exhausted_src = tree.src[0];
    loser_tree_push(tree, OVC_LATE_FENCE, exhausted_src, out_ovc, out_src);
    return true;
}

// ── Replace root OVC without full push (duplicate shortcut) ────────
// Maps from: LoserTreeOVC::replace_top_ovc (tree_of_losers_ovc.rs)
// Used when consecutive records have identical keys (DuplicateValue flag).
// Swaps OVC at root without disturbing tree structure.
__device__ inline uint32_t loser_tree_replace_top_ovc(
    LoserTreeState& tree,
    uint32_t new_ovc
) {
    uint32_t old_ovc = tree.ovc[0];
    tree.ovc[0] = new_ovc;
    return old_ovc;
}
