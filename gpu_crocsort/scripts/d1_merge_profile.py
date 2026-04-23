#!/usr/bin/env python3
"""D1: K-way merge profiling with perf stat.

Instrument the CPU K-way merge at different K values:
- Branch misprediction rate
- L1/L2/L3 miss rate
- Cycles per merged row

Uses synthetic pre-sorted runs to isolate merge cost.

Output: results/overnight/d1_merge_profile.csv
"""
import os, sys, subprocess, time, csv, struct, tempfile
import numpy as np

OUT = "results/overnight/d1_merge_profile.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def generate_sorted_runs(n_total, k, record_size=120):
    """Generate K sorted runs of n_total/K records each, interleaved for K-way merge."""
    run_size = n_total // k
    # Each record: 8-byte key (big-endian uint64) + padding
    data = bytearray(n_total * record_size)
    for run_idx in range(k):
        base = run_idx * run_size
        for i in range(run_size):
            # Key that sorts within each run
            key = struct.pack('>Q', i * k + run_idx)
            offset = (base + i) * record_size
            data[offset:offset+8] = key
    return bytes(data), run_size

def run_with_perf(cmd, env=None):
    """Run command under perf stat, parse counters."""
    perf_cmd = [
        'perf', 'stat', '-e',
        'cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses',
        '--'] + cmd

    try:
        result = subprocess.run(perf_cmd, capture_output=True, text=True, timeout=300, env=env)
        return result.stderr  # perf outputs to stderr
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None

def parse_perf(output):
    """Parse perf stat output into dict."""
    counters = {}
    if not output:
        return counters
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Format: "1,234,567      cycles" or "1,234,567      branches  # 45.6% of all ..."
        parts = line.split()
        if len(parts) >= 2:
            try:
                val = int(parts[0].replace(',', ''))
                name = parts[1]
                counters[name] = val
            except (ValueError, IndexError):
                pass
    return counters

def main():
    # Check perf is available
    try:
        subprocess.run(['perf', '--version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WARNING: perf not available. Using timing-only measurements.")
        use_perf = False
    else:
        use_perf = True

    results = []
    n_total = 100_000_000  # 100M records

    # Write a C program that does K-way merge so we can perf it
    merge_src = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Simple loser tree K-way merge
typedef struct {
    uint64_t key;
    int run_idx;
} Entry;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: merge_bench N K\n"); return 1; }
    long N = atol(argv[1]);
    int K = atoi(argv[2]);
    long run_size = N / K;

    // Generate K sorted runs in memory
    uint64_t** runs = (uint64_t**)malloc(K * sizeof(uint64_t*));
    long* pos = (long*)calloc(K, sizeof(long));
    for (int k = 0; k < K; k++) {
        runs[k] = (uint64_t*)malloc(run_size * sizeof(uint64_t));
        for (long i = 0; i < run_size; i++) {
            runs[k][i] = (uint64_t)i * K + k;  // interleaved sorted
        }
    }

    // K-way merge using tournament tree (loser tree)
    // Simplified: just use a heap
    Entry* heap = (Entry*)malloc(K * sizeof(Entry));
    int heap_size = K;

    // Initialize heap with first element from each run
    for (int k = 0; k < K; k++) {
        heap[k].key = runs[k][0];
        heap[k].run_idx = k;
        pos[k] = 1;
    }

    // Build min-heap
    for (int i = heap_size/2 - 1; i >= 0; i--) {
        int j = i;
        while (1) {
            int smallest = j;
            int left = 2*j + 1, right = 2*j + 2;
            if (left < heap_size && heap[left].key < heap[smallest].key) smallest = left;
            if (right < heap_size && heap[right].key < heap[smallest].key) smallest = right;
            if (smallest == j) break;
            Entry tmp = heap[j]; heap[j] = heap[smallest]; heap[smallest] = tmp;
            j = smallest;
        }
    }

    // Merge
    uint64_t prev = 0;
    long merged = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (heap_size > 0) {
        // Extract min
        Entry min_entry = heap[0];
        merged++;
        prev = min_entry.key;

        // Advance the run that produced the min
        int k = min_entry.run_idx;
        if (pos[k] < run_size) {
            heap[0].key = runs[k][pos[k]];
            pos[k]++;
        } else {
            // Run exhausted — move last heap element to root
            heap_size--;
            if (heap_size > 0) heap[0] = heap[heap_size];
        }

        // Sift down
        int j = 0;
        while (1) {
            int smallest = j;
            int left = 2*j + 1, right = 2*j + 2;
            if (left < heap_size && heap[left].key < heap[smallest].key) smallest = left;
            if (right < heap_size && heap[right].key < heap[smallest].key) smallest = right;
            if (smallest == j) break;
            Entry tmp = heap[j]; heap[j] = heap[smallest]; heap[smallest] = tmp;
            j = smallest;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("K=%d N=%ld merged=%ld time=%.3fs rate=%.1f Mrows/s\n",
           K, N, merged, elapsed, merged / elapsed / 1e6);

    // Cleanup
    for (int k = 0; k < K; k++) free(runs[k]);
    free(runs); free(pos); free(heap);
    return 0;
}
"""
    # Write and compile the merge benchmark
    src_path = "/tmp/merge_bench.c"
    bin_path = "/tmp/merge_bench"
    with open(src_path, 'w') as f:
        f.write(merge_src)

    subprocess.run(['gcc', '-O3', '-o', bin_path, src_path, '-lrt'], check=True)
    print(f"Compiled merge benchmark to {bin_path}")

    for K in [2, 4, 8, 16]:
        print(f"\n=== K={K}, N={n_total:,} ===")

        if use_perf:
            perf_output = run_with_perf([bin_path, str(n_total), str(K)])
            counters = parse_perf(perf_output)

            cycles = counters.get('cycles', 0)
            instructions = counters.get('instructions', 0)
            branches = counters.get('branches', 0)
            branch_misses = counters.get('branch-misses', 0)
            l1_loads = counters.get('L1-dcache-loads', 0)
            l1_misses = counters.get('L1-dcache-load-misses', 0)
            llc_loads = counters.get('LLC-loads', 0)
            llc_misses = counters.get('LLC-load-misses', 0)

            mispredict_rate = branch_misses / branches * 100 if branches > 0 else 0
            l1_miss_rate = l1_misses / l1_loads * 100 if l1_loads > 0 else 0
            llc_miss_rate = llc_misses / llc_loads * 100 if llc_loads > 0 else 0
            cycles_per_row = cycles / n_total if n_total > 0 else 0
            ipc = instructions / cycles if cycles > 0 else 0

            print(f"  Cycles/row: {cycles_per_row:.1f}")
            print(f"  IPC: {ipc:.2f}")
            print(f"  Branch mispredict: {mispredict_rate:.1f}%")
            print(f"  L1 miss rate: {l1_miss_rate:.2f}%")
            print(f"  LLC miss rate: {llc_miss_rate:.1f}%")

            if perf_output:
                # Also extract the timing line from the program itself
                for line in (perf_output or '').split('\n'):
                    if 'time=' in line:
                        print(f"  {line.strip()}")

            results.append({
                'K': K, 'N': n_total,
                'cycles_per_row': round(cycles_per_row, 1),
                'ipc': round(ipc, 2),
                'branch_mispredict_pct': round(mispredict_rate, 2),
                'l1_miss_rate_pct': round(l1_miss_rate, 2),
                'llc_miss_rate_pct': round(llc_miss_rate, 2),
                'total_cycles': cycles,
                'branches': branches,
                'branch_misses': branch_misses,
            })
        else:
            # Timing only
            result = subprocess.run([bin_path, str(n_total), str(K)],
                                   capture_output=True, text=True, timeout=300)
            print(f"  {result.stdout.strip()}")
            results.append({
                'K': K, 'N': n_total,
                'cycles_per_row': 0, 'ipc': 0,
                'branch_mispredict_pct': 0, 'l1_miss_rate_pct': 0,
                'llc_miss_rate_pct': 0,
                'total_cycles': 0, 'branches': 0, 'branch_misses': 0,
            })

    # Write CSV
    if results:
        with open(OUT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {len(results)} rows to {OUT}")

if __name__ == '__main__':
    main()
