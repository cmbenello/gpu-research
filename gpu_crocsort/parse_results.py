#!/usr/bin/env python3
"""
GPU CrocSort -- Benchmark Results Parser

Parses CSV output from the benchmark suite, generates summary statistics,
ASCII bar charts, and a strategy comparison report.

Usage:
    python3 parse_results.py [benchmark_results.csv]
    python3 parse_results.py --help
"""

import csv
import sys
import os
from collections import defaultdict

# ── CSV parsing ──────────────────────────────────────────────────────

def parse_csv(filename):
    """Parse benchmark CSV into a list of result dicts."""
    results = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "gpu": row["gpu"],
                "num_records": int(row["num_records"]),
                "distribution": row["distribution"],
                "strategy": row["strategy"],
                "total_ms": float(row["total_ms"]),
                "run_gen_ms": float(row["run_gen_ms"]),
                "merge_ms": float(row["merge_ms"]),
                "throughput_gbs": float(row["throughput_gbs"]),
                "throughput_mrec_s": float(row["throughput_mrec_s"]),
                "merge_passes": int(row["merge_passes"]),
                "hbm_traffic_gb": float(row["hbm_traffic_gb"]),
                "verify": row["verify"],
            })
    return results


def human_size(n):
    """Format record count for display."""
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    elif n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


# ── Summary statistics ───────────────────────────────────────────────

def print_summary(results):
    """Print overall summary statistics."""
    if not results:
        print("No results to summarize.")
        return

    gpu_name = results[0]["gpu"]
    total = len(results)
    fails = sum(1 for r in results if r["verify"] != "pass")

    throughputs = [r["throughput_gbs"] for r in results]
    mrec_s = [r["throughput_mrec_s"] for r in results]
    times = [r["total_ms"] for r in results]

    print("=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  GPU              : {gpu_name}")
    print(f"  Total tests      : {total}")
    print(f"  Verify failures  : {fails}")
    print(f"  Best throughput  : {max(throughputs):.3f} GB/s ({max(mrec_s):.2f} Mrec/s)")
    print(f"  Worst throughput : {min(throughputs):.3f} GB/s")
    print(f"  Mean throughput  : {sum(throughputs)/len(throughputs):.3f} GB/s")
    print(f"  Fastest sort     : {min(times):.2f} ms")
    print(f"  Slowest sort     : {max(times):.2f} ms")
    print("=" * 70)
    print()


# ── Per-size statistics ──────────────────────────────────────────────

def print_per_size_stats(results):
    """Print statistics grouped by data size."""
    by_size = defaultdict(list)
    for r in results:
        by_size[r["num_records"]].append(r)

    print("=" * 70)
    print("  THROUGHPUT BY DATA SIZE")
    print("=" * 70)
    print(f"{'Size':>10s}  {'Tests':>5s}  {'Min GB/s':>10s}  {'Mean GB/s':>10s}  "
          f"{'Max GB/s':>10s}  {'Mean Mrec/s':>12s}")
    print("-" * 70)

    for size in sorted(by_size.keys()):
        group = by_size[size]
        thr = [r["throughput_gbs"] for r in group]
        mrec = [r["throughput_mrec_s"] for r in group]
        print(f"{human_size(size):>10s}  {len(group):>5d}  {min(thr):>10.3f}  "
              f"{sum(thr)/len(thr):>10.3f}  {max(thr):>10.3f}  "
              f"{sum(mrec)/len(mrec):>12.2f}")

    print("=" * 70)
    print()


# ── ASCII bar chart ──────────────────────────────────────────────────

def ascii_bar(value, max_value, width=40):
    """Generate an ASCII bar of proportional length."""
    if max_value <= 0:
        return ""
    bar_len = int(round(value / max_value * width))
    bar_len = max(0, min(width, bar_len))
    return "#" * bar_len


def print_throughput_chart(results, strategy_filter=None):
    """Print ASCII bar chart of throughput by size and distribution."""
    filtered = results
    if strategy_filter:
        filtered = [r for r in results if r["strategy"] == strategy_filter]

    if not filtered:
        return

    title = f"  THROUGHPUT (GB/s) -- {strategy_filter or 'ALL'}"
    print("=" * 70)
    print(title)
    print("=" * 70)

    # Group by size, then average over distributions
    by_size = defaultdict(list)
    for r in filtered:
        by_size[r["num_records"]].append(r)

    max_thr = max(r["throughput_gbs"] for r in filtered) if filtered else 1.0

    for size in sorted(by_size.keys()):
        group = by_size[size]
        avg_thr = sum(r["throughput_gbs"] for r in group) / len(group)
        bar = ascii_bar(avg_thr, max_thr, 40)
        print(f"  {human_size(size):>8s}  {bar}  {avg_thr:.3f}")

    print()

    # Now per-distribution for the largest size
    largest_size = max(by_size.keys())
    largest = [r for r in filtered if r["num_records"] == largest_size]

    if largest:
        print(f"  Distribution breakdown @ {human_size(largest_size)} records:")
        print(f"  {'Distribution':<20s}  {'Bar':40s}  {'GB/s':>8s}")
        print("  " + "-" * 72)
        max_dist_thr = max(r["throughput_gbs"] for r in largest)
        for r in sorted(largest, key=lambda x: -x["throughput_gbs"]):
            bar = ascii_bar(r["throughput_gbs"], max_dist_thr, 40)
            print(f"  {r['distribution']:<20s}  {bar}  {r['throughput_gbs']:.3f}")

    print("=" * 70)
    print()


# ── Phase breakdown ──────────────────────────────────────────────────

def print_phase_breakdown(results):
    """Show run-gen vs merge time breakdown."""
    print("=" * 70)
    print("  PHASE BREAKDOWN (% of total time)")
    print("=" * 70)
    print(f"{'Size':>10s}  {'Strategy':>8s}  {'RunGen%':>8s}  {'Merge%':>8s}  "
          f"{'RunGen ms':>10s}  {'Merge ms':>10s}  {'Total ms':>10s}")
    print("-" * 70)

    # Average across distributions for each (size, strategy)
    groups = defaultdict(list)
    for r in results:
        groups[(r["num_records"], r["strategy"])].append(r)

    for (size, strat) in sorted(groups.keys()):
        group = groups[(size, strat)]
        avg_rg = sum(r["run_gen_ms"] for r in group) / len(group)
        avg_mg = sum(r["merge_ms"] for r in group) / len(group)
        avg_total = avg_rg + avg_mg
        if avg_total > 0:
            rg_pct = 100.0 * avg_rg / avg_total
            mg_pct = 100.0 * avg_mg / avg_total
        else:
            rg_pct = mg_pct = 0.0
        print(f"{human_size(size):>10s}  {strat:>8s}  {rg_pct:>7.1f}%  {mg_pct:>7.1f}%  "
              f"{avg_rg:>10.2f}  {avg_mg:>10.2f}  {avg_total:>10.2f}")

    print("=" * 70)
    print()


# ── Strategy comparison ──────────────────────────────────────────────

def print_strategy_comparison(results):
    """Compare 2-way vs 8-way merge strategies."""
    print("=" * 70)
    print("  2-WAY vs 8-WAY MERGE COMPARISON")
    print("=" * 70)

    # Group by (size, distribution)
    pairs = defaultdict(dict)
    for r in results:
        key = (r["num_records"], r["distribution"])
        pairs[key][r["strategy"]] = r

    print(f"{'Size':>10s}  {'Distribution':<20s}  {'2-way ms':>10s}  "
          f"{'8-way ms':>10s}  {'Speedup':>9s}  {'Pass 2w':>7s}  {'Pass 8w':>7s}")
    print("-" * 80)

    speedups = []

    for (size, dist) in sorted(pairs.keys()):
        entry = pairs[(size, dist)]
        t2 = entry.get("2-way", {}).get("total_ms", -1)
        t8 = entry.get("8-way", {}).get("total_ms", -1)
        p2 = entry.get("2-way", {}).get("merge_passes", -1)
        p8 = entry.get("8-way", {}).get("merge_passes", -1)

        if t2 > 0 and t8 > 0:
            speedup = t2 / t8
            speedups.append(speedup)
            print(f"{human_size(size):>10s}  {dist:<20s}  {t2:>10.2f}  "
                  f"{t8:>10.2f}  {speedup:>8.2f}x  {p2:>7d}  {p8:>7d}")
        elif t2 > 0:
            print(f"{human_size(size):>10s}  {dist:<20s}  {t2:>10.2f}  "
                  f"{'N/A':>10s}  {'N/A':>9s}  {p2:>7d}  {'N/A':>7s}")
        elif t8 > 0:
            print(f"{human_size(size):>10s}  {dist:<20s}  {'N/A':>10s}  "
                  f"{t8:>10.2f}  {'N/A':>9s}  {'N/A':>7s}  {p8:>7d}")

    if speedups:
        print("-" * 80)
        avg_speedup = sum(speedups) / len(speedups)
        geo_speedup = 1.0
        for s in speedups:
            geo_speedup *= s
        geo_speedup = geo_speedup ** (1.0 / len(speedups))

        print(f"{'':>10s}  {'AVERAGE':<20s}  {'':>10s}  {'':>10s}  "
              f"{avg_speedup:>8.2f}x")
        print(f"{'':>10s}  {'GEOMETRIC MEAN':<20s}  {'':>10s}  {'':>10s}  "
              f"{geo_speedup:>8.2f}x")
        winner = "8-way" if avg_speedup > 1.0 else "2-way"
        print(f"\n  Winner: {winner} merge (avg {avg_speedup:.2f}x)")

    print("=" * 70)
    print()


# ── HBM traffic analysis ────────────────────────────────────────────

def print_hbm_analysis(results):
    """Analyze HBM traffic patterns."""
    print("=" * 70)
    print("  HBM TRAFFIC ANALYSIS")
    print("=" * 70)

    groups = defaultdict(list)
    for r in results:
        groups[(r["num_records"], r["strategy"])].append(r)

    print(f"{'Size':>10s}  {'Strategy':>8s}  {'Data GB':>8s}  {'HBM GB':>8s}  "
          f"{'Amplif.':>8s}  {'Passes':>7s}")
    print("-" * 60)

    for (size, strat) in sorted(groups.keys()):
        group = groups[(size, strat)]
        data_gb = size * 100 / 1e9  # RECORD_SIZE = 100
        avg_hbm = sum(r["hbm_traffic_gb"] for r in group) / len(group)
        avg_passes = sum(r["merge_passes"] for r in group) / len(group)
        amplification = avg_hbm / data_gb if data_gb > 0 else 0
        print(f"{human_size(size):>10s}  {strat:>8s}  {data_gb:>8.3f}  {avg_hbm:>8.3f}  "
              f"{amplification:>7.1f}x  {avg_passes:>7.1f}")

    print("=" * 70)
    print()


# ── Distribution sensitivity ────────────────────────────────────────

def print_distribution_sensitivity(results):
    """Show how throughput varies by distribution for each size."""
    print("=" * 70)
    print("  DISTRIBUTION SENSITIVITY")
    print("=" * 70)

    sizes = sorted(set(r["num_records"] for r in results))

    for size in sizes:
        size_results = [r for r in results if r["num_records"] == size]
        if not size_results:
            continue

        print(f"\n  {human_size(size)} records:")
        print(f"  {'Distribution':<20s}  {'2-way GB/s':>12s}  {'8-way GB/s':>12s}")
        print("  " + "-" * 48)

        by_dist = defaultdict(dict)
        for r in size_results:
            by_dist[r["distribution"]][r["strategy"]] = r["throughput_gbs"]

        for dist in sorted(by_dist.keys()):
            t2 = by_dist[dist].get("2-way", -1)
            t8 = by_dist[dist].get("8-way", -1)
            t2_str = f"{t2:.3f}" if t2 > 0 else "N/A"
            t8_str = f"{t8:.3f}" if t8 > 0 else "N/A"
            print(f"  {dist:<20s}  {t2_str:>12s}  {t8_str:>12s}")

    print()
    print("=" * 70)
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print(__doc__)
        if len(sys.argv) < 2:
            # Try default filename
            default = "benchmark_results.csv"
            if os.path.exists(default):
                filename = default
                print(f"Using default: {default}\n")
            else:
                print(f"No {default} found. Provide a CSV file path as argument.")
                sys.exit(1)
        else:
            sys.exit(0)
    else:
        filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"ERROR: File not found: {filename}")
        sys.exit(1)

    print(f"\nParsing: {filename}\n")
    results = parse_csv(filename)

    if not results:
        print("No results found in CSV.")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark results.\n")

    # Generate all reports
    print_summary(results)
    print_per_size_stats(results)
    print_throughput_chart(results, "2-way")
    print_throughput_chart(results, "8-way")
    print_phase_breakdown(results)
    print_strategy_comparison(results)
    print_hbm_analysis(results)
    print_distribution_sensitivity(results)

    print("Report complete.")


if __name__ == "__main__":
    main()
