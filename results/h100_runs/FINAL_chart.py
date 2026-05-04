#!/usr/bin/env python3
"""Final chart: throughput vs scale across all engines + multi-GPU.

Plots:
  - gpu_crocsort post-0.3.1 (single-GPU): SF10/50/100/300
  - gpu_crocsort 4-GPU partition+sort (15.4): SF500
  - gpu_crocsort distributed (15.5.3 paired): SF500
  - DuckDB: SF50, SF100
  - Polars: SF50, SF100
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Data tabulated from the run results.
# Updated 2026-05-04: single-GPU and 4-GPU-partition use numactl wrap.
gpu_crocsort_1g = [
    ("SF10",    7.20, 0.352, 20.4),    # not yet retested under numactl
    ("SF50",   36.00, 1.41,  25.6),    # 17.3.2.2 numactl warm best
    ("SF100",  72.00, 2.95,  24.4),    # 17.3.2.2 numactl warm best
    ("SF300", 216.00, 6.59,  32.7),    # 17.3.2.7.2 numactl --preferred best of 8 warm
]
gpu_crocsort_4g_partition = [
    ("SF500-4G-partition",  360.00, 5.48, 65.7),  # 17.3.2.3 numactl spread, run 2 best
]
gpu_crocsort_4g_distributed = [
    ("SF500-4G-distributed", 360.00, 430.0, 0.84),   # 17.3.2.3.3 numactl --preferred, 7m10s
    ("SF1000-4G-distributed", 720.00, 1208.0, 0.60), # 17.3.2.7.6 SF1000 with --preferred 20m08s
]
duckdb_1g = [
    ("SF50",   36.00, 33.0, 1.09),
    ("SF100",  72.00, 106.0, 0.68),
]
polars_1g = [
    ("SF50",   36.00, 14.5, 2.48),
    ("SF100",  72.00, 28.6, 2.52),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

def plot_series(ax, data, label, marker, color, linestyle="-"):
    xs = [d[1] for d in data]  # GB
    ys_t = [d[2] for d in data]  # seconds
    ys_g = [d[3] for d in data]  # GB/s
    ax_t = ax  # left axis = wall time
    ax_t.plot(xs, ys_t, marker=marker, color=color, linestyle=linestyle,
              label=label, markersize=10, linewidth=2)

# Wall time chart (log-log)
plot_series(ax1, gpu_crocsort_1g, "gpu_crocsort 1×H100 (post-0.3.1)", "o", "#1f77b4")
plot_series(ax1, gpu_crocsort_4g_partition, "gpu_crocsort 4×H100 partition (15.4 sort phase)", "P", "#2ca02c")
plot_series(ax1, gpu_crocsort_4g_distributed, "gpu_crocsort 4×H100 distributed sort (15.5.3 paired)", "*", "#d62728")
plot_series(ax1, duckdb_1g, "DuckDB 1.3.2 (CPU)", "s", "#ff7f0e", "--")
plot_series(ax1, polars_1g, "Polars 1.8.2 (CPU)", "v", "#9467bd", "--")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Input size (GB raw lineitem)")
ax1.set_ylabel("Wall time (s, log)")
ax1.set_title("Sort wall time vs scale on 4× H100 NVL box")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="upper left", fontsize=8)

# Throughput chart
def plot_thr(ax, data, label, marker, color, linestyle="-"):
    xs = [d[1] for d in data]
    ys = [d[3] for d in data]
    ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle,
            label=label, markersize=10, linewidth=2)

plot_thr(ax2, gpu_crocsort_1g, "gpu_crocsort 1×H100", "o", "#1f77b4")
plot_thr(ax2, gpu_crocsort_4g_partition, "gpu_crocsort 4×H100 partition", "P", "#2ca02c")
plot_thr(ax2, gpu_crocsort_4g_distributed, "gpu_crocsort 4×H100 distributed", "*", "#d62728")
plot_thr(ax2, duckdb_1g, "DuckDB 1.3.2", "s", "#ff7f0e", "--")
plot_thr(ax2, polars_1g, "Polars 1.8.2", "v", "#9467bd", "--")
ax2.set_xscale("log")
ax2.set_xlabel("Input size (GB raw lineitem)")
ax2.set_ylabel("Effective throughput (GB / s)")
ax2.set_title("Sort throughput vs scale")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper left", fontsize=8)
ax2.set_ylim(0.1, 100)
ax2.set_yscale("log")

# Annotations
ax1.annotate("DuckDB / Polars hit single-CPU\n→ ~10-30× slower than gpu_crocsort",
             xy=(72, 30), xytext=(8, 100),
             fontsize=8, ha="left", color="#666666",
             arrowprops=dict(arrowstyle="->", color="#666666", lw=0.5))
ax1.annotate("Single-GPU OOMs at SF500;\n4-GPU only path beyond this",
             xy=(360, 6.77), xytext=(20, 0.5),
             fontsize=8, ha="left", color="#888888",
             arrowprops=dict(arrowstyle="->", color="#888888", lw=0.5))

fig.suptitle("H100 NVL TPC-H lineitem sort — gpu_crocsort vs CPU baselines",
             y=1.02, fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "FINAL_throughput.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
