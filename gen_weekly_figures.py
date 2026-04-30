#!/usr/bin/env python3
"""Generate the weekly-progress charts from real CSV data."""
import os, csv, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "results", "overnight_pulled")
OUT  = os.path.join(ROOT, "results", "figures_weekly")
os.makedirs(OUT, exist_ok=True)

DARK   = "#0f172a"
INK    = "#1f2937"
GRAY   = "#6b7280"
GPU    = "#10b981"
CPU    = "#dc2626"
ACCENT = "#2563eb"
AMBER  = "#d97706"
PURPLE = "#7c3aed"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlepad": 12,
    "figure.dpi": 130,
})

def title(ax, t, sub=None):
    pad = 28 if sub else 12
    ax.set_title(t, color=DARK, loc="left", pad=pad)
    if sub:
        # Place subtitle just above the axes, below the title
        ax.text(0, 1.02, sub, transform=ax.transAxes,
                fontsize=9, color=GRAY, style="italic", va="bottom")

def save(fig, name):
    fig.tight_layout()
    fp = os.path.join(OUT, name + ".png")
    fig.savefig(fp, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {fp}")

# ─────────────────────────────────────────────────────────────
# Figure 1: DuckDB vs CrocSort wall time per machine
# ─────────────────────────────────────────────────────────────
def fig_duckdb_vs_crocsort():
    # Pulled from baselines + GPU runs (real measured numbers, best of 3 warm).
    data = [
        # (machine, sf, gpu_s, duckdb_s, polars_s_or_None)
        ("RTX 2080\n8 GB",  "SF10",  1.81,   8.41,  13.42),
        ("RTX 2080\n8 GB",  "SF20",  2.76,  21.74,  27.61),
        ("P5000\n16 GB",    "SF10",  2.66,   7.81,  13.34),
        ("P5000\n16 GB",    "SF20",  5.54,  19.19,  28.67),
        ("P5000\n16 GB",    "SF50",  7.72, 223.64,  None),  # Polars OOM
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(data))
    w = 0.27
    duck = [d[3] for d in data]
    polars = [d[4] if d[4] else 0 for d in data]
    croc = [d[2] for d in data]
    labels = [f"{d[0]}\n{d[1]}" for d in data]

    ax.bar(x - w, duck,   w, label="DuckDB",   color=CPU,    alpha=0.85)
    ax.bar(x,     polars, w, label="Polars",   color=PURPLE, alpha=0.85)
    ax.bar(x + w, croc,   w, label="CrocSort", color=GPU,    alpha=0.85)

    for i, (d, p, c) in enumerate(zip(duck, polars, croc)):
        # Speedup vs DuckDB above CrocSort bar
        s_d = d / c
        ax.text(i + w, c + max(duck)*0.015, f"{s_d:.1f}×",
                ha="center", fontsize=11, fontweight="bold", color=GPU)
        # Polars bar value (when present)
        if polars[i] > 0:
            ax.text(i, p + max(duck)*0.015, f"{p:.1f}s",
                    ha="center", fontsize=8.5, color=GRAY)
        else:
            ax.text(i, 1.0, "Polars\nOOM", ha="center", fontsize=8,
                    color=PURPLE, style="italic")
        # DuckDB bar value
        ax.text(i - w, d + max(duck)*0.015, f"{d:.1f}s",
                ha="center", fontsize=8.5, color=GRAY)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Wall time (seconds)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 500)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    title(ax, "DuckDB and Polars vs CrocSort — TPC-H ORDER BY",
          "Best of 3 warm runs · log scale · CPU baselines + GPU on the same machine")
    save(fig, "f1_duckdb_vs_crocsort")

# ─────────────────────────────────────────────────────────────
# Figure 2: memory envelope per GPU
# ─────────────────────────────────────────────────────────────
def fig_memory_envelope():
    gpus = ["RTX 2080\n8 GB", "P5000\n16 GB", "RTX 6000\n24 GB"]
    sf   = [20, 50, 100]
    gb   = [14.4, 36.0, 72.0]
    time_s = [2.76, 7.72, 3.74]
    color  = [AMBER, PURPLE, GPU]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(gpus, gb, color=color, alpha=0.85, width=0.55)
    for i, (b, s, t, g) in enumerate(zip(bars, sf, time_s, gb)):
        ax.text(b.get_x() + b.get_width()/2, g + 1.5,
                f"SF{s}\n{g} GB sorted in {t}s",
                ha="center", fontsize=10, color=DARK, fontweight="bold")
    ax.set_ylabel("Largest TPC-H scale that fits (GB)")
    ax.set_ylim(0, 90)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    title(ax, "Memory envelope per GPU",
          "Capped by merge-phase arena = num_records × 24 B + key buffers")
    save(fig, "f2_memory_envelope")

# ─────────────────────────────────────────────────────────────
# Figure 3: per-column FOR+bitpack compression
# ─────────────────────────────────────────────────────────────
def fig_codec_ratios():
    cols = []
    raw  = []
    comp = []
    for r in csv.DictReader(open(os.path.join(SRC, "a3_codec_ratios.csv"))):
        if int(r["scale_factor"]) != 10: continue
        if r["codec"] != "FOR+bitpack": continue
        cols.append(r["column"].replace("l_", ""))
        raw.append(float(r["raw_bytes"]))
        comp.append(float(r["compressed_bytes"]))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(cols))
    w = 0.38
    ax.bar(x - w/2, raw,  w, color=CPU, alpha=0.7, label="raw bytes")
    ax.bar(x + w/2, comp, w, color=GPU, alpha=0.85, label="FOR + bit-pack")
    for i, (rb, cb) in enumerate(zip(raw, comp)):
        ratio = rb/cb if cb > 0 else 0
        ax.text(i, max(rb, cb) + 0.25, f"{ratio:.1f}×",
                ha="center", fontsize=9, fontweight="bold", color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Bytes per row")
    ax.set_ylim(0, 10)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    total_raw  = sum(raw)
    total_comp = sum(comp)
    title(ax, "Per-column FOR + bit-packing on TPC-H sort key",
          f"Total: {total_raw:.0f} B raw → {total_comp:.1f} B compressed → "
          f"{total_raw/total_comp:.2f}× extra on top of compact-key scan")
    save(fig, "f3_codec_ratios")

# ─────────────────────────────────────────────────────────────
# Figure 4: GPU codec decode bandwidth vs PCIe ceiling
# ─────────────────────────────────────────────────────────────
def fig_decode_bw():
    fig, ax = plt.subplots(figsize=(9, 5))

    # Combine FOR + bitpack data
    rows = []
    for fn, codec in [("b1_for_decode.csv", "FOR"),
                      ("b2_bitpack_decode.csv", "Bit-pack")]:
        for r in csv.reader(open(os.path.join(SRC, fn))):
            if r[0].startswith("experiment"): continue
            try:
                gb = float(r[-1])
                rows.append((codec, r[2], gb))
            except ValueError: continue

    # Get max throughput per codec
    best = defaultdict(float)
    for codec, w, gb in rows:
        best[(codec, w)] = max(best[(codec, w)], gb)

    items = sorted(best.items())
    labels = [f"{c}\n{w}" + ("B" if c=="FOR" else " bits") for (c,w),_ in items]
    vals   = [v for _, v in items]
    colors = [GPU if c=="FOR" else PURPLE for (c,_),_ in items]

    bars = ax.bar(range(len(items)), vals, color=colors, alpha=0.85, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 12, f"{v:.0f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(12, color=CPU, linestyle="--", linewidth=1.5, alpha=0.8,
               label="PCIe 3.0 ceiling (12 GB/s)")
    ax.text(len(items)-0.3, 14, "PCIe 3.0", color=CPU, fontsize=9,
            fontweight="bold", ha="right")

    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Decode throughput (GB/s)")
    ax.set_ylim(0, max(vals)*1.15)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    title(ax, "GPU codec decode throughput vs PCIe ceiling",
          "Decode is ~40× faster than PCIe — codec is essentially free")
    save(fig, "f4_decode_bandwidth")

# ─────────────────────────────────────────────────────────────
# Figure 5: sort time vs key width
# ─────────────────────────────────────────────────────────────
def fig_sort_vs_keywidth():
    rows = []
    for r in csv.reader(open(os.path.join(SRC, "b3_direct_sort.csv"))):
        try:
            w = int(r[2]); t = float(r[3]); thru = float(r[4])
            rows.append((w, t, thru))
        except ValueError: continue
    rows.sort()
    widths   = [r[0] for r in rows]
    times    = [r[1] for r in rows]
    thrupts  = [r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.plot(widths, times, "o-", color=CPU, linewidth=2, markersize=8)
    for w, t in zip(widths, times):
        ax1.annotate(f"{t:.1f} ms", (w, t), textcoords="offset points",
                     xytext=(8, 6), fontsize=9, color=DARK)
    ax1.set_xlabel("Key width (bytes)")
    ax1.set_ylabel("Sort time (ms) for 100M records")
    ax1.set_xticks(widths)
    ax1.grid(True, alpha=0.2)
    title(ax1, "Sort time vs key width")

    ax2.plot(widths, thrupts, "o-", color=GPU, linewidth=2, markersize=8)
    for w, th in zip(widths, thrupts):
        ax2.annotate(f"{th:.0f} GB/s", (w, th), textcoords="offset points",
                     xytext=(8, -12), fontsize=9, color=DARK)
    ax2.set_xlabel("Key width (bytes)")
    ax2.set_ylabel("Throughput (GB/s)")
    ax2.set_xticks(widths)
    ax2.grid(True, alpha=0.2)
    title(ax2, "Throughput vs key width",
          "Smaller keys → fewer radix passes → faster sort")
    save(fig, "f5_sort_vs_keywidth")

# ─────────────────────────────────────────────────────────────
# Figure 6: K-way merge slowdown
# ─────────────────────────────────────────────────────────────
def fig_kway_merge():
    K, throughput, slowdown = [], [], []
    for r in csv.reader(open(os.path.join(SRC, "d1_merge_profile.csv"))):
        try:
            K.append(int(r[0])); throughput.append(float(r[3]))
            slowdown.append(float(r[5]))
        except ValueError: continue

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar([str(k) for k in K], slowdown,
                  color=[GPU if s < 1.5 else CPU for s in slowdown],
                  alpha=0.85, width=0.55)
    for b, s, t in zip(bars, slowdown, throughput):
        ax.text(b.get_x() + b.get_width()/2, s + 0.04,
                f"{s:.2f}×\n{t:.0f} M/s",
                ha="center", fontsize=9, color=DARK)
    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("K (merge fan-in)")
    ax.set_ylabel("Slowdown vs K=2")
    ax.set_ylim(0, max(slowdown)*1.18)
    ax.grid(True, axis="y", alpha=0.2)
    title(ax, "K-way CPU merge cost grows fast in K",
          "Branch-divergent + cache-hostile — the phase GPU sample-sort is meant to remove")
    save(fig, "f6_kway_merge")

# ─────────────────────────────────────────────────────────────
# Figure 7: time breakdown of the pipeline (SF100 on RTX 6000)
# ─────────────────────────────────────────────────────────────
def fig_time_breakdown():
    phases = [
        ("CPU key encoding",   55, CPU),
        ("PCIe H→D upload",    29, AMBER),
        ("Permutation gather",  7, PURPLE),
        ("GPU radix sort",      9, GPU),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    left = 0
    narrow_below = True   # alternate narrow labels above/below to avoid overlap
    for name, pct, color in phases:
        ax.barh(0, pct, left=left, color=color, alpha=0.85, height=0.4)
        if pct >= 12:
            ax.text(left + pct/2, 0, f"{name}\n{pct}%",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white")
        else:
            y = -0.40 if narrow_below else 0.40
            va = "top" if narrow_below else "bottom"
            # leader line from bar edge to label
            ax.plot([left + pct/2, left + pct/2], [0, y*0.55],
                    color=color, linewidth=1, alpha=0.6)
            ax.text(left + pct/2, y, f"{name} · {pct}%",
                    ha="center", va=va, fontsize=9,
                    fontweight="bold", color=color)
            narrow_below = not narrow_below
        left += pct
    ax.set_xlim(-2, 102); ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)
    title(ax, "Where the time goes — SF100 on RTX 6000",
          "GPU sort kernel is only 9% of wall time. PCIe + encoding is 84%.")
    save(fig, "f7_time_breakdown")

# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Figure 8: bitpack pipeline integration — measured PCIe + chunks
# ─────────────────────────────────────────────────────────────
def fig_bitpack_integration():
    # Measured 2026-04-30 on the actual sort path
    rows = [
        # (machine, scale, label, baseline_pcie_gb, bitpack_pcie_gb, baseline_chunks, bitpack_chunks)
        ("RTX 2080\n8 GB",  "SF20\n14 GB", 3.84, 2.88, 3, 2),
        ("P5000\n16 GB",    "SF50\n36 GB", 9.60, 7.20, 3, 2),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: PCIe bytes
    labels = [f"{r[0]}\n{r[1]}" for r in rows]
    base   = [r[2] for r in rows]
    bp     = [r[3] for r in rows]
    x = np.arange(len(rows)); w = 0.35
    ax1.bar(x - w/2, base, w, color=CPU,  alpha=0.85, label="baseline")
    ax1.bar(x + w/2, bp,   w, color=GPU,  alpha=0.85, label="USE_BITPACK")
    for i, (b, p) in enumerate(zip(base, bp)):
        ratio = b/p if p > 0 else 0
        ax1.text(i, max(b, p) + 0.4, f"{(1-p/b)*100:.0f}% less PCIe",
                 ha="center", fontsize=10, fontweight="bold", color=GPU)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("H→D PCIe bytes (GB)")
    ax1.set_ylim(0, max(base) * 1.30)
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, axis="y", alpha=0.2, zorder=0)
    title(ax1, "PCIe bytes uploaded — actual measurement",
          "CPU FOR + bit-pack → ~25% fewer bytes on the wire")

    # Right: chunks
    base_c = [r[4] for r in rows]
    bp_c   = [r[5] for r in rows]
    ax2.bar(x - w/2, base_c, w, color=CPU,  alpha=0.85, label="baseline")
    ax2.bar(x + w/2, bp_c,   w, color=GPU,  alpha=0.85, label="USE_BITPACK")
    for i, (b, p) in enumerate(zip(base_c, bp_c)):
        ax2.text(i - w/2, b + 0.05, str(b), ha="center", fontsize=10)
        ax2.text(i + w/2, p + 0.05, str(p), ha="center", fontsize=10,
                 fontweight="bold", color=GPU)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Number of chunks")
    ax2.set_ylim(0, max(base_c) + 1.5)
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(True, axis="y", alpha=0.2, zorder=0)
    title(ax2, "Chunk count drops too",
          "Smaller keys → more records fit per GPU chunk")
    save(fig, "f8_bitpack_pipeline")

print("Generating weekly figures...")
fig_duckdb_vs_crocsort()
fig_memory_envelope()
fig_codec_ratios()
fig_decode_bw()
fig_sort_vs_keywidth()
fig_kway_merge()
fig_time_breakdown()
fig_bitpack_integration()
print(f"\nAll charts → {OUT}/")
