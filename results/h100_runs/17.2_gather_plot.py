#!/usr/bin/env python3
# 17.2 — Plot gather GB/s vs thread count.
import csv, sys, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    out = []
    with open(path) as f:
        rdr = csv.reader(f)
        rows = [r for r in rdr if r and r[0] != 'threads' and not r[0].startswith('#')]
        for r in rows:
            try:
                out.append((int(r[0]), int(r[1]), float(r[2]), float(r[3])))
            except (ValueError, IndexError):
                continue
    return out

base = "/home/cc/gpu-research/results/h100_runs"
sources = [
    (f"{base}/17.2_gather_sweep.csv", "12 GB (100M records)"),
    (f"{base}/17.2_gather_sf100.csv", "72 GB (600M records)"),
    (f"{base}/17.2_gather_sf300.csv", "216 GB (1.8B records)"),
]

fig, ax = plt.subplots(figsize=(9, 5))
markers = ['o', 's', '^']
for (path, label), m in zip(sources, markers):
    if not os.path.exists(path):
        continue
    data = load(path)
    if not data:
        continue
    threads = [d[0] for d in data]
    gbs = [d[3] for d in data]
    ax.plot(threads, gbs, marker=m, label=label, linewidth=2)
    # Annotate peak
    peak_i = max(range(len(gbs)), key=lambda i: gbs[i])
    ax.annotate(f"peak {gbs[peak_i]:.0f} GB/s @ {threads[peak_i]}t",
                xy=(threads[peak_i], gbs[peak_i]),
                xytext=(5, -10), textcoords="offset points", fontsize=9)

ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 48, 64, 96, 128])
ax.set_xticklabels([1, 2, 4, 8, 16, 32, 48, 64, 96, 128])
ax.set_xlabel("Thread count (log2)")
ax.set_ylabel("Gather throughput (GB/s)")
ax.set_title("17.2 — CPU gather scaling on Xeon 8468 (random perm, 120 B records)")
ax.grid(True, alpha=0.3)
ax.legend()
ax.axhline(400, color="#888", linestyle="--", alpha=0.4, label="DDR5 ceiling")
ax.text(96, 400, "DDR5 ceiling ~400 GB/s", fontsize=8, color="#666",
        verticalalignment="bottom", horizontalalignment="right")
plt.tight_layout()
out = f"{base}/17.2_gather_scaling.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")
