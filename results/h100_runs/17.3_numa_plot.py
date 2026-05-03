#!/usr/bin/env python3
# 17.3 — Bar chart of NUMA gather policies.
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "/home/cc/gpu-research/results/h100_runs/17.3_numa_gather.csv"
rows = []
with open(path) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append((r["policy"], int(r["wall_ms"]), float(r["gbs"])))

policies = [r[0] for r in rows]
gbs = [r[2] for r in rows]
walls = [r[1]/1000.0 for r in rows]

colors = ["#c33", "#2c7", "#3c4", "#888"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(policies, gbs, color=colors)
default = next(g for p, g in zip(policies, gbs) if p == "default")
for bar, g, w in zip(bars, gbs, walls):
    speedup = g / default
    ax.text(bar.get_x() + bar.get_width()/2, g + 0.5,
            f"{g:.1f} GB/s\n{w:.1f} s\n{speedup:.1f}x",
            ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("Gather throughput (GB/s)")
ax.set_title("17.3 — NUMA gather: SF100 (72 GB), 32 threads, cold-cache\n"
             "Single-node binding gives 3.9× speedup over kernel default")
ax.set_ylim(0, max(gbs) * 1.25)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = "/home/cc/gpu-research/results/h100_runs/17.3_numa_gather.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")
