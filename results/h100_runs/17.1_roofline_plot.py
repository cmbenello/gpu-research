#!/usr/bin/env python3
# 17.1.1 — Roofline efficiency bar chart
#
# For each phase of the SF300 single-GPU sort, plot:
#   measured throughput   |||||||
#   theoretical ceiling   ============
# The gap is the optimization headroom.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

phases = [
    "CPU compact extract",
    "PCIe upload",
    "GPU LSD radix sort",
    "GPU OVC merge (HBM)",
    "CPU gather (random)",
]
# Effective ceilings tied to the real bottleneck of each phase
# (DDR5 host RAM, PCIe5, HBM3 effective from 6.2)
ceilings_gbs  = [400.0, 32.0, 2191.0, 2191.0, 400.0]
measured_gbs  = [ 90.0, 19.0,  900.0, 1300.0,  44.0]
notes = [
    "DDR5 host (~400 GB/s)",
    "PCIe 5 x16 (~32 GB/s)",
    "HBM3 read+write peak (6.2: 2191 GB/s)",
    "HBM3 read+write peak",
    "DDR5 host (~400 GB/s)",
]

x = np.arange(len(phases))
fig, ax = plt.subplots(figsize=(11, 6))

w = 0.4
b1 = ax.bar(x - w/2, ceilings_gbs, w, color="#888", label="Theoretical ceiling")
b2 = ax.bar(x + w/2, measured_gbs, w, color="#2c7", label="Measured (SF300)")

for i, (m, c) in enumerate(zip(measured_gbs, ceilings_gbs)):
    eff = m / c * 100
    ax.text(x[i] + w/2, m + max(ceilings_gbs)*0.02,
            f"{eff:.0f}%", ha="center", fontsize=9, color="#163")

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(phases, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("GB/s (log scale)")
ax.set_title("17.1 — H100 NVL sort phases vs hardware ceilings (SF300)")
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")

# Annotate notes under each bar pair
for i, n in enumerate(notes):
    ax.text(x[i], 1.0, n, fontsize=7, ha="center", color="#666",
            transform=ax.get_xaxis_transform(), rotation=0,
            verticalalignment="top")

plt.tight_layout()
out = "/home/cc/gpu-research/results/h100_runs/17.1_roofline.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")

# Also: a simpler "% of ceiling" chart
fig2, ax2 = plt.subplots(figsize=(9, 5))
eff_pct = [m/c*100 for m, c in zip(measured_gbs, ceilings_gbs)]
colors = ["#2c7" if e >= 50 else "#e90" if e >= 25 else "#c33" for e in eff_pct]
ax2.barh(phases, eff_pct, color=colors)
for i, e in enumerate(eff_pct):
    ax2.text(e + 1, i, f"{e:.0f}%", va="center", fontsize=10)
ax2.set_xlabel("Efficiency vs hardware ceiling (%)")
ax2.set_xlim(0, 100)
ax2.set_title("17.1 — Phase efficiency: gather is the biggest gap")
ax2.axvline(50, color="#888", linestyle="--", alpha=0.5)
ax2.axvline(80, color="#888", linestyle=":", alpha=0.5)
plt.tight_layout()
out2 = "/home/cc/gpu-research/results/h100_runs/17.1_efficiency.png"
plt.savefig(out2, dpi=140)
print(f"wrote {out2}")
