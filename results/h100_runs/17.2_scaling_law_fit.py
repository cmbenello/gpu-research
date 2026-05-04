#!/usr/bin/env python3
"""17.2 — Scaling law fit for gpu_crocsort single-GPU sort.

Fits two models to warm-best wall times:
  (A) linear:   t = α·N + β
  (B) N log N: t = α·N·log2(N) + β·N + γ

Then predicts SF500 / SF1000 / SF10000 wall times.

Single-GPU data points (numactl --preferred, warm best from 17.3.2.x):
  SF10  : 60M records  → 0.353 s (fast path, no Phase 2)
  SF50  : 300M         → 1.408 s
  SF100 : 600M         → 2.952 s
  SF300 : 1.8B         → 6.594 s
"""
import numpy as np
from scipy.optimize import curve_fit
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (records, wall_seconds)
data_1gpu = [
    (    60_000_000,  0.353),  # SF10  - fast path
    (   300_000_000,  1.408),  # SF50
    (   600_000_000,  2.952),  # SF100
    ( 1_800_000_000,  6.594),  # SF300
]

# Multi-GPU distributed (full pipeline: partition + sort)
data_4gpu_distributed = [
    ( 3_000_000_000,  430),    # SF500 distributed (7m10s)
    ( 6_000_000_000, 1208),    # SF1000 distributed (20m08s)
]

N_arr = np.array([d[0] for d in data_1gpu], dtype=np.float64)
T_arr = np.array([d[1] for d in data_1gpu], dtype=np.float64)

# Model A: pure linear in N (radix sort theoretical)
def f_lin(N, alpha, beta):
    return alpha * N + beta

popt_lin, _ = curve_fit(f_lin, N_arr, T_arr)
print(f"Model A (linear):  t = {popt_lin[0]:.3e}·N + {popt_lin[1]:.3e}")
print(f"  Predictions:")
for N, T in data_1gpu:
    pred = f_lin(N, *popt_lin)
    print(f"    SF{int(N/6e6):4d} ({N/1e9:.1f}B): predicted {pred:.2f}s vs measured {T:.2f}s ({(pred-T)/T*100:+.1f}%)")

# Model B: N log N + N (comparison-based-ish)
def f_nlogn(N, alpha, beta, gamma):
    return alpha * N * np.log2(N) + beta * N + gamma

# Use bounds to keep coefficients reasonable
popt_nlogn, _ = curve_fit(f_nlogn, N_arr, T_arr,
                           p0=[1e-11, 1e-9, 0],
                           maxfev=10000)
print(f"\nModel B (N log N): t = {popt_nlogn[0]:.3e}·N·log2(N) + {popt_nlogn[1]:.3e}·N + {popt_nlogn[2]:.3e}")
print(f"  Predictions:")
for N, T in data_1gpu:
    pred = f_nlogn(N, *popt_nlogn)
    print(f"    SF{int(N/6e6):4d} ({N/1e9:.1f}B): predicted {pred:.2f}s vs measured {T:.2f}s ({(pred-T)/T*100:+.1f}%)")

# Out-of-sample predictions (using the linear model as it fits better
# for radix sort)
print("\n=== Predictions for unmeasured single-GPU scales (linear model) ===")
for N_pred, label in [(3e9, "SF500"), (6e9, "SF1000"), (60e9, "SF10000")]:
    pred = f_lin(N_pred, *popt_lin)
    print(f"  {label} ({N_pred/1e9:.0f}B records): {pred:.1f}s = {pred/60:.1f} min")

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
N_plot = np.logspace(7, 11, 100)
ax.loglog(N_arr, T_arr, 'o', markersize=10, color='C0', label='measured')
ax.loglog(N_plot, f_lin(N_plot, *popt_lin), '--', color='C1', label='linear fit (α·N+β)')
ax.loglog(N_plot, f_nlogn(N_plot, *popt_nlogn), ':', color='C2', label='N log N fit')

# Annotate measured points
for N, T in data_1gpu:
    ax.annotate(f"SF{int(N/6e6)}\n{T:.2f}s",
                xy=(N, T), xytext=(8, -8), textcoords='offset points',
                fontsize=9)

# Show the SF500 OOM
ax.axvline(3e9, color='red', linestyle='--', alpha=0.4)
ax.text(3e9, 0.3, '   SF500 (1-GPU OOM)', fontsize=9, color='red',
        verticalalignment='bottom')

ax.set_xlabel('Number of records')
ax.set_ylabel('Wall time (s)')
ax.set_title('17.2 — gpu_crocsort scaling law on single H100 NVL\n'
             '(numactl --preferred=0, warm best)')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('/home/cc/gpu-research/results/h100_runs/17.2_scaling_law.png', dpi=140)
print("\nWrote 17.2_scaling_law.png")
