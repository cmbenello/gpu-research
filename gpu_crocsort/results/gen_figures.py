#!/usr/bin/env python3
"""Generate figures for CrocSort project proposal."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "figures"

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
})

COLORS = {
    'gpu': '#2ecc71',
    'polars': '#3498db',
    'arrow': '#e74c3c',
    'numpy': '#9b59b6',
    'pandas': '#e67e22',
    'datafusion': '#1abc9c',
    'sqlite': '#e74c3c',
    'duckdb': '#f39c12',
    'stdsort': '#95a5a6',
    'dark': '#2c3e50',
}

# ════════════════════════════════════════════════════════════════
# Fig 1: Speedup bar chart — GPU vs everything at 100M rows
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

engines = ['PyArrow', 'pandas', 'NumPy', 'DataFusion', 'Polars']
times_cpu = [24.8, 9.5, 7.0, 1.56, 0.604]
time_gpu = 0.280
speedups = [t / time_gpu for t in times_cpu]
colors = [COLORS['arrow'], COLORS['pandas'], COLORS['numpy'],
          COLORS['datafusion'], COLORS['polars']]

bars = ax.barh(engines, speedups, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
for bar, s, t in zip(bars, speedups, times_cpu):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{s:.0f}x  ({t:.1f}s → {time_gpu*1000:.0f}ms)',
            va='center', fontsize=9, color=COLORS['dark'])

ax.set_xlabel('GPU Speedup (×)', fontsize=12)
ax.set_title('100M int32 rows — GPU sort vs CPU engines', fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0, max(speedups) * 1.45)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(f'{OUT}/fig1_speedup_100m.png', bbox_inches='tight')
plt.close()
print("fig1 done")

# ════════════════════════════════════════════════════════════════
# Fig 2: Scaling — GPU time vs row count
# ════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: absolute times
rows_m = [10, 50, 100, 300]
gpu_ms = [28, 90, 280, 1660]   # approx from benchmarks
polars_ms = [55, 260, 604, 2100]  # approx
arrow_ms = [1800, 8500, 24800, 96800]

ax1.plot(rows_m, gpu_ms, 'o-', color=COLORS['gpu'], linewidth=2, markersize=6, label='GPU (CrocSort)')
ax1.plot(rows_m, polars_ms, 's-', color=COLORS['polars'], linewidth=2, markersize=6, label='Polars')
ax1.plot(rows_m, arrow_ms, '^-', color=COLORS['arrow'], linewidth=2, markersize=6, label='PyArrow')
ax1.set_xlabel('Rows (millions)')
ax1.set_ylabel('Time (ms)')
ax1.set_yscale('log')
ax1.set_title('Sort time scaling', fontweight='bold')
ax1.legend(frameon=False, fontsize=9)
ax1.grid(True, alpha=0.2)

# Right: GPU time breakdown at 300M (4-col Arrow bench)
phases = ['Encode', 'Upload', 'Sort', 'Gather', 'Download']
phase_ms = [2045, 1057, 338, 259, 0]  # from the 4-col bench, adjusted
phase_colors = ['#95a5a6', '#3498db', '#2ecc71', '#e67e22', '#9b59b6']

wedges, texts, autotexts = ax2.pie(
    [max(p, 1) for p in phase_ms],  # avoid zero
    labels=phases, colors=phase_colors,
    autopct=lambda p: f'{p:.0f}%' if p > 2 else '',
    startangle=90, textprops={'fontsize': 9}
)
ax2.set_title('GPU pipeline breakdown\n(300M × 4 cols)', fontweight='bold', fontsize=11)

fig.tight_layout()
fig.savefig(f'{OUT}/fig2_scaling_breakdown.png', bbox_inches='tight')
plt.close()
print("fig2 done")

# ════════════════════════════════════════════════════════════════
# Fig 3: The killer chart — CREATE INDEX
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))

rows_idx = ['1M', '10M', '100M']
sqlite_s = [0.703, 7.40, 87.9]
gpu_s = [0.021, 0.074, 0.723]

x = np.arange(len(rows_idx))
w = 0.35
b1 = ax.bar(x - w/2, sqlite_s, w, color=COLORS['sqlite'], label='SQLite CPU', edgecolor='white')
b2 = ax.bar(x + w/2, gpu_s, w, color=COLORS['gpu'], label='GPU (CrocSort)', edgecolor='white')

# Speedup annotations
for i, (s, g) in enumerate(zip(sqlite_s, gpu_s)):
    sp = s / g
    ax.text(i + w/2, g * 1.5, f'{sp:.0f}x', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['dark'])

ax.set_yscale('log')
ax.set_ylabel('Time (seconds, log scale)')
ax.set_xticks(x)
ax.set_xticklabels(rows_idx)
ax.set_xlabel('Number of rows')
ax.set_title('CREATE INDEX: SQLite vs GPU sort', fontsize=13, fontweight='bold', pad=10)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f'{OUT}/fig3_create_index.png', bbox_inches='tight')
plt.close()
print("fig3 done")

# ════════════════════════════════════════════════════════════════
# Fig 4: External sort — TPC-H CrocSort vs DuckDB
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))

sf = ['SF10\n(60M rows)', 'SF50\n(300M rows)', 'SF100\n(600M rows)']
duckdb_s = [8.0, 66.6, 72]
croc_s = [1.71, 6.56, 8.02]

x = np.arange(len(sf))
w = 0.35
ax.bar(x - w/2, duckdb_s, w, color=COLORS['duckdb'], label='DuckDB CPU', edgecolor='white')
ax.bar(x + w/2, croc_s, w, color=COLORS['gpu'], label='CrocSort (GPU)', edgecolor='white')

for i, (d, c) in enumerate(zip(duckdb_s, croc_s)):
    sp = d / c
    ax.text(i + w/2, c + 1.5, f'{sp:.1f}x', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['dark'])

ax.set_ylabel('Time (seconds)')
ax.set_xticks(x)
ax.set_xticklabels(sf)
ax.set_title('TPC-H lineitem full sort — CrocSort vs DuckDB', fontsize=13, fontweight='bold', pad=10)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f'{OUT}/fig4_tpch.png', bbox_inches='tight')
plt.close()
print("fig4 done")

# ════════════════════════════════════════════════════════════════
# Fig 5: Architecture pipeline diagram (using matplotlib patches)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.5)
ax.axis('off')

def box(x, y, w, h, text, color, fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='#2c3e50', linewidth=1.2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white' if color != '#ecf0f1' else '#2c3e50')

def arrow(x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.15
        ax.text(mx, my, label, ha='center', fontsize=7, color='#7f8c8d')

# Host side
box(0.2, 2.2, 2.0, 0.9, 'Columnar Data\n(Arrow/NumPy)', '#3498db')
box(0.2, 0.5, 2.0, 0.9, 'Key Encode\nsign-flip + BE', '#95a5a6')

# PCIe
box(3.0, 1.2, 1.4, 1.0, 'PCIe 3.0\n12 GB/s', '#ecf0f1', fontsize=8)

# GPU
box(5.2, 1.8, 2.2, 1.0, 'CUB Radix Sort\nO(N) — 70ms/300M', COLORS['gpu'])
box(5.2, 0.3, 2.2, 1.0, 'GPU Gather\n(optional)', '#e67e22')

# Output
box(8.2, 1.2, 1.5, 1.0, 'Sorted\nOutput', '#2c3e50')

# Arrows
arrow(2.2, 2.65, 3.0, 1.9)  # data -> encode
arrow(1.2, 2.2, 1.2, 1.4)   # data -> encode (down)
arrow(2.2, 0.95, 3.0, 1.5)  # encode -> pcie
arrow(4.4, 1.9, 5.2, 2.3)   # pcie -> sort
arrow(7.4, 2.3, 8.2, 1.9)   # sort -> output
arrow(7.4, 0.8, 8.2, 1.4)   # gather -> output
arrow(6.3, 1.8, 6.3, 1.3)   # sort -> gather

ax.set_title('CrocSort Pipeline', fontsize=14, fontweight='bold', pad=8)
fig.tight_layout()
fig.savefig(f'{OUT}/fig5_pipeline.png', bbox_inches='tight')
plt.close()
print("fig5 done")

# ════════════════════════════════════════════════════════════════
# Fig 6: Hardware portability projections
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))

hw = ['RTX 6000\n(current)', 'A100\nPCIe 4', 'H100\nPCIe 5', 'GH200\nNVLink C2C']
times = [6.56, 3.5, 2.0, 1.2]
hw_colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']

bars = ax.bar(hw, times, color=hw_colors, width=0.55, edgecolor='white', linewidth=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{t:.1f}s', ha='center', fontsize=11, fontweight='bold', color=COLORS['dark'])

ax.set_ylabel('TPC-H SF50 sort time (seconds)')
ax.set_title('Projected performance on newer hardware', fontsize=13, fontweight='bold', pad=10)
ax.set_ylim(0, 8)
fig.tight_layout()
fig.savefig(f'{OUT}/fig6_hardware.png', bbox_inches='tight')
plt.close()
print("fig6 done")

print("\nAll figures saved to", OUT)
