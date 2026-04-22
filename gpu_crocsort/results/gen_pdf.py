#!/usr/bin/env python3
"""Generate PDF proposal using matplotlib's PdfPages (no system deps needed)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
import os
import textwrap

SRC = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(SRC, "figures")
OUT = os.path.join(SRC, "gpu_sort_research_summary.pdf")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DARK = '#2c3e50'
COLORS = {
    'gpu': '#2ecc71', 'polars': '#3498db', 'arrow': '#e74c3c',
    'numpy': '#9b59b6', 'pandas': '#e67e22', 'datafusion': '#1abc9c',
    'sqlite': '#e74c3c', 'duckdb': '#f39c12', 'dark': DARK,
}

def text_page(pdf, lines, fontsize=11):
    """Render a page of wrapped text."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.90])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.98
    for line in lines:
        if line.startswith('# '):
            ax.text(0, y, line[2:], fontsize=18, fontweight='bold', color=DARK,
                    va='top', transform=ax.transAxes)
            # underline
            ax.plot([0, 0.95], [y - 0.03, y - 0.03], color=DARK, linewidth=1.5,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.06
        elif line.startswith('## '):
            ax.text(0, y, line[3:], fontsize=14, fontweight='bold', color=DARK,
                    va='top', transform=ax.transAxes)
            ax.plot([0, 0.95], [y - 0.022, y - 0.022], color='#ddd', linewidth=0.8,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.05
        elif line.startswith('**') and line.endswith('**'):
            ax.text(0, y, line.strip('*'), fontsize=fontsize, fontweight='bold',
                    color=DARK, va='top', transform=ax.transAxes)
            y -= 0.03
        elif line == '---':
            ax.plot([0, 0.95], [y, y], color='#ccc', linewidth=0.5,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.015
        elif line == '':
            y -= 0.012
        else:
            # wrap long lines
            wrapped = textwrap.wrap(line, width=95)
            for wl in wrapped:
                if wl.startswith('- '):
                    ax.text(0.02, y, '\u2022', fontsize=fontsize, va='top',
                            transform=ax.transAxes, color=DARK)
                    ax.text(0.04, y, wl[2:], fontsize=fontsize, va='top',
                            transform=ax.transAxes, color='#333')
                else:
                    ax.text(0, y, wl, fontsize=fontsize, va='top',
                            transform=ax.transAxes, color='#333')
                y -= 0.026
    pdf.savefig(fig)
    plt.close(fig)

def table_page(pdf, title, headers, rows, col_widths=None):
    """Render a table on its own page section."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.90])
    ax.axis('off')

    ax.text(0, 0.98, title, fontsize=14, fontweight='bold', color=DARK,
            va='top', transform=ax.transAxes)

    ncols = len(headers)
    nrows = len(rows)
    if col_widths is None:
        col_widths = [1.0 / ncols] * ncols

    table_data = [headers] + rows
    tbl = ax.table(cellText=table_data, cellLoc='left',
                   loc='upper center', bbox=[0, 0.98 - 0.04*(nrows+2), 0.95, 0.04*(nrows+2)])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#ccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('white')

    pdf.savefig(fig)
    plt.close(fig)

def image_page(pdf, img_path, caption=''):
    """Full-page image with optional caption."""
    fig = plt.figure(figsize=(8.27, 11.69))
    img = mpimg.imread(img_path)
    h, w = img.shape[:2]
    aspect = w / h
    # fit within page
    img_w = 0.84
    img_h = img_w / aspect * (8.27 / 11.69)
    ax = fig.add_axes([0.08, 0.95 - img_h, img_w, img_h])
    ax.imshow(img)
    ax.axis('off')
    if caption:
        fig.text(0.5, 0.95 - img_h - 0.02, caption, fontsize=10, ha='center',
                 style='italic', color='#555')
    pdf.savefig(fig)
    plt.close(fig)

def combo_page(pdf, img_path1, img_path2, caption1='', caption2=''):
    """Two images stacked vertically."""
    fig = plt.figure(figsize=(8.27, 11.69))
    for i, (path, cap) in enumerate([(img_path1, caption1), (img_path2, caption2)]):
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        aspect = w / h
        img_w = 0.84
        img_h = img_w / aspect * (8.27 / 11.69)
        top = 0.95 - i * 0.48
        ax = fig.add_axes([0.08, top - img_h, img_w, img_h])
        ax.imshow(img)
        ax.axis('off')
        if cap:
            fig.text(0.5, top - img_h - 0.015, cap, fontsize=9, ha='center',
                     style='italic', color='#555')
    pdf.savefig(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# Build the PDF
# ═══════════════════════════════════════════════════════════════
with PdfPages(OUT) as pdf:

    # ── Page 1: Title + intro ──
    text_page(pdf, [
        '# CrocSort — GPU-Accelerated Sorting',
        '',
        '**what it is:** a system that offloads sorting to the GPU using radix sort. works as a',
        'drop-in for pyarrow/polars/numpy, and we also tested it for database index creation and',
        'external sort on TPC-H. runs on a single RTX 6000 over PCIe 3.0.',
        '',
        '**hardware:** Quadro RTX 6000 (24 GB, Turing), 2x Xeon Gold (48 threads), 192 GB DDR4, PCIe 3.0',
        '',
        '---',
        '',
        '## how it works',
        '',
        'the basic idea: convert sort keys to a byte-comparable format (sign-flip + big-endian),',
        'ship them to the GPU over PCIe, run CUB radix sort (O(N) instead of O(N log N)), get',
        'back a permutation, and reorder the original data.',
        '',
        'for in-memory columnar data (arrow tables, numpy arrays, etc.) this is the whole pipeline.',
        'the GPU sort kernel itself takes ~70ms for 300M rows — most of the time is actually spent',
        'on PCIe transfer and encoding, not sorting.',
        '',
        'for datasets bigger than GPU memory (like TPC-H SF100 at 72 GB), we chunk the data,',
        'sort each chunk on the GPU, and do a K-way merge on CPU. we also do a "compact key"',
        'trick where we detect which byte positions actually vary across the dataset and only',
        'upload those — saves 2-8x on PCIe transfer for real data.',
        '',
        'key encoding supports: int8/16/32/64, uint8/16/32/64, float32/64, date32, fixed-width strings.',
    ])

    # ── Page 2: Pipeline diagram ──
    image_page(pdf, f'{FIG}/fig5_pipeline.png',
               'Figure 1: CrocSort pipeline — encode keys on CPU, sort on GPU, gather results')

    # ── Page 3: Columnar results ──
    text_page(pdf, [
        '## results — columnar analytics engines',
        '',
        'tested against the main python sort engines on 100M int32 rows.',
        'GPU sort is 2-89x faster depending on the baseline.',
        '',
        'the more interesting comparison is polars, which is the fastest CPU engine — we still',
        'get 2.2x on single-column and 9.5x on multi-column sorts.',
        '',
        'at 300M rows with 4 sort columns (Arrow C++ benchmark), the full GPU pipeline',
        '(encode + upload + sort + gather) takes 4.3s vs pyarrow\'s 97s — about 22x.',
        '',
        'the actual GPU sort is only 9% of our time; most is key encoding (55%) and PCIe upload',
        '(29%). this means faster interconnects (PCIe 5, NVLink C2C) would help a lot.',
        '',
        'the crossover point where GPU starts winning is around 5,000 rows.',
        '',
        '',
        '## multi-column and strings',
        '',
        'multi-column sorts show bigger speedups because CPU comparison-sort cost grows with',
        'key width, while GPU radix sort cost grows linearly (just more bytes to upload).',
        '',
        'string sort works by encoding to fixed-width null-padded byte keys. UUIDs (36B) get',
        '38x speedup, email addresses (variable-length, avg ~24B) get 26x.',
    ])

    # ── Page 4: Speedup chart + scaling ──
    combo_page(pdf,
               f'{FIG}/fig1_speedup_100m.png',
               f'{FIG}/fig2_scaling_breakdown.png',
               'Figure 2: GPU speedup vs CPU engines at 100M rows',
               'Figure 3: Scaling from 10M to 300M rows (left); GPU time breakdown at 300M (right)')

    # ── Page 5: CREATE INDEX ──
    text_page(pdf, [
        '## results — database index creation',
        '',
        'this is probably the most compelling application. CREATE INDEX is basically just a big',
        'sort, and databases do it single-threaded (or poorly parallelized).',
        '',
        'benchmarked against SQLite — 100M rows: 87.9s vs 0.72s on GPU = 122x faster.',
        'PostgreSQL numbers from literature are similar (~52s for 100M rows single-threaded,',
        '~10 min for 500M rows).',
        '',
        'this matters operationally: index creation locks the table. if you can do it in under a',
        'second instead of a minute, you don\'t need maintenance windows for schema migrations.',
        '',
        'also relevant for LSM-tree compaction (RocksDB/LevelDB) — prior work (LUDA, 2020)',
        'already showed 2x throughput by offloading the sort to GPU.',
        '',
        '',
        '## results — external sort (TPC-H)',
        '',
        'full multi-column sort of TPC-H lineitem (9 sort columns, 66-byte records).',
        '4.7-10x faster than DuckDB depending on scale factor.',
        '',
        'SF100 (600M rows, 72 GB) sorts in 8 seconds at ~9 GB/s throughput.',
        'for context, MendSort (JouleSort 2023 winner) does 3.3 GB/s on GenSort,',
        'and we hit 8 GB/s on wider records with a single GPU.',
        '',
        'we also tried integrating directly into DuckDB as a custom operator, but the',
        'row-serialization overhead (DuckDB converts columns to rows for sorting) ate all the',
        'GPU advantage — end-to-end query time was basically 1.0x. this is why we pivoted to',
        'the columnar approach where data is already in the right format.',
    ])

    # ── Page 6: CREATE INDEX chart + TPC-H chart ──
    combo_page(pdf,
               f'{FIG}/fig3_create_index.png',
               f'{FIG}/fig4_tpch.png',
               'Figure 4: CREATE INDEX — SQLite vs GPU sort (log scale)',
               'Figure 5: TPC-H lineitem full sort — CrocSort vs DuckDB')

    # ── Page 7: Hardware + what's next ──
    text_page(pdf, [
        '## hardware projections',
        '',
        'the current system is bottlenecked by PCIe 3.0 (12 GB/s) and host DRAM bandwidth.',
        '',
        'the big one is GH200 (Grace Hopper) — its 900 GB/s NVLink C2C between CPU and GPU',
        'means the gather phase (26% of time on RTX 6000) goes to zero. the GPU writes sorted',
        'output directly into unified memory. projected ~1.2s for SF50 vs current 6.6s.',
        '',
        '',
        '## what\'s next',
        '',
        '- postgres integration — replace the sort in tuplesort_performsort() for CREATE INDEX.',
        '  postgres already normalizes keys to byte-comparable format so this is straightforward.',
        '',
        '- rocksdb compaction — GPU sort in background compaction to reduce write stalls.',
        '',
        '- GH200 testing — unified memory should give ~5x over current numbers.',
        '',
        '- multi-GPU — NVLink all-to-all for larger datasets.',
        '',
        '',
        '## where it makes sense (and where it doesn\'t)',
        '',
        '**good fit:**',
        '- columnar analytics (pyarrow, polars, datafusion) — 2-89x, drop-in replacement',
        '- CREATE INDEX in postgres/mysql/sqlite — 30-122x, eliminates maintenance windows',
        '- LSM compaction (rocksdb) — background sort, no PCIe latency on query path',
        '- any bulk sort > 5,000 rows',
        '',
        '**bad fit:**',
        '- per-query OLTP sorts (< 100K rows) — PCIe latency dominates',
        '- row-oriented engines (DuckDB) — serialization overhead eats the GPU advantage',
    ])

    # ── Page 8: Hardware chart ──
    image_page(pdf, f'{FIG}/fig6_hardware.png',
               'Figure 6: Projected sort time on newer GPU hardware (TPC-H SF50)')

    # ── Page 9: Summary table ──
    table_page(pdf, 'Summary of Results',
               ['Workload', 'CPU Baseline', 'GPU (CrocSort)', 'Speedup'],
               [
                   ['100M int32 sort', '0.60s (Polars)', '0.28s', '2.2x'],
                   ['100M 2-col int32', '3.68s (Polars)', '0.39s', '9.5x'],
                   ['300M int32 (Arrow C++)', '96.8s', '4.3s', '22x'],
                   ['100M numpy int32', '7.0s', '0.28s', '25x'],
                   ['10M UUIDs (36B strings)', '4.46s (Polars)', '0.12s', '38x'],
                   ['10M emails (var-len)', '3.64s (Polars)', '0.14s', '26x'],
                   ['100M CREATE INDEX', '87.9s (SQLite)', '0.72s', '122x'],
                   ['TPC-H SF50 (300M rows)', '66.6s (DuckDB)', '6.56s', '10x'],
                   ['TPC-H SF100 (600M rows)', '~72s (DuckDB)', '8.02s', '9x'],
                   ['60 GB GenSort (100B rec)', 'N/A', '7.5s', '8 GB/s throughput'],
               ])

print(f"PDF written to {OUT}")
