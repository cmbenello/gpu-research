#!/usr/bin/env python3
"""Render the H100 single-GPU scale envelope.

Reads results/h100_runs/1.7_envelope.csv and produces:
  - 1.7_envelope_walltime.png — best warm wall time vs scale (log-log)
  - 1.7_envelope_throughput.png — best warm GB/s vs scale (log-x linear-y)

PASS scales are markers + line, FAIL scales are red 'X' markers with
annotations stating why they failed.
"""
from __future__ import annotations
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "1.7_envelope.csv")

def load_rows():
    with open(CSV) as f:
        return [r for r in csv.DictReader(f)]

def plot_walltime(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    base = [r for r in rows if r["config"] == "compact_baseline" and r["status"] == "PASS"]
    bp   = [r for r in rows if r["config"] == "compact_bitpack"  and r["status"] == "PASS"]
    fail = [r for r in rows if r["status"] == "FAIL"]

    bx = [float(r["gb"]) for r in base]
    by = [float(r["best_warm_ms"]) / 1000.0 for r in base]
    pbx = [float(r["gb"]) for r in bp]
    pby = [float(r["best_warm_ms"]) / 1000.0 for r in bp]

    ax.plot(bx, by, "o-", color="#1f77b4", label="compact baseline (warm best)")
    ax.plot(pbx, pby, "s--", color="#ff7f0e", label="compact + USE_BITPACK (warm best)")

    for r in fail:
        gb = float(r["gb"])
        ax.scatter([gb], [1.0], marker="x", s=120, color="red", zorder=5)
        ax.annotate(f"FAIL: {r['note']}", (gb, 1.0),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, color="darkred", ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size (GB raw lineitem records)")
    ax.set_ylabel("Best-warm wall time (s, log)")
    ax.set_title("H100 NVL single-GPU scale envelope: TPC-H lineitem sort")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    out = os.path.join(HERE, "1.7_envelope_walltime.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

def plot_throughput(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    base = [r for r in rows if r["config"] == "compact_baseline" and r["status"] == "PASS"]
    bp   = [r for r in rows if r["config"] == "compact_bitpack"  and r["status"] == "PASS"]
    fail = [r for r in rows if r["status"] == "FAIL"]

    bx = [float(r["gb"]) for r in base]
    by = [float(r["best_warm_gb_per_s"]) for r in base]
    pbx = [float(r["gb"]) for r in bp]
    pby = [float(r["best_warm_gb_per_s"]) for r in bp]

    ax.plot(bx, by, "o-", color="#1f77b4", label="compact baseline")
    ax.plot(pbx, pby, "s--", color="#ff7f0e", label="compact + USE_BITPACK")

    for r in fail:
        gb = float(r["gb"])
        ax.scatter([gb], [0.5], marker="x", s=120, color="red", zorder=5)
        ax.annotate(r["note"][:50] + "…" if len(r["note"]) > 50 else r["note"],
                    (gb, 0.5), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color="darkred", ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Input size (GB raw lineitem records)")
    ax.set_ylabel("Effective throughput (GB / s)")
    ax.set_title("H100 NVL single-GPU effective throughput vs scale")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 30)

    out = os.path.join(HERE, "1.7_envelope_throughput.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

def main():
    rows = load_rows()
    plot_walltime(rows)
    plot_throughput(rows)

if __name__ == "__main__":
    main()
