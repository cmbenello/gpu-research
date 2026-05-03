#!/usr/bin/env python3
"""2.2 — chart H→D + D→H PCIe bytes vs scale, baseline vs bitpack."""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "2.2_pcie_sweep.csv")

def load():
    with open(CSV) as f:
        return [r for r in csv.DictReader(f)]

def main():
    rows = [r for r in load() if r["best_warm_ms"] != "0"]
    base = [r for r in rows if r["config"] == "baseline_compact"]
    bp   = [r for r in rows if r["config"] == "bitpack_compact"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bx = [float(r["gb"]) for r in base]
    by_pcie = [float(r["pcie_total_gb"]) for r in base]
    by_raw  = [float(r["raw_record_gb"]) for r in base]
    pbx = [float(r["gb"]) for r in bp]
    pby = [float(r["pcie_total_gb"]) for r in bp]

    ax1.plot(bx, by_raw, "k:", label="raw records (2 × scale)", linewidth=1)
    ax1.plot(bx, by_pcie, "o-", color="#1f77b4", label="baseline compact PCIe")
    ax1.plot(pbx, pby, "s--", color="#ff7f0e", label="bitpack compact PCIe")
    ax1.set_xscale("log")
    ax1.set_xlabel("Input size (GB)")
    ax1.set_ylabel("PCIe traffic H→D + D→H (GB)")
    ax1.set_title("PCIe traffic vs scale")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left")

    by_ratio = [float(r["compression_ratio"]) for r in base]
    pby_ratio = [float(r["compression_ratio"]) for r in bp]
    ax2.plot(bx, by_ratio, "o-", color="#1f77b4", label="baseline compact")
    ax2.plot(pbx, pby_ratio, "s--", color="#ff7f0e", label="bitpack compact")
    ax2.axhline(1.0, color="grey", linestyle=":", linewidth=1, label="no compression")
    ax2.set_xscale("log")
    ax2.set_xlabel("Input size (GB)")
    ax2.set_ylabel("PCIe compression ratio (raw / on-the-wire)")
    ax2.set_title("PCIe compression ratio vs scale")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 5)

    fig.suptitle("H100 NVL: PCIe traffic during sort (compact + USE_BITPACK)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "2.2_pcie_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
