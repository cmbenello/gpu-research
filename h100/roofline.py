#!/usr/bin/env python3
"""Tier 17.1 — roofline ceiling for radix sort on H100.

Computes upper-bound throughput from hardware specs, compares to measured
numbers from results/overnight_pulled/*.csv.
"""
import argparse, csv, json, os, sys
from collections import defaultdict

# Hardware specs (per-GPU, single device)
HW = {
    "H100_PCIe":    {"hbm_GB_s": 2_039,  "pcie_GB_s": 32,   "fp32_TFLOPS": 51,  "memory_GB": 80},
    "H100_SXM":     {"hbm_GB_s": 3_350,  "pcie_GB_s": 32,   "fp32_TFLOPS": 67,  "memory_GB": 80},
    "RTX_6000":     {"hbm_GB_s": 672,    "pcie_GB_s": 12,   "fp32_TFLOPS": 16,  "memory_GB": 24},
    "P5000":        {"hbm_GB_s": 288,    "pcie_GB_s": 12,   "fp32_TFLOPS": 9,   "memory_GB": 16},
    "RTX_2080":     {"hbm_GB_s": 448,    "pcie_GB_s": 12,   "fp32_TFLOPS": 10,  "memory_GB": 8},
}

# Radix sort cost model: per pass, read+write all keys = 2× key bytes of HBM traffic
# Number of passes = ceil(key_bytes / 1) (one pass per byte for byte-radix)
# So total HBM traffic per record = 2 * key_bytes * key_bytes (in bytes)
# But CUB does 8-bit radix, so passes = key_bytes
# Memory traffic = 2 × key_bytes × n × passes  (read + write each pass)

def roofline_sort_throughput(gpu, key_bytes_unpacked, key_bytes_packed,
                              record_bytes, total_records, in_memory=True):
    """
    Predict end-to-end sort throughput in GB/s based on the roofline model.

    Phases:
    - PCIe upload (only when records don't fit in HBM):
        bytes = record_bytes × n  (full record) OR key_bytes × n (compact)
    - HBM traffic during sort: 2 × key_bytes × n × passes (n radix passes)
    - PCIe download of permutation: 4 × n bytes (uint32 perm) - small

    Returns dict with predicted phase times + total throughput.
    """
    h = HW[gpu]
    n = total_records
    total_bytes_input = n * record_bytes / 1e9   # GB

    pcie = h["pcie_GB_s"]
    hbm  = h["hbm_GB_s"]

    # Compact upload: only the key portion goes over PCIe
    pcie_bytes = key_bytes_packed * n / 1e9

    # HBM traffic: each radix pass reads + writes the key buffer
    # Number of byte-radix passes:
    n_passes = key_bytes_packed
    hbm_bytes_sort = 2 * key_bytes_packed * n * n_passes / 1e9

    # Permutation download (constant 4B per record)
    perm_bytes = 4 * n / 1e9

    # Phase times
    t_pcie_up = pcie_bytes / pcie
    t_sort    = hbm_bytes_sort / hbm
    t_pcie_dn = perm_bytes / pcie

    t_total = t_pcie_up + t_sort + t_pcie_dn
    throughput = total_bytes_input / t_total if t_total > 0 else 0

    return {
        "gpu": gpu, "n_records": n,
        "key_bytes_unpacked": key_bytes_unpacked,
        "key_bytes_packed": key_bytes_packed,
        "record_bytes": record_bytes,
        "data_GB": round(total_bytes_input, 2),
        "pcie_up_s": round(t_pcie_up, 3),
        "hbm_sort_s": round(t_sort, 3),
        "pcie_dn_s": round(t_pcie_dn, 3),
        "total_s": round(t_total, 3),
        "throughput_GB_s": round(throughput, 2),
        "bottleneck": "pcie_up" if t_pcie_up > t_sort and t_pcie_up > t_pcie_dn else \
                      "hbm_sort" if t_sort > t_pcie_dn else "pcie_dn",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/overnight_pulled/roofline.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # TPC-H sort sizes
    SCALES = [(10, 60_000_000), (50, 300_000_000), (100, 600_000_000),
              (300, 1_800_000_000), (500, 3_000_000_000), (1000, 6_000_000_000)]

    rows = []
    for gpu in HW:
        for sf, n in SCALES:
            data_gb = n * 120 / 1e9
            if data_gb > HW[gpu]["memory_GB"] * 5:
                continue   # skip if extreme out-of-core
            # Two configs: 32B compact (no bitpack), 24B compact (bitpack)
            for label, packed in [("compact_32B", 32), ("bitpack_24B", 24)]:
                r = roofline_sort_throughput(gpu, 32, packed, 120, n)
                r["sf"] = sf
                r["config"] = label
                rows.append(r)

    fields = ["gpu", "sf", "config", "n_records", "data_GB",
              "key_bytes_unpacked", "key_bytes_packed", "record_bytes",
              "pcie_up_s", "hbm_sort_s", "pcie_dn_s", "total_s",
              "throughput_GB_s", "bottleneck"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n=== Roofline predictions ({len(rows)} configs) ===")
    print(f"{'GPU':<12s} {'SF':>4s} {'config':<14s} {'data':>7s} {'pred':>7s} {'GB/s':>7s} {'bottleneck':<10s}")
    for r in rows:
        print(f"{r['gpu']:<12s} {r['sf']:>4d} {r['config']:<14s} "
              f"{r['data_GB']:>5.1f}GB {r['total_s']:>6.2f}s {r['throughput_GB_s']:>6.1f} {r['bottleneck']}")
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
