#!/bin/bash
# Overnight experiment runner
# Runs Groups A, B, C, E (skip D2) from results/overnight_experiments.md
#
# Usage: bash scripts/run_overnight.sh 2>&1 | tee results/overnight/run.log

set -e
cd "$(dirname "$0")/.."

echo "=== Overnight Experiments ==="
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo ""

# Setup checks
echo "--- Setup checks ---"
nvidia-smi --query-gpu=gpu_name,memory.total,clocks.gr --format=csv,noheader
echo ""

mkdir -p results/overnight

# ═════════════════════════════════════════════════════════════
# GROUP A: Compressibility profiling (CPU-only, ~45 min)
# ═════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  GROUP A: Compressibility Profiling   ║"
echo "╚═══════════════════════════════════════╝"

echo "--- A1: Per-column byte entropy ---"
python3 scripts/a1_byte_entropy.py 2>&1
echo ""

echo "--- A2: Compact-key scan baseline ---"
python3 scripts/a2_compact_key_baseline.py 2>&1
echo ""

echo "--- A3: Codec compression ratios ---"
python3 scripts/a3_codec_ratios.py 2>&1
echo ""

# ═════════════════════════════════════════════════════════════
# GROUP B: GPU codec microbenchmarks (~1.5h)
# ═════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  GROUP B: GPU Codec Microbenchmarks   ║"
echo "╚═══════════════════════════════════════╝"

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "Building codec benchmarks for sm_${ARCH}..."
nvcc -O3 -std=c++17 -arch=sm_${ARCH} -o b_codec_bench scripts/b_codec_benchmarks.cu 2>&1
echo "Running B1/B2/B3..."
./b_codec_bench 2>&1
rm -f b_codec_bench
echo ""

# ═════════════════════════════════════════════════════════════
# GROUP C: End-to-end sort with compression (~90 min)
# ═════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  GROUP C: E2E Sort with Compression   ║"
echo "╚═══════════════════════════════════════╝"

echo "--- C1: TPC-H with FOR keys ---"
# Needs normalized binary data — generate if not present
for SF in 10 50; do
    BIN="/tmp/lineitem_sf${SF}_normalized.bin"
    if [ ! -f "$BIN" ]; then
        echo "Generating SF${SF} normalized binary..."
        python3 gen_tpch_normalized.py ${SF} "$BIN" 2>&1
    fi
done
python3 scripts/c1_tpch_for.py 2>&1
echo ""

# ═════════════════════════════════════════════════════════════
# GROUP E: Out-of-core regime (~90 min)
# ═════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  GROUP E: Out-of-core Regime          ║"
echo "╚═══════════════════════════════════════╝"

echo "--- E1: TPC-H SF100 with FOR + compact ---"
BIN100="/tmp/lineitem_sf100_normalized.bin"
if [ ! -f "$BIN100" ]; then
    echo "Generating SF100 normalized binary..."
    python3 gen_tpch_normalized.py 100 "$BIN100" 2>&1
fi
# E1 uses the same C1 script logic on SF100
echo "Running SF100 baseline..."
./external_sort_tpch_compact --input "$BIN100" --runs 3 2>&1
echo ""

# ════════════════════════════════════════════════════════════��
# GROUP D1: Merge profiling (~30 min)
# ═════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  GROUP D1: K-way Merge Profiling      ║"
echo "╚═══════════════════════════════════════╝"
python3 scripts/d1_merge_profile.py 2>&1
echo ""

echo "=== All overnight experiments complete ==="
echo "Date: $(date)"
echo ""
echo "Output files:"
ls -la results/overnight/*.csv 2>/dev/null
