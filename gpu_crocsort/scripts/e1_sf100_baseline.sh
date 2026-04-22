#!/bin/bash
# E1: TPC-H SF100 baseline sort timing
# Runs external_sort_tpch_compact on SF100 data and records detailed timing
#
# Usage: bash scripts/e1_sf100_baseline.sh

set -e
cd "$(dirname "$0")/.."

BINARY=./external_sort_tpch_compact
INPUT=/tmp/lineitem_sf100_normalized.bin
OUT=results/overnight/e1_sf100_compressed.csv
RUNS=5

if [ ! -f "$BINARY" ]; then
    echo "Building binary..."
    make external-sort-tpch-compact
fi

if [ ! -f "$INPUT" ]; then
    echo "SF100 data not found at $INPUT. Generating..."
    python3 gen_tpch_normalized.py 100 "$INPUT"
fi

echo "=== E1: TPC-H SF100 Baseline External Sort ==="
echo "Input: $INPUT ($(ls -lh $INPUT | awk '{print $5}'))"
echo "Runs: $RUNS"
echo ""

$BINARY --input "$INPUT" --runs $RUNS 2>&1 | tee results/overnight/e1_log.txt

echo ""
echo "Results saved to results/overnight/e1_log.txt"
