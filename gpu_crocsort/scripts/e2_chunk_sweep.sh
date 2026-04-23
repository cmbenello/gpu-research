#!/bin/bash
# E2: Chunk-size sweep on SF100
# Tests different GPU chunk sizes to find optimal for this hardware
#
# Usage: bash scripts/e2_chunk_sweep.sh

set -e
cd "$(dirname "$0")/.."

BINARY=./external_sort_tpch_compact
INPUT=/tmp/lineitem_sf100_normalized.bin
OUT=results/overnight/e2_chunk_sweep.csv
RUNS=3

if [ ! -f "$BINARY" ]; then
    echo "Building binary..."
    make external-sort-tpch-compact
fi

if [ ! -f "$INPUT" ]; then
    echo "SF100 data not found at $INPUT"
    exit 1
fi

echo "chunk_size_M,wall_time_s,sort_ms,merge_ms,gather_ms" > "$OUT"

for CHUNK in 50000000 100000000 150000000 200000000; do
    CHUNK_M=$((CHUNK / 1000000))
    echo "=== Chunk size: ${CHUNK_M}M ==="
    # Set chunk size via environment variable
    result=$(CHUNK_SIZE=$CHUNK $BINARY --input "$INPUT" --runs $RUNS 2>&1)
    echo "$result" | tail -20

    # Extract timing from output
    wall=$(echo "$result" | grep -oP 'Wall time: \K[\d.]+' | tail -1)
    sort=$(echo "$result" | grep -oP 'sort_ms=\K[\d.]+' | tail -1)
    merge=$(echo "$result" | grep -oP 'merge_ms=\K[\d.]+' | tail -1)
    gather=$(echo "$result" | grep -oP 'gather_ms=\K[\d.]+' | tail -1)

    echo "${CHUNK_M},${wall:-0},${sort:-0},${merge:-0},${gather:-0}" >> "$OUT"
    echo ""
done

echo "Wrote $OUT"
cat "$OUT"
