#!/bin/bash
# Scaling curve: performance from 1M to 300M records (single-chunk to multi-chunk)
# Uses random 120B records (66B key + 54B value, all bytes varying)
set -e

BINARY=./external_sort_tpch_compact
RUNS=3
OUTDIR=/dev/shm/paper_experiments
LOG="${OUTDIR}/scaling_curve.log"

echo "=== Scaling Curve Experiment ===" | tee "$LOG"
echo "Binary: $BINARY" | tee -a "$LOG"
echo "Warm runs per size: $RUNS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Sizes: 1M, 5M, 10M, 30M, 60M, 120M, 200M
# 200M × 120B = 24 GB (just fits in GPU, single chunk but tight)
# Skip 300M (SF50) — already have that data
for NREC in 1000000 5000000 10000000 30000000 60000000 120000000 200000000; do
    LABEL=$(echo "$NREC / 1000000" | bc)M
    SIZE_GB=$(echo "scale=2; $NREC * 120 / 1073741824" | bc)
    INPUT="/dev/shm/scaling_${LABEL}_normalized.bin"

    echo "--- ${LABEL}: ${NREC} records, ${SIZE_GB} GB ---" | tee -a "$LOG"

    # Generate
    python3 scripts/gen_random_records.py $NREC "$INPUT"

    # Sort
    $BINARY --input "$INPUT" --runs $RUNS 2>&1 | tee -a "$LOG"

    # Extract CSV
    echo "CSV summary for ${LABEL}:" | tee -a "$LOG"
    grep "^CSV" "$LOG" | tail -$RUNS | tee -a /dev/null
    echo "" | tee -a "$LOG"

    # Clean up
    rm -f "$INPUT"
    echo "  Cleaned up $INPUT" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "=== Scaling Curve Complete ===" | tee -a "$LOG"
