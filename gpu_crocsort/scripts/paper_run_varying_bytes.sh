#!/bin/bash
# Run GPU sort on varying-byte-count datasets, one at a time to manage /dev/shm space
set -e

BINARY=./external_sort_tpch_compact
RUNS=3
OUTDIR=/dev/shm/paper_experiments
NREC=60000000  # 60M records = 7.2 GB

echo "=== Varying Byte Count Sweep ==="
echo "Binary: $BINARY"
echo "Records per dataset: ${NREC}"
echo "Warm runs per dataset: $RUNS"
echo ""

for NVARY in 8 16 24 32 40 48 56 66; do
    INPUT="/dev/shm/vary${NVARY}_60M_normalized.bin"
    LOG="${OUTDIR}/vary${NVARY}.log"

    echo "--- Generating: ${NVARY}/66 varying bytes ---"
    python3 scripts/gen_varying_bytes.py $NVARY $NREC "$INPUT"
    echo ""

    echo "--- Sorting: ${NVARY}/66 varying bytes ---"
    $BINARY --input "$INPUT" --runs $RUNS 2>&1 | tee "$LOG"
    echo ""

    # Extract CSV lines
    echo "CSV summary for vary${NVARY}:"
    grep "^CSV" "$LOG" || true
    echo ""

    # Clean up to save space
    rm -f "$INPUT"
    echo "  Cleaned up $INPUT"
    echo ""
done

echo "=== Varying Byte Sweep Complete ==="
