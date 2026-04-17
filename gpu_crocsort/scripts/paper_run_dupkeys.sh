#!/bin/bash
# Run GPU sort on duplicate-key datasets and collect results
set -e

BINARY=./external_sort_tpch_compact
RUNS=3
OUTDIR=/dev/shm/paper_experiments

echo "=== Duplicate Key Experiments ==="
echo "Binary: $BINARY"
echo "Warm runs per dataset: $RUNS"
echo ""

for MODE in pool1000 pool10 zipf; do
    INPUT="/dev/shm/dupkeys_${MODE}_60M_normalized.bin"
    LOG="${OUTDIR}/dupkeys_${MODE}.log"

    if [ ! -f "$INPUT" ]; then
        echo "SKIP: $INPUT not found"
        continue
    fi

    SIZE_GB=$(echo "scale=2; $(stat -c%s "$INPUT") / 1073741824" | bc)
    NREC=$(echo "$(stat -c%s "$INPUT") / 120" | bc)

    echo "--- ${MODE}: ${NREC} records, ${SIZE_GB} GB ---"
    $BINARY --input "$INPUT" --runs $RUNS 2>&1 | tee "$LOG"
    echo ""

    # Extract CSV lines
    echo "CSV summary for ${MODE}:"
    grep "^CSV" "$LOG" || true
    echo ""
done

echo "=== Duplicate Key Experiments Complete ==="
