#!/bin/bash
# Record size sensitivity: fixed 66B key, varying value size
# Shows key-value separation advantage: GPU sorts keys only,
# larger values increase gather cost but not sort cost.
set -e

RUNS=3
OUTDIR=/dev/shm/paper_experiments
LOG="${OUTDIR}/recsize_sweep.log"

echo "=== Record Size Sensitivity Experiment ===" | tee "$LOG"
echo "Key size: 66B (fixed)" | tee -a "$LOG"
echo "Warm runs per size: $RUNS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Value sizes to test. Record size = 66 + VSIZE.
# At 60M records: 66B=3.96GB, 120B=7.2GB, 194B=11.6GB, 322B=19.3GB, 578B=34.7GB
# Use 60M for sizes up to 322B (fit in GPU memory)
# Use 30M for 578B (30M × 578B = 17.3 GB, fits in GPU)
for VSIZE in 0 54 128 256 512; do
    RSIZE=$((66 + VSIZE))
    BINARY="./external_sort_v${VSIZE}"

    if [ ! -f "$BINARY" ]; then
        echo "SKIP: $BINARY not found" | tee -a "$LOG"
        continue
    fi

    if [ $VSIZE -le 256 ]; then
        NREC=60000000
    else
        NREC=30000000  # 578B records need less count to fit
    fi

    SIZE_GB=$(echo "scale=2; $NREC * $RSIZE / 1073741824" | bc)
    INPUT="/dev/shm/recsize_v${VSIZE}_normalized.bin"

    echo "--- VALUE=${VSIZE}B, RECORD=${RSIZE}B, ${NREC} records, ${SIZE_GB} GB ---" | tee -a "$LOG"

    # Generate random records
    python3 scripts/gen_random_records.py $NREC "$INPUT" --record-size $RSIZE 2>&1 | tail -1

    # Sort
    $BINARY --input "$INPUT" --runs $RUNS 2>&1 | tee -a "$LOG"

    # Extract CSV
    echo "CSV summary for v${VSIZE}:" | tee -a "$LOG"
    grep "^CSV" "$LOG" | tail -$RUNS
    echo "" | tee -a "$LOG"

    # Clean up
    rm -f "$INPUT"
    echo "  Cleaned up $INPUT" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "=== Record Size Sensitivity Complete ===" | tee -a "$LOG"
