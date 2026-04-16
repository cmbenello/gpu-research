#!/bin/bash
set -euo pipefail
INPUT=${1:-/tmp/lineitem_sf50_normalized.bin}
RUNS=${2:-6}
OUTDIR=${3:-results/2026-04-16-fixup-thread-scaling}
BINARY=./external_sort_tpch_compact

for T in 1 2 4 8 16 24 48; do
    echo "============================================"
    echo "FIXUP_THREADS=$T  ($RUNS warm runs)"
    echo "============================================"
    LOGFILE="$OUTDIR/sf50_threads_${T}.log"
    FIXUP_THREADS=$T $BINARY --input "$INPUT" --runs "$RUNS" 2>&1 | tee "$LOGFILE"
    echo
done

echo "=== SUMMARY ==="
for T in 1 2 4 8 16 24 48; do
    LOGFILE="$OUTDIR/sf50_threads_${T}.log"
    RG=$(grep "runs in" "$LOGFILE" | tail -1 | grep -oP '\d+ ms' | head -1)
    MGF=$(grep "Total merge" "$LOGFILE" | tail -1 | grep -oP '\d+ ms' | head -1)
    FIX=$(grep "Fixup:" "$LOGFILE" | tail -1 | grep -oP 'Fixup: \d+ ms' | grep -oP '\d+')
    GD=$(grep "group-detect" "$LOGFILE" | tail -1 | grep -oP 'group-detect \d+' | grep -oP '\d+')
    PAR=$(grep "parallel" "$LOGFILE" | tail -1 | grep -oP 'parallel \d+' | grep -oP '\d+')
    echo "T=$T  fixup=${FIX}ms  group-detect=${GD}ms  parallel-sort=${PAR}ms"
done
