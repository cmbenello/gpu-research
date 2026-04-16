#!/bin/bash
set -uo pipefail
# Resume overnight experiments from Section 2b (taxi 6mo)
# Sections already completed: TPC-H SF10/50/100, Taxi 1mo sort+compact

BINARY=./external_sort_tpch_compact
RUNS=3
DATE=$(date +%Y-%m-%d)
OUTDIR=/dev/shm/overnight_${DATE}
SCRIPTS=$(dirname "$0")

mkdir -p "$OUTDIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTDIR/master.log"; }

run_sort() {
    local label=$1 input=$2 extra_env=${3:-}
    local logfile="$OUTDIR/${label}.log"
    log "=== SORT: $label ($RUNS warm runs) ==="
    if [ -n "$extra_env" ]; then
        env $extra_env $BINARY --input "$input" --runs "$RUNS" 2>&1 | tee "$logfile"
    else
        $BINARY --input "$input" --runs "$RUNS" 2>&1 | tee "$logfile"
    fi
    log "  Done: $label"
    echo
}

run_sort_compact_off() {
    local label=$1 input=$2
    local logfile="$OUTDIR/${label}_compact_off.log"
    log "=== SORT (compact OFF): $label ($RUNS warm runs) ==="
    DISABLE_COMPACT=1 $BINARY --input "$input" --runs "$RUNS" 2>&1 | tee "$logfile"
    log "  Done: $label (compact OFF)"
    echo
}

# ── Section 2b: NYC Taxi 6 months ──
log "╔════════════════════════════════════════════╗"
log "║  Section 2b: NYC Taxi 6 months (resume)    ║"
log "╚════════════════════════════════════════════╝"

TAXI_6MO=/dev/shm/nyctaxi_6mo_normalized.bin
if [ ! -f "$TAXI_6MO" ]; then
    log "Generating NYC Taxi 6-month dataset..."
    python3 "$SCRIPTS/gen_nyctaxi_normalized.py" 6 "$TAXI_6MO" 2>&1 | tee "$OUTDIR/gen_taxi_6mo.log"
fi
if [ -f "$TAXI_6MO" ]; then
    run_sort "taxi_6mo" "$TAXI_6MO"
    run_sort_compact_off "taxi_6mo" "$TAXI_6MO"

    # Thread scaling
    log "Running fixup thread scaling on taxi_6mo..."
    for T in 1 4 8 24; do
        LOGFILE="$OUTDIR/taxi_6mo_threads_${T}.log"
        log "  FIXUP_THREADS=$T"
        FIXUP_THREADS=$T $BINARY --input "$TAXI_6MO" --runs "$RUNS" 2>&1 | tee "$LOGFILE"
    done
fi

# ── Section 2c: NYC Taxi 12 months ──
TAXI_12MO=/dev/shm/nyctaxi_12mo_normalized.bin
if [ ! -f "$TAXI_12MO" ]; then
    rm -f "$TAXI_6MO"  # free space
    log "Generating NYC Taxi 12-month dataset..."
    python3 "$SCRIPTS/gen_nyctaxi_normalized.py" 12 "$TAXI_12MO" 2>&1 | tee "$OUTDIR/gen_taxi_12mo.log"
fi
if [ -f "$TAXI_12MO" ]; then
    run_sort "taxi_12mo" "$TAXI_12MO"
    rm -f "$TAXI_12MO"
fi

# ── Section 3: Uniform Random ──
log "╔════════════════════════════════════════════╗"
log "║  Section 3: Uniform Random 66B Keys        ║"
log "╚════════════════════════════════════════════╝"

RAND_60M=/dev/shm/random_60M_normalized.bin
log "Generating 60M random records..."
python3 "$SCRIPTS/gen_random_normalized.py" 60000000 "$RAND_60M" 2>&1 | tee "$OUTDIR/gen_random_60M.log"
if [ -f "$RAND_60M" ]; then
    run_sort "random_60M" "$RAND_60M"
    run_sort_compact_off "random_60M" "$RAND_60M"
    rm -f "$RAND_60M"
fi

RAND_300M=/dev/shm/random_300M_normalized.bin
log "Generating 300M random records..."
python3 "$SCRIPTS/gen_random_normalized.py" 300000000 "$RAND_300M" 2>&1 | tee "$OUTDIR/gen_random_300M.log"
if [ -f "$RAND_300M" ]; then
    run_sort "random_300M" "$RAND_300M"
    run_sort_compact_off "random_300M" "$RAND_300M"
    rm -f "$RAND_300M"
fi

# ── Section 4: Adversarial variants ──
log "╔════════════════════════════════════════════╗"
log "║  Section 4: Adversarial Variants           ║"
log "╚════════════════════════════════════════════╝"

if [ -f /tmp/lineitem_sf50_normalized.bin ]; then
    run_sort_compact_off "tpch_sf50" /tmp/lineitem_sf50_normalized.bin
    run_sort "tpch_sf50_zero16" /tmp/lineitem_sf50_normalized.bin "ADV_ZERO_BYTES=16"
    run_sort "tpch_sf50_zero32" /tmp/lineitem_sf50_normalized.bin "ADV_ZERO_BYTES=32"
fi

# ── Summary ──
log "╔════════════════════════════════════════════╗"
log "║  ALL EXPERIMENTS COMPLETE                   ║"
log "╚════════════════════════════════════════════╝"

log "Generating summary..."
echo "" >> "$OUTDIR/master.log"
echo "=== TIMING SUMMARY ===" >> "$OUTDIR/master.log"
for f in "$OUTDIR"/*.log; do
    [ "$f" = "$OUTDIR/master.log" ] && continue
    name=$(basename "$f" .log)
    csvlines=$(grep "^CSV," "$f" 2>/dev/null || true)
    if [ -n "$csvlines" ]; then
        echo "--- $name ---" >> "$OUTDIR/master.log"
        echo "$csvlines" >> "$OUTDIR/master.log"
    fi
done

# Clean up parquet cache
rm -rf /dev/shm/nyctaxi_parquet

PERSIST_DIR=$(cd "$(dirname "$0")/.."; pwd)/results/overnight_${DATE}
mkdir -p "$PERSIST_DIR" 2>/dev/null || true
cp "$OUTDIR/master.log" "$PERSIST_DIR/master.log" 2>/dev/null || log "WARNING: disk full"

log "Done."
