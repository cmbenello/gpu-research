#!/bin/bash
set -uo pipefail
# Scale experiments: larger datasets for paper credibility
# - NYC Taxi 2019-2023 (~200M records, ~24 GB) — multi-year real-world
# - Random 600M (~72 GB) — worst-case at SF100 scale

BINARY=./external_sort_tpch_compact
RUNS=3
OUTDIR=/dev/shm/scale_experiments
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

# ── NYC Taxi 2019-2023 ──
log "╔════════════════════════════════════════════╗"
log "║  NYC Taxi 2019-2023 (5 years)              ║"
log "╚════════════════════════════════════════════╝"

TAXI_5Y=/dev/shm/nyctaxi_2019_2023_normalized.bin
if [ ! -f "$TAXI_5Y" ]; then
    log "Generating NYC Taxi 2019-2023 dataset..."
    python3 "$SCRIPTS/gen_nyctaxi_normalized.py" "2019-2023" "$TAXI_5Y" 2>&1 | tee "$OUTDIR/gen_taxi_5y.log"
fi
if [ -f "$TAXI_5Y" ]; then
    run_sort "taxi_2019_2023" "$TAXI_5Y"
    run_sort_compact_off "taxi_2019_2023" "$TAXI_5Y"
    rm -f "$TAXI_5Y"
fi

# Clean up parquet cache between datasets
rm -rf /dev/shm/nyctaxi_parquet

# ── Random 600M (72 GB — SF100 equivalent) ──
log "╔════════════════════════════════════════════╗"
log "║  Random 600M (72 GB, SF100 scale)          ║"
log "╚════════════════════════════════════════════╝"

RAND_600M=/dev/shm/random_600M_normalized.bin
log "Generating 600M random records..."
python3 "$SCRIPTS/gen_random_normalized.py" 600000000 "$RAND_600M" 2>&1 | tee "$OUTDIR/gen_random_600M.log"
if [ -f "$RAND_600M" ]; then
    run_sort "random_600M" "$RAND_600M"
    rm -f "$RAND_600M"
fi

# ── Summary ──
log "╔════════════════════════════════════════════╗"
log "║  SCALE EXPERIMENTS COMPLETE                 ║"
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

log "Done."
