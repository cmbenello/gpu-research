#!/bin/bash
set -euo pipefail
# ──────────────────────────────────────────────────────────
# Overnight experiment runner — multiple datasets + analyses
# ──────────────────────────────────────────────────────────
#
# Runs on: exp/fixup-fast-comparator binary (external_sort_tpch_compact)
# Output:  /dev/shm/overnight_YYYY-MM-DD/
#
# Datasets generated:
#   1. NYC Yellow Taxi 2023 (1mo, 6mo, 12mo)
#   2. Uniform random 66B keys (60M, 300M)
#   3. TPC-H (existing SF10, SF50, SF100 from /tmp)
#
# Experiments per dataset:
#   - Sort with 3 warm runs (timing + phase breakdown)
#   - Compact ON vs OFF (where applicable)
#   - DuckDB baseline (NYC Taxi only)
#
# Space management: generates, runs, and deletes sequentially
# to stay within /dev/shm (94 GB) + /tmp (108 GB existing).

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

# ──────────────────────────────────────────────────────────
# SECTION 1: TPC-H (existing data on /tmp)
# ──────────────────────────────────────────────────────────
log "╔════════════════════════════════════════════╗"
log "║  Section 1: TPC-H Scaling (existing data)  ║"
log "╚════════════════════════════════════════════╝"

if [ -f /tmp/lineitem_sf10_normalized.bin ]; then
    run_sort "tpch_sf10" /tmp/lineitem_sf10_normalized.bin
fi
if [ -f /tmp/lineitem_sf50_normalized.bin ]; then
    run_sort "tpch_sf50" /tmp/lineitem_sf50_normalized.bin
fi
if [ -f /tmp/lineitem_sf100_normalized.bin ]; then
    run_sort "tpch_sf100" /tmp/lineitem_sf100_normalized.bin
fi

# ──────────────────────────────────────────────────────────
# SECTION 2: NYC Yellow Taxi 2023
# ──────────────────────────────────────────────────────────
log "╔════════════════════════════════════════════╗"
log "║  Section 2: NYC Yellow Taxi 2023           ║"
log "╚════════════════════════════════════════════╝"

# 2a: 1 month (~38M records, ~4.5 GB) — quick test
TAXI_1MO=/dev/shm/nyctaxi_1mo_normalized.bin
if [ ! -f "$TAXI_1MO" ]; then
    log "Generating NYC Taxi 1-month dataset..."
    python3 "$SCRIPTS/gen_nyctaxi_normalized.py" 1 "$TAXI_1MO" 2>&1 | tee "$OUTDIR/gen_taxi_1mo.log"
fi
run_sort "taxi_1mo" "$TAXI_1MO"
run_sort_compact_off "taxi_1mo" "$TAXI_1MO"

# 2b: 6 months (~230M records, ~28 GB)
TAXI_6MO=/dev/shm/nyctaxi_6mo_normalized.bin
if [ ! -f "$TAXI_6MO" ]; then
    log "Generating NYC Taxi 6-month dataset..."
    python3 "$SCRIPTS/gen_nyctaxi_normalized.py" 6 "$TAXI_6MO" 2>&1 | tee "$OUTDIR/gen_taxi_6mo.log"
fi
# Delete 1mo first to save space
rm -f "$TAXI_1MO"
run_sort "taxi_6mo" "$TAXI_6MO"
run_sort_compact_off "taxi_6mo" "$TAXI_6MO"

# Thread scaling on 6mo (if fixup takes >500ms)
log "Running fixup thread scaling on taxi_6mo..."
for T in 1 4 8 24; do
    LOGFILE="$OUTDIR/taxi_6mo_threads_${T}.log"
    log "  FIXUP_THREADS=$T"
    FIXUP_THREADS=$T $BINARY --input "$TAXI_6MO" --runs "$RUNS" 2>&1 | tee "$LOGFILE"
done

# 2c: 12 months (~456M records, ~55 GB)
TAXI_12MO=/dev/shm/nyctaxi_12mo_normalized.bin
if [ ! -f "$TAXI_12MO" ]; then
    log "Generating NYC Taxi 12-month dataset..."
    # Delete 6mo to make room (55 GB needed, only ~66 GB free with 6mo present)
    rm -f "$TAXI_6MO"
    python3 "$SCRIPTS/gen_nyctaxi_normalized.py" 12 "$TAXI_12MO" 2>&1 | tee "$OUTDIR/gen_taxi_12mo.log"
fi
# Ensure 6mo is deleted
rm -f "$TAXI_6MO"
run_sort "taxi_12mo" "$TAXI_12MO"
# Clean up for next section
rm -f "$TAXI_12MO"

# ──────────────────────────────────────────────────────────
# SECTION 3: Uniform Random (worst case for compact key)
# ──────────────────────────────────────────────────────────
log "╔════════════════════════════════════════════╗"
log "║  Section 3: Uniform Random 66B Keys        ║"
log "╚════════════════════════════════════════════╝"

# 3a: 60M records (~7.2 GB, SF10-equivalent)
RAND_60M=/dev/shm/random_60M_normalized.bin
log "Generating 60M random records..."
python3 "$SCRIPTS/gen_random_normalized.py" 60000000 "$RAND_60M" 2>&1 | tee "$OUTDIR/gen_random_60M.log"
run_sort "random_60M" "$RAND_60M"
run_sort_compact_off "random_60M" "$RAND_60M"
rm -f "$RAND_60M"

# 3b: 300M records (~36 GB, SF50-equivalent)
RAND_300M=/dev/shm/random_300M_normalized.bin
log "Generating 300M random records..."
python3 "$SCRIPTS/gen_random_normalized.py" 300000000 "$RAND_300M" 2>&1 | tee "$OUTDIR/gen_random_300M.log"
run_sort "random_300M" "$RAND_300M"
run_sort_compact_off "random_300M" "$RAND_300M"
rm -f "$RAND_300M"

# ──────────────────────────────────────────────────────────
# SECTION 4: Adversarial variants
# ──────────────────────────────────────────────────────────
log "╔════════════════════════════════════════════╗"
log "║  Section 4: Adversarial Variants           ║"
log "╚════════════════════════════════════════════╝"

# TPC-H SF50 with compact disabled (how much does compact help on the hard case?)
if [ -f /tmp/lineitem_sf50_normalized.bin ]; then
    run_sort_compact_off "tpch_sf50" /tmp/lineitem_sf50_normalized.bin
fi

# TPC-H SF50 with ADV_ZERO_BYTES — make SF50 look like SF100 (fewer varying bytes)
if [ -f /tmp/lineitem_sf50_normalized.bin ]; then
    run_sort "tpch_sf50_zero16" /tmp/lineitem_sf50_normalized.bin "ADV_ZERO_BYTES=16"
    run_sort "tpch_sf50_zero32" /tmp/lineitem_sf50_normalized.bin "ADV_ZERO_BYTES=32"
fi

# ──────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────
log "╔════════════════════════════════════════════╗"
log "║  ALL EXPERIMENTS COMPLETE                   ║"
log "╚════════════════════════════════════════════╝"

log "Generating summary..."
echo "" >> "$OUTDIR/master.log"
echo "=== TIMING SUMMARY ===" >> "$OUTDIR/master.log"
for f in "$OUTDIR"/*.log; do
    [ "$f" = "$OUTDIR/master.log" ] && continue
    name=$(basename "$f" .log)
    # Extract CSV lines (warm runs only — skip first if multiple)
    csvlines=$(grep "^CSV," "$f" 2>/dev/null || true)
    if [ -n "$csvlines" ]; then
        echo "--- $name ---" >> "$OUTDIR/master.log"
        echo "$csvlines" >> "$OUTDIR/master.log"
    fi
done

log "Results in: $OUTDIR/"
log "Total overnight time: $(( ($(date +%s) - $(date -d "$DATE" +%s 2>/dev/null || echo $(date +%s))) )) seconds (approx)"

# Copy results to persistent storage (if space allows)
PERSIST_DIR=$(cd "$(dirname "$0")/.."; pwd)/results/overnight_${DATE}
mkdir -p "$PERSIST_DIR" 2>/dev/null || true
cp "$OUTDIR/master.log" "$PERSIST_DIR/master.log" 2>/dev/null || log "WARNING: could not copy to persistent storage (disk full?)"

log "Done."
