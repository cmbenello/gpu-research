#!/usr/bin/env bash
# Tier 1.1/1.3/1.5 — generate the big TPC-H scales the H100 needs.
# Idempotent: skips scales whose binary file already exists at the right size.
# Disk-safe: stops if generating the next scale would exceed 80% of $DATA_DIR.
#
# Usage:  bash h100/gen_all_scales.sh [SF1 SF2 ...]
# Default scales: 50 100 300 500 1000
set -uo pipefail

WORK_DIR="${WORK_DIR:-$HOME/gpu-research}"
DATA_DIR="${DATA_DIR:-$HOME/data}"
mkdir -p "$DATA_DIR"

# Each lineitem record is 120 bytes; row counts from TPC-H spec.
declare -A ROW_COUNTS=(
    [10]=59986052
    [50]=300005811
    [100]=600037902
    [300]=1799989091
    [500]=3000028242
    [1000]=5999989709
)

SCALES=("${@:-50 100 300 500 1000}")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

disk_ok() {
    local need_gb=$1
    local avail_kb=$(df -P "$DATA_DIR" | tail -1 | awk '{print $4}')
    local avail_gb=$((avail_kb / 1024 / 1024))
    local total_kb=$(df -P "$DATA_DIR" | tail -1 | awk '{print $2}')
    local total_gb=$((total_kb / 1024 / 1024))
    local cap_gb=$((total_gb * 80 / 100))
    log "  disk: ${avail_gb}GB free of ${total_gb}GB (cap ${cap_gb}GB)"
    [ "$avail_gb" -gt "$need_gb" ] || return 1
    return 0
}

for sf in $SCALES; do
    out="$DATA_DIR/lineitem_sf${sf}.bin"
    rows=${ROW_COUNTS[$sf]:-0}
    if [ "$rows" -eq 0 ]; then
        log "WARN: unknown row count for SF${sf}, skipping"
        continue
    fi
    expected_bytes=$((rows * 120))
    expected_gb=$((expected_bytes / 1000000000 + 1))

    if [ -f "$out" ]; then
        actual=$(stat -c %s "$out" 2>/dev/null || stat -f %z "$out")
        if [ "$actual" -eq "$expected_bytes" ]; then
            log "SF${sf}: already done ($expected_gb GB)"
            continue
        fi
        log "SF${sf}: file present but wrong size ($actual vs $expected_bytes), regenerating"
        rm -f "$out"
    fi

    log "SF${sf}: need ${expected_gb}GB"
    if ! disk_ok "$expected_gb"; then
        log "SF${sf}: insufficient disk, stopping"
        break
    fi

    log "SF${sf}: starting gen via h100/gen_tpch_fast.py..."
    t0=$(date +%s)
    python3 "$WORK_DIR/h100/gen_tpch_fast.py" "$sf" "$out" 2>&1 | tail -5
    t1=$(date +%s)
    elapsed=$((t1 - t0))

    if [ -f "$out" ]; then
        actual=$(stat -c %s "$out" 2>/dev/null || stat -f %z "$out")
        if [ "$actual" -eq "$expected_bytes" ]; then
            log "SF${sf}: done in ${elapsed}s ($((expected_bytes / elapsed / 1000000)) MB/s effective)"
        else
            log "SF${sf}: WARN size mismatch ($actual vs $expected_bytes)"
        fi
    else
        log "SF${sf}: FAILED"
    fi
done

log "All requested scales attempted."
log "Files in $DATA_DIR:"
ls -lah "$DATA_DIR"/lineitem_sf*.bin 2>/dev/null
