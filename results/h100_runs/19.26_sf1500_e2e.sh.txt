#!/bin/bash
# 19.26 — Streaming partition+sort + gather to produce full sorted records.
# True end-to-end vs the 49m baseline output format.
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
GATHER=/home/cc/gpu-research/gpu_crocsort/experiments/gather_records
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.26_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2

T0=$(date +%s.%N)
echo "=== Phase 1: stream partition + sort (offsets only) ==="
$BIN $INPUT $NVME/sf1500 16
T1=$(date +%s.%N)
PART_SORT_WALL=$(echo "$T1 - $T0" | bc)
echo "Stream wall: ${PART_SORT_WALL} s"

echo "=== Phase 2: gather records (sorted offsets → full sorted records) ==="
T2=$(date +%s.%N)
# Pass all 16 sorted_* files in correct order (b0..b15)
ARGS=""
for B in $(seq 0 15); do
    ARGS="$ARGS $NVME/sf1500.sorted_${B}.bin"
done
$GATHER $INPUT $NVME/sorted_records.bin $ARGS
T3=$(date +%s.%N)
GATHER_WALL=$(echo "$T3 - $T2" | bc)
echo "Gather wall: ${GATHER_WALL} s"

T_TOTAL=$(echo "$T3 - $T0" | bc)
echo "=== TOTAL E2E (stream + gather): ${T_TOTAL} s ==="

ls -la $NVME/sorted_records.bin

# Verify a sample of the sorted records
echo "=== Verifying sorted_records.bin (1M record sample) ==="
python3 -c "
import sys
KEY_SIZE = 66
RECORD_SIZE = 120
N_CHECK = 1_000_000
with open('$NVME/sorted_records.bin', 'rb') as f:
    bad = 0
    prev = None
    for i in range($N_CHECK):
        rec = f.read(RECORD_SIZE)
        if not rec or len(rec) < RECORD_SIZE: break
        key = rec[:KEY_SIZE]
        if prev is not None and prev > key:
            bad += 1
            if bad < 5: print(f'  violation at {i}', file=sys.stderr)
        prev = key
print(f'Checked {N_CHECK} records, {bad} violations: {\"PASS\" if bad==0 else \"FAIL\"}')
"

rm -rf $NVME
date '+%H:%M:%S — done'
