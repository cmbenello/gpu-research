#!/bin/bash
# 19.37 — SF1000 stream + gather full e2e (sorted records output).
set -uo pipefail
source ~/gpu-research/h100/env.sh

STREAM=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
GATHER=/home/cc/gpu-research/gpu_crocsort/experiments/gather_records
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1000.bin
NVME=/mnt/data/19.37_sf1000
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2

T0=$(date +%s.%N)
echo "=== Phase 1: stream partition + sort (offsets only) ==="
$STREAM $INPUT $NVME/sf1000 8
T1=$(date +%s.%N)
STREAM_WALL=$(echo "$T1 - $T0" | bc)

echo "=== Phase 2: gather full sorted records ==="
ARGS=""
for B in $(seq 0 7); do
    ARGS="$ARGS $NVME/sf1000.sorted_${B}.bin"
done
T2=$(date +%s.%N)
$GATHER $INPUT $NVME/sorted_records.bin $ARGS
T3=$(date +%s.%N)
GATHER_WALL=$(echo "$T3 - $T2" | bc)

T_TOTAL=$(echo "$T3 - $T0" | bc)
echo "=== Stream wall: ${STREAM_WALL} s ==="
echo "=== Gather wall: ${GATHER_WALL} s ==="
echo "=== TOTAL E2E: ${T_TOTAL} s ==="

ls -la $NVME/sorted_records.bin

# Verify first 1M records
python3 << 'EOF'
KEY_SIZE = 66; RECORD_SIZE = 120; N = 1000000
import os
fname = '/mnt/data/19.37_sf1000/sorted_records.bin'
bad = 0; prev = None
with open(fname, 'rb') as f:
    for i in range(N):
        rec = f.read(RECORD_SIZE)
        if len(rec) < RECORD_SIZE: break
        key = rec[:KEY_SIZE]
        if prev is not None and prev > key: bad += 1
        prev = key
print(f'Verify {N} records, {bad} violations: {"PASS" if bad==0 else "FAIL"}')
EOF

rm -rf $NVME
date '+%H:%M:%S — done'
