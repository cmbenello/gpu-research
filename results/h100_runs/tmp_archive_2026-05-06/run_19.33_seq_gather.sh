#!/bin/bash
# 19.33 — Test gather_records_seq (sequential read input + scattered writes)
# vs gather_records (random read + sequential write).
# Run on SF300 (smaller scale, faster iteration).
set -uo pipefail
source ~/gpu-research/h100/env.sh

STREAM=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
GATHER_RAND=/home/cc/gpu-research/gpu_crocsort/experiments/gather_records
GATHER_SEQ=/home/cc/gpu-research/gpu_crocsort/experiments/gather_records_seq
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf300.bin
NVME=/mnt/data/19.33_sf300
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2

# Phase 1: stream sort to get sorted offsets
echo "=== Stream sort (SF300) ==="
T0=$(date +%s.%N)
$STREAM $INPUT $NVME/sf300 8
T1=$(date +%s.%N)
echo "Stream wall: $(echo "$T1 - $T0" | bc) s"

# Build args for both gathers
ARGS=""
for B in $(seq 0 7); do
    ARGS="$ARGS $NVME/sf300.sorted_${B}.bin"
done

# Phase 2a: random-read gather
echo "=== gather_records (random reads) ==="
T0=$(date +%s.%N)
$GATHER_RAND $INPUT $NVME/sorted_records_rand.bin $ARGS
T1=$(date +%s.%N)
RAND_WALL=$(echo "$T1 - $T0" | bc)
echo "Random gather: ${RAND_WALL} s"
rm -f $NVME/sorted_records_rand.bin
$EVICT $INPUT  # reset cache state

# Phase 2b: sequential-read gather
echo "=== gather_records_seq (sequential reads) ==="
T0=$(date +%s.%N)
$GATHER_SEQ $INPUT $NVME/sorted_records_seq.bin $ARGS
T1=$(date +%s.%N)
SEQ_WALL=$(echo "$T1 - $T0" | bc)
echo "Sequential gather: ${SEQ_WALL} s"

echo "=========================="
echo "Random: ${RAND_WALL} s"
echo "Sequential: ${SEQ_WALL} s"

# Verify seq output
python3 -c "
KEY_SIZE = 66; RECORD_SIZE = 120; N = 1000000
bad = 0; prev = None
with open('$NVME/sorted_records_seq.bin', 'rb') as f:
    for i in range(N):
        rec = f.read(RECORD_SIZE)
        if len(rec) < RECORD_SIZE: break
        key = rec[:KEY_SIZE]
        if prev is not None and prev > key: bad += 1
        prev = key
print(f'Seq verify {N} records, {bad} violations: {\"PASS\" if bad==0 else \"FAIL\"}')
"

rm -rf $NVME
date '+%H:%M:%S — done'
