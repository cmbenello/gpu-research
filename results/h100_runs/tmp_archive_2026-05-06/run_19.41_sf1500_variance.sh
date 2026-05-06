#!/bin/bash
# 19.41 — SF1500 stream pre-pin n=5 for tighter variance bound.
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.41_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

WALLS=()
for RUN in 1 2 3 4 5; do
    echo "=========================================="
    echo "Run ${RUN}"
    echo "=========================================="
    $EVICT $INPUT
    sleep 3
    free -g | head -2

    T0=$(date +%s.%N)
    $BIN $INPUT $NVME/sf1500 8 2>&1 | grep -E "Pre-pin|Sort phase|Total wall"
    T1=$(date +%s.%N)
    WALL=$(echo "$T1 - $T0" | bc)
    WALLS+=("$WALL")
    echo "Run ${RUN} wrapper wall: ${WALL} s"

    # Verify quickly
    if [ -f $NVME/sf1500.sorted_0.bin ]; then
        CHECK_LIMIT=100000 /home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets $INPUT $NVME/sf1500.sorted_0.bin 2>&1 | tail -1
    fi
    rm -f $NVME/sf1500.*
done

echo "=========================================="
echo "n=5 walls: ${WALLS[*]}"

rm -rf $NVME
date '+%H:%M:%S — done'
