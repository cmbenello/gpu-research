#!/bin/bash
# 19.32 — Stream pre-pin at SF100 and SF300 for full scaling chart.
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
VERIFY=/home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets

run_scale() {
    local SF=$1
    local INPUT=/mnt/data/lineitem_sf${SF}.bin
    local NVME=/mnt/data/19.32_sf${SF}
    mkdir -p $NVME

    echo "=========================================="
    echo "SF${SF}"
    echo "=========================================="
    date '+%H:%M:%S — start'
    $EVICT $INPUT
    sleep 2

    T0=$(date +%s.%N)
    $BIN $INPUT $NVME/sf 8
    T1=$(date +%s.%N)
    WALL=$(echo "$T1 - $T0" | bc)
    echo "SF${SF} wall: ${WALL} s"

    if [ -f $NVME/sf.sorted_0.bin ]; then
        CHECK_LIMIT=1000000 $VERIFY $INPUT $NVME/sf.sorted_0.bin 2>&1 | tail -2
    fi
    rm -rf $NVME
    date '+%H:%M:%S — done'
}

run_scale 100
run_scale 300
