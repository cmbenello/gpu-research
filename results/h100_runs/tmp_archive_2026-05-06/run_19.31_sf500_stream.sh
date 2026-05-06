#!/bin/bash
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf500.bin
NVME=/mnt/data/19.31_sf500
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2
free -g | head -2

$BIN $INPUT $NVME/sf500 8
date '+%H:%M:%S — done'

if [ -f $NVME/sf500.sorted_0.bin ]; then
    CHECK_LIMIT=1000000 /home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets $INPUT $NVME/sf500.sorted_0.bin 2>&1 | tail -2
fi

rm -rf $NVME
date '+%H:%M:%S — cleaned'
