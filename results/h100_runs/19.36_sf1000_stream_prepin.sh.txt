#!/bin/bash
# 19.36 — SF1000 stream pre-pin (option 3 final).
# Predicted ~5min based on linear scaling from SF1500 (1.08 TB / 6m23s).
# SF1000 = 720 GB → predicted 720/1080 × 6.4 = 4.3 min.
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1000.bin
NVME=/mnt/data/19.36_sf1000
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2
free -g | head -2

$BIN $INPUT $NVME/sf1000 8
date '+%H:%M:%S — done'

if [ -f $NVME/sf1000.sorted_0.bin ]; then
    CHECK_LIMIT=1000000 /home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets $INPUT $NVME/sf1000.sorted_0.bin 2>&1 | tail -2
fi

rm -rf $NVME
date '+%H:%M:%S — cleaned'
