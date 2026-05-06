#!/bin/bash
# 19.25 — SF1500 streaming partition + sort. No intermediate NVMe write.
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.25_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2
free -g | head -2

# Stream version with K=16
$BIN $INPUT $NVME/sf1500 16
date '+%H:%M:%S — done'

# Quick verify
ls -la $NVME/sf1500.sorted_*.bin | head -16
if [ -f $NVME/sf1500.sorted_0.bin ]; then
    CHECK_LIMIT=1000000 /home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets $INPUT $NVME/sf1500.sorted_0.bin 2>&1 | tail -2
fi

rm -rf $NVME
date '+%H:%M:%S — cleaned'
