#!/bin/bash
# 19.27 — Streaming partition+sort with K=8 (vs K=16 in 19.25).
# Fewer larger buckets → fewer rounds with more amortized per-round overhead.
# Projected ~6m40s (vs 7m17s mean at K=16).
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.27_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2
free -g | head -2

$BIN $INPUT $NVME/sf1500 8
date '+%H:%M:%S — done'

if [ -f $NVME/sf1500.sorted_0.bin ]; then
    CHECK_LIMIT=1000000 /home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets $INPUT $NVME/sf1500.sorted_0.bin 2>&1 | tail -2
fi

rm -rf $NVME
date '+%H:%M:%S — cleaned'
