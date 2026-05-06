#!/bin/bash
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/gds_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2
free -g | head -2

T0=$(date +%s.%N)
$BIN $INPUT /tmp/19.45_sf1500 8
T1=$(date +%s.%N)
echo "GDS partition wall: $(echo "$T1 - $T0" | bc) s"

date '+%H:%M:%S — done'
