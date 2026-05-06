#!/bin/bash
# 19.7 — SF1500 with SKIP_HOST_PIN=1, sequential 1-GPU strategy.
# Tests the predicted 31% win from skipping cudaHostRegister.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.7_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition ==="
$PART $INPUT $NVME/part 4
date '+%H:%M:%S — partitioned'

echo "=== Step 2: SEQUENTIAL 1-GPU sort with SKIP_HOST_PIN=1 ==="
SORTSTART=$(date +%s.%N)
for B in 0 1 2 3; do
    date "+%H:%M:%S — bucket $B start"
    SKIP_HOST_PIN=1 NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --preferred=0 \
        $BIN --input $NVME/part.bucket_${B}.bin --output-file $NVME/sorted_${B}.bin \
        --runs 1 --no-verify > /tmp/19.7_b${B}.log 2>&1
    rm -f $NVME/part.bucket_${B}.bin
    grep "^CSV" /tmp/19.7_b${B}.log | tail -1
done
SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo
echo "=== Sort-only wall (sequential SKIP_HOST_PIN): ${SORTWALL} s ==="

rm -rf $NVME
date '+%H:%M:%S — done'
