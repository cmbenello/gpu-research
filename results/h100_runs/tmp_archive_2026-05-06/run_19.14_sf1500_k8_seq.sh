#!/bin/bash
# 19.14 — SF1500 K=8 + SEQUENTIAL 1-GPU sort.
# 8 buckets × 135 GB each, one at a time. Should never OOM.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.14_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition K=8 ==="
$PART $INPUT $NVME/part 8
date '+%H:%M:%S — partitioned'

echo "=== Step 2: SEQUENTIAL 1-GPU sort, 8 buckets ==="
SORTSTART=$(date +%s.%N)
for B in 0 1 2 3 4 5 6 7; do
    date "+%H:%M:%S — bucket $B start"
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --preferred=0 \
        $BIN --input $NVME/part.bucket_${B}.bin --output-file $NVME/sorted_${B}.bin \
        --runs 1 --no-verify > /tmp/19.14_b${B}.log 2>&1
    rm -f $NVME/part.bucket_${B}.bin
    grep "^CSV" /tmp/19.14_b${B}.log | tail -1
done
SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo
echo "=== Sort-only wall (sequential K=8): ${SORTWALL} s ==="

rm -rf $NVME
date '+%H:%M:%S — done'
