#!/bin/bash
# 19.1.2 — SF1500 with RELAXED NUMA: --preferred only, no --cpunodebind.
# Tests if the cudaMallocHost failure was due to CPU constraint vs memory.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.1.2_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition ==="
$PART $INPUT $NVME/part 4
date '+%H:%M:%S — partitioned'

echo "=== Step 2: 2-GPU one-per-node, RELAXED NUMA (--preferred only) ==="
SORTSTART=$(date +%s.%N)

# Round 1: --preferred only, no --cpunodebind
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --preferred=0 \
    $BIN --input $NVME/part.bucket_0.bin --output-file $NVME/sorted_0.bin \
    --runs 1 --no-verify > /tmp/19.1.2_g0_b0.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --preferred=1 \
    $BIN --input $NVME/part.bucket_2.bin --output-file $NVME/sorted_2.bin \
    --runs 1 --no-verify > /tmp/19.1.2_g2_b2.log 2>&1 &
wait
rm -f $NVME/part.bucket_0.bin $NVME/part.bucket_2.bin
date "+%H:%M:%S — round 1 done"

NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --preferred=0 \
    $BIN --input $NVME/part.bucket_1.bin --output-file $NVME/sorted_1.bin \
    --runs 1 --no-verify > /tmp/19.1.2_g0_b1.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --preferred=1 \
    $BIN --input $NVME/part.bucket_3.bin --output-file $NVME/sorted_3.bin \
    --runs 1 --no-verify > /tmp/19.1.2_g2_b3.log 2>&1 &
wait
rm -f $NVME/part.bucket_1.bin $NVME/part.bucket_3.bin
date "+%H:%M:%S — round 2 done"

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo
echo "=== Sort-only wall: ${SORTWALL} s ==="
for log in /tmp/19.1.2_g0_b0 /tmp/19.1.2_g2_b2 /tmp/19.1.2_g0_b1 /tmp/19.1.2_g2_b3; do
    bucket=$(echo $log | grep -o "b[0-9]")
    csv=$(grep "^CSV" $log.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed" $log.log | tail -1)
        echo "$bucket: FAILED — $err"
    else
        echo "$bucket: $csv"
    fi
done

rm -rf $NVME
date '+%H:%M:%S — done'
