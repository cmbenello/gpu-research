#!/bin/bash
# 19.12 — SF1500 with K=8 partitioning + 4-GPU concurrent sort.
# 8 buckets × 135 GB each, 2 rounds of 4-GPU concurrent.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.12_sf1500_k8
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition K=8 ==="
$PART $INPUT $NVME/part 8
date '+%H:%M:%S — partitioned'
df -h /mnt/data | tail -1
ls -la $NVME/

echo "=== Step 2: 4-GPU concurrent sort, 2 rounds × 4 buckets ==="
SORTSTART=$(date +%s.%N)

date "+%H:%M:%S — round 1 start"
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_0.bin --output-file $NVME/sorted_0.bin --runs 1 --no-verify > /tmp/19.12_b0.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_1.bin --output-file $NVME/sorted_1.bin --runs 1 --no-verify > /tmp/19.12_b1.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_2.bin --output-file $NVME/sorted_2.bin --runs 1 --no-verify > /tmp/19.12_b2.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_3.bin --output-file $NVME/sorted_3.bin --runs 1 --no-verify > /tmp/19.12_b3.log 2>&1 &
wait
rm -f $NVME/part.bucket_0.bin $NVME/part.bucket_1.bin $NVME/part.bucket_2.bin $NVME/part.bucket_3.bin
date "+%H:%M:%S — round 1 done"

date "+%H:%M:%S — round 2 start"
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_4.bin --output-file $NVME/sorted_4.bin --runs 1 --no-verify > /tmp/19.12_b4.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_5.bin --output-file $NVME/sorted_5.bin --runs 1 --no-verify > /tmp/19.12_b5.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_6.bin --output-file $NVME/sorted_6.bin --runs 1 --no-verify > /tmp/19.12_b6.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_7.bin --output-file $NVME/sorted_7.bin --runs 1 --no-verify > /tmp/19.12_b7.log 2>&1 &
wait
rm -f $NVME/part.bucket_4.bin $NVME/part.bucket_5.bin $NVME/part.bucket_6.bin $NVME/part.bucket_7.bin
date "+%H:%M:%S — round 2 done"

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo "=== Sort-only wall: ${SORTWALL} s ==="
for B in 0 1 2 3 4 5 6 7; do
    csv=$(grep "^CSV" /tmp/19.12_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed" /tmp/19.12_b$B.log | tail -1)
        echo "b$B: FAILED — $err"
    else
        echo "b$B: $csv"
    fi
done

rm -rf $NVME
date '+%H:%M:%S — done'
