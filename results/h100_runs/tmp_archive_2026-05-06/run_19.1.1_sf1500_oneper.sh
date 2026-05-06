#!/bin/bash
# 19.1.1 — SF1500 (1.08 TB) with 2-GPU one-per-node strategy (18.5c style).
# 2 rounds: GPU 0 (node 0) + GPU 2 (node 1) sort 2 buckets each round.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.1.1_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition (one-time, ~18 min) ==="
$PART $INPUT $NVME/part 4
date '+%H:%M:%S — partitioned'
df -h /mnt/data | tail -1

echo "=== Step 2: 2-GPU one-per-node, 2 rounds ==="
SORTSTART=$(date +%s.%N)

date "+%H:%M:%S — round 1 start (GPU 0 → bucket 0, GPU 2 → bucket 2)"
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_0.bin --output-file $NVME/sorted_0.bin \
    --runs 1 --no-verify > /tmp/19.1.1_g0_b0.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_2.bin --output-file $NVME/sorted_2.bin \
    --runs 1 --no-verify > /tmp/19.1.1_g2_b2.log 2>&1 &
wait
rm -f $NVME/part.bucket_0.bin $NVME/part.bucket_2.bin
date "+%H:%M:%S — round 1 done"
df -h /mnt/data | tail -1

date "+%H:%M:%S — round 2 start (GPU 0 → bucket 1, GPU 2 → bucket 3)"
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_1.bin --output-file $NVME/sorted_1.bin \
    --runs 1 --no-verify > /tmp/19.1.1_g0_b1.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_3.bin --output-file $NVME/sorted_3.bin \
    --runs 1 --no-verify > /tmp/19.1.1_g2_b3.log 2>&1 &
wait
rm -f $NVME/part.bucket_1.bin $NVME/part.bucket_3.bin
date "+%H:%M:%S — round 2 done"

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo
echo "=== Sort-only wall (no partition): ${SORTWALL} s ==="
for log in /tmp/19.1.1_g0_b0 /tmp/19.1.1_g2_b2 /tmp/19.1.1_g0_b1 /tmp/19.1.1_g2_b3; do
    bucket=$(echo $log | grep -o "b[0-9]")
    last=$(grep "^CSV" $log.log | tail -1)
    echo "$bucket: $last"
done

echo "=== cleanup ==="
rm -rf $NVME
date '+%H:%M:%S — done'
