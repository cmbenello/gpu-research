#!/bin/bash
# 19.18 — single round of K=16 4-GPU, no output write.
# Measure how much of round time is NVMe write of sorted output.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.18_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'
$PART $INPUT $NVME/part 16
date '+%H:%M:%S — partitioned'
$EVICT $INPUT $NVME/part.bucket_*.bin
sleep 2

echo "=== Round 1 — WITH output write (baseline) ==="
date '+%H:%M:%S — round 1 start'
T0=$(date +%s.%N)
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_0.bin --output-file $NVME/sorted_0.bin --runs 1 --no-verify > /tmp/19.18_b0_with_write.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_1.bin --output-file $NVME/sorted_1.bin --runs 1 --no-verify > /tmp/19.18_b1_with_write.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_2.bin --output-file $NVME/sorted_2.bin --runs 1 --no-verify > /tmp/19.18_b2_with_write.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_3.bin --output-file $NVME/sorted_3.bin --runs 1 --no-verify > /tmp/19.18_b3_with_write.log 2>&1 &
wait
T1=$(date +%s.%N)
WW=$(echo "$T1 - $T0" | bc)
echo "Round 1 WITH write: ${WW} sec"
$EVICT $NVME/sorted_*.bin 2>/dev/null || true
rm -f $NVME/sorted_*.bin

echo "=== Round 2 — WITHOUT output write ==="
date '+%H:%M:%S — round 2 start'
T0=$(date +%s.%N)
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_4.bin --runs 1 --no-verify > /tmp/19.18_b4_no_write.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_5.bin --runs 1 --no-verify > /tmp/19.18_b5_no_write.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_6.bin --runs 1 --no-verify > /tmp/19.18_b6_no_write.log 2>&1 &
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_7.bin --runs 1 --no-verify > /tmp/19.18_b7_no_write.log 2>&1 &
wait
T1=$(date +%s.%N)
NW=$(echo "$T1 - $T0" | bc)
echo "Round 2 WITHOUT write: ${NW} sec"

echo "=========================="
echo "WITH    write: ${WW} sec"
echo "WITHOUT write: ${NW} sec"
DIFF=$(echo "$WW - $NW" | bc)
echo "Output write contributes: ${DIFF} sec per 4-bucket round"

rm -rf $NVME
date '+%H:%M:%S — done'
