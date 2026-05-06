#!/bin/bash
# 19.13 — SF1500 K=8 + 2-GPU one-per-node × 4 rounds.
# 8 buckets × 135 GB each, 4 rounds of 2 GPUs (no concurrency on same node).
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.13_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition K=8 ==="
$PART $INPUT $NVME/part 8
date '+%H:%M:%S — partitioned'

echo "=== Step 2: 2-GPU one-per-node × 4 rounds ==="
SORTSTART=$(date +%s.%N)

# Pair node-0 buckets with node-1 buckets for concurrent execution
# Round 1: GPU 0 → b0, GPU 2 → b1
# Round 2: GPU 0 → b2, GPU 2 → b3
# Round 3: GPU 0 → b4, GPU 2 → b5
# Round 4: GPU 0 → b6, GPU 2 → b7
for ROUND in 1 2 3 4; do
    B_A=$(( (ROUND-1) * 2 ))
    B_B=$(( B_A + 1 ))
    date "+%H:%M:%S — round $ROUND (GPU 0 → b$B_A, GPU 2 → b$B_B)"
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        $BIN --input $NVME/part.bucket_${B_A}.bin --output-file $NVME/sorted_${B_A}.bin \
        --runs 1 --no-verify > /tmp/19.13_b${B_A}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
        $BIN --input $NVME/part.bucket_${B_B}.bin --output-file $NVME/sorted_${B_B}.bin \
        --runs 1 --no-verify > /tmp/19.13_b${B_B}.log 2>&1 &
    wait
    rm -f $NVME/part.bucket_${B_A}.bin $NVME/part.bucket_${B_B}.bin
done

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
date "+%H:%M:%S — sort phase done"
echo "=== Sort-only wall: ${SORTWALL} s ==="
for B in 0 1 2 3 4 5 6 7; do
    csv=$(grep "^CSV" /tmp/19.13_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed" /tmp/19.13_b$B.log | tail -1)
        echo "b$B: FAILED — $err"
    else
        echo "b$B: $csv"
    fi
done

rm -rf $NVME
date '+%H:%M:%S — done'
