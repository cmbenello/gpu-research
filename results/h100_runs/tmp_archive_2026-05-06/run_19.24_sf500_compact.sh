#!/bin/bash
# 19.24 — SF500 K=16 with compact pipeline. Same recipe as 19.23 but at SF500.
# Predicted ~3-4 min total (vs SF1500 7m56s).
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/experiments/partition_by_range_compact_v2
BIN=/home/cc/gpu-research/gpu_crocsort/experiments/sort_compact_bucket
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf500.bin
NVME=/mnt/data/19.24_sf500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: SINGLE-PASS compact partition K=16 ==="
PSTART=$(date +%s.%N)
$PART $INPUT $NVME/part 16
PEND=$(date +%s.%N)
PWALL=$(echo "$PEND - $PSTART" | bc)
date '+%H:%M:%S — partitioned'
echo "Partition wall: ${PWALL} s"

echo "=== Evicting INPUT only (keep bucket cache) ==="
$EVICT $INPUT
sleep 2

echo "=== Step 2: 4-GPU concurrent × 4 rounds ==="
SORTSTART=$(date +%s.%N)

for ROUND in 1 2 3 4; do
    B0=$(( (ROUND-1) * 4 ))
    B1=$(( B0 + 1 ))
    B2=$(( B0 + 2 ))
    B3=$(( B0 + 3 ))
    date "+%H:%M:%S — round $ROUND"
    CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        $BIN $NVME/part.bucket_${B0}.bin $NVME/sorted_${B0}.bin > /tmp/19.24_b${B0}.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
        $BIN $NVME/part.bucket_${B1}.bin $NVME/sorted_${B1}.bin > /tmp/19.24_b${B1}.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
        $BIN $NVME/part.bucket_${B2}.bin $NVME/sorted_${B2}.bin > /tmp/19.24_b${B2}.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
        $BIN $NVME/part.bucket_${B3}.bin $NVME/sorted_${B3}.bin > /tmp/19.24_b${B3}.log 2>&1 &
    wait
    rm -f $NVME/part.bucket_${B0}.bin $NVME/part.bucket_${B1}.bin $NVME/part.bucket_${B2}.bin $NVME/part.bucket_${B3}.bin
done

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo "=== Sort-only wall: ${SORTWALL} s ==="

PASS=0; FAIL=0
for B in $(seq 0 15); do
    csv=$(grep "^CSV" /tmp/19.24_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed|invalid argument" /tmp/19.24_b$B.log | tail -1)
        echo "b$B: FAILED — $err"
        FAIL=$((FAIL + 1))
    else
        echo "b$B: $csv"
        PASS=$((PASS + 1))
    fi
done
echo "=== $PASS PASS / $FAIL FAIL ==="

if [ -f $NVME/sorted_0.bin ]; then
    CHECK_LIMIT=1000000 /home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets $INPUT $NVME/sorted_0.bin 2>&1 | tail -2
fi

rm -rf $NVME
date '+%H:%M:%S — done'
