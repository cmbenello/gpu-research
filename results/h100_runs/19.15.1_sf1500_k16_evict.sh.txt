#!/bin/bash
# 19.15.1 — SF1500 K=16 + 4-GPU concurrent (4 GPUs × 4 rounds), with cache eviction.
# K=16 → 67.5 GB buckets. With 4 GPUs running concurrently and one per node-pair,
# we should be able to fit two buckets per NUMA node simultaneously.
# But to stay safe (memory-wise) start with 2-GPU one-per-node × 8 rounds.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.15.1_sf1500_k16
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition K=16 ==="
PSTART=$(date +%s.%N)
$PART $INPUT $NVME/part 16
PEND=$(date +%s.%N)
PWALL=$(echo "$PEND - $PSTART" | bc)
date '+%H:%M:%S — partitioned'
echo "Partition wall: ${PWALL} s"
df -h /mnt/data | tail -1
ls -la $NVME/part.bucket_*.bin | head -16

echo "=== Cache eviction after partition ==="
$EVICT $INPUT $NVME/part.bucket_*.bin
sleep 2
free -g | head -2

echo "=== Step 2: 2-GPU one-per-node × 8 rounds ==="
SORTSTART=$(date +%s.%N)

for ROUND in 1 2 3 4 5 6 7 8; do
    B_A=$(( (ROUND-1) * 2 ))
    B_B=$(( B_A + 1 ))
    date "+%H:%M:%S — round $ROUND (GPU 0 → b$B_A, GPU 2 → b$B_B)"
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        $BIN --input $NVME/part.bucket_${B_A}.bin --output-file $NVME/sorted_${B_A}.bin \
        --runs 1 --no-verify > /tmp/19.15.1_b${B_A}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
        $BIN --input $NVME/part.bucket_${B_B}.bin --output-file $NVME/sorted_${B_B}.bin \
        --runs 1 --no-verify > /tmp/19.15.1_b${B_B}.log 2>&1 &
    wait
    rm -f $NVME/part.bucket_${B_A}.bin $NVME/part.bucket_${B_B}.bin
    $EVICT $NVME/part.bucket_*.bin $NVME/sorted_*.bin 2>/dev/null || true
done

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo "=== Sort-only wall: ${SORTWALL} s ==="

PASS=0
FAIL=0
for B in $(seq 0 15); do
    csv=$(grep "^CSV" /tmp/19.15.1_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed" /tmp/19.15.1_b$B.log | tail -1)
        echo "b$B: FAILED — $err"
        FAIL=$((FAIL + 1))
    else
        echo "b$B: $csv"
        PASS=$((PASS + 1))
    fi
done
echo "=== $PASS PASS / $FAIL FAIL ==="

rm -rf $NVME
date '+%H:%M:%S — done'
