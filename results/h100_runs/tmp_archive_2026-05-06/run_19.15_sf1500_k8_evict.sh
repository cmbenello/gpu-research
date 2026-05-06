#!/bin/bash
# 19.15 — SF1500 K=8 + 2-GPU one-per-node, with explicit cache eviction.
# Uses posix_fadvise(DONTNEED) between partition and sort, and between rounds.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.15_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: partition K=8 ==="
$PART $INPUT $NVME/part 8
date '+%H:%M:%S — partitioned'
df -h /mnt/data | tail -1

echo "=== Cache eviction after partition (free up RAM for sort) ==="
$EVICT $INPUT $NVME/part.bucket_*.bin
sleep 2
free -g | head -2

echo "=== Step 2: 2-GPU one-per-node × 4 rounds (with eviction between rounds) ==="
SORTSTART=$(date +%s.%N)

for ROUND in 1 2 3 4; do
    B_A=$(( (ROUND-1) * 2 ))
    B_B=$(( B_A + 1 ))
    date "+%H:%M:%S — round $ROUND (GPU 0 → b$B_A, GPU 2 → b$B_B)"
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        $BIN --input $NVME/part.bucket_${B_A}.bin --output-file $NVME/sorted_${B_A}.bin \
        --runs 1 --no-verify > /tmp/19.15_b${B_A}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
        $BIN --input $NVME/part.bucket_${B_B}.bin --output-file $NVME/sorted_${B_B}.bin \
        --runs 1 --no-verify > /tmp/19.15_b${B_B}.log 2>&1 &
    wait
    rm -f $NVME/part.bucket_${B_A}.bin $NVME/part.bucket_${B_B}.bin
    # Evict cache for any remaining bucket files (keep them physically but drop cache)
    $EVICT $NVME/part.bucket_*.bin $NVME/sorted_*.bin 2>/dev/null || true
done

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo "=== Sort-only wall: ${SORTWALL} s ==="

PASS=0
FAIL=0
for B in 0 1 2 3 4 5 6 7; do
    csv=$(grep "^CSV" /tmp/19.15_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed" /tmp/19.15_b$B.log | tail -1)
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
