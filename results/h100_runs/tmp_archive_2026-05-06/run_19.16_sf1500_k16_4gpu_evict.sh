#!/bin/bash
# 19.16 — SF1500 K=16 + 4-GPU concurrent × 4 rounds, with cache eviction.
# Hypothesis: at K=16, buckets are 67 GB so 4 concurrent buckets = 268 GB
# pinned simultaneously, well within 1 TB RAM. The 19.12 K=8 4-GPU OOM
# happened with 135 GB × 4 = 540 GB which combined with cache pressure
# pushed past 1 TB. K=16 + cache eviction should work.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.16_sf1500_k16_4gpu
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

echo "=== Cache eviction after partition ==="
$EVICT $INPUT $NVME/part.bucket_*.bin
sleep 2
free -g | head -2

echo "=== Step 2: 4-GPU concurrent × 4 rounds ==="
echo "  GPUs 0,1 on NUMA node 0; GPUs 2,3 on NUMA node 1"
SORTSTART=$(date +%s.%N)

for ROUND in 1 2 3 4; do
    B0=$(( (ROUND-1) * 4 ))
    B1=$(( B0 + 1 ))
    B2=$(( B0 + 2 ))
    B3=$(( B0 + 3 ))
    date "+%H:%M:%S — round $ROUND (b$B0,b$B1,b$B2,b$B3)"
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        $BIN --input $NVME/part.bucket_${B0}.bin --output-file $NVME/sorted_${B0}.bin \
        --runs 1 --no-verify > /tmp/19.16_b${B0}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
        $BIN --input $NVME/part.bucket_${B1}.bin --output-file $NVME/sorted_${B1}.bin \
        --runs 1 --no-verify > /tmp/19.16_b${B1}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
        $BIN --input $NVME/part.bucket_${B2}.bin --output-file $NVME/sorted_${B2}.bin \
        --runs 1 --no-verify > /tmp/19.16_b${B2}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
        $BIN --input $NVME/part.bucket_${B3}.bin --output-file $NVME/sorted_${B3}.bin \
        --runs 1 --no-verify > /tmp/19.16_b${B3}.log 2>&1 &
    wait
    rm -f $NVME/part.bucket_${B0}.bin $NVME/part.bucket_${B1}.bin $NVME/part.bucket_${B2}.bin $NVME/part.bucket_${B3}.bin
    $EVICT $NVME/part.bucket_*.bin $NVME/sorted_*.bin 2>/dev/null || true
done

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo "=== Sort-only wall: ${SORTWALL} s ==="

PASS=0
FAIL=0
for B in $(seq 0 15); do
    csv=$(grep "^CSV" /tmp/19.16_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed|invalid argument" /tmp/19.16_b$B.log | tail -1)
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
