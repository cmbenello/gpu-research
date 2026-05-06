#!/bin/bash
# 19.21 — SF1500 K=16 with COMPACT partition (40-byte records, 32 key + 8 offset)
# + sort_compact_bucket. Projected: ~12-14m total (vs 22m31s PERM_ONLY).
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/experiments/partition_by_range_compact
BIN=/home/cc/gpu-research/gpu_crocsort/experiments/sort_compact_bucket
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.21_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'

echo "=== Step 1: COMPACT partition K=16 (40-byte records) ==="
PSTART=$(date +%s.%N)
$PART $INPUT $NVME/part 16
PEND=$(date +%s.%N)
PWALL=$(echo "$PEND - $PSTART" | bc)
date '+%H:%M:%S — partitioned'
echo "Partition wall: ${PWALL} s"
df -h /mnt/data | tail -1

echo "=== Cache eviction ==="
$EVICT $INPUT $NVME/part.bucket_*.bin
sleep 2
free -g | head -2

echo "=== Step 2: 4-GPU concurrent × 4 rounds, sort_compact_bucket ==="
SORTSTART=$(date +%s.%N)

for ROUND in 1 2 3 4; do
    B0=$(( (ROUND-1) * 4 ))
    B1=$(( B0 + 1 ))
    B2=$(( B0 + 2 ))
    B3=$(( B0 + 3 ))
    date "+%H:%M:%S — round $ROUND (b$B0,b$B1,b$B2,b$B3)"
    CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        $BIN $NVME/part.bucket_${B0}.bin $NVME/sorted_offsets_${B0}.bin > /tmp/19.21_b${B0}.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
        $BIN $NVME/part.bucket_${B1}.bin $NVME/sorted_offsets_${B1}.bin > /tmp/19.21_b${B1}.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
        $BIN $NVME/part.bucket_${B2}.bin $NVME/sorted_offsets_${B2}.bin > /tmp/19.21_b${B2}.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
        $BIN $NVME/part.bucket_${B3}.bin $NVME/sorted_offsets_${B3}.bin > /tmp/19.21_b${B3}.log 2>&1 &
    wait
    rm -f $NVME/part.bucket_${B0}.bin $NVME/part.bucket_${B1}.bin $NVME/part.bucket_${B2}.bin $NVME/part.bucket_${B3}.bin
    $EVICT $NVME/part.bucket_*.bin $NVME/sorted_offsets_*.bin 2>/dev/null || true
done

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo "=== Sort-only wall: ${SORTWALL} s ==="

PASS=0; FAIL=0
for B in $(seq 0 15); do
    csv=$(grep "^CSV" /tmp/19.21_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed" /tmp/19.21_b$B.log | tail -1)
        echo "b$B: FAILED — $err"
        FAIL=$((FAIL + 1))
    else
        echo "b$B: $csv"
        PASS=$((PASS + 1))
    fi
done
echo "=== $PASS PASS / $FAIL FAIL ==="

ls -la $NVME/sorted_offsets_*.bin 2>&1 | head -16
echo "Total offsets size: $(du -sh $NVME)"

# Quick verify a few buckets (1M pair check, fast)
echo "=== Verifying buckets 0,4,8,12 (1M pairs each) ==="
for B in 0 4 8 12; do
    if [ -f $NVME/sorted_offsets_$B.bin ]; then
        echo "-- bucket $B --"
        CHECK_LIMIT=1000000 /home/cc/gpu-research/gpu_crocsort/experiments/verify_compact_offsets $INPUT $NVME/sorted_offsets_$B.bin 2>&1 | tail -2
    fi
done

rm -rf $NVME
date '+%H:%M:%S — done'
