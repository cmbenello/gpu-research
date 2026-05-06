#!/bin/bash
# 19.17 — SF1500 K=16 + 4-GPU + WILLNEED prefetch overlap.
# While round N is running on GPUs, posix_fadvise(WILLNEED) on round N+1's
# bucket files. Should populate OS cache so round N+1's fread is fast.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.17_sf1500
mkdir -p $NVME

# Helper: posix_fadvise WILLNEED on a list of files (background)
willneed_files() {
    for f in "$@"; do
        [ -f "$f" ] && cat "$f" > /dev/null 2>&1 &
    done
}

date '+%H:%M:%S — start'

echo "=== Step 1: partition K=16 ==="
PSTART=$(date +%s.%N)
$PART $INPUT $NVME/part 16
PEND=$(date +%s.%N)
PWALL=$(echo "$PEND - $PSTART" | bc)
date '+%H:%M:%S — partitioned'
echo "Partition wall: ${PWALL} s"

echo "=== Cache eviction after partition ==="
$EVICT $INPUT $NVME/part.bucket_*.bin
sleep 2
free -g | head -2

echo "=== Step 2: 4-GPU concurrent × 4 rounds, with WILLNEED prefetch ==="
SORTSTART=$(date +%s.%N)

# Round 1: pre-warm cache for round 1 buckets first (no sort running)
echo "  pre-warming round 1 buckets..."
willneed_files $NVME/part.bucket_0.bin $NVME/part.bucket_1.bin $NVME/part.bucket_2.bin $NVME/part.bucket_3.bin
wait  # wait for cat to finish

for ROUND in 1 2 3 4; do
    B0=$(( (ROUND-1) * 4 ))
    B1=$(( B0 + 1 ))
    B2=$(( B0 + 2 ))
    B3=$(( B0 + 3 ))
    date "+%H:%M:%S — round $ROUND (b$B0,b$B1,b$B2,b$B3)"

    # Launch sort processes (they'll fread bucket from OS cache that was pre-warmed)
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
        $BIN --input $NVME/part.bucket_${B0}.bin --output-file $NVME/sorted_${B0}.bin \
        --runs 1 --no-verify > /tmp/19.17_b${B0}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
        $BIN --input $NVME/part.bucket_${B1}.bin --output-file $NVME/sorted_${B1}.bin \
        --runs 1 --no-verify > /tmp/19.17_b${B1}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
        $BIN --input $NVME/part.bucket_${B2}.bin --output-file $NVME/sorted_${B2}.bin \
        --runs 1 --no-verify > /tmp/19.17_b${B2}.log 2>&1 &
    NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
        $BIN --input $NVME/part.bucket_${B3}.bin --output-file $NVME/sorted_${B3}.bin \
        --runs 1 --no-verify > /tmp/19.17_b${B3}.log 2>&1 &

    # Background prefetch next round's buckets WHILE current round sorts
    if [ $ROUND -lt 4 ]; then
        NB0=$(( ROUND * 4 ))
        NB1=$(( NB0 + 1 ))
        NB2=$(( NB0 + 2 ))
        NB3=$(( NB0 + 3 ))
        sleep 30  # let pin/read of current round get past contention point
        echo "  background prefetch round $((ROUND+1)) buckets..."
        willneed_files $NVME/part.bucket_${NB0}.bin $NVME/part.bucket_${NB1}.bin $NVME/part.bucket_${NB2}.bin $NVME/part.bucket_${NB3}.bin &
        PREFETCH_PID=$!
    fi

    wait
    [ -n "${PREFETCH_PID:-}" ] && kill $PREFETCH_PID 2>/dev/null || true
    rm -f $NVME/part.bucket_${B0}.bin $NVME/part.bucket_${B1}.bin $NVME/part.bucket_${B2}.bin $NVME/part.bucket_${B3}.bin
    # Evict sorted output to free RAM (next round's pin needs it)
    $EVICT $NVME/sorted_*.bin 2>/dev/null || true
done

SORTEND=$(date +%s.%N)
SORTWALL=$(echo "$SORTEND - $SORTSTART" | bc)
echo "=== Sort-only wall: ${SORTWALL} s ==="

PASS=0; FAIL=0
for B in $(seq 0 15); do
    csv=$(grep "^CSV" /tmp/19.17_b$B.log | tail -1)
    if [ -z "$csv" ]; then
        err=$(grep -E "CUDA error|FAIL|Killed|invalid argument" /tmp/19.17_b$B.log | tail -1)
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
