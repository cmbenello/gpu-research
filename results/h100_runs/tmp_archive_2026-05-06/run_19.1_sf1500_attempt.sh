#!/bin/bash
# 19.1 — SF1500 (1.08 TB > 1 TB host RAM) attempt with NUMA --preferred per pair.
# Tests whether gpu_crocsort architecture handles above-RAM workloads.
set -uo pipefail
source ~/gpu-research/h100/env.sh

PART=/home/cc/gpu-research/gpu_crocsort/partition_by_range
BIN=/home/cc/gpu-research/gpu_crocsort/external_sort_tpch_compact
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.1_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'
df -h /mnt/data | tail -1
free -g | head -2

echo "=== Step 1: partition (output → NVMe) — expected ~30 min ==="
$PART $INPUT $NVME/part 4
date '+%H:%M:%S — partitioned'
df -h /mnt/data | tail -1

echo "=== Step 2: paired-concurrent sorts ==="
date "+%H:%M:%S — pair A start"
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_0.bin --output-file $NVME/sorted_0.bin \
    --runs 1 --no-verify > /tmp/19.1_g0.log 2>&1 &
P0=$!
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=0 --preferred=0 \
    $BIN --input $NVME/part.bucket_1.bin --output-file $NVME/sorted_1.bin \
    --runs 1 --no-verify > /tmp/19.1_g1.log 2>&1 &
P1=$!
wait $P0 $P1
rm -f $NVME/part.bucket_0.bin $NVME/part.bucket_1.bin
date "+%H:%M:%S — pair A done"
df -h /mnt/data | tail -1

date "+%H:%M:%S — pair B start"
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_2.bin --output-file $NVME/sorted_2.bin \
    --runs 1 --no-verify > /tmp/19.1_g2.log 2>&1 &
P2=$!
NO_MAP_POPULATE=1 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 --preferred=1 \
    $BIN --input $NVME/part.bucket_3.bin --output-file $NVME/sorted_3.bin \
    --runs 1 --no-verify > /tmp/19.1_g3.log 2>&1 &
P3=$!
wait $P2 $P3
rm -f $NVME/part.bucket_2.bin $NVME/part.bucket_3.bin
date "+%H:%M:%S — pair B done"

echo "=== Per-bucket times ==="
for g in 0 1 2 3; do
    last=$(grep "^CSV" /tmp/19.1_g$g.log | tail -1)
    echo "GPU $g: $last"
done

echo "=== Cleanup ==="
rm -rf $NVME
date '+%H:%M:%S — done'
