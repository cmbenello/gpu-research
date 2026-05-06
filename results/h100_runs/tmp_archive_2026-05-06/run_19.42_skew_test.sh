#!/bin/bash
# 19.42 — Adversarial skew test on SF50.
# Inject 30% heavy-hitter, then run v2 (baseline) vs v3 (adaptive).
set -uo pipefail
source ~/gpu-research/h100/env.sh

BASE=/home/cc/gpu-research/gpu_crocsort/experiments
INPUT_ORIG=/mnt/data/lineitem_sf50.bin
INPUT_SKEW=/mnt/data/lineitem_sf50_skew30.bin

date '+%H:%M:%S — start'

echo "=== Inject 30% heavy-hitter skew ==="
$BASE/inject_skew $INPUT_ORIG $INPUT_SKEW 30
ls -la $INPUT_SKEW

mkdir -p /tmp/19.42_v2_skew /tmp/19.42_v3_skew

echo "=== Baseline (v2) on skewed SF50 ==="
T0=$(date +%s.%N)
$BASE/partition_by_range_compact_v2 $INPUT_SKEW /tmp/19.42_v2_skew/part 8 2>&1 | grep -E "bucket\[|Pass|imbalance|Total"
T1=$(date +%s.%N)
echo "v2 wall: $(echo "$T1 - $T0" | bc) s"

echo "=== Adaptive (v3) on skewed SF50 ==="
T0=$(date +%s.%N)
$BASE/partition_by_range_compact_v3 $INPUT_SKEW /tmp/19.42_v3_skew/part 8 2>&1 | grep -E "bucket\[|WARNING|Adaptive|imbalance|Total"
T1=$(date +%s.%N)
echo "v3 wall: $(echo "$T1 - $T0" | bc) s"

echo "=== Cleanup ==="
rm -rf /tmp/19.42_v2_skew /tmp/19.42_v3_skew
rm -f $INPUT_SKEW
date '+%H:%M:%S — done'
