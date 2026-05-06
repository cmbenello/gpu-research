#!/bin/bash
# 19.34 — Measure GPU + CPU power during a SF1500 stream pre-pin run.
# Compute joules per byte sorted.
set -uo pipefail
source ~/gpu-research/h100/env.sh

BIN=/home/cc/gpu-research/gpu_crocsort/experiments/stream_partition_sort
EVICT=/home/cc/gpu-research/gpu_crocsort/experiments/evict_cache
INPUT=/mnt/data/lineitem_sf1500.bin
NVME=/mnt/data/19.34_sf1500
mkdir -p $NVME

date '+%H:%M:%S — start'
$EVICT $INPUT
sleep 2

# Sample idle power baseline
echo "=== Idle GPU power (5 samples) ==="
nvidia-smi --query-gpu=index,power.draw --format=csv,noheader -lms 200 -c 5 2>&1 | head -25
sleep 3

# Start power monitoring in background, log every 200ms
nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv,noheader -lms 200 > $NVME/power.csv 2>&1 &
NVMON_PID=$!
sleep 1

T0=$(date +%s.%N)
$BIN $INPUT $NVME/sf1500 8 2>&1 | tail -5
T1=$(date +%s.%N)

kill $NVMON_PID 2>&1 || true
wait $NVMON_PID 2>&1 || true

WALL=$(echo "$T1 - $T0" | bc)
echo "Wall: ${WALL} s"

# Compute average power per GPU (rough integration)
python3 << EOF
import csv
power_per_gpu = {0: [], 1: [], 2: [], 3: []}
with open('$NVME/power.csv') as f:
    for line in f:
        parts = line.strip().split(', ')
        if len(parts) < 3: continue
        try:
            ts = parts[0]
            gpu = int(parts[1])
            pw = float(parts[2].replace(' W', '').replace('W', ''))
            power_per_gpu[gpu].append(pw)
        except: pass

print('GPU avg power (W) during run:')
total_avg = 0
for g in sorted(power_per_gpu):
    if power_per_gpu[g]:
        avg = sum(power_per_gpu[g]) / len(power_per_gpu[g])
        peak = max(power_per_gpu[g])
        print(f'  GPU {g}: avg={avg:.0f}W, peak={peak:.0f}W, samples={len(power_per_gpu[g])}')
        total_avg += avg

print(f'\\nTotal 4-GPU avg power: {total_avg:.0f}W')
WALL = $WALL
TOTAL_BYTES = 1080e9
energy_J = total_avg * WALL  # joules from 4 GPUs
print(f'GPU energy (run): {energy_J:.0f} J ({energy_J/3600:.3f} Wh)')
print(f'Per byte sorted: {energy_J / TOTAL_BYTES * 1e9:.3f} nJ/byte')
print(f'Per record sorted: {energy_J / 9e9 * 1e6:.3f} uJ/record')
EOF

rm -rf $NVME
date '+%H:%M:%S — done'
