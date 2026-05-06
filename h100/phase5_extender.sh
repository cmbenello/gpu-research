#!/bin/bash
# Phase 5+: extends master runner with much more aggressive experimentation.
# Runs after current master completes (around 22:00 UTC).
# Total budget: ~36h until lease expires (2026-05-08 16:49 UTC).
set -uo pipefail
source ~/gpu-research/h100/env.sh

REPO=/home/cc/gpu-research
RESULTS=$REPO/results/h100_runs/master_overnight
PART=$REPO/gpu_crocsort/experiments/gds_partition_multistream
EVICT=$REPO/gpu_crocsort/experiments/evict_cache
VERIFY=$REPO/gpu_crocsort/experiments/verify_compact_offsets

LOG=$RESULTS/phase5.log
echo "=========================================" | tee -a $LOG
echo "Phase 5+ start: $(date)" | tee -a $LOG
echo "=========================================" | tee -a $LOG

commit_result() {
    local TAG=$1
    local DESC=$2
    cd $REPO
    git add results/h100_runs/master_overnight/ 2>&1 > /dev/null
    git commit -m "h100/phase5: $TAG — $DESC" 2>&1 | tail -2 | tee -a $LOG
}

run_one() {
    local SF=$1
    local TAG=$2
    local INPUT=/mnt/data/lineitem_sf${SF}.bin
    [ ! -f "$INPUT" ] && return 1
    local NVME=/mnt/data/p5_${TAG}
    rm -rf $NVME && mkdir -p $NVME

    sudo sysctl vm.drop_caches=3 2>&1 > /dev/null
    $EVICT $INPUT 2>&1 > /dev/null
    sleep 3

    local T0=$(date +%s.%N)
    RUN_SORT=1 $PART $INPUT $NVME/sf 8 > $RESULTS/${TAG}.log 2>&1
    local T1=$(date +%s.%N)
    local WALL=$(echo "$T1 - $T0" | bc)

    local STATUS="OK"
    if [ -f "$NVME/sf.sorted_0.bin" ]; then
        STATUS=$(CHECK_LIMIT=100000 $VERIFY $INPUT $NVME/sf.sorted_0.bin 2>&1 | tail -1)
    fi
    rm -rf $NVME
    echo "[$(date '+%H:%M:%S')] $TAG: ${WALL}s — $STATUS" | tee -a $LOG
}

# Wait for master to finish (poll for "Master overnight done" line)
while ! grep -q "Master overnight done" $RESULTS/master.log 2>/dev/null; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] Master finished, starting phase 5+" | tee -a $LOG

# ── Phase 5: 30 more SF1500 runs (target n=50 total, ultra-tight variance) ──
echo "=== Phase 5: SF1500 n=30 more ===" | tee -a $LOG
for i in $(seq 1 30); do
    run_one 1500 "p5_sf1500_run${i}"
    if [ $((i % 5)) -eq 0 ]; then
        commit_result "p5_sf1500_runs1-${i}" "phase 5 SF1500 batch"
    fi
done

# ── Phase 6: cross-scale n=10 each (was n=5 in phase 2) ──
echo "=== Phase 6: cross-scale n=10 each ===" | tee -a $LOG
for SF in 100 300 500 1000; do
    for i in $(seq 1 10); do
        run_one $SF "p6_sf${SF}_run${i}"
    done
    commit_result "p6_sf${SF}_n10" "cross-scale n=10 SF${SF}"
done

# ── Phase 7: hugepages stream pre-pin n=20 cold for COMPARISON ──
# (rebuild with NO_HUGETLB unset, NO RUN_SORT, use stream_partition_sort)
echo "=== Phase 7: hugepages comparison n=20 ===" | tee -a $LOG
HUGE=$REPO/gpu_crocsort/experiments/stream_partition_sort
INPUT=/mnt/data/lineitem_sf1500.bin
for i in $(seq 1 20); do
    NVME=/mnt/data/p7_run${i}
    rm -rf $NVME && mkdir -p $NVME
    sudo sysctl vm.drop_caches=3 2>&1 > /dev/null
    $EVICT $INPUT 2>&1 > /dev/null
    sleep 3
    T0=$(date +%s.%N)
    $HUGE $INPUT $NVME/sf 8 > $RESULTS/p7_sf1500_huge_run${i}.log 2>&1
    T1=$(date +%s.%N)
    WALL=$(echo "$T1 - $T0" | bc)
    rm -rf $NVME
    echo "[$(date '+%H:%M:%S')] p7_sf1500_huge_run${i}: ${WALL}s" | tee -a $LOG
    if [ $((i % 5)) -eq 0 ]; then
        commit_result "p7_huge_runs1-${i}" "hugepages comparison batch"
    fi
done

# ── Phase 8: SF2000 generation + run (if disk allows after Phase 6 cleanup) ──
echo "=== Phase 8: SF2000 attempt ===" | tee -a $LOG
df -h /mnt/data | tail -1 | tee -a $LOG
SF2000=/mnt/data/lineitem_sf2000.bin
if [ ! -f "$SF2000" ]; then
    # Need 1.44 TB. Disk should have ~1.7 TB free at this point.
    DATA_DIR=/mnt/data WORK_DIR=$REPO bash $REPO/h100/gen_all_scales.sh 2000 2>&1 | tail -10 | tee -a $LOG
fi
if [ -f "$SF2000" ]; then
    for i in 1 2 3; do
        run_one 2000 "p8_sf2000_run${i}"
    done
    commit_result "p8_sf2000_n3" "SF2000 attempt"
fi

echo "=========================================" | tee -a $LOG
echo "Phase 5+ done: $(date)" | tee -a $LOG
