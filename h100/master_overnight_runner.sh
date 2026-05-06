#!/bin/bash
# 47-hour master experiment runner.
# Each result gets committed immediately so nothing is lost on timeout.
#
# Order (most-valuable-first):
#  1. n=20 SF1500 GDS+integrated variance lock (definitive)
#  2. cross-scale n=5 GDS at SF100/300/500/1000 (variance + scaling chart)
#  3. adversarial distributions at SF50
#  4. multi-NVMe scaling test
#  5. SF1500 energy under GDS
#  6. SF2000 dbgen + run (if disk allows)
#
# Each iteration: run experiment, copy log to repo, git commit, push.
set -uo pipefail
source ~/gpu-research/h100/env.sh

REPO=/home/cc/gpu-research
RESULTS=$REPO/results/h100_runs/master_overnight
mkdir -p $RESULTS

PART=$REPO/gpu_crocsort/experiments/gds_partition_multistream
EVICT=$REPO/gpu_crocsort/experiments/evict_cache
VERIFY=$REPO/gpu_crocsort/experiments/verify_compact_offsets

LOG=$RESULTS/master.log
echo "=========================================" | tee -a $LOG
echo "Master overnight start: $(date)" | tee -a $LOG
echo "=========================================" | tee -a $LOG

commit_result() {
    local TAG=$1
    local DESC=$2
    cd $REPO
    git add results/h100_runs/master_overnight/ 2>&1 > /dev/null
    git commit -m "h100/master-overnight: $TAG — $DESC" 2>&1 | tail -2 | tee -a $LOG
}

run_one() {
    local SF=$1
    local TAG=$2
    local INPUT=/mnt/data/lineitem_sf${SF}.bin
    [ ! -f "$INPUT" ] && return 1
    local NVME=/mnt/data/master_${TAG}
    rm -rf $NVME && mkdir -p $NVME

    sudo sysctl vm.drop_caches=3 2>&1 > /dev/null
    $EVICT $INPUT 2>&1 > /dev/null
    sleep 3

    local T0=$(date +%s.%N)
    RUN_SORT=1 $PART $INPUT $NVME/sf 8 > $RESULTS/${TAG}.log 2>&1
    local T1=$(date +%s.%N)
    local WALL=$(echo "$T1 - $T0" | bc)

    local CSV=$(grep "^CSV" $RESULTS/${TAG}.log | tail -1)
    local STATUS="OK"
    if [ -f "$NVME/sf.sorted_0.bin" ]; then
        local VERIFY_RESULT=$(CHECK_LIMIT=100000 $VERIFY $INPUT $NVME/sf.sorted_0.bin 2>&1 | tail -1)
        STATUS="$VERIFY_RESULT"
    fi
    rm -rf $NVME
    echo "[$(date '+%H:%M:%S')] $TAG: ${WALL}s — $STATUS" | tee -a $LOG
}

# ── 1. n=20 SF1500 GDS+integrated variance lock ──
echo "=== Phase 1: n=20 SF1500 ===" | tee -a $LOG
for i in $(seq 1 20); do
    run_one 1500 "p1_sf1500_run${i}"
    if [ $((i % 5)) -eq 0 ]; then
        commit_result "p1_sf1500_runs1-${i}" "n=20 SF1500 batch"
    fi
done
commit_result "p1_sf1500_n20_done" "n=20 SF1500 LOCK"

# ── 2. cross-scale n=5 ──
echo "=== Phase 2: cross-scale n=5 ===" | tee -a $LOG
for SF in 100 300 500 1000; do
    for i in 1 2 3 4 5; do
        run_one $SF "p2_sf${SF}_run${i}"
    done
    commit_result "p2_sf${SF}_n5_done" "cross-scale SF${SF} n=5"
done

# ── 3. adversarial ──
echo "=== Phase 3: adversarial ===" | tee -a $LOG
bash /tmp/run_19.61_adversarial.sh > $RESULTS/p3_adversarial.log 2>&1
commit_result "p3_adversarial" "adversarial robustness"

# ── 4. extra n=20 GDS+integrated for variance whiskers ──
echo "=== Phase 4: extra SF1500 reps ===" | tee -a $LOG
for i in $(seq 21 40); do
    run_one 1500 "p4_sf1500_run${i}"
    if [ $((i % 5)) -eq 0 ]; then
        commit_result "p4_sf1500_runs21-${i}" "extra SF1500"
    fi
done
commit_result "p4_sf1500_n40_done" "n=40 SF1500 mega-lock"

echo "=========================================" | tee -a $LOG
echo "Master overnight done: $(date)" | tee -a $LOG
