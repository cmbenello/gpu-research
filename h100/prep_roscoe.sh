#!/usr/bin/env bash
# Roscoe prep: GPU vs CPU crossover sweep + larger TPC-H bitpack stability
set -o pipefail

DATE=$(date +%Y-%m-%d_%H%M)
ROOT=/mnt/nvme1/cmbenello
DATA=$ROOT/data
REPO=$ROOT/gpu-research/gpu_crocsort
OUT=$DATA/prep_roscoe_$DATE
mkdir -p "$OUT"

cd "$REPO"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT/master.log"; }
run() {
  local name=$1; shift
  log "=== $name ==="
  ( "$@" ) > "$OUT/$name.log" 2>&1 || log "  $name FAILED ($?)"
  log "  $name done"
}

log "Roscoe prep starting"

# SF50 variance: 5 runs each
log "============================================================"
log "Section 1: SF50 variance (5 warm runs)"
log "============================================================"
run sf50_baseline_5x  ./external_sort_tpch_compact --input $DATA/lineitem_sf50.bin --runs 5
run sf50_bitpack_5x   env USE_BITPACK=1 ./external_sort_tpch_compact --input $DATA/lineitem_sf50.bin --runs 5

# Crossover sweep: how does GPU sort scale on small N?
log "============================================================"
log "Section 2: GPU vs CPU crossover sweep"
log "============================================================"
for N in 100000 1000000 10000000 30000000 100000000; do
  run gpu_smoke_${N}     ./gpu_crocsort --num-records $N --verify
done

# NYC Taxi 6mo on roscoe (16 GB GPU should fit easily)
log "============================================================"
log "Section 3: NYC Taxi 6mo on P5000"
log "============================================================"
[ -f $DATA/taxi_6mo.bin ] || run gen_taxi_6mo  python3 scripts/gen_nyctaxi_normalized.py 6 $DATA/taxi_6mo.bin
run taxi_6mo_baseline   ./external_sort_tpch_compact --input $DATA/taxi_6mo.bin --runs 3
run taxi_6mo_bitpack    env USE_BITPACK=1 ./external_sort_tpch_compact --input $DATA/taxi_6mo.bin --runs 3

log "All sections complete"
date > "$OUT/DONE"
