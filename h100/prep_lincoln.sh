#!/usr/bin/env bash
# Lincoln prep: NYC taxi (real-world data) end-to-end with bitpack
# + variance run on the bitpack patch
set -o pipefail

DATE=$(date +%Y-%m-%d_%H%M)
ROOT=/mnt/nvme1/cmbenello
DATA=$ROOT/data
REPO=$ROOT/gpu-research/gpu_crocsort
OUT=$DATA/prep_lincoln_$DATE
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

log "Lincoln prep starting"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# NYC Taxi: real-world dirty data
log "============================================================"
log "Section 1: NYC Yellow Taxi 6mo + bitpack"
log "============================================================"
[ -f $DATA/taxi_6mo.bin ] || run gen_taxi_6mo  python3 scripts/gen_nyctaxi_normalized.py 6 $DATA/taxi_6mo.bin
run taxi_6mo_baseline   ./external_sort_tpch_compact --input $DATA/taxi_6mo.bin --runs 3
run taxi_6mo_bitpack    env USE_BITPACK=1 ./external_sort_tpch_compact --input $DATA/taxi_6mo.bin --runs 3
run taxi_6mo_codec      python3 scripts/a3_codec_ratios.py $DATA/taxi_6mo.bin

# Variance run: 5 warm runs of SF20 with bitpack to pin down stability
log "============================================================"
log "Section 2: SF20 variance (5 warm runs each)"
log "============================================================"
run sf20_baseline_5x    ./external_sort_tpch_compact --input $DATA/lineitem_sf20.bin --runs 5
run sf20_bitpack_5x     env USE_BITPACK=1 ./external_sort_tpch_compact --input $DATA/lineitem_sf20.bin --runs 5

# Random 100M for crossover (small)
log "============================================================"
log "Section 3: Random 100M baseline"
log "============================================================"
run gen_random_100M  python3 scripts/gen_random_normalized.py 100000000 $DATA/random_100M.bin
run random_100M_baseline  ./external_sort_tpch_compact --input $DATA/random_100M.bin --runs 3
run random_100M_bitpack   env USE_BITPACK=1 ./external_sort_tpch_compact --input $DATA/random_100M.bin --runs 3
rm -f $DATA/random_100M.bin

log "All sections complete"
date > "$OUT/DONE"
