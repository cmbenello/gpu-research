#!/bin/bash
# Overnight queue for lincoln (RTX 2080 8 GB).
# Stays under 8 GB GPU memory by capping at SF20.
set -o pipefail

DATE=$(date +%Y-%m-%d_%H%M)
ROOT=/mnt/nvme1/cmbenello
DATA=$ROOT/data
REPO=$ROOT/gpu-research/gpu_crocsort
OUT=$DATA/overnight_lincoln_$DATE
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

log "Lincoln overnight queue starting"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "Output: $OUT"

# ── Section 1: TPC-H scaling, compact ON vs OFF ────────────────
log "============================================================"
log "Section 1: TPC-H SF10 + SF20, compact ON vs OFF"
log "============================================================"
run sf10_compact     ./external_sort_tpch_compact --input $DATA/lineitem_sf10.bin --runs 3
run sf10_noncompact  env DISABLE_COMPACT=1 ./external_sort_tpch_compact --input $DATA/lineitem_sf10.bin --runs 3
run sf20_compact     ./external_sort_tpch_compact --input $DATA/lineitem_sf20.bin --runs 3
run sf20_noncompact  env DISABLE_COMPACT=1 ./external_sort_tpch_compact --input $DATA/lineitem_sf20.bin --runs 3

# ── Section 2: codec ratio analysis (CPU-only, fast) ────────────
log "============================================================"
log "Section 2: codec compression ratios (Python only)"
log "============================================================"
cd "$REPO"
# A1 byte entropy on SF20
run a1_byte_entropy_sf20  python3 scripts/a1_byte_entropy.py $DATA/lineitem_sf20.bin
# A2 compact key baseline
run a2_compact_baseline   python3 scripts/a2_compact_key_baseline.py $DATA/lineitem_sf20.bin
# A3 codec ratios
run a3_codec_ratios_sf20  python3 scripts/a3_codec_ratios.py $DATA/lineitem_sf20.bin

# ── Section 3: random 60M (fits in 8 GB) ─────────────────────────
log "============================================================"
log "Section 3: uniform random 60M records (~7 GB)"
log "============================================================"
run gen_random_60M python3 scripts/gen_random_normalized.py 60000000 $DATA/random_60M.bin
run random_60M_compact     ./external_sort_tpch_compact --input $DATA/random_60M.bin --runs 3
run random_60M_noncompact  env DISABLE_COMPACT=1 ./external_sort_tpch_compact --input $DATA/random_60M.bin --runs 3
rm -f $DATA/random_60M.bin

# ── Section 4: NYC Taxi 1mo (small, fits) ────────────────────────
log "============================================================"
log "Section 4: NYC Yellow Taxi 1 month"
log "============================================================"
run gen_taxi_1mo  python3 scripts/gen_nyctaxi_normalized.py 1 $DATA/taxi_1mo.bin
run taxi_1mo_compact     ./external_sort_tpch_compact --input $DATA/taxi_1mo.bin --runs 3
run taxi_1mo_noncompact  env DISABLE_COMPACT=1 ./external_sort_tpch_compact --input $DATA/taxi_1mo.bin --runs 3
rm -f $DATA/taxi_1mo.bin

log "All experiments complete"
date > "$OUT/DONE"
