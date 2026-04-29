#!/bin/bash
# Roscoe (P5000 16 GB) session: rebuild with lower budget, try SF100 via OVC.
# Plus a baseline TPC-H sweep with compact ON/OFF.
set -o pipefail

DATE=$(date +%Y-%m-%d_%H%M)
ROOT=/mnt/nvme1/cmbenello
DATA=$ROOT/data
REPO=$ROOT/gpu-research/gpu_crocsort
OUT=$DATA/roscoe_run_$DATE
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

log "Roscoe session starting"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "Output: $OUT"

# ── Section 1: TPC-H baseline at the working scales ─────────────
log "============================================================"
log "Section 1: TPC-H SF10 + SF50 baseline (current 0.65 budget)"
log "============================================================"
run sf10_baseline_compact     ./external_sort_tpch_compact --input $DATA/lineitem_sf10.bin --runs 3
run sf10_baseline_noncompact  env DISABLE_COMPACT=1 ./external_sort_tpch_compact --input $DATA/lineitem_sf10.bin --runs 3
run sf50_baseline_compact     ./external_sort_tpch_compact --input $DATA/lineitem_sf50.bin --runs 3
run sf50_baseline_noncompact  env DISABLE_COMPACT=1 ./external_sort_tpch_compact --input $DATA/lineitem_sf50.bin --runs 3

# ── Section 2: SF100 with progressively lower budget ────────────
log "============================================================"
log "Section 2: SF100 budget sweep to push OVC path active"
log "============================================================"
# Make sure the original 0.65 binary is preserved
cp -f external_sort_tpch_compact external_sort_tpch_compact.b65

# Try budget 0.50 → 0.40 → 0.30 → 0.25 → 0.20 in sequence, stop on first PASS
for budget in 0.50 0.40 0.30 0.25 0.20; do
  log ">>> Rebuilding with budget=$budget"
  sed -i "s/free_mem \* 0\.[0-9][0-9]\([^0-9]\)/free_mem * $budget\1/" src/external_sort.cu
  grep "gpu_budget = (size_t)" src/external_sort.cu | tee -a "$OUT/master.log"
  make external-sort-tpch-compact 2>&1 | tail -3 | tee -a "$OUT/master.log"
  log ">>> SF100 attempt with budget=$budget"
  ./external_sort_tpch_compact --input $DATA/lineitem_sf100.bin 2>&1 | tee "$OUT/sf100_b${budget}.log"
  if grep -q "PASS:" "$OUT/sf100_b${budget}.log"; then
    log "  SF100 SUCCEEDED at budget=$budget"
    break
  fi
done

# Restore original
log ">>> Restoring budget to 0.65"
sed -i "s/free_mem \* 0\.[0-9][0-9]\([^0-9]\)/free_mem * 0.65\1/" src/external_sort.cu
make external-sort-tpch-compact > /dev/null 2>&1

# ── Section 3: codec analysis on SF50 ───────────────────────────
log "============================================================"
log "Section 3: codec ratios + byte entropy on SF50"
log "============================================================"
run a1_byte_entropy_sf50  python3 scripts/a1_byte_entropy.py $DATA/lineitem_sf50.bin
run a2_compact_baseline   python3 scripts/a2_compact_key_baseline.py $DATA/lineitem_sf50.bin
run a3_codec_ratios_sf50  python3 scripts/a3_codec_ratios.py $DATA/lineitem_sf50.bin

log "All sections complete"
date > "$OUT/DONE"
