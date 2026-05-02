#!/usr/bin/env bash
# Tier 19.2 — One-shot script that regenerates every figure in the paper.
# After H100 has run the autoresearch loop, this script collects all CSVs
# and re-runs gen_weekly_figures.py with full data.
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/gpu-research}"
RESULTS_DIR="${RESULTS_DIR:-$WORK_DIR/results/h100_runs}"
FIGS_DIR="$WORK_DIR/results/figures_paper"
mkdir -p "$FIGS_DIR"

cd "$WORK_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  Repro paper figures"
echo "════════════════════════════════════════════════════════════════"

# 1. Hardware fingerprint (once per machine)
bash h100/hardware_inventory.sh > "$FIGS_DIR/hardware.json" 2>&1

# 2. Roofline predictions (deterministic)
python3 h100/roofline.py --out "$FIGS_DIR/roofline.csv"

# 3. Codec matrix (deterministic if data files exist)
if [ -f "$RESULTS_DIR/codec_matrix.csv" ]; then
    cp "$RESULTS_DIR/codec_matrix.csv" "$FIGS_DIR/"
else
    echo "  [warn] no codec_matrix.csv yet — run h100/codec_matrix.py first"
fi

# 4. Find all sort CSV outputs, collate
echo "── Collating sort results ──"
find "$RESULTS_DIR" -name "*.csv" -newer "$WORK_DIR/h100/repro_paper.sh" 2>/dev/null \
    | head -20 | while read csv; do
    echo "  found: $csv"
done

# 5. Regenerate the chart pack
python3 gen_weekly_figures.py
cp -r results/figures_weekly/* "$FIGS_DIR/" 2>/dev/null || true

echo
echo "Figures: $FIGS_DIR/"
ls -la "$FIGS_DIR/"
