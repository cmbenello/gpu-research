#!/usr/bin/env bash
# Helper: run experiment with N runs across SF10/50/100, parse stats, append to LOG.tsv
# Usage: run_experiment.sh <slug> <branch> [SF1 SF2 ...]
set -e
SLUG="$1"
BRANCH="$2"
shift 2
SFS=("$@")
[ ${#SFS[@]} -eq 0 ] && SFS=(sf10 sf50 sf100)
RUNS=${RUNS:-3}

UTC_DATE=$(cat /tmp/overnight_date.txt)
D=results/overnight_$UTC_DATE
LOG=$D/LOG.tsv
RESULT=$D/${SLUG}.md

echo "## Experiment: $SLUG" >> $RESULT
echo "Branch: $BRANCH ($(git rev-parse --short HEAD))" >> $RESULT
echo "Date: $(date -u)" >> $RESULT
echo "Runs per SF: $RUNS" >> $RESULT
echo >> $RESULT
echo "### Raw timings (ms)" >> $RESULT
echo >> $RESULT
echo "| SF | run1 | run2 | run3 | run4 | run5 | min | median | stdev | verified |" >> $RESULT
echo "|---:|-----:|-----:|-----:|-----:|-----:|----:|-------:|------:|---------:|" >> $RESULT

for SF in "${SFS[@]}"; do
  TOTALS=()
  VERIFIED=true
  for i in $(seq 1 $RUNS); do
    OUT=$(./external_sort_tpch_compact --input /tmp/lineitem_${SF}_normalized.bin --runs 1 2>&1)
    CSV=$(echo "$OUT" | grep '^CSV,' | head -1)
    PASS_S=$(echo "$OUT" | grep -c 'PASS sortedness')
    PASS_M=$(echo "$OUT" | grep -c 'PASS multiset')
    if [ "$PASS_S" -ne 1 ] || [ "$PASS_M" -ne 1 ]; then
      VERIFIED=false
      echo "  $SF run $i: VERIFY FAILED" >&2
    fi
    TOT=$(echo "$CSV" | awk -F, '{print $8}')
    TOTALS+=("$TOT")
  done
  # Pad missing runs
  while [ ${#TOTALS[@]} -lt 5 ]; do TOTALS+=("-"); done
  # Compute stats from numerics
  NUMS=$(printf "%s\n" "${TOTALS[@]}" | grep -v '^-$' | sort -g)
  MIN=$(echo "$NUMS" | head -1)
  COUNT=$(echo "$NUMS" | wc -l)
  if [ "$COUNT" -ge 1 ]; then
    MED=$(echo "$NUMS" | awk -v c=$COUNT 'NR==int((c+1)/2){printf "%.0f",$1}')
    AVG=$(echo "$NUMS" | awk '{s+=$1} END{printf "%.0f",s/NR}')
    SD=$(echo "$NUMS" | awk -v a=$AVG '{s+=($1-a)*($1-a)} END{printf "%.0f",sqrt(s/NR)}')
  else
    MED="-"; SD="-"
  fi
  ROW="| $SF | ${TOTALS[0]:-} | ${TOTALS[1]:-} | ${TOTALS[2]:-} | ${TOTALS[3]:-} | ${TOTALS[4]:-} | $MIN | $MED | $SD | $VERIFIED |"
  echo "$ROW" >> $RESULT
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$BRANCH" "$SLUG" "$SF" "$(awk -v ms=$MED 'BEGIN{printf "%.3f", ms/1000}')" "$VERIFIED" "ok" \
    >> $LOG
done
echo >> $RESULT
cat $RESULT | tail -20
