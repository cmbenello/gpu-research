#!/bin/bash
# perf-compare.sh — Run GPU CrocSort at multiple sizes and compare results
# Usage: ./perf-compare.sh [--quick|--full]
# Outputs a formatted comparison table + CSV

set -e
cd "$(dirname "$0")/../gpu_crocsort"

BOLD='\033[1m'
GREEN='\033[32m'
CYAN='\033[36m'
DIM='\033[2m'
RESET='\033[0m'

MODE=${1:---quick}
CSV_FILE="../results/perf_results_$(date +%Y%m%d_%H%M%S).csv"

if [ ! -f gpu_crocsort ]; then
    echo "gpu_crocsort binary not found. Building..."
    ../scripts/quick-build.sh
fi

case "$MODE" in
    --quick)
        SIZES=(10000 100000 1000000)
        ;;
    --full)
        SIZES=(1000 10000 100000 500000 1000000 5000000 10000000)
        ;;
    *)
        echo "Usage: $0 [--quick|--full]"
        exit 1
        ;;
esac

mkdir -p ../results

echo -e "${BOLD}GPU CrocSort Performance Comparison${RESET}"
echo -e "${DIM}$(date)${RESET}"
echo ""

# GPU info
if command -v nvidia-smi &>/dev/null; then
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${CYAN}GPU: $gpu${RESET}"
fi
echo ""

# CSV header
echo "records,data_mb,sort_time_ms,throughput_gbps,records_per_sec_m,merge_passes,strategy" > "$CSV_FILE"

# Table header
printf "${BOLD}%-12s %-10s %-12s %-12s %-12s %-8s${RESET}\n" \
    "Records" "Data(MB)" "Time(ms)" "GB/s" "M rec/s" "Passes"
printf "%-12s %-10s %-12s %-12s %-12s %-8s\n" \
    "--------" "--------" "--------" "--------" "--------" "------"

for n in "${SIZES[@]}"; do
    data_mb=$(echo "scale=2; $n * 100 / 1048576" | bc)

    # Run sort, capture output
    output=$(./gpu_crocsort --num-records "$n" 2>&1)

    # Parse results
    sort_time=$(echo "$output" | grep "SORT COMPLETE" | grep -oP '[\d.]+(?= ms)' | head -1)
    throughput=$(echo "$output" | grep "Throughput:" | grep -oP '[\d.]+(?= GB/s)' | head -1)
    rec_per_sec=$(echo "$output" | grep "records/sec" | grep -oP '[\d.]+(?= M)' | head -1)
    passes=$(echo "$output" | grep "Merge passes:" | grep -oP '(?<=Merge passes: )\d+' | head -1)
    strategy=$(echo "$output" | grep "Strategy:" | sed 's/.*Strategy: //')

    # Default values if parsing fails
    sort_time=${sort_time:-"N/A"}
    throughput=${throughput:-"N/A"}
    rec_per_sec=${rec_per_sec:-"N/A"}
    passes=${passes:-"?"}

    # Print table row
    printf "%-12s %-10s %-12s %-12s %-12s %-8s\n" \
        "$n" "$data_mb" "$sort_time" "$throughput" "$rec_per_sec" "$passes"

    # CSV row
    echo "$n,$data_mb,$sort_time,$throughput,$rec_per_sec,$passes,$strategy" >> "$CSV_FILE"
done

echo ""
echo -e "${DIM}Results saved to: $CSV_FILE${RESET}"

# ── Also test with duplicates if doing full run ──
if [ "$MODE" = "--full" ]; then
    echo ""
    echo -e "${BOLD}Duplicate-heavy data:${RESET}"
    printf "%-12s %-10s %-12s %-12s %-12s %-8s\n" \
        "Records" "Data(MB)" "Time(ms)" "GB/s" "M rec/s" "Passes"
    printf "%-12s %-10s %-12s %-12s %-12s %-8s\n" \
        "--------" "--------" "--------" "--------" "--------" "------"

    for n in 100000 1000000 10000000; do
        data_mb=$(echo "scale=2; $n * 100 / 1048576" | bc)
        output=$(./gpu_crocsort --num-records "$n" --duplicates 2>&1)
        sort_time=$(echo "$output" | grep "SORT COMPLETE" | grep -oP '[\d.]+(?= ms)' | head -1)
        throughput=$(echo "$output" | grep "Throughput:" | grep -oP '[\d.]+(?= GB/s)' | head -1)
        rec_per_sec=$(echo "$output" | grep "records/sec" | grep -oP '[\d.]+(?= M)' | head -1)
        passes=$(echo "$output" | grep "Merge passes:" | grep -oP '(?<=Merge passes: )\d+' | head -1)

        printf "%-12s %-10s %-12s %-12s %-12s %-8s\n" \
            "${n}(dup)" "$data_mb" "${sort_time:-N/A}" "${throughput:-N/A}" "${rec_per_sec:-N/A}" "${passes:-?}"
    done
fi

echo ""
echo -e "${GREEN}Done!${RESET}"
