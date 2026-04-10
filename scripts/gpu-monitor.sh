#!/bin/bash
# gpu-monitor.sh — Compact GPU monitoring dashboard
# Usage: ./gpu-monitor.sh [interval_seconds]
# Press Ctrl+C to stop

INTERVAL=${1:-2}

BOLD='\033[1m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
CYAN='\033[36m'
DIM='\033[2m'
RESET='\033[0m'

if ! command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi not found"
    exit 1
fi

# Stash cursor, clear screen
bar() {
    local val=$1 max=$2 width=${3:-30}
    local filled=$(( val * width / max ))
    local empty=$(( width - filled ))
    local color=$GREEN
    if [ "$val" -gt $(( max * 80 / 100 )) ]; then color=$RED
    elif [ "$val" -gt $(( max * 50 / 100 )) ]; then color=$YELLOW
    fi
    printf "${color}"
    printf '%0.s#' $(seq 1 $filled 2>/dev/null) || true
    printf "${DIM}"
    printf '%0.s-' $(seq 1 $empty 2>/dev/null) || true
    printf "${RESET}"
}

cleanup() {
    tput cnorm 2>/dev/null  # Show cursor
    echo ""
    exit 0
}
trap cleanup INT TERM

tput civis 2>/dev/null  # Hide cursor

while true; do
    clear
    echo -e "${BOLD}GPU Monitor${RESET}  ${DIM}(every ${INTERVAL}s, Ctrl+C to stop)${RESET}"
    echo ""

    # Query GPU stats
    IFS=',' read -r name temp power power_max util mem_used mem_total fan \
        <<< "$(nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,fan.speed --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"

    echo -e "${CYAN}${name:-GPU}${RESET}"
    echo ""

    # Temperature
    temp_int=${temp%%.*}
    temp_int=${temp_int:-0}
    printf "  Temp:    %3s C  " "${temp_int}"
    bar "$temp_int" 100 25
    echo ""

    # GPU Utilization
    util_int=${util%%.*}
    util_int=${util_int:-0}
    printf "  GPU:     %3s %%  " "${util_int}"
    bar "$util_int" 100 25
    echo ""

    # Memory
    mem_used_int=${mem_used%%.*}
    mem_total_int=${mem_total%%.*}
    mem_used_int=${mem_used_int:-0}
    mem_total_int=${mem_total_int:-1}
    mem_pct=$(( mem_used_int * 100 / mem_total_int ))
    printf "  Mem:  %5s MB  " "${mem_used_int}"
    bar "$mem_pct" 100 25
    printf "  / %s MB\n" "${mem_total_int}"

    # Power
    power_int=${power%%.*}
    power_max_int=${power_max%%.*}
    power_int=${power_int:-0}
    power_max_int=${power_max_int:-1}
    power_pct=$(( power_int * 100 / power_max_int ))
    printf "  Power: %4s W   " "${power_int}"
    bar "$power_pct" 100 25
    printf "  / %s W\n" "${power_max_int}"

    # Fan
    if [ "${fan}" != "[N/A]" ] && [ -n "$fan" ]; then
        fan_int=${fan%%.*}
        fan_int=${fan_int:-0}
        printf "  Fan:     %3s %%  " "${fan_int}"
        bar "$fan_int" 100 25
        echo ""
    fi

    echo ""

    # Running GPU processes
    echo -e "${BOLD}Processes:${RESET}"
    procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -z "$procs" ]; then
        echo -e "  ${DIM}(none)${RESET}"
    else
        echo "$procs" | while IFS=',' read -r pid pname pmem; do
            pid=$(echo "$pid" | tr -d ' ')
            pname=$(echo "$pname" | tr -d ' ' | xargs -I{} basename "{}")
            pmem=$(echo "$pmem" | tr -d ' ')
            printf "  PID %-8s %-30s %s\n" "$pid" "$pname" "$pmem"
        done
    fi

    echo ""
    echo -e "${DIM}$(date '+%H:%M:%S')${RESET}"

    sleep "$INTERVAL"
done
