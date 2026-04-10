#!/bin/bash
# project-stats.sh — Quick overview of the GPU CrocSort project
# Shows code stats, git info, GPU info, and build status

set -e
cd "$(dirname "$0")/.."

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[32m'
YELLOW='\033[33m'
CYAN='\033[36m'
RED='\033[31m'
RESET='\033[0m'

echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  GPU CrocSort — Project Stats${RESET}"
echo -e "${BOLD}========================================${RESET}"
echo ""

# ── Code Stats ──
echo -e "${CYAN}Code:${RESET}"
cuda_lines=$(find gpu_crocsort -name '*.cu' -o -name '*.cuh' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
cuda_files=$(find gpu_crocsort -name '*.cu' -o -name '*.cuh' 2>/dev/null | wc -l)
echo "  CUDA source:    ${cuda_lines:-0} lines across ${cuda_files:-0} files"

kernel_count=$(grep -r '__global__' gpu_crocsort/src/ gpu_crocsort/experiments/ 2>/dev/null | wc -l)
echo "  CUDA kernels:   ${kernel_count:-0}"

device_funcs=$(grep -r '__device__' gpu_crocsort/include/ gpu_crocsort/src/ 2>/dev/null | grep -v '//' | wc -l)
echo "  Device funcs:   ${device_funcs:-0}"

if [ -d crocsort_repo ]; then
    rust_lines=$(find crocsort_repo -name '*.rs' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
    echo "  Rust reference: ${rust_lines:-0} lines"
fi

echo ""

# ── File Breakdown ──
echo -e "${CYAN}File breakdown:${RESET}"
for f in gpu_crocsort/src/*.cu gpu_crocsort/include/*.cuh; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        name=$(basename "$f")
        printf "  %-28s %4d lines\n" "$name" "$lines"
    fi
done
echo ""
echo -e "  ${DIM}Experiments:${RESET}"
for f in gpu_crocsort/experiments/*.cu; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        name=$(basename "$f")
        printf "  %-28s %4d lines\n" "$name" "$lines"
    fi
done
echo ""

# ── Git Stats ──
echo -e "${CYAN}Git:${RESET}"
branch=$(git branch --show-current 2>/dev/null || echo "unknown")
commits=$(git rev-list --count HEAD 2>/dev/null || echo "?")
last_commit=$(git log -1 --format="%h %s" 2>/dev/null || echo "unknown")
uncommitted=$(git status --porcelain 2>/dev/null | wc -l)
echo "  Branch:         $branch"
echo "  Commits:        $commits"
echo "  Last commit:    $last_commit"
if [ "$uncommitted" -gt 0 ]; then
    echo -e "  Uncommitted:    ${YELLOW}${uncommitted} changes${RESET}"
else
    echo -e "  Uncommitted:    ${GREEN}clean${RESET}"
fi
echo ""

# ── GPU Info ──
echo -e "${CYAN}GPU:${RESET}"
if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:            ${gpu_name:-not detected}"
    echo "  Memory:         ${gpu_free:-?} free / ${gpu_mem:-?} total"
    echo "  Utilization:    ${gpu_util:-?}"
    echo "  Temperature:    ${gpu_temp:-?}C"
    echo "  Driver:         ${driver:-?}"
else
    echo -e "  ${RED}nvidia-smi not found${RESET}"
fi

if command -v nvcc &>/dev/null; then
    cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "  CUDA toolkit:   ${cuda_ver:-?}"
else
    echo -e "  ${RED}nvcc not found${RESET}"
fi
echo ""

# ── Build Status ──
echo -e "${CYAN}Binaries:${RESET}"
cd gpu_crocsort
for bin in gpu_crocsort bottleneck_bench layout_bench radix_vs_merge cpu_vs_gpu \
           profile_sort test_edge_cases external_sort kv_bench sample_partition benchmark_suite; do
    if [ -f "$bin" ]; then
        mod_time=$(stat -c '%Y' "$bin" 2>/dev/null || stat -f '%m' "$bin" 2>/dev/null)
        now=$(date +%s)
        age=$(( (now - mod_time) / 3600 ))
        echo -e "  ${GREEN}[built]${RESET}  $bin  ${DIM}(${age}h ago)${RESET}"
    else
        echo -e "  ${DIM}[  --  ]${RESET}  $bin"
    fi
done
echo ""

# ── Research Ideas ──
if [ -f "../results/summary.md" ]; then
    top_ideas=$(grep -c "^### " ../results/summary.md 2>/dev/null || echo "0")
    echo -e "${CYAN}Research:${RESET}"
    echo "  Ideas documented: ${top_ideas}"
    implemented=$(grep -c '\[x\]' ../gpu_crocsort/README.md 2>/dev/null || echo "0")
    todo=$(grep -c '\[ \]' ../gpu_crocsort/README.md 2>/dev/null || echo "0")
    echo -e "  Implemented:      ${GREEN}${implemented}${RESET}"
    echo -e "  TODO:             ${YELLOW}${todo}${RESET}"
fi

echo ""
echo -e "${BOLD}========================================${RESET}"
