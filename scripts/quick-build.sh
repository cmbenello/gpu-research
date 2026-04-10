#!/bin/bash
# quick-build.sh — Smart incremental build for GPU CrocSort
# Detects GPU arch, only rebuilds what changed, runs tests on success

set -e
cd "$(dirname "$0")/../gpu_crocsort"

BOLD='\033[1m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
DIM='\033[2m'
RESET='\033[0m'

# ── Detect GPU arch ──
detect_arch() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        case "$GPU_NAME" in
            *H100*|*H200*)              echo "sm_90" ;;
            *L4*|*L40*|*4090*|*4080*)   echo "sm_89" ;;
            *A10*|*3090*|*3080*|*A6000*) echo "sm_86" ;;
            *A100*)                     echo "sm_80" ;;
            *V100*)                     echo "sm_70" ;;
            *T4*)                       echo "sm_75" ;;
            *)                          echo "sm_80" ;;
        esac
    else
        echo "sm_80"
    fi
}

ARCH=${1:-$(detect_arch)}
TARGET=${2:-all}

echo -e "${BOLD}Quick Build${RESET} [arch=$ARCH, target=$TARGET]"

# ── Check for changes ──
needs_rebuild=false
if [ ! -f gpu_crocsort ]; then
    needs_rebuild=true
else
    binary_time=$(stat -c '%Y' gpu_crocsort 2>/dev/null || echo 0)
    for src in src/*.cu include/*.cuh; do
        src_time=$(stat -c '%Y' "$src" 2>/dev/null || echo 0)
        if [ "$src_time" -gt "$binary_time" ]; then
            needs_rebuild=true
            echo -e "  ${YELLOW}changed:${RESET} $src"
        fi
    done
fi

if [ "$needs_rebuild" = false ] && [ "$TARGET" = "all" ]; then
    echo -e "  ${GREEN}Up to date${RESET} - no source changes detected"
    echo -e "  ${DIM}(use 'make clean && $0' to force rebuild)${RESET}"
    exit 0
fi

# ── Build ──
START=$(date +%s%N)

case "$TARGET" in
    all)
        echo -n "  Building main binary..."
        if make ARCH=$ARCH 2>/tmp/build_err.log; then
            echo -e " ${GREEN}OK${RESET}"
        else
            echo -e " ${RED}FAIL${RESET}"
            cat /tmp/build_err.log
            exit 1
        fi
        ;;
    experiments)
        echo -n "  Building experiments..."
        if make experiments ARCH=$ARCH 2>/tmp/build_err.log; then
            echo -e " ${GREEN}OK${RESET}"
        else
            echo -e " ${RED}FAIL${RESET}"
            cat /tmp/build_err.log
            exit 1
        fi
        ;;
    test)
        echo -n "  Building + testing..."
        if make ARCH=$ARCH 2>/tmp/build_err.log && ./gpu_crocsort --num-records 10000 --verify 2>&1 | tail -3; then
            echo -e "  ${GREEN}Tests passed${RESET}"
        else
            echo -e "  ${RED}Tests failed${RESET}"
            exit 1
        fi
        ;;
    clean)
        make clean
        echo -e "  ${GREEN}Cleaned${RESET}"
        exit 0
        ;;
    *)
        echo -n "  Building $TARGET..."
        if make "$TARGET" ARCH=$ARCH 2>/tmp/build_err.log; then
            echo -e " ${GREEN}OK${RESET}"
        else
            echo -e " ${RED}FAIL${RESET}"
            cat /tmp/build_err.log
            exit 1
        fi
        ;;
esac

END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
echo -e "  ${DIM}Build time: ${ELAPSED}ms${RESET}"

# ── Quick smoke test on successful build ──
if [ "$TARGET" = "all" ] && [ -f gpu_crocsort ]; then
    echo -n "  Smoke test (1000 records)..."
    if ./gpu_crocsort --num-records 1000 --verify 2>&1 | grep -q "PASS"; then
        echo -e " ${GREEN}PASS${RESET}"
    else
        echo -e " ${YELLOW}skipped (no GPU?)${RESET}"
    fi
fi
