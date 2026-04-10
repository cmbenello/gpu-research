#!/bin/bash
# cuda-debug.sh — Quick CUDA debugging helpers
# Usage: ./cuda-debug.sh <command>
#
# Commands:
#   memcheck [args]   — Run with cuda-memcheck (detects OOB, races)
#   sanitize [args]   — Run with compute-sanitizer (CUDA 11.6+)
#   profile [args]    — Run with nvprof or nsys profiling
#   ptx               — Show PTX assembly for merge kernel
#   occupancy         — Show kernel occupancy analysis
#   errors            — Decode CUDA error codes

set -e
cd "$(dirname "$0")/../gpu_crocsort"

BOLD='\033[1m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
DIM='\033[2m'
RESET='\033[0m'

cmd=${1:-help}
shift 2>/dev/null || true

# Default args if none provided
ARGS="${*:---num-records 10000 --verify}"

case "$cmd" in
    memcheck)
        echo -e "${BOLD}CUDA Memcheck${RESET}"
        if command -v compute-sanitizer &>/dev/null; then
            compute-sanitizer --tool memcheck ./gpu_crocsort $ARGS
        elif command -v cuda-memcheck &>/dev/null; then
            cuda-memcheck ./gpu_crocsort $ARGS
        else
            echo -e "${RED}Neither compute-sanitizer nor cuda-memcheck found${RESET}"
            exit 1
        fi
        ;;

    racecheck)
        echo -e "${BOLD}CUDA Race Check${RESET}"
        if command -v compute-sanitizer &>/dev/null; then
            compute-sanitizer --tool racecheck ./gpu_crocsort $ARGS
        else
            echo -e "${RED}compute-sanitizer not found${RESET}"
            exit 1
        fi
        ;;

    sanitize)
        echo -e "${BOLD}Compute Sanitizer (all checks)${RESET}"
        if command -v compute-sanitizer &>/dev/null; then
            echo -e "${DIM}Running memcheck...${RESET}"
            compute-sanitizer --tool memcheck ./gpu_crocsort $ARGS
            echo ""
            echo -e "${DIM}Running racecheck...${RESET}"
            compute-sanitizer --tool racecheck ./gpu_crocsort $ARGS
            echo ""
            echo -e "${DIM}Running initcheck...${RESET}"
            compute-sanitizer --tool initcheck ./gpu_crocsort $ARGS
        else
            echo -e "${RED}compute-sanitizer not found (needs CUDA 11.6+)${RESET}"
            exit 1
        fi
        ;;

    profile)
        echo -e "${BOLD}Profiling${RESET}"
        if command -v nsys &>/dev/null; then
            echo -e "Using ${GREEN}nsys${RESET} (Nsight Systems)"
            report="nsys_report_$(date +%Y%m%d_%H%M%S)"
            nsys profile -o "$report" --stats=true ./gpu_crocsort $ARGS
            echo -e "\nReport: ${report}.nsys-rep"
            echo -e "View:   nsys-ui ${report}.nsys-rep"
        elif command -v nvprof &>/dev/null; then
            echo -e "Using ${GREEN}nvprof${RESET}"
            nvprof --print-gpu-trace ./gpu_crocsort $ARGS
        else
            echo -e "${YELLOW}No profiler found. Install nsys:${RESET}"
            echo "  apt install nsight-systems"
            exit 1
        fi
        ;;

    ptx)
        echo -e "${BOLD}Generating PTX for merge kernels${RESET}"
        ARCH=${1:-sm_80}
        nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=$ARCH \
            -ptx -Iinclude src/merge.cu -o merge.ptx 2>/dev/null

        if [ -f merge.ptx ]; then
            echo -e "${GREEN}Generated merge.ptx${RESET}"
            echo ""
            echo "Key kernel entry points:"
            grep '\.visible \.entry' merge.ptx | sed 's/.*\.entry /  /'
            echo ""
            total_inst=$(wc -l < merge.ptx)
            echo "Total PTX instructions: $total_inst"
            echo ""
            echo -e "${DIM}Full PTX: merge.ptx${RESET}"
        fi
        ;;

    occupancy)
        echo -e "${BOLD}Kernel Occupancy Analysis${RESET}"
        echo ""
        echo "Kernel configurations:"
        echo ""
        echo "  run_generation_kernel:"
        echo "    Threads/block: 256"
        echo "    Shared mem:    $(echo "512 * (8+2+4)" | bc) bytes (keys+indices)"
        echo "    Registers:     ~32 (estimate)"
        echo ""
        echo "  merge_2way_kernel:"
        echo "    Threads/block: 256"
        echo "    Shared mem:    0 bytes"
        echo "    Items/thread:  8"
        echo ""
        echo "  smem_kway_merge_kernel:"
        echo "    Threads/block: 256"
        echo "    Shared mem:    2 * records * 100 bytes (dynamic)"
        echo "    Max records:   ~495 per partition (99KB cap)"
        echo ""

        if command -v nvidia-smi &>/dev/null; then
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            echo "  GPU: $gpu_name"

            # Get SM count and shared mem
            echo ""
            echo -e "  ${DIM}For detailed occupancy, use:${RESET}"
            echo "    nvcc --resource-usage -arch=sm_80 -Iinclude src/merge.cu"
        fi
        ;;

    errors)
        echo -e "${BOLD}Common CUDA Error Codes${RESET}"
        echo ""
        echo "  0   cudaSuccess"
        echo "  1   cudaErrorInvalidValue"
        echo "  2   cudaErrorMemoryAllocation (out of GPU memory)"
        echo "  4   cudaErrorLaunchFailure"
        echo "  6   cudaErrorLaunchTimeout"
        echo "  7   cudaErrorLaunchOutOfResources (too many registers/smem)"
        echo "  9   cudaErrorInvalidConfiguration (bad launch params)"
        echo "  11  cudaErrorInvalidDevice"
        echo "  35  cudaErrorInsufficientDriver"
        echo "  46  cudaErrorDevicesUnavailable"
        echo "  71  cudaErrorIllegalAddress (segfault on GPU)"
        echo "  72  cudaErrorMisalignedAddress"
        echo "  98  cudaErrorNotReady (async not done)"
        echo "  100 cudaErrorNoDevice"
        echo "  700 cudaErrorIllegalInstruction"
        echo "  701 cudaErrorAssert (__assert_fail in kernel)"
        echo "  710 cudaErrorInvalidPc"
        echo "  719 cudaErrorLaunchFailure (unspecified)"
        echo ""
        echo -e "${DIM}Full list: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html${RESET}"
        ;;

    help|*)
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  memcheck [args]    Run with CUDA memcheck"
        echo "  racecheck [args]   Check for race conditions"
        echo "  sanitize [args]    Run all sanitizer checks"
        echo "  profile [args]     Profile with nsys/nvprof"
        echo "  ptx [arch]         Generate PTX assembly"
        echo "  occupancy          Show kernel occupancy info"
        echo "  errors             Common CUDA error reference"
        echo ""
        echo "Default args: --num-records 10000 --verify"
        echo "Override: $0 profile --num-records 1000000"
        ;;
esac
