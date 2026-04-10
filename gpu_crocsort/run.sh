#!/bin/bash
set -e

echo "════════════════════════════════════════════════════"
echo "  GPU CrocSort — One-Click Build & Run"
echo "════════════════════════════════════════════════════"

# Detect GPU architecture
detect_arch() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "Detected GPU: $GPU_NAME"

        # Map GPU name to architecture
        case "$GPU_NAME" in
            *H100*|*H200*)          echo "sm_90" ;;
            *L4*|*L40*|*4090*|*4080*) echo "sm_89" ;;
            *A10*|*3090*|*3080*|*A6000*) echo "sm_86" ;;
            *A100*)                 echo "sm_80" ;;
            *V100*)                 echo "sm_70" ;;
            *T4*)                   echo "sm_75" ;;
            *)                      echo "sm_80" ;; # Default
        esac
    else
        echo "nvidia-smi not found, defaulting to sm_80"
        echo "sm_80"
    fi
}

ARCH=$(detect_arch)
echo "Using architecture: $ARCH"
echo ""

# Check for nvcc
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit first:"
    echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "  Or: module load cuda  (on HPC clusters)"
    echo "  Or: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

echo "nvcc version: $(nvcc --version | tail -1)"
echo ""

# Build
echo "══ Building ══"
make clean 2>/dev/null || true
make ARCH=$ARCH
echo ""

# Run bottleneck analysis first
echo "══ Step 1: Bottleneck Analysis ══"
echo "(This tells us WHERE the time goes on YOUR GPU)"
echo ""
make bottleneck-nocub ARCH=$ARCH 2>/dev/null || make bottleneck ARCH=$ARCH
./bottleneck_bench 1000000
echo ""

# Small correctness test
echo "══ Step 2: Correctness Test (10K records) ══"
./gpu_crocsort --num-records 10000 --verify
echo ""

# Medium correctness test
echo "══ Step 3: Correctness Test (1M records) ══"
./gpu_crocsort --num-records 1000000 --verify
echo ""

# Benchmark
echo "══ Step 4: Benchmark (10M records = 1 GB) ══"
./gpu_crocsort --num-records 10000000
echo ""

# Larger benchmark if enough GPU memory
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [ "${GPU_MEM_MB:-0}" -gt 10000 ]; then
    echo "══ Step 5: Large Benchmark (100M records = 10 GB) ══"
    ./gpu_crocsort --num-records 100000000
    echo ""
fi

echo "════════════════════════════════════════════════════"
echo "  Done. Results above."
echo "════════════════════════════════════════════════════"
