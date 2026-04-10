#!/bin/bash
set -e

echo "════════════════════════════════════════════════════"
echo "  GPU CrocSort — Full Build, Test & Benchmark Suite"
echo "════════════════════════════════════════════════════"

# Detect GPU architecture
detect_arch() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "Detected GPU: $GPU_NAME" >&2
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

ARCH=$(detect_arch)
echo "Architecture: $ARCH"
echo ""

# Check nvcc
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit:"
    echo "  sudo apt install nvidia-cuda-toolkit"
    echo "  -or- module load cuda"
    echo "  -or- export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi
echo "CUDA: $(nvcc --version | grep release)"
echo ""

# ── Build everything ──
echo "══ Building ══"
make clean 2>/dev/null || true
make ARCH=$ARCH 2>&1 | tail -1
echo "  Built: gpu_crocsort"

# Build experiments (ignore failures — some need CUB)
for exp in bottleneck-nocub layout-bench radix-vs-merge; do
    if make $exp ARCH=$ARCH 2>/dev/null; then
        echo "  Built: $exp"
    fi
done
echo ""

# ── Step 1: Bottleneck Analysis ──
echo "══ Step 1: Bottleneck Analysis ══"
echo "  (Measures raw HBM bandwidth, merge pass cost, fanin tradeoff)"
echo ""
if [ -f bottleneck_bench ]; then
    ./bottleneck_bench 1000000
else
    echo "  (skipped — build failed)"
fi
echo ""

# ── Step 2: Memory Layout Comparison ──
echo "══ Step 2: Memory Layout Benchmark ══"
echo "  (AoS vs SoA vs Key+Index — which is fastest for merge?)"
echo ""
if [ -f layout_bench ]; then
    ./layout_bench
else
    echo "  (skipped — build failed)"
fi
echo ""

# ── Step 3: Correctness Tests ──
echo "══ Step 3: Correctness Tests ══"
./gpu_crocsort --num-records 100 --verify
./gpu_crocsort --num-records 10000 --verify
./gpu_crocsort --num-records 100000 --verify
echo ""

# ── Step 4: Scaling Benchmark ──
echo "══ Step 4: Scaling Benchmark ══"
for n in 100000 500000 1000000 5000000 10000000; do
    ./gpu_crocsort --num-records $n 2>&1 | grep -E "SORT COMPLETE|Throughput"
done
echo ""

# ── Step 5: Large benchmark if enough memory ──
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [ "${GPU_MEM_MB:-0}" -gt 15000 ]; then
    echo "══ Step 5: Large Benchmark (50M records = 5 GB) ══"
    ./gpu_crocsort --num-records 50000000
    echo ""
fi
if [ "${GPU_MEM_MB:-0}" -gt 35000 ]; then
    echo "══ Step 6: XL Benchmark (100M records = 10 GB) ══"
    ./gpu_crocsort --num-records 100000000
    echo ""
fi

# ── Step 6: Radix sort comparison ──
echo "══ Step 7: Radix Sort vs Merge Sort Analysis ══"
if [ -f radix_vs_merge ]; then
    ./radix_vs_merge
else
    echo "  (skipped — needs CUB)"
fi
echo ""

echo "════════════════════════════════════════════════════"
echo "  COMPLETE"
echo ""
echo "  Additional experiments you can run:"
echo "    make cpu-vs-gpu && ./cpu_vs_gpu --num-records 1000000"
echo "    make test-edge && ./test_edge_cases"
echo "    make profile && ./profile_sort 1000000"
echo "    make external-sort && ./external_sort --total-gb 2"
echo "    make kv-bench && ./kv_bench"
echo "    make sample-part && ./sample_partition"
echo "════════════════════════════════════════════════════"
