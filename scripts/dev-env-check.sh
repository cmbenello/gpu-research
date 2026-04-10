#!/bin/bash
# dev-env-check.sh — Verify everything needed for GPU CrocSort development
# Run this on a new machine or after setup changes

set -e

BOLD='\033[1m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
DIM='\033[2m'
RESET='\033[0m'

pass=0
fail=0
warn=0

check_pass() { echo -e "  ${GREEN}[PASS]${RESET} $1"; ((pass++)); }
check_fail() { echo -e "  ${RED}[FAIL]${RESET} $1"; ((fail++)); }
check_warn() { echo -e "  ${YELLOW}[WARN]${RESET} $1"; ((warn++)); }

echo -e "${BOLD}Development Environment Check${RESET}"
echo ""

# ── CUDA Toolkit ──
echo -e "${BOLD}CUDA:${RESET}"
if command -v nvcc &>/dev/null; then
    cuda_ver=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
    major=$(echo "$cuda_ver" | cut -d. -f1)
    if [ "$major" -ge 11 ]; then
        check_pass "nvcc $cuda_ver (>= 11.0 required)"
    else
        check_fail "nvcc $cuda_ver (>= 11.0 required)"
    fi
else
    check_fail "nvcc not found — install CUDA toolkit"
fi

# ── GPU Driver ──
echo ""
echo -e "${BOLD}GPU Driver:${RESET}"
if command -v nvidia-smi &>/dev/null; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    check_pass "nvidia-smi available (driver $driver)"

    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    check_pass "GPU: $gpu_name ($gpu_mem)"

    compute=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 2>/dev/null || echo "")
    if [ -n "$compute" ]; then
        check_pass "Compute capability: $compute"
    fi
else
    check_fail "nvidia-smi not found — GPU driver not installed"
fi

# ── C++ Compiler ──
echo ""
echo -e "${BOLD}C++ Toolchain:${RESET}"
if command -v g++ &>/dev/null; then
    gpp_ver=$(g++ --version | head -1)
    check_pass "g++: $gpp_ver"
else
    check_warn "g++ not found (nvcc uses its own, but useful for CPU baselines)"
fi

if command -v make &>/dev/null; then
    check_pass "make: $(make --version | head -1)"
else
    check_fail "make not found — install build-essential"
fi

# ── Python (for result parsing) ──
echo ""
echo -e "${BOLD}Python:${RESET}"
if command -v python3 &>/dev/null; then
    py_ver=$(python3 --version 2>&1)
    check_pass "$py_ver"

    # Check useful packages
    for pkg in matplotlib numpy pandas; do
        if python3 -c "import $pkg" 2>/dev/null; then
            check_pass "  $pkg installed"
        else
            check_warn "  $pkg not installed (optional, for plotting)"
        fi
    done
else
    check_warn "python3 not found (optional, for result parsing)"
fi

# ── Git ──
echo ""
echo -e "${BOLD}Git:${RESET}"
if command -v git &>/dev/null; then
    check_pass "git: $(git --version)"

    if git remote -v 2>/dev/null | grep -q origin; then
        remote=$(git remote get-url origin 2>/dev/null)
        check_pass "remote: $remote"
    else
        check_warn "no git remote configured"
    fi

    # SSH key check
    if [ -f ~/.ssh/id_ed25519 ] || [ -f ~/.ssh/id_rsa ]; then
        check_pass "SSH key found"
        if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
            check_pass "GitHub SSH auth working"
        else
            check_warn "GitHub SSH auth failed (run: ssh-keygen -t ed25519)"
        fi
    else
        check_warn "No SSH key — git push will need HTTPS token"
    fi
else
    check_fail "git not found"
fi

# ── Disk Space ──
echo ""
echo -e "${BOLD}Disk:${RESET}"
avail=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${avail:-0}" -gt 10 ]; then
    check_pass "${avail}G available (10G+ recommended for large benchmarks)"
elif [ "${avail:-0}" -gt 2 ]; then
    check_warn "${avail}G available (10G+ recommended for large benchmarks)"
else
    check_fail "${avail}G available (need at least 2G)"
fi

# ── Memory ──
echo ""
echo -e "${BOLD}System Memory:${RESET}"
total_mem=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
if [ "${total_mem:-0}" -gt 8 ]; then
    check_pass "${total_mem}G RAM"
else
    check_warn "${total_mem:-?}G RAM (8G+ recommended)"
fi

# ── Project Files ──
echo ""
echo -e "${BOLD}Project:${RESET}"
cd "$(dirname "$0")/.."

if [ -f gpu_crocsort/Makefile ]; then
    check_pass "Makefile found"
else
    check_fail "gpu_crocsort/Makefile missing"
fi

for f in gpu_crocsort/src/main.cu gpu_crocsort/src/merge.cu gpu_crocsort/include/record.cuh gpu_crocsort/include/ovc.cuh; do
    if [ -f "$f" ]; then
        check_pass "$f"
    else
        check_fail "$f missing"
    fi
done

# ── Summary ──
echo ""
echo -e "${BOLD}════════════════════════════════${RESET}"
echo -e "  ${GREEN}Pass: $pass${RESET}  ${YELLOW}Warn: $warn${RESET}  ${RED}Fail: $fail${RESET}"
if [ "$fail" -eq 0 ]; then
    echo -e "  ${GREEN}Ready to build!${RESET} Run: cd gpu_crocsort && make"
else
    echo -e "  ${RED}Fix failures above before building${RESET}"
fi
echo -e "${BOLD}════════════════════════════════${RESET}"

exit $fail
