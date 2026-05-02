#!/usr/bin/env bash
# Tier 19.5 — record exact hardware fingerprint for paper reproducibility.
# Outputs a single JSON file listing every relevant detail of the host.
set -uo pipefail

OUT="${1:-results/overnight_pulled/hardware_$(hostname)_$(date +%Y%m%d_%H%M%S).json}"
mkdir -p "$(dirname "$OUT")"

j() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))"; }

cat > "$OUT" <<EOF
{
  "host":          $(hostname              | j),
  "kernel":        $(uname -srv            | j),
  "os":            $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '"' | j),
  "cpu_model":     $(lscpu | grep -m1 "Model name:" | sed 's/.*: *//'   | j),
  "cpu_sockets":   $(lscpu | grep -m1 "Socket"      | awk '{print $NF}' | j),
  "cpu_cores":     $(lscpu | grep -m1 "Core(s) per socket:" | awk '{print $NF}' | j),
  "cpu_threads":   $(nproc | j),
  "cpu_freq_mhz":  $(lscpu | grep -m1 "CPU MHz:" | awk '{print $NF}' | j),
  "cpu_flags":     $(grep -m1 flags /proc/cpuinfo | sed 's/^flags.*: //' | j),
  "ram_total_gb":  $(free -g | awk '/^Mem/{print $2}' | j),
  "ram_speed":     $(sudo dmidecode -t memory 2>/dev/null | grep -m1 "Speed:" | sed 's/.*: *//' | j),
  "numa_nodes":    $(lscpu | grep -m1 "NUMA node(s):" | awk '{print $NF}' | j),
  "gpu_summary":   $(nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,compute_cap --format=csv,noheader | j),
  "cuda_version":  $(nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+' | j),
  "nvidia_driver": $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | j),
  "disk_root":     $(df -h / | tail -1 | awk '{print $2,"used",$3,"avail",$4,$1}' | j),
  "disk_data":     $(df -h "${DATA_DIR:-/tmp}" 2>/dev/null | tail -1 | awk '{print $2,"used",$3,"avail",$4,$1}' | j),
  "git_commit":    $(git -C "$(dirname "$0")/.." rev-parse HEAD 2>/dev/null | j),
  "git_branch":    $(git -C "$(dirname "$0")/.." branch --show-current 2>/dev/null | j),
  "timestamp":     $(date -u +%Y-%m-%dT%H:%M:%SZ | j)
}
EOF

# Pretty print
python3 -m json.tool "$OUT" 2>/dev/null || cat "$OUT"
echo
echo "Inventory written to $OUT"
