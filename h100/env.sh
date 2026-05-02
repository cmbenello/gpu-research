# Generated manually 2026-05-02 (setup.sh failed before reaching this step due
# to non-compact build error — see KNOWN_ISSUES.md). Sourced by autoresearch.
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export GPU_ARCH=sm_90
export WORK_DIR=$HOME/gpu-research
export DATA_DIR=/mnt/data
export RESULTS_DIR=$WORK_DIR/results/h100_runs
export REPO_BRANCH=h100/discoveries-2026-05-02
