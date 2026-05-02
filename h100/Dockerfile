# Tier 19.1 — Reproducible Docker image for the gpu_crocsort sort + autoresearch.
# Builds on top of the official NVIDIA CUDA 12.4 dev image so nvcc + drivers are present.
#
# Build:  docker build -t gpu-crocsort -f h100/Dockerfile .
# Run:    docker run --gpus all --rm -v $(pwd)/data:/data gpu-crocsort \
#             ./external_sort_tpch_compact --input /data/lineitem_sf10.bin --runs 3
#
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        build-essential git tmux htop numactl python3 python3-pip \
        curl wget jq parallel pigz vim && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --quiet --no-cache-dir \
    duckdb polars pyarrow numpy matplotlib pandas tabulate

# tpchgen-cli (Rust binary for fast TPC-H gen)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable >/dev/null && \
    . /root/.cargo/env && \
    cargo install tpchgen-cli >/dev/null && \
    cp /root/.cargo/bin/tpchgen-cli /usr/local/bin/

# Build gpu_crocsort
WORKDIR /repo
COPY . /repo
RUN cd gpu_crocsort && \
    make ARCH=sm_90 && \
    make ARCH=sm_90 external-sort-tpch-compact && \
    make ARCH=sm_90 external-sort-tpch && \
    make ARCH=sm_90 external-sort

WORKDIR /repo/gpu_crocsort
ENTRYPOINT ["bash", "-c"]
CMD ["./external_sort_tpch_compact --help || true; echo 'Container ready. Mount data at /data and pass --input /data/lineitem_sfN.bin'"]
