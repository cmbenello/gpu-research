# GPU CrocSort — Makefile
# Build: make
# Run:   ./gpu_crocsort --num-records 1000000 --verify
# All experiments: make experiments
# Clean: make clean

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 --expt-relaxed-constexpr --extended-lambda -lineinfo
INCLUDES = -Iinclude

# Auto-detect the actual GPU's compute capability. Cross-arch JIT (e.g. sm_80
# PTX → sm_75 SASS) silently produces no-op kernels on Turing in our config,
# corrupting sort output without raising an error. Always build native SASS.
# Override with: make ARCH=sm_86
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifneq ($(DETECTED_CC),)
ARCH ?= sm_$(DETECTED_CC)
else
ARCH ?= sm_80
endif
NVCC_FLAGS += -arch=$(ARCH)

# Source files
SRCS = src/main.cu src/run_generation.cu src/merge.cu src/host_sort.cu
HDRS = include/record.cuh include/ovc.cuh include/loser_tree.cuh

TARGET = gpu_crocsort

# ── Main build ──────────────────────────────────────────────────────

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)

debug: NVCC_FLAGS = -G -g -std=c++17 --expt-relaxed-constexpr -arch=$(ARCH)
debug: $(TARGET)

# ── Tests ───────────────────────────────────────────────────────────

test: $(TARGET)
	./$(TARGET) --num-records 10000 --verify

test-medium: $(TARGET)
	./$(TARGET) --num-records 1000000 --verify

# ── Benchmarks ──────────────────────────────────────────────────────

bench: $(TARGET)
	./$(TARGET) --num-records 10000000

bench-dup: $(TARGET)
	./$(TARGET) --num-records 10000000 --duplicates

bench-large: $(TARGET)
	./$(TARGET) --num-records 100000000

# ── Benchmark Suite ────────────────────────────────────────────────

BENCH_SUITE_SRCS = benchmark_suite.cu src/run_generation.cu src/merge.cu

benchmark-suite: $(BENCH_SUITE_SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(BENCH_SUITE_SRCS) -o benchmark_suite

bench-quick: benchmark-suite
	./benchmark_suite --quick

bench-full: benchmark-suite
	./benchmark_suite --full

bench-default: benchmark-suite
	./benchmark_suite

parse-results:
	python3 parse_results.py benchmark_results.csv

bench-and-parse: benchmark-suite
	./benchmark_suite
	python3 parse_results.py benchmark_results.csv

# ── Experiments ─────────────────────────────────────────────────────

# Bottleneck analysis (run FIRST — tells you what to optimize)
bottleneck: experiments/bottleneck_bench.cu
	$(NVCC) $(NVCC_FLAGS) experiments/bottleneck_bench.cu -o bottleneck_bench

bottleneck-nocub: experiments/bottleneck_bench.cu
	$(NVCC) $(NVCC_FLAGS) -DNO_CUB experiments/bottleneck_bench.cu -o bottleneck_bench

# CUB radix sort vs merge sort crossover analysis
radix-vs-merge: experiments/radix_vs_merge.cu
	$(NVCC) $(NVCC_FLAGS) experiments/radix_vs_merge.cu -o radix_vs_merge

# Memory layout comparison (AoS vs SoA vs Key+Index)
layout-bench: experiments/memory_layout_bench.cu
	$(NVCC) $(NVCC_FLAGS) experiments/memory_layout_bench.cu -o layout_bench

# CPU vs GPU comparison
cpu-vs-gpu: experiments/cpu_baseline.cu $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) experiments/cpu_baseline.cu src/run_generation.cu src/merge.cu src/host_sort.cu -o cpu_vs_gpu

# Key-value separation benchmark
kv-bench: experiments/kv_separate_bench.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) experiments/kv_separate_bench.cu -o kv_bench

# Profiling instrumentation (self-contained: has own main + merge orchestration)
profile: experiments/profile_sort.cu src/run_generation.cu src/merge.cu $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) experiments/profile_sort.cu src/run_generation.cu src/merge.cu -o profile_sort

# Sample-based partitioning test
sample-part: experiments/sample_partition.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) experiments/sample_partition.cu -o sample_partition

# Edge case test suite
test-edge: experiments/test_edge_cases.cu src/run_generation.cu src/merge.cu src/host_sort.cu $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) experiments/test_edge_cases.cu src/run_generation.cu src/merge.cu src/host_sort.cu -o test_edge_cases
	./test_edge_cases

# External sort (data > GPU memory) — GenSort format (10B key, 100B record)
external-sort: src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DEXTERNAL_SORT_MAIN src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu -o external_sort

# External sort — TPC-H lineitem format (88B key, 120B record)
external-sort-tpch: src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DEXTERNAL_SORT_MAIN -DKEY_SIZE=88 -DVALUE_SIZE=32 src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu -o external_sort_tpch

# External sort — TPC-H lineitem, compact varying-byte key (26 varying bytes → 32B key, 4 LSD passes)
external-sort-tpch-compact: src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DEXTERNAL_SORT_MAIN -DKEY_SIZE=66 -DVALUE_SIZE=54 -DUSE_COMPACT_KEY src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu -o external_sort_tpch_compact

# External sort — TPC-H lineitem, compact key (66B significant key, same 120B record)
# The 88B key has 22B zero padding (bytes 66-87). Using KEY_SIZE=66 skips
# sorting those zeros: 9 LSD passes instead of 11 (18% fewer). Same sort order
# since zero bytes can't affect ordering.
external-sort-tpch66: src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DEXTERNAL_SORT_MAIN -DKEY_SIZE=66 -DVALUE_SIZE=54 src/external_sort.cu src/run_generation.cu src/merge.cu src/host_sort.cu -o external_sort_tpch66

# ── Build all experiments ───────────────────────────────────────────

experiments: bottleneck-nocub radix-vs-merge layout-bench
	@echo ""
	@echo "Built all experiments. Run them:"
	@echo "  ./bottleneck_bench 10000000"
	@echo "  ./radix_vs_merge"
	@echo "  ./layout_bench"

# ── Run everything ──────────────────────────────────────────────────

run-all: $(TARGET) experiments
	@echo "═══ Running complete benchmark suite ═══"
	./bottleneck_bench 1000000
	./radix_vs_merge
	./layout_bench
	./$(TARGET) --num-records 1000000 --verify
	./$(TARGET) --num-records 10000000

# ── Clean ───────────────────────────────────────────────────────────

clean:
	rm -f $(TARGET) bottleneck_bench radix_vs_merge layout_bench cpu_vs_gpu \
	      kv_bench profile_sort sample_partition test_edge_cases external_sort \
	      benchmark_suite

.PHONY: all debug test test-medium bench bench-dup bench-large \
        benchmark-suite bench-quick bench-full bench-default \
        parse-results bench-and-parse \
        bottleneck bottleneck-nocub radix-vs-merge layout-bench \
        cpu-vs-gpu kv-bench profile test-edge external-sort \
        experiments run-all clean
