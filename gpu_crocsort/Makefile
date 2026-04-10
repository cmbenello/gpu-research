# GPU CrocSort — Makefile
# Build: make
# Run:   ./gpu_crocsort --num-records 1000000 --verify
# Clean: make clean

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 --expt-relaxed-constexpr -lineinfo
INCLUDES = -Iinclude

# Auto-detect GPU architecture (default to sm_80 for A100)
# Override with: make ARCH=sm_86
ARCH ?= sm_80
NVCC_FLAGS += -arch=$(ARCH)

# Source files
SRCS = src/main.cu src/run_generation.cu src/merge.cu src/host_sort.cu
HDRS = include/record.cuh include/ovc.cuh include/loser_tree.cuh

TARGET = gpu_crocsort

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)

# Debug build (with device debug info, no optimization)
debug: NVCC_FLAGS = -G -g -std=c++17 --expt-relaxed-constexpr -arch=$(ARCH)
debug: $(TARGET)

# Quick test: 10K records with verification
test: $(TARGET)
	./$(TARGET) --num-records 10000 --verify

# Medium test: 1M records with verification
test-medium: $(TARGET)
	./$(TARGET) --num-records 1000000 --verify

# Benchmark: 10M records, no verification
bench: $(TARGET)
	./$(TARGET) --num-records 10000000

# Benchmark with duplicates (tests OVC duplicate shortcut)
bench-dup: $(TARGET)
	./$(TARGET) --num-records 10000000 --duplicates

# Large benchmark: 100M records (~10GB)
bench-large: $(TARGET)
	./$(TARGET) --num-records 100000000

# Bottleneck benchmark (run this FIRST to know what to optimize)
bottleneck: experiments/bottleneck_bench.cu
	$(NVCC) $(NVCC_FLAGS) experiments/bottleneck_bench.cu -o bottleneck_bench

# If CUB headers not found, compile without CUB baseline
bottleneck-nocub: experiments/bottleneck_bench.cu
	$(NVCC) $(NVCC_FLAGS) -DNO_CUB experiments/bottleneck_bench.cu -o bottleneck_bench

# Run bottleneck analysis
run-bottleneck: bottleneck
	./bottleneck_bench 10000000

clean:
	rm -f $(TARGET) bottleneck_bench

.PHONY: all debug test test-medium bench bench-dup bench-large bottleneck bottleneck-nocub run-bottleneck clean
