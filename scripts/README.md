# Scripts & Tools

Developer scripts for the GPU CrocSort project. All scripts auto-detect the project root and GPU architecture.

## Quick Reference

```bash
# First time? Check your environment
./scripts/dev-env-check.sh

# Build and test
./scripts/quick-build.sh              # smart incremental build + smoke test
./scripts/quick-build.sh sm_90 test   # build for H100 and run tests
./scripts/watch-build.sh              # auto-rebuild on file changes

# Run benchmarks
./scripts/perf-compare.sh --quick     # quick perf table (3 sizes)
./scripts/perf-compare.sh --full      # full perf table (7 sizes + duplicates)
cd gpu_crocsort && make bench-quick   # benchmark suite (7 distributions)

# Monitor GPU
./scripts/gpu-monitor.sh              # live dashboard (Ctrl+C to stop)
./scripts/gpu-monitor.sh 5            # update every 5 seconds

# Git workflow
./scripts/git-smart.sh status         # enhanced status
./scripts/git-smart.sh save "msg"     # quick commit all
./scripts/git-smart.sh sync           # pull --rebase + push

# Debug CUDA
./scripts/cuda-debug.sh memcheck      # memory error check
./scripts/cuda-debug.sh profile       # nsys/nvprof profiling
./scripts/cuda-debug.sh ptx           # generate PTX assembly

# Housekeeping
./scripts/auto-format.sh              # fix whitespace/tabs in CUDA files
./scripts/project-stats.sh            # project overview
```

## Scripts

### `dev-env-check.sh`

Verifies everything needed to build and run GPU CrocSort: CUDA toolkit (>= 11.0), GPU driver, C++ compiler, make, Python + optional packages (matplotlib, numpy, pandas), git + SSH keys, disk space, system memory, and project files. Prints PASS/WARN/FAIL for each check.

```bash
./scripts/dev-env-check.sh
```

### `quick-build.sh`

Smart incremental build. Detects GPU architecture, checks if source files changed since last build, only rebuilds if needed, and runs a smoke test (1000 records) on success.

```bash
./scripts/quick-build.sh                # auto-detect arch, build main binary
./scripts/quick-build.sh sm_89          # build for L4/4090
./scripts/quick-build.sh sm_80 test     # build + run verification tests
./scripts/quick-build.sh "" experiments # build all experiments
./scripts/quick-build.sh "" clean       # clean build artifacts
```

### `watch-build.sh`

Watches CUDA source files for changes and auto-rebuilds. Uses `inotifywait` if available, falls back to polling every 3 seconds.

```bash
./scripts/watch-build.sh              # watch and rebuild main binary
./scripts/watch-build.sh experiments  # watch and rebuild experiments

# install inotify for instant rebuilds (optional)
sudo apt install inotify-tools
```

### `perf-compare.sh`

Runs GPU CrocSort at multiple data sizes and outputs a formatted comparison table plus CSV for plotting.

```bash
./scripts/perf-compare.sh --quick     # 10K, 100K, 1M records
./scripts/perf-compare.sh --full      # 1K to 10M + duplicate-heavy tests
```

Results saved to `results/perf_results_YYYYMMDD_HHMMSS.csv`.

### `gpu-monitor.sh`

Live GPU monitoring dashboard showing temperature, utilization, memory, power draw, and running processes. Updates every N seconds.

```bash
./scripts/gpu-monitor.sh              # update every 2 seconds (default)
./scripts/gpu-monitor.sh 10           # update every 10 seconds
```

### `git-smart.sh`

Git workflow helpers with enhanced output.

```bash
./scripts/git-smart.sh status         # branch info + staged/modified/untracked + recent commits
./scripts/git-smart.sh save "message" # git add -A && git commit -m "message"
./scripts/git-smart.sh sync           # git pull --rebase && git push
./scripts/git-smart.sh log            # pretty commit graph (last 15)
./scripts/git-smart.sh log 30         # last 30 commits
./scripts/git-smart.sh diff           # diff summary with stats
./scripts/git-smart.sh wip            # create WIP commit
./scripts/git-smart.sh unwip          # undo last WIP commit (keeps changes staged)
./scripts/git-smart.sh changed        # files changed since last commit
./scripts/git-smart.sh size           # repo size breakdown by directory
```

### `cuda-debug.sh`

CUDA debugging and profiling helpers.

```bash
./scripts/cuda-debug.sh memcheck                    # detect out-of-bounds, leaks
./scripts/cuda-debug.sh racecheck                   # detect race conditions
./scripts/cuda-debug.sh sanitize                     # run all sanitizer checks
./scripts/cuda-debug.sh profile                      # profile with nsys or nvprof
./scripts/cuda-debug.sh profile --num-records 1000000  # profile with custom args
./scripts/cuda-debug.sh ptx                          # generate PTX assembly for merge.cu
./scripts/cuda-debug.sh ptx sm_90                    # PTX for H100
./scripts/cuda-debug.sh occupancy                    # kernel occupancy analysis
./scripts/cuda-debug.sh errors                       # CUDA error code reference
```

### `auto-format.sh`

Cleans up CUDA source files: removes trailing whitespace, converts tabs to 4 spaces, ensures newline at end of file. Warns about lines over 120 characters.

```bash
./scripts/auto-format.sh
```

### `project-stats.sh`

Prints a full project overview: line counts by file, CUDA kernel count, git branch/commit info, GPU hardware info, build status of all binaries, and research progress (implemented vs TODO items).

```bash
./scripts/project-stats.sh
```

## Benchmark Suite

Separate from the scripts above, the benchmark suite lives in `gpu_crocsort/` and is built via make:

```bash
cd gpu_crocsort

# Build
make benchmark-suite

# Run
make bench-quick          # 3 sizes, 7 distributions
make bench-default        # 5 sizes, 7 distributions
make bench-full           # 6 sizes (up to 50M), 7 distributions

# Parse results
make parse-results        # analyze benchmark_results.csv

# Build, run, and parse in one step
make bench-and-parse
```

The benchmark tests 7 data distributions (random, sorted, reverse, nearly-sorted, all-duplicates, few-unique, zipf) and compares 2-way merge path vs 8-way shared-memory merge tree.

## Paper

The research paper draft is at `paper/gpu_crocsort.tex` (IEEE conference format).

```bash
cd paper
pdflatex gpu_crocsort.tex
bibtex gpu_crocsort
pdflatex gpu_crocsort.tex
pdflatex gpu_crocsort.tex
```
