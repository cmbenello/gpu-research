# First 10 minutes on the H100 box

Use this to verify everything's healthy before letting the loop run unattended.

## After `bash h100/setup.sh` finishes

```bash
# 1. Verify the bootstrap landed
cat ~/gpu-research/h100/.bootstrap_done   # should print a date
cat ~/gpu-research/h100/env.sh             # should show GPU_ARCH=sm_90 etc.

# 2. Run the dry-run validator
bash ~/gpu-research/h100/dry_run.sh        # all ✓ except maybe ANTHROPIC_API_KEY

# 3. Hand-run the smoke experiment to make sure CUDA path works
cd ~/gpu-research/gpu_crocsort
./external_sort_tpch_compact --input ~/data/lineitem_sf10.bin --runs 3 \
    | grep -E "PASS|^CSV|GB/s"
# Expect 3 PASS lines and ~5-10 GB/s on H100
```

If any of those fail, fix before starting the loop.

## After `bash h100/start_autoresearch.sh`

```bash
tmux ls           # should show "autoresearch" session
tmux attach -t autoresearch     # follow the live loop, Ctrl-b d to detach

# In another shell, watch new commits land
watch -n 30 'git -C ~/gpu-research log --oneline -5 origin/research/overnight-runs-cs-uchicago'

# Watch the queue tick off
watch -n 30 'grep -cE "^- \[" ~/gpu-research/h100/research_queue.md; \
             grep -cE "^- \[x\]" ~/gpu-research/h100/research_queue.md'
```

After 30 minutes you should see:
- At least one item ticked `[x]` in `research_queue.md`
- At least one new commit on the feature branch
- A new file under `results/h100_runs/0.1_baseline_smoke.md` (or similar)

## Failure modes + fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `setup.sh` says "No CUDA 12.x" | toolkit not installed | `apt install cuda-toolkit-12-4` |
| `setup.sh` says "nvcc: No such file" during build | env not propagating | `source ~/gpu-research/h100/env.sh` and re-run make |
| Loop says "GPU busy" every iteration | another tenant on the box | `nvidia-smi` to see PID; coordinate or wait |
| Items get stuck `[~]` | claude crashed mid-experiment | Mark as `[!]` manually, loop will move on |
| All experiments fail with `OOM` | wrong arch detected | check `$GPU_ARCH` in env.sh, force `make ARCH=sm_90` if needed |
| API errors / 429 | rate-limited | bump `INTERVAL_MIN` in start_autoresearch.sh |
| `git push` rejected | another loop pushed first | git pull --rebase, retry — should self-heal next iteration |
| No commits after 1 hour | loop logged but no progress | check `tail -50 results/h100_runs/loop_logs/iter_*.log` |

## Sanity numbers to expect on H100 (rough projections vs RTX 6000 base)

| Workload | RTX 6000 baseline | H100 projection |
|---|---|---|
| TPC-H SF10 (7.2 GB)  | 0.28 s     | ≤ 0.15 s     |
| TPC-H SF50 (36 GB)   | 1.6 s      | ≤ 0.7 s      |
| TPC-H SF100 (72 GB)  | 3.74 s     | ≤ 1.5 s      |
| TPC-H SF300 (216 GB) | OOM       | ≤ 5 s        |
| TPC-H SF500 (360 GB) | OOM       | ≤ 9 s        |

If H100 numbers are not at least 2× faster than the RTX 6000 baselines, something's wrong (probably the binary was built without `-arch=sm_90` and is JIT-compiling).

## When to step in

The loop is designed to be unattended. Step in only if:
- The first commit doesn't appear within 90 minutes
- Multiple consecutive iterations fail
- The H100 is hot or its clock is throttling (`nvidia-smi -q -d CLOCK,TEMPERATURE`)
- You want to add an experiment manually — append to `h100/research_queue.md` and commit

Otherwise: let it cook.
