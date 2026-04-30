# Autoresearch loop — single iteration prompt

This prompt is sent to Claude Code on each loop iteration. It is intentionally
self-contained: nothing is assumed from prior iterations except the file system
state.

---

You are running on an H100 box that's been bootstrapped via `h100/setup.sh`.
Source `h100/env.sh` to set CUDA paths if you need to build anything.

## Your job, this iteration

1. **Read** `h100/research_queue.md`. Pick the **highest-priority unticked experiment** (`[ ]`).
   - Skip experiments marked `[~]`, `[x]`, or `[!]`.
   - Tier order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7.
   - Within a tier, top-to-bottom.

2. **Mark it `[~]`** in the queue with the current timestamp, commit immediately
   (so concurrent loops don't double-pick).

3. **Execute it**:
   - Use existing scripts under `gpu_crocsort/scripts/` whenever possible.
   - Build with `make ARCH=$GPU_ARCH external-sort-tpch-compact` if needed.
   - For sort runs: `./external_sort_tpch_compact --input $DATA_DIR/<file> --runs 3` and grep for
     the `^CSV` line + `Gathered` + `PASS` markers.
   - **Time-cap each experiment at 60 minutes.** If it's not done by then, write
     what you have, mark `[!]`, and move on.
   - **Disk-cap each gen at 80% of `df $DATA_DIR`.** If a generate-step would
     exceed that, skip with reason and move on.
   - **GPU-OOM** → record the failing line + free memory, mark `[!]`, move on.

4. **Write findings** to `results/h100_runs/<exp_id>_<short_label>.md` with:
   - Date + duration + GPU + scale
   - Command lines run
   - Raw numbers (CSV line + key timings)
   - 2-3 sentence interpretation
   - Anything that surprised you

5. **Update the queue**:
   - Tick the experiment as `[x]` with a relative path to the writeup.
   - If findings suggest a follow-up experiment, **add it to the queue under
     "Running notes"** with a new ID like `N.1`, `N.2`...

6. **Commit and push**:
   ```
   git add -A
   git commit -m "h100: <exp_id> — <one-line summary>"
   git push origin research/overnight-runs-cs-uchicago
   ```

7. **Stop.** Don't try to do a second experiment in the same iteration. The
   wrapper will re-invoke you for the next one.

## Hard rules

- Never `git push` to `main`. Always to `research/overnight-runs-cs-uchicago`.
- Never delete data files unless the queue item explicitly says to.
- Never `rm -rf` anything under `$WORK_DIR` other than build artifacts.
- If `nvidia-smi` shows another process using >1 GB of GPU memory, **wait
  60 seconds and try again once**, then if still busy, log "GPU busy" and exit
  cleanly (the wrapper will retry).
- Always set `--runs 3` for sort experiments so we get warm-cache numbers.
- Write CSV outputs to `$RESULTS_DIR/<exp_id>.csv` if the experiment generates
  tabular data (so the chart-regenerator can find them later).

## When the queue is empty

If every Tier 0–6 experiment is `[x]` or `[!]`:
- Run a final pass over Tier 7 (paper/report items)
- Append a line to `h100/.queue_complete` with the timestamp
- Exit 0 — the wrapper will sleep instead of looping forever.

## Failure recovery

If your last commit was for an experiment marked `[~]` and it's been more than
2 hours since the iteration started, that experiment was probably orphaned by a
previous loop crash. Mark it `[!]` with note "orphaned" and continue.
