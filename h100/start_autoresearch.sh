#!/usr/bin/env bash
# h100/start_autoresearch.sh — kick off the autoresearch loop in tmux.
#
# Each iteration:
#   1. Checks .queue_complete and bails if research is done
#   2. Invokes Claude Code in headless (--print) mode with the iterate_prompt
#   3. Sleeps INTERVAL_MIN minutes (default 5) before the next iteration
#
# The loop runs forever in tmux. Stop it with ./stop_autoresearch.sh.
#
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/gpu-research}"
cd "$WORK_DIR"
# shellcheck disable=SC1091
source h100/env.sh

# ── Configuration ────────────────────────────────────────────────────────────
SESSION="${SESSION:-autoresearch}"
INTERVAL_MIN="${INTERVAL_MIN:-5}"
LOG_DIR="${RESULTS_DIR}/loop_logs"
mkdir -p "$LOG_DIR"

# Sanity checks
[ -f h100/.bootstrap_done ] || { echo "Run h100/setup.sh first."; exit 1; }
command -v claude >/dev/null || { echo "claude CLI not on PATH. Run setup.sh."; exit 1; }
[ -n "${ANTHROPIC_API_KEY:-}" ] || \
    [ -f "$HOME/.claude/.credentials.json" ] || \
    { echo "Auth missing: set ANTHROPIC_API_KEY or run 'claude login' first."; exit 1; }

# Kill any existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Loop body — written to a temp script so tmux can re-invoke it ────────────
cat > "$WORK_DIR/h100/.loop_body.sh" <<'LOOP_EOF'
#!/usr/bin/env bash
set -uo pipefail
WORK_DIR="${WORK_DIR:-$HOME/gpu-research}"
cd "$WORK_DIR"
# shellcheck disable=SC1091
source h100/env.sh

LOG_DIR="${RESULTS_DIR}/loop_logs"
INTERVAL_MIN="${INTERVAL_MIN:-5}"
ITER=0

# Trap so Ctrl-C cleanly exits the loop
trap 'echo "[loop] caught signal — exiting"; exit 0' INT TERM

while true; do
    ITER=$((ITER + 1))
    NOW=$(date '+%Y-%m-%d_%H%M%S')
    LOG="$LOG_DIR/iter_${NOW}.log"

    # Check if research is complete
    if [ -f h100/.queue_complete ]; then
        echo "[loop] queue complete — sleeping 1 hour" | tee -a "$LOG"
        sleep 3600
        continue
    fi

    echo "════════════════════════════════════════════════════════════════"
    echo " iteration $ITER — $(date)"
    echo " log: $LOG"
    echo "════════════════════════════════════════════════════════════════"

    # Pull latest before iterating (in case another loop or human pushed)
    git pull --ff-only --rebase=false 2>&1 | tail -2 | tee -a "$LOG"

    # Read iterate prompt
    PROMPT=$(cat h100/iterate_prompt.md)

    # Run claude in headless mode. --dangerously-skip-permissions lets it run
    # bash/edit/git without prompting, which is required for unattended ops.
    # It writes results to the file system; we just stream stdout to the log.
    {
        echo "── prompt sent at $(date) ──"
        echo
        timeout 5400 claude \
            --print \
            --dangerously-skip-permissions \
            "$PROMPT" 2>&1 || echo "[loop] claude exited non-zero (or timed out at 90 min)"
        echo
        echo "── iteration ended at $(date) ──"
    } | tee -a "$LOG"

    # Brief breathing room (cooldown + cache TTL)
    echo "[loop] sleeping ${INTERVAL_MIN} min before next iteration..."
    sleep $((INTERVAL_MIN * 60))
done
LOOP_EOF
chmod +x "$WORK_DIR/h100/.loop_body.sh"

# ── Spawn tmux session ───────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" \
    "WORK_DIR=$WORK_DIR INTERVAL_MIN=$INTERVAL_MIN bash $WORK_DIR/h100/.loop_body.sh"

sleep 2
tmux ls
echo
echo "════════════════════════════════════════════════════════════════"
echo "  AUTORESEARCH STARTED — session: $SESSION"
echo "════════════════════════════════════════════════════════════════"
echo
echo "Watch with:    tmux attach -t $SESSION   (then Ctrl-b d to detach)"
echo "Tail logs:     tail -f $LOG_DIR/iter_*.log"
echo "Stop with:     bash $WORK_DIR/h100/stop_autoresearch.sh"
echo "Queue:         $WORK_DIR/h100/research_queue.md"
echo "Results:       $RESULTS_DIR/"
echo
