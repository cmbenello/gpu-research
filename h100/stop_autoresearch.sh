#!/usr/bin/env bash
# Stop the autoresearch loop. Safe to run multiple times.
set -euo pipefail
SESSION="${SESSION:-autoresearch}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Stopped tmux session: $SESSION"
else
    echo "No tmux session named '$SESSION' — already stopped"
fi

# Best-effort: kill any orphaned claude processes
pkill -f "claude --print" 2>/dev/null || true
pkill -f "loop_body.sh" 2>/dev/null || true
pkill -f "external_sort_tpch_compact" 2>/dev/null || true

echo "Done. (existing experiments mid-run may need manual cleanup — check 'ps -ef | grep -E claude\|external_sort')"
