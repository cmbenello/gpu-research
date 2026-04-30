#!/usr/bin/env bash
# Dry-run smoke test for the autoresearch loop. Doesn't call Claude — just
# verifies that the queue + scripts are well-formed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  H100 autoresearch dry-run"
echo "============================================================"
echo

# 1. Validate scripts are syntactically OK
for f in h100/setup.sh h100/start_autoresearch.sh h100/stop_autoresearch.sh \
         h100/prep_lincoln.sh h100/prep_roscoe.sh; do
    if bash -n "$f" 2>&1; then
        echo "  ✓ $f parses"
    else
        echo "  ✗ $f FAILED"
    fi
done

echo

# 2. Validate Python scripts parse
for f in h100/gen_tpch_fast.py baseline_runner.py gen_weekly_figures.py; do
    if python3 -c "import ast; ast.parse(open('$f').read())" 2>&1; then
        echo "  ✓ $f parses"
    else
        echo "  ✗ $f FAILED"
    fi
done

echo

# 3. Show what the loop would pick first
echo "── First 10 unticked items in research_queue.md ──"
grep -E "^- \[ \]" h100/research_queue.md | head -10
echo

# 4. Count queue items by tier
echo "── Queue stats ──"
total=$(grep -cE "^- \[" h100/research_queue.md || true)
done=$(grep -cE "^- \[x\]" h100/research_queue.md || true)
todo=$(grep -cE "^- \[ \]" h100/research_queue.md || true)
inprog=$(grep -cE "^- \[~\]" h100/research_queue.md || true)
fail=$(grep -cE "^- \[!\]" h100/research_queue.md || true)
echo "  Total: $total | Done: $done | InProgress: $inprog | Failed: $fail | Todo: $todo"
echo

# 5. Show what env vars are needed
echo "── Required environment ──"
echo "  ANTHROPIC_API_KEY: $([ -n "${ANTHROPIC_API_KEY:-}" ] && echo set || echo MISSING)"
echo "  Claude credentials: $([ -f "$HOME/.claude/.credentials.json" ] && echo present || echo none)"
echo "  Repo branch: $(git branch --show-current)"
echo

# 6. Validate iterate_prompt is non-empty + has all critical sections
prompt="h100/iterate_prompt.md"
echo "── iterate_prompt.md sanity ──"
for keyword in "research_queue.md" "Mark it" "60 minutes" "git push" "research/overnight"; do
    if grep -q "$keyword" "$prompt"; then
        echo "  ✓ contains: $keyword"
    else
        echo "  ✗ MISSING: $keyword"
    fi
done

echo
echo "============================================================"
echo "  Dry-run complete. If all ✓ above, the harness should work."
echo "  Real loop start:  bash h100/start_autoresearch.sh"
echo "============================================================"
