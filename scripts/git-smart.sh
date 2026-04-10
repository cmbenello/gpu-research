#!/bin/bash
# git-smart.sh — Git workflow helpers for GPU CrocSort
# Usage: ./git-smart.sh <command> [args]
#
# Commands:
#   status    — Enhanced git status with branch info and recent commits
#   save      — Quick save: stage all, commit with auto-message
#   sync      — Pull, rebase, push
#   log       — Pretty commit log
#   diff      — Show diff summary with stats
#   wip       — Create a work-in-progress commit
#   unwip     — Undo the last WIP commit (keeps changes staged)
#   changed   — List files changed since last commit
#   size      — Show repo size breakdown

set -e
cd "$(dirname "$0")/.."

BOLD='\033[1m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
CYAN='\033[36m'
DIM='\033[2m'
RESET='\033[0m'

cmd=${1:-status}
shift 2>/dev/null || true

case "$cmd" in
    status|s)
        echo -e "${BOLD}Branch:${RESET} $(git branch --show-current)"
        echo ""

        # Ahead/behind
        upstream=$(git rev-parse --abbrev-ref @{upstream} 2>/dev/null || echo "")
        if [ -n "$upstream" ]; then
            ahead=$(git rev-list --count @{upstream}..HEAD 2>/dev/null || echo 0)
            behind=$(git rev-list --count HEAD..@{upstream} 2>/dev/null || echo 0)
            if [ "$ahead" -gt 0 ] || [ "$behind" -gt 0 ]; then
                echo -e "${YELLOW}  ahead: $ahead, behind: $behind${RESET} (vs $upstream)"
            else
                echo -e "  ${GREEN}Up to date${RESET} with $upstream"
            fi
            echo ""
        fi

        # Status grouped
        staged=$(git diff --cached --name-only 2>/dev/null)
        modified=$(git diff --name-only 2>/dev/null)
        untracked=$(git ls-files --others --exclude-standard 2>/dev/null)

        if [ -n "$staged" ]; then
            echo -e "${GREEN}Staged:${RESET}"
            echo "$staged" | while read f; do echo "  + $f"; done
            echo ""
        fi
        if [ -n "$modified" ]; then
            echo -e "${YELLOW}Modified:${RESET}"
            echo "$modified" | while read f; do echo "  ~ $f"; done
            echo ""
        fi
        if [ -n "$untracked" ]; then
            echo -e "${RED}Untracked:${RESET}"
            echo "$untracked" | while read f; do echo "  ? $f"; done
            echo ""
        fi
        if [ -z "$staged" ] && [ -z "$modified" ] && [ -z "$untracked" ]; then
            echo -e "  ${GREEN}Working tree clean${RESET}"
            echo ""
        fi

        # Recent commits
        echo -e "${BOLD}Recent:${RESET}"
        git log --oneline -5 --format="  %C(yellow)%h%C(reset) %s %C(dim)(%cr)%C(reset)"
        ;;

    save)
        msg="${*:-Auto-save $(date '+%Y-%m-%d %H:%M')}"
        git add -A
        git commit -m "$msg"
        echo -e "${GREEN}Saved:${RESET} $msg"
        ;;

    sync)
        echo "Pulling..."
        git pull --rebase
        echo "Pushing..."
        git push
        echo -e "${GREEN}Synced${RESET}"
        ;;

    log|l)
        count=${1:-15}
        git log --oneline -"$count" --format="%C(yellow)%h%C(reset) %C(bold)%s%C(reset)%n    %C(dim)%an, %cr%C(reset)" --graph
        ;;

    diff|d)
        echo -e "${BOLD}Changes:${RESET}"
        git diff --stat
        echo ""
        echo -e "${BOLD}Staged:${RESET}"
        git diff --cached --stat
        ;;

    wip)
        git add -A
        git commit -m "WIP: $(date '+%Y-%m-%d %H:%M')"
        echo -e "${YELLOW}WIP commit created${RESET} (use 'git-smart.sh unwip' to undo)"
        ;;

    unwip)
        last_msg=$(git log -1 --format="%s")
        if [[ "$last_msg" == WIP:* ]]; then
            git reset --soft HEAD~1
            echo -e "${GREEN}WIP undone${RESET} — changes are staged"
        else
            echo -e "${RED}Last commit is not a WIP:${RESET} $last_msg"
            exit 1
        fi
        ;;

    changed|c)
        echo -e "${BOLD}Files changed since last commit:${RESET}"
        git diff --name-status HEAD~1 2>/dev/null || git diff --name-status HEAD
        ;;

    size)
        echo -e "${BOLD}Repo size:${RESET}"
        total=$(du -sh . 2>/dev/null | cut -f1)
        git_size=$(du -sh .git 2>/dev/null | cut -f1)
        echo "  Total:     $total"
        echo "  .git:      $git_size"
        echo ""
        echo -e "${BOLD}By directory:${RESET}"
        du -sh */ .* 2>/dev/null | sort -rh | head -10 | while read size dir; do
            printf "  %-10s %s\n" "$size" "$dir"
        done
        ;;

    *)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  status (s)   Enhanced git status"
        echo "  save [msg]   Quick commit all changes"
        echo "  sync         Pull rebase + push"
        echo "  log (l) [n]  Pretty log (default: 15)"
        echo "  diff (d)     Diff summary"
        echo "  wip          WIP commit"
        echo "  unwip        Undo WIP commit"
        echo "  changed (c)  Files changed since last commit"
        echo "  size         Repo size breakdown"
        ;;
esac
