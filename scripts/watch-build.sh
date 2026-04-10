#!/bin/bash
# watch-build.sh — Watch for file changes and auto-rebuild
# Usage: ./watch-build.sh [target]
# Requires: inotifywait (apt install inotify-tools)

set -e
cd "$(dirname "$0")/.."

BOLD='\033[1m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
DIM='\033[2m'
RESET='\033[0m'

TARGET=${1:-all}

if ! command -v inotifywait &>/dev/null; then
    echo -e "${YELLOW}inotifywait not found.${RESET}"
    echo "Install: sudo apt install inotify-tools"
    echo ""
    echo "Falling back to polling mode (checks every 3s)..."
    echo ""

    # Polling fallback
    last_hash=""
    while true; do
        current_hash=$(find gpu_crocsort/src gpu_crocsort/include -name '*.cu' -o -name '*.cuh' 2>/dev/null | \
                        xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
        if [ "$current_hash" != "$last_hash" ] && [ -n "$last_hash" ]; then
            echo -e "\n${YELLOW}Change detected!${RESET} Rebuilding..."
            if ./scripts/quick-build.sh "" "$TARGET" 2>&1; then
                echo -e "${GREEN}Build succeeded${RESET} at $(date +%H:%M:%S)"
            else
                echo -e "${RED}Build failed${RESET} at $(date +%H:%M:%S)"
            fi
        fi
        last_hash="$current_hash"
        sleep 3
    done
else
    echo -e "${BOLD}Watching for changes...${RESET} (target: $TARGET)"
    echo -e "${DIM}Press Ctrl+C to stop${RESET}"
    echo ""

    while true; do
        inotifywait -q -r -e modify,create,delete \
            --include '\.(cu|cuh)$' \
            gpu_crocsort/src/ gpu_crocsort/include/ 2>/dev/null

        echo -e "\n${YELLOW}Change detected!${RESET} Rebuilding..."
        sleep 0.5  # Debounce

        if ./scripts/quick-build.sh "" "$TARGET" 2>&1; then
            echo -e "${GREEN}Build succeeded${RESET} at $(date +%H:%M:%S)"
        else
            echo -e "${RED}Build failed${RESET} at $(date +%H:%M:%S)"
        fi
        echo ""
        echo -e "${DIM}Watching...${RESET}"
    done
fi
