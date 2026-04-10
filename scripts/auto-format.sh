#!/bin/bash
# auto-format.sh — Format CUDA source files consistently
# Fixes common formatting issues: trailing whitespace, tabs vs spaces,
# missing newlines at EOF, long lines warning

set -e
cd "$(dirname "$0")/.."

BOLD='\033[1m'
GREEN='\033[32m'
YELLOW='\033[33m'
DIM='\033[2m'
RESET='\033[0m'

fixes=0
warnings=0

echo -e "${BOLD}Auto-Format CUDA Sources${RESET}"
echo ""

for f in $(find gpu_crocsort -name '*.cu' -o -name '*.cuh' | sort); do
    file_fixes=0

    # Trailing whitespace
    if grep -qP '\s+$' "$f" 2>/dev/null; then
        sed -i 's/[[:space:]]*$//' "$f"
        ((file_fixes++))
    fi

    # No newline at end of file
    if [ -s "$f" ] && [ "$(tail -c1 "$f" | wc -l)" -eq 0 ]; then
        echo "" >> "$f"
        ((file_fixes++))
    fi

    # Tabs to spaces (4 spaces per tab)
    if grep -qP '\t' "$f" 2>/dev/null; then
        sed -i 's/\t/    /g' "$f"
        ((file_fixes++))
    fi

    if [ "$file_fixes" -gt 0 ]; then
        echo -e "  ${GREEN}fixed${RESET}  $f ($file_fixes issues)"
        fixes=$((fixes + file_fixes))
    fi

    # Long lines warning (>120 chars)
    long_lines=$(awk 'length > 120' "$f" | wc -l)
    if [ "$long_lines" -gt 0 ]; then
        echo -e "  ${YELLOW}warn${RESET}   $f ($long_lines lines > 120 chars)"
        ((warnings++))
    fi
done

echo ""
if [ "$fixes" -gt 0 ]; then
    echo -e "${GREEN}Fixed $fixes issues${RESET}"
else
    echo -e "${GREEN}All files clean${RESET}"
fi
if [ "$warnings" -gt 0 ]; then
    echo -e "${YELLOW}$warnings files with long lines${RESET}"
fi
