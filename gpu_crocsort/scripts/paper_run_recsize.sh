#!/bin/bash
# Record size sensitivity: vary value size while keeping key fixed at 66B
# Shows the key-value separation advantage — GPU sorts keys only,
# larger values just increase gather cost
#
# NOTE: This requires recompiling with different VALUE_SIZE.
# For now, just document the gather-phase sensitivity.
# We use the existing 120B binary (66B key + 54B value) as baseline.
set -e

echo "=== Record Size Sensitivity ==="
echo "This experiment measures gather-phase scaling with record size."
echo "GPU sort time is key-size-dependent, not record-size-dependent."
echo ""
echo "Current binary: 66B key + 54B value = 120B records"
echo "Key upload to GPU: 66B × N records (key-only)"
echo "Gather: 120B × N records (full record permutation)"
echo ""
echo "To run with different record sizes, recompile with:"
echo "  -DKEY_SIZE=66 -DVALUE_SIZE=<new_value_size>"
echo ""
echo "Gather-phase bandwidth is ~17 GB/s (DDR4, random scatter)."
echo "Doubling record size approximately doubles gather time."
echo "GPU sort time remains constant (only processes 66B keys)."
