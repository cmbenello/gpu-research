#!/usr/bin/env python3
"""Generate sorted/reverse/nearly-sorted variants of a binary input file.

Usage: python3 gen_sorted_variants.py INPUT_FILE OUTPUT_DIR

Records are 120 bytes each (66B key + 54B value). The script produces:
  OUTPUT_DIR/sorted.bin           — sorted by 66B key (lexicographic)
  OUTPUT_DIR/reverse.bin          — reverse of sorted order
  OUTPUT_DIR/nearly_sorted_99.bin — sorted, then 1% of records displaced
  OUTPUT_DIR/nearly_sorted_95.bin — sorted, then 5% of records displaced

"Displaced" means: pick k% of indices uniformly at random, remove those
records from the array, then re-insert each at a random position.

Seed=42 for reproducibility. Uses numpy for performance.
"""

import sys
import os
import time
import numpy as np

RECORD_SIZE = 120
KEY_SIZE = 66

def read_records(path):
    """Read binary file into (N, 120) uint8 array, chunked to limit peak RAM."""
    file_size = os.path.getsize(path)
    if file_size % RECORD_SIZE != 0:
        print(f"ERROR: file size {file_size} is not a multiple of {RECORD_SIZE}")
        sys.exit(1)
    nrec = file_size // RECORD_SIZE
    print(f"Reading {nrec:,} records ({file_size / 1e9:.2f} GB) from {path}")
    t0 = time.time()
    data = np.fromfile(path, dtype=np.uint8).reshape(nrec, RECORD_SIZE)
    elapsed = time.time() - t0
    print(f"  Read complete in {elapsed:.1f}s")
    return data


def sort_records(data):
    """Return index array that sorts records by their 66B key lexicographically."""
    nrec = data.shape[0]
    print(f"Sorting {nrec:,} records by 66B key (lexicographic)...")
    t0 = time.time()

    # Convert 66-byte keys into 9 big-endian uint64 columns (8 × 8B + 1 × 2B
    # padded to 8B).  np.lexsort on 9 columns is much faster than argsort on
    # a structured dtype with 66 uint8 fields.
    keys = data[:, :KEY_SIZE].copy()  # contiguous (N, 66)

    cols = []
    for i in range(8):
        # bytes i*8 .. i*8+7 → big-endian uint64
        chunk = keys[:, i*8:i*8+8].copy()
        # Convert big-endian bytes to uint64: view as big-endian u8
        col = np.zeros(nrec, dtype=np.uint64)
        for b in range(8):
            col = col | (keys[:, i*8+b].astype(np.uint64) << np.uint64(56 - b*8))
        cols.append(col)

    # Last 2 bytes (64-65) → padded uint64
    col_last = keys[:, 64].astype(np.uint64) << np.uint64(56)
    col_last |= keys[:, 65].astype(np.uint64) << np.uint64(48)
    cols.append(col_last)

    print(f"  Key columns built in {time.time() - t0:.1f}s")

    # np.lexsort sorts by last key first, so pass columns in REVERSE order
    # (most significant = cols[0] should be last argument)
    order = np.lexsort(cols[::-1])

    elapsed = time.time() - t0
    print(f"  Sort complete in {elapsed:.1f}s")
    return order


def write_records(data, indices, path):
    """Write records in the given index order to path, chunked for large arrays."""
    nrec = len(indices)
    print(f"Writing {nrec:,} records to {path}")
    t0 = time.time()

    CHUNK = 5_000_000
    with open(path, 'wb') as f:
        written = 0
        while written < nrec:
            end = min(written + CHUNK, nrec)
            chunk_idx = indices[written:end]
            f.write(data[chunk_idx].tobytes())
            written = end
            if written % 10_000_000 == 0 or written == nrec:
                elapsed = time.time() - t0
                print(f"  {written / 1e6:.1f}M / {nrec / 1e6:.1f}M written "
                      f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    size_gb = os.path.getsize(path) / 1e9
    print(f"  Done: {size_gb:.2f} GB in {elapsed:.1f}s")


def displace_records(sorted_indices, frac, rng):
    """Displace frac of records: remove them, re-insert at random positions."""
    nrec = len(sorted_indices)
    k = int(nrec * frac)
    print(f"  Displacing {k:,} records ({frac*100:.0f}% of {nrec:,})...")

    # Pick k unique indices to displace
    displaced_pos = rng.choice(nrec, size=k, replace=False)
    displaced_pos.sort()  # sort so we can remove in order

    # Separate displaced records from the rest
    mask = np.ones(nrec, dtype=bool)
    mask[displaced_pos] = False
    remaining = sorted_indices[mask]
    displaced_vals = sorted_indices[displaced_pos]

    # Shuffle the displaced records themselves so insertion order is random
    rng.shuffle(displaced_vals)

    # Insert each displaced record at a random position in the growing array
    # For efficiency, compute all insertion points at once then use np.insert
    # sequentially would be O(k*n); instead, batch:
    # - Pick k insertion points in [0, len(remaining)+i] for i in 0..k-1
    #   We approximate by picking insertion points in [0, nrec) uniformly
    #   and sorting them, then using a single vectorized insert.
    n_remaining = len(remaining)
    insert_positions = rng.integers(0, n_remaining + 1, size=k)
    insert_positions.sort()

    # Build result: merge remaining and displaced by insertion positions
    result = np.empty(nrec, dtype=sorted_indices.dtype)
    # Offset insertion positions to account for prior inserts
    adjusted = insert_positions + np.arange(k)

    # Place remaining records first, then insert displaced
    # Use a mask-based approach for O(n) construction
    insert_mask = np.zeros(nrec, dtype=bool)
    insert_mask[adjusted] = True

    result[insert_mask] = displaced_vals
    result[~insert_mask] = remaining

    return result


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 gen_sorted_variants.py INPUT_FILE OUTPUT_DIR")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isfile(input_path):
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    t_total = time.time()

    # 1. Read input
    data = read_records(input_path)
    nrec = data.shape[0]

    # 2. Sort by 66B key
    sorted_idx = sort_records(data)

    # 3. Write sorted
    sorted_path = os.path.join(output_dir, "sorted.bin")
    write_records(data, sorted_idx, sorted_path)

    # 4. Write reverse
    reverse_path = os.path.join(output_dir, "reverse.bin")
    reverse_idx = sorted_idx[::-1].copy()
    print(f"Reversing sorted order...")
    write_records(data, reverse_idx, reverse_path)

    # 5. Nearly sorted (99% sorted, 1% displaced)
    rng = np.random.default_rng(seed=42)
    print(f"\n--- Nearly sorted (99%) ---")
    ns99_idx = displace_records(sorted_idx.copy(), 0.01, rng)
    ns99_path = os.path.join(output_dir, "nearly_sorted_99.bin")
    write_records(data, ns99_idx, ns99_path)

    # 6. Nearly sorted (95% sorted, 5% displaced)
    rng = np.random.default_rng(seed=42)
    print(f"\n--- Nearly sorted (95%) ---")
    ns95_idx = displace_records(sorted_idx.copy(), 0.05, rng)
    ns95_path = os.path.join(output_dir, "nearly_sorted_95.bin")
    write_records(data, ns95_idx, ns95_path)

    elapsed = time.time() - t_total
    print(f"\nAll variants written in {elapsed:.1f}s:")
    for p in [sorted_path, reverse_path, ns99_path, ns95_path]:
        sz = os.path.getsize(p) / 1e9
        print(f"  {sz:.2f} GB  {p}")


if __name__ == "__main__":
    main()
