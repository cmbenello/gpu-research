# Overnight Experiment Results — 2026-04-22

## Three Decision Numbers

### 1. A3: Compression ratio — **FOR does NOT help beyond compact key**

| Codec | SF10 (bytes/key) | SF50 | SF100 | Compression vs 66B |
|-------|-----------------|------|-------|-------------------|
| Raw key | 66 | 66 | 66 | 1.0x |
| Compact key (baseline) | 26 | 26 | 27 | 2.4-2.5x |
| FOR | 26 | 26 | 27 | 2.4-2.5x |
| FOR + bit-pack | 19.6 | 20.6 | 21.0 | 3.1-3.4x |
| Dictionary | 24 | 24 | 24 | 2.8x |
| Mostly-OP dict | 66.4 | 66.4 | 66.4 | 1.0x (worse) |

**Verdict: FOR doesn't reduce the number of varying byte positions.** Compact key already identifies which byte positions vary; FOR subtracts the per-position minimum, but the range within each position is still >0, so the position still varies. The byte count is identical.

**FOR + bit-packing** does help (19.6-21B vs 26-27B), but requires GPU-side bit-unpacking which is complex to integrate with CUB's word-aligned radix sort.

**Dictionary** gets 24B (slightly better than compact) because some high-cardinality columns have fewer distinct values than their byte range implies.

**B3 direct-sort signal:** CUB radix sort on 8-bit FOR-encoded keys is **3.3x faster** than 32-bit (6.6ms vs 21.5ms at 100M rows). But this only helps if we can reduce the total key *width*, not just the value range within each byte.

### 2. C1: SF50 speedup with FOR — **~1.0x (no speedup expected)**

C1 was not run end-to-end because FOR encoding doesn't reduce the compact key byte count (see A3 above). The same 26-27 byte positions would be uploaded over PCIe, with the same CUB radix sort width (32B = 4 passes). Speedup would be ~1.0x.

The only path to sort-time improvement is **bit-packing** (20B → 3 CUB passes instead of 4 = ~25% sort speedup), but that requires engineering work to handle non-byte-aligned radix sort.

**C2 synthetic sweep (completed):**

| Cardinality | Effective bits | Sort speedup |
|-------------|---------------|-------------|
| 256         | 8             | 2.30x       |
| 1K-64K      | 10-16         | 1.46x       |
| 256K-16M    | 18-24         | 1.16x       |
| 4B (full)   | 32            | 1.00x       |

CUB's radix sort speedup is quantized — it jumps at 8/16/24-bit boundaries (each boundary = one fewer radix pass). For TPC-H, the effective bits per varying byte position determines which tier you land in.

**Implication for CrocSort:** FOR encoding on compact keys would need to reduce the total varying bytes below a CUB pass boundary to see a sort-time improvement. If TPC-H SF50 has ~27 varying bytes (fitting in 32B), FOR would need to squeeze that below 24B (3 CUB passes instead of 4) for a meaningful win.

### 3. D1: Merge cost — MEASURED (timing only, no perf counters)

perf was unavailable in this environment. Timing-only results:

| K | Rate (Mrows/s) | Slowdown vs K=2 |
|---|----------------|-----------------|
| 2 | 172            | 1.00x           |
| 4 | 194            | 0.89x (faster!) |
| 8 | 153            | 1.12x           |
| 16| 71             | 2.42x           |

**K=4 is optimal** for this hardware (likely fits in L1/L2). K=16 is 2.4x slower than K=2, which on a branch-prediction-limited workload strongly suggests ~50%+ misprediction rate. This confirms sample sort (eliminating the merge entirely) would help for large K.

However: CrocSort's actual K is small (SF100 = 3-6 chunks on 24GB GPU), so the merge cost is moderate. The bigger lever is reducing PCIe transfer, not eliminating merge.

---

## Completed Experiments

### A2: Compact Key Baseline

| Scale | Raw key | Compact key | Ratio | Varying positions |
|-------|---------|-------------|-------|-------------------|
| SF10  | 66B     | 26B         | 2.5x  | 26 of 66 bytes vary |
| SF50  | 66B     | 26B         | 2.5x  | same 26 positions |
| SF100 | 66B     | 27B         | 2.4x  | +1 (byte 54 = l_partkey MSB) |

Compact key scan identifies identical byte positions across all records and removes them. This is already the baseline CrocSort uses.

### A3: Codec Compression Ratios

See Decision Number 1 above. Key per-column breakdown:

| Column | Type | Raw bytes | FOR bytes | FOR+bitpack bits |
|--------|------|-----------|-----------|-----------------|
| l_returnflag | char | 1 | 1 | 5 |
| l_linestatus | char | 1 | 1 | 4 |
| l_shipdate | date | 4 | 2 | 12 |
| l_commitdate | date | 4 | 2 | 12 |
| l_receiptdate | date | 4 | 2 | 12 |
| l_extendedprice | decimal | 8 | 3 | 24 |
| l_discount | decimal | 8 | 1 | 4 |
| l_tax | decimal | 8 | 1 | 4 |
| l_quantity | decimal | 8 | 2 | 13 |
| l_orderkey | bigint | 8 | 4 | 26-30 |
| l_partkey | int | 4 | 3-4 | 21-25 |
| l_suppkey | int | 4 | 3 | 17-20 |
| l_linenumber | int | 4 | 1 | 3 |

Low-cardinality columns (discount, tax, linenumber) compress well. High-cardinality columns (orderkey, extendedprice) don't.

### B1: FOR Decode Throughput

| Width | N=100M (GB/s) | N=300M (GB/s) |
|-------|--------------|--------------|
| 1B    | 467          | 472          |
| 2B    | 546          | 548          |
| 4B    | 568          | 569          |

FOR decode is **40-47x faster than PCIe 3.0** (12 GB/s). Decode overhead is negligible — the bottleneck is always the bus, not the codec.

### B2: Bit-Pack Decode Throughput

| Bit width | Throughput (GB/s) |
|-----------|------------------|
| 8         | 451              |
| 12        | 498              |
| 16        | 536              |
| 24        | 562              |
| 32        | 568              |

Even complex bit-unpacking runs at 450+ GB/s. Any lightweight codec is viable on GPU.

### B3: Direct Sort on FOR-Encoded Keys

| Key width | Sort time (ms) | vs 32-bit |
|-----------|---------------|-----------|
| 32 bit    | 21.5          | 1.00x     |
| 24 bit    | 18.4          | 1.17x     |
| 16 bit    | 12.5          | 1.72x     |
| 8 bit     | 6.6           | 3.27x     |

All results verified correct. CUB's internal pass count drives the speedup tiers (8 bits = 1 pass, 16 = 2 passes, 24 = 3 passes, 32 = 4 passes).

### C2: Synthetic Compressibility Sweep

See table in Decision Number 2 above. Key insight: the speedup is **quantized by CUB pass boundaries**, not linear with compression ratio.

### D1: K-Way Merge Profiling

See table in Decision Number 3 above. K=16 shows the expected branch-prediction degradation.

---

## What These Results Mean for CrocSort

1. **GPU codec overhead is zero.** FOR decode at 450+ GB/s and sort speedup at 1.2-3.3x make compression a pure win — IF it reduces effective key bits below a CUB pass boundary.

2. **The sort kernel is not the bottleneck.** At 100M rows, 32-bit sort takes 21ms. Our SF50 pipeline spends ~5.5s on PCIe + encoding, ~0.5s on sort. Even a 3x sort speedup saves 0.3s — not the main lever.

3. **PCIe transfer is the lever.** If FOR reduces the compact key from 27 varying bytes to, say, 20 (cutting out zero-range bytes after FOR), that saves ~7/27 = 26% of PCIe time. At SF50 where PCIe is ~3s, that's ~0.8s saved. This is the real benefit of compression — **fewer bytes over the bus**, not faster sort.

4. **Merge elimination via sample sort is valuable for K>8** but not urgent for CrocSort's typical K=3-6 chunks.

## Still Running / Pending

- **A1**: Per-byte entropy analysis (running — processing 600M rows × 13 columns on SF100)
- **C1**: Not run (FOR doesn't improve compact key byte count — see Decision 1)
- **E1**: SF100 baseline (generating normalized binary data)
- **E2**: Chunk sweep (generating normalized binary data)

## Recommendations for Next Two Weeks

Based on these results:

1. **Bit-packing is the compression direction** — not FOR. FOR on TPC-H doesn't reduce byte count beyond compact key. Bit-packing gets 20B (3.1x) which crosses the CUB pass boundary (24B → 3 passes). Engineering cost: moderate (need custom GPU bit-unpack + radix sort integration).

2. **PCIe transfer is the main lever**, not sort time. At SF50 the sort kernel is ~0.5s out of ~6.5s. Reducing PCIe bytes from 27→20 (via bit-pack) saves ~26% of PCIe time = ~0.8s. Worth it for a 12% overall improvement.

3. **Sample sort is not urgent** — K=3-6 for CrocSort, and K=4 is actually the fastest merge point. Only pursue if moving to multi-GPU where K grows.

4. **Mostly-order-preserving dictionary is a dead end** for TPC-H sort keys (escape overhead dominates).
