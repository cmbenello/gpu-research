# CrocSort weekly progress — week of 2026-04-23

## TL;DR

1. Stood up the sort on two new GPUs (P5000 16 GB on roscoe, RTX 2080 8 GB on lincoln) in addition to the existing RTX 6000 24 GB box. Verified end-to-end on TPC-H SF10–SF50.
2. Confirmed the memory envelope is the binding constraint, not throughput: each GPU caps at a different TPC-H scale because the merge-phase arena scales with total record count.
3. Quantified the compression headroom: FOR + bit-packing on TPC-H lineitem sort key adds 3.36× on top of the existing compact-key scan. GPU decode of the codec runs at ~500 GB/s, ~40× above the PCIe ceiling, so the codec is essentially free.
4. Got DuckDB head-to-head numbers on the same machines: **2.9× to 29× speedup** over DuckDB across SF10–SF50, with the gap widening sharply when the dataset crosses RAM thresholds (DuckDB falls off a cliff, GPU continues to scale linearly).

## Experiment results from this week

### 1. Memory envelope per GPU (TPC-H lineitem ORDER BY)

| GPU | Memory | Largest SF that fits | Wall time | Throughput |
|---|---|---|---|---|
| RTX 6000 | 24 GB | SF100 (72 GB) | 3.74 s | 19.8 GB/s |
| P5000 | 16 GB | SF50 (36 GB) | 7.72 s | 4.7 GB/s |
| RTX 2080 | 8 GB | SF20 (14 GB) | 2.76 s | 5.2 GB/s |

The cap on each GPU is set by the merge-phase arena, which is `num_records × 24 bytes` (sort keys + permutation + CUB scratch). For SF100 that's ~14 GB just for the arena, so anything below 24 GB cannot complete SF100 with the current code path — verified by a budget sweep on the P5000 that pushed Phase 1 through but OOM'd in Phase 2 at every budget setting tested (0.50, 0.40, 0.30, 0.25, 0.20).

This is exactly the constraint the compression direction is meant to relax.

### 2. DuckDB head-to-head (apples to apples per machine)

DuckDB ORDER BY on identical 13-column TPC-H sort key, best of 3 warm runs:

| Machine | SF | DuckDB | CrocSort | **Speedup** |
|---|---|---|---|---|
| lincoln (RTX 2080 8 GB) | SF10 | 8.41 s | 1.81 s | **4.6×** |
| lincoln | SF20 | 21.74 s | 2.76 s | **7.9×** |
| roscoe (P5000 16 GB) | SF10 | 7.81 s | 2.66 s | **2.9×** |
| roscoe | SF50 | 223.64 s | 7.72 s | **29×** |

The SF50 number is the headline. DuckDB took ~3.7 minutes with high run-to-run variance (224, 248, 267 s) suggesting external sort-to-disk kicked in. CrocSort sorted the same 36 GB in 7.72 s. Once the dataset crosses RAM thresholds, CPU sort hits a wall while GPU sort scales linearly.

### 3. Compression headroom on TPC-H sort key

Per-column FOR + bit-packing ratios from `scripts/a3_codec_ratios.py`:

| Column | Raw | FOR+bitpack | Ratio |
|---|---|---|---|
| l_returnflag | 1 | 0.625 | 1.6× |
| l_linestatus | 1 | 0.5 | 2.0× |
| l_shipdate / l_commitdate / l_receiptdate | 4 | 1.5 | 2.67× |
| l_extendedprice | 8 | 3.0 | 2.67× |
| l_discount | 8 | 0.5 | **16×** |
| l_tax | 8 | 0.5 | **16×** |
| l_quantity | 8 | 1.625 | 4.9× |
| l_orderkey | 8 | 3.25 | 2.46× |
| l_partkey | 4 | 2.625 | 1.52× |
| l_suppkey | 4 | 2.125 | 1.88× |
| l_linenumber | 4 | 0.375 | 10.7× |
| **Total** | **66** | **~19.6** | **3.36×** |

Compact-key scan (drop invariant bytes) already gets us 88 → 32. Layering FOR + bit-packing on top projects to ~10 byte keys end-to-end.

### 4. GPU codec throughput is essentially free

| Codec | Width | Decode throughput |
|---|---|---|
| FOR | 1 byte | 466 GB/s |
| FOR | 4 byte | 568 GB/s |
| Bit-pack | 8 bits/value | 451 GB/s |
| Bit-pack | 24 bits/value | 562 GB/s |

PCIe 3.0 caps at 12 GB/s. Decode is 40-50× faster than data can arrive, so a "compress on host, decode on device" pipeline costs almost nothing.

### 5. Direct sort on smaller keys

| Key width | 100 M record sort time | Throughput |
|---|---|---|
| 32 B | 21.48 ms | 37.3 GB/s |
| 16 B | 12.45 ms | **64.3 GB/s** |
| 8 B | 6.56 ms | **122.0 GB/s** |

Smaller keys are faster to sort because each radix pass works on one byte at a time. Compression compounds: smaller keys move faster across PCIe and sort faster on device.

### 6. K-way CPU merge cost grows fast in K

| K | Throughput | Slowdown vs K=2 |
|---|---|---|
| 2 | 172 Mrows/s | 1.00× |
| 4 | 194 Mrows/s | 0.89× |
| 8 | 153 Mrows/s | 1.12× |
| 16 | 71 Mrows/s | **2.42×** |

This is the cost the GPU sample-sort path is designed to remove. At K=16 the CPU merge is the dominant phase.

## Next steps

### Immediate (next 1–2 weeks)

1. **Wire FOR + bit-packing into the actual sort pipeline.** Building blocks exist (encoder in Python, decoder in CUDA, both verified). Need to integrate end-to-end so PCIe traffic actually drops.
2. **Validate the envelope-extension hypothesis on small GPUs.** With FOR-compressed keys, can the 2080 sort SF40? Can the P5000 sort SF100? If yes, this is the strongest single argument for the compression direction.
3. **Replace CPU K-way merge with GPU sample-sort.** Sample 1024 splitters, partition input on GPU, radix-sort each partition independently. No merge phase at all.

### Medium term (next month)

4. **NYC Taxi and other real-world datasets.** TPC-H is synthetic. Validating on messier real data (taxi, postgres logs, BTC blockchain) tells us whether the codec choices generalize.
5. **PostgreSQL CREATE INDEX integration.** `tuplesort_performsort()` is a clean integration point; PostgreSQL already byte-normalizes keys.
6. **DuckDB ORDER BY operator.** Earlier attempt failed because of column→row serialization overhead; the compression path may change the calculus by reducing the bytes to serialize.

### Longer term (next quarter)

7. **GH200 evaluation.** NVLink-C2C at 900 GB/s eliminates the PCIe bottleneck. Projecting ~5× over current numbers. Need cluster access.
8. **Multi-GPU scaling via NVLink.** Once single-GPU is solid, partition-then-merge across multiple GPUs.
9. **Submission target: VLDB 2027 or DaMoN 2027.** Story arc: byte-comparable order-preserving compression as a new design point for GPU sorting, verified end-to-end with 4–29× speedups over DuckDB on commodity hardware.

## Risks worth flagging

- **NVIDIA absorption.** If the compression idea works cleanly, it could be folded into CUB / cuDF as a 6-month engineering project, eating the research contribution. Mitigation: focus on the codec design space and the accelerator-aware merge story, not just "we made it faster."
- **Encoding overhead.** CrocSort already spends 55% of wall time in CPU-side key encoding. Adding compression has to be very cheap or it breaks the model. This forces us toward lightweight codes (FOR, bit-pack), which constrains the design space but is a manageable constraint.
- **GH200 / PCIe 5/6 erosion.** Faster interconnects reduce the bandwidth pressure that motivates compression. The work needs to argue why it still matters in 2-3 years (HBM bandwidth, energy efficiency, OOM datasets that remain PCIe-bound on cheaper hardware).

## Related work informing direction

- Stehle & Jacobsen, SIGMOD 2017: established that GPU radix sort is memory-bandwidth-bound.
- Nicholson, Chasialis, Boffa, Ailamaki, DaMoN 2025: showed compression widens the bandwidth envelope for GPU queries on out-of-memory datasets. Did not target sort specifically.
- Zhang et al., HOPE (SIGMOD 2020): order-preserving encoder for in-memory search trees. Byte-comparable output.
- Liu et al., Mostly Order Preserving Dictionaries (ICDE 2019): UChicago, same ecosystem as Chien.
- IEICE 2017: FPGA sorter that combines sorting networks with bit-packing — direct prior art for the architectural pattern, different substrate.

Full survey in [`results/related_accelerators.md`](related_accelerators.md). All experiment scripts in [`gpu_crocsort/scripts/`](../gpu_crocsort/scripts/).
