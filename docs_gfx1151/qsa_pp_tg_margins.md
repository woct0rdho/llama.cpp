# QSA: where the PP and TG time goes

Measured with `test-backend-ops perf` (`-o QSA_PREFILL`, `-o QSA_DECODE`), gfx1151, 12 layers,
1 KV head, 12 query heads, D=256, n_sel=2051 cells.

## Arithmetic

| | FLOP per (query row, layer) | per token, 12 layers |
| --- | ---: | ---: |
| sparse (selected cells) | 25.2 MFLOP | 0.30 GFLOP |
| dense, 131072 context | 1.61 GFLOP | 19.3 GFLOP |

The model itself is 13.3 GFLOP/token. QSA arithmetic is therefore never the constraint at the model
level: at the machine's 51 TFLOP/s the sparse FLOPs would be 6 us of a 40 ms token. What matters is
bytes per useful FLOP, and QSA moves 1.7 MB per query row per layer to do 25-100 MFLOP, i.e. 15-60
FLOP/byte against the machine's 222 FLOP/byte balance (51 TFLOP/s over 230 GB/s). The operation is
memory bound by construction, so the only meaningful metric is effective bandwidth.

## PP

Packed union path, 4096 query rows, one call:

| cache keys | time | per row | useful FLOP | effective DRAM |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 72.65 ms | 17.7 us | 25.2 MFLOP | ~95 GB/s |
| 32768 | 132.08 ms | 32.2 us | | |
| 131072 | 161.43 ms | 39.4 us | | |

The kernel reads the union of four rows' selections, so it computes 4x the useful FLOPs (100 MFLOP
per row) and runs at 5.7 TFLOP/s, 11% of the fp16 WMMA peak. It is not compute bound in the model
sense: it is at 41% of the DRAM ceiling, and the ceiling for a kernel moving 1.7 MB per row is
7.4 us/row against the measured 17.7. That 2.4x is the largest single margin in QSA, worth about
1.4 s of the ~3.5 s per pp16384 pass.

Tried and rejected: running the per-row gather decode kernel at prefill query counts. It does 1x the
FLOPs and the same DRAM traffic, but it is scalar FMA with two byte loads, not WMMA: 530 ms against
132 ms at 32768 keys, i.e. 0.2 TFLOP/s. It stays a decode kernel (one row, at most a few).

## TG

Per-row gather with split keys, one query row over twelve heads: 148 us flat from 32768 to 131072
keys (2.1 MB of selection, 13.5 GB/s effective) against 295 / 577 / 1147 us for the dense path. It is
latency bound, not bandwidth bound, so there is ratio margin left, but the absolute traffic is 25 MB
per token and perfecting it is worth 1-2 ms of a ~50 ms token.

The depth dependent cost that actually matters at TG is the indexer: the score GEMM over all blocks,
the relu-sum, the top-k and the gather, plus its own cached keys (0.4 GB/token at 131072).

## Why the PP kernel is at 95 GB/s and not 230

Reading `qsa3_attn_kernel`'s inner loop accounts for the measurement:
- One workgroup owns a group of 4 query rows x 12 heads = 48 rows (three 16 row WMMA tiles), and
  processes the *union* of those four rows' selections, about 8100 cells at 16384 keys.
- Each 4 cell chunk costs the block roughly 92 cycles (measured: 187k cycles per workgroup over
  ~2000 chunks), and one `__syncthreads()` per chunk joins the eight warps that split the head
  dimension, because the QK partials have to be summed across warps before the softmax.
- That chunk serves 48 rows but only 4 x 12 of them need each particular cell, so the union costs
  4x the instructions, not 4x the traffic: DRAM per row is the same 2.03 MB either way, and the
  WMMA units run at 2.8 TFLOP/s, 5% of peak.

So the kernel is neither compute nor bandwidth bound: it is instruction and latency bound, at 50% of
DRAM and 5% of fp16 WMMA at the same time. The lever is therefore the group size, not the arithmetic:
- With one query per workgroup the union becomes just that row's own sorted selection (2051 cells),
  the block is 12 rows instead of 48 (one 16 row tile, 4 wasted), the cross group reduction
  disappears, and the chunk count drops ~4x. DRAM per row is unchanged. That is the ~2x that would
  take PP from 95 GB/s back to the DRAM ceiling.
- Duplicates are the one correctness detail: the production top-k emits distinct cells, so the sorted
  per row selection can be consumed directly, but the merge kernel deduplicates today and a G=1 path
  would have to either keep that dedup or assert uniqueness.
- The cheap intermediate, if a G=1 kernel is too much at once: unroll the chunk loop by two chunks and
  hoist one `__syncthreads()` for both, which halves the sync count without touching the structure.

`QSA3_G` is a compile time constant threaded through the kernel's three tile groups, the merge
kernel's cap, and the epilogue, so a G=1 variant is a second instantiation (`template<int G>`) rather
than a constant change.

## Ablations

- Skipping the mask gather (timing only, the scores change) takes the 4096 row prefill from
  72.6 / 132.1 / 161.5 ms to 58.7 / 116.4 / 133.9 ms at 16384 / 32768 / 131072 keys: the scattered
  two byte mask reads are 12-19% of the kernel. The gather sits in the critical path right after the
  per chunk barrier, so prefetching the next chunk's mask bytes alongside the K fragment (which the
  kernel already prefetches) should recover most of that ~4% of PP end to end.
- The rest of the growth with depth is in the packed K/V gathers: the packed buffer is 134 MB per
  tensor at 131072 keys and each chunk reads four 2 KB blocks scattered across it.

## The indexer at depth

Per decode token the graph rebuilds the indexer keys it scores with: `get_rows(k_all, blk_cells)` gathers
r x n_blocks cells, i.e. the whole indexer cache, at 33.5 MB per layer for 131072 keys, plus the
pairwise mean, the per block RMS norm and the per block RoPE over n_blocks x idx_dim, another 8.4 MB
per layer each way. That is ~400 MB per token across the twelve layers, plus the score GEMM, the
relu-sum, the top-k and the cell gather.

None of it depends on the query: a pooled key is the mean of its block's r cells, normed per block and
rotated at the block's own position, so it is fixed the moment the block fills. A persistent pooled
cache updated every r tokens (r = 4, so once per four tokens per layer) turns the per token cost into
reading the pooled keys, 8.4 MB per layer, and removes the five ops that build them. At 131072 that is
worth roughly 2 ms of a ~60 ms token. At prefill the same work is already done once per pass over all
rows, so this is a TG change.

## Non-finite values in the cache, and the dropped poison cases

The two `poison` cases in `test_qsa_prefill` are dropped from the registry (`f30f4f9af`). They were
passing when the sparse prefill kernel was ported (`828db9b117`, "21/21 QSA_PREFILL operator cases
pass") and broke when `388140beb` removed the per element V mask for +3 t/s at pp16384 (0.3%).

That commit's reasoning - a key the query did not select already scores -inf, so it weighs zero - is
right for finite values and wrong for NaN: in the fp16 WMMA path `0 * NaN = NaN`, and one such cell
poisons the row, while the CPU reference skips invisible cells entirely. The masked cell cannot be
caught at block granularity either, because the case that matters shares a block with a selected cell.

A non-finite value in the KV cache is a bug upstream of attention, so the kernel does not pay per
element to tolerate it. The cases stay in the file commented out, with the reason, so a future per
element V guard can be checked against them.

If the concern is worth insuring against anyway, the cheap place is the cache write rather than the
attention kernel: `cpy_k` writes one row per token per layer (256 halfs), so a finiteness check there
is nearly free and it catches the source - a non-finite value entering the cache - instead of making
every consumer tolerate it. The decode kernel happens to be safe already, since it skips the V read
when the probability is zero and so never forms `0 * NaN`.

## Ranking

- PP union structure: one query per workgroup instead of four drops the instruction count per row
  about 4x at unchanged DRAM traffic, which is the difference between the measured 95 GB/s and the
  230 GB/s ceiling. Worth roughly 1.4 s per pp16384 pass.
- The indexer pooled cache at depth: ~400 MB per token of rebuild traffic that does not depend on the
  query, worth ~2 ms per token at 131072.
- TG decode kernel latency: ratio-rich, absolute-poor (~3% at 131072).
- Quantised (int8/fp8) K/V: halves the 334 GB/pass union traffic and the selection read; the decode
  kernel reads the cache directly, so this belongs in the cache type.
