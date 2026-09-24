# Strix Halo flash-attention tuning plan

Upstream context:
- PR #21344 added the original gfx1151 attention tile experiment.
- No upstream change in the rebase altered `fattn-tile.cuh`.

Current retained settings:
- For RDNA3.5 and `DKQ=DV=256, ncols=32`, use `nbatch_K=64` instead of the generic `nbatch_K=128` choice.
- Keep the generic configuration for `ncols=8`.
- Do not enable rocWMMA; it is slower on the target machine.

Implementation:
- Add an RDNA3.5-only override that falls through to the generic RDNA table for all other shapes.
- Use the same override in the host and HIP device dispatch paths.

Benchmark target:
- Model: `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Quality.gguf`
- Focus: single GPU token generation at long context, with attention depths of 8192 and 32768.

Log:
- Flash-attention depth baseline for the generic `DKQ=256`, `DV=256`, `ncols=8`, `nbatch_K=256` tile:
  - `tg128 @ d8192 56.31 +/- 0.27`, `tg128 @ d32768 50.03 +/- 0.17`.
- Candidate I: add an RDNA3.5 `ncols=8` override with `nbatch_K=128`.
  - Result: `tg128 @ d8192 56.22 +/- 0.28`, `tg128 @ d32768 50.09 +/- 0.17`.
  - Mixed changes of about -0.16% at 8K and +0.11% at 32K were within variance; rejected.
- Candidate J: lower the RDNA3.5 `ncols=8` override to `nbatch_K=64`.
  - Result: `tg128 @ d8192 56.08 +/- 0.27`, `tg128 @ d32768 49.91 +/- 0.15`.
  - Regressed about 0.42% at 8K and 0.25% at 32K; rejected. The generic `nbatch_K=256` configuration is retained.

The retained `ncols=32` override remains the only RDNA3.5-specific flash-attention change.

## qwen4exp sparse attention (QSA)

Model: `~/models/qwen4/Qwen3.8-Flash-Next-IQ4_NL.gguf`, 12 full attention layers with ratio 4 compression and
indexer top_k 2048, plus 36 gated delta net layers. Prefill 16384 per pass, measured with rocprofv3:

| kernel | ms/pass |
| --- | --- |
| `qsa3_attn_kernel` | 977 |
| `qsa3_rows_kernel` | 125 |
| `qsa3_merge_kernel` | 40 |
| indexer score/relu/top_k/gather | ~200 |
| `gated_delta_net_tiled_cuda` | 996 |
| `gdn_conv_direct_kernel` | 248 |

Each query scans its own union of 2048 selected cells plus 3 tail cells, so the K/V read volume is
16384 x 2051 x 256 x 2 B x 2 = 413 GB per pass across the 12 layers. At 230 GB/s that is 1.8 s; the measured
1.0-1.1 s means a good part is served from L2. This is the cost of top_k=2048 at f16 K/V, not a kernel defect.

- Retained: drop the per-element V masking from `qsa3_attn_kernel` (score masking alone is sufficient, the
  weights are exactly zero for unselected keys). `pp16384 1125 -> 1128`, `PPL 1.2047 -> 1.2041`.
- Rejected: sort the selected block ids by key before the merge (as the old lineage did). The argsort cost
  650 ms/pass and changed the attention kernel by 0 ms.
- Rejected: visible-prefix trim (bound each 512-query strip's selection to the blocks it can reach, as the old
  lineage did). The attention kernel shrank by 94 ms/pass but the extra fill/copy/top-k work ate the gain;
  two-pass total 29.07 -> 29.48 s. The union size is therefore not the remaining gap.
- Remaining gap: `qsa3_attn_kernel` is 1.95 s/pass against 1.54 s for the old lineage kernel. Masking, id order
  and union size are all measured and excluded, so the difference is inside the kernel's per-chunk work or in the
  data it is fed (the old launcher consumed graph-provided packed K/V tensors, the current one packs the cache
  inside every FA call and drives the merge from a `4*ns+3` capacity union).
- Decode does not use the sparse path at all: `ggml_cuda_flash_attn_ext_qsa_prefill_supported` requires
  `q->ne[1] >= 128`, so TG runs the dense `flash_attn_tile` over the whole KV cache. At 131072 context that is
  12 x 131072 x 512 x 2 B = 1.6 GB of KV per token plus a 0.4 GB indexer K scan for block scoring, against 25 MB
  for a 2048-cell selection.
