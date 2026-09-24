# MMB tiny-M tile

Operator: the small-output-width variant of the dense MMB kernels in `ggml/src/ggml-cuda/mmb.cu`
(`mmb_dense_kernel<16, 128, 16, 16, WT>` and `mmb_f32split_kernel<16, 128, 16, 16, ...>`, dispatched
for `M <= 64`). Model independent: it applies to any weight whose output width is a handful of rows.

## Why

MMB's dense dispatch used a 128-row tile. For output widths of 4 to 48 rows that leaves 75-97% of
every tile empty while still issuing the full set of WMMA ops, and at 512 tokens the grid is a
handful of blocks for 160 SIMDs. In qwen4exp those weights are the hyper-connection injects
(`hc_attn_inject`, `hc_ffn_inject`, `[10240, 4]`, 96 ops per pass) and the GDN alpha/beta
(`ssm_alpha`, `ssm_beta`, `[2560, 48]`, 72 ops per pass) - 168 of the 226 F32 MMB ops in a pass and
about 9% of UD-IQ1_S pp512 before the fix.

## Tile

`<BM=16, BN=128, WTM=16, WTN=16>` with the standard 256 threads (8 waves):

- `WAVES_M = BM/WTM = 1`, so `wn = wave` and the eight waves split the 128 columns over tokens.
- `TM = WTM/16 = 1`, `TN = WTN/16 = 1`, so one WMMA per k-step per wave; the A and B fragment loads
  are one each per k-step, which the 64-clock WMMA easily hides.
- Grid `(ceil(M/16), ceil(T/128))`, LDS 2.5 KB for A plus 20.5 KB for B in the split kernel.

Matrix work drops by up to 8x for `M = 4` and 2.7x for `M = 48` (three 16-row tiles instead of one
128-row tile) with identical weight traffic. The math is bit-identical: the same hi/lo splits and the
same k order, only the row blocking changes.

## Measurements

F32 shapes, `test-backend-ops perf -o MMB_PERF` (m = output width, n = tokens, k = reduction):

| shape | before | after |
| --- | --- | --- |
| m=48, n=512, k=2560 | 1.34 TFLOP/s | (tiny tile) |
| m=512, n=512, k=2560 (router) | 5.65 TFLOP/s | unchanged, not tiny |
| m=512, n=16384, k=2560 (router) | 12.99 TFLOP/s | unchanged, not tiny |

The per-shape F32 numbers are noisy because these ops are latency bound; the model is the real
measurement:

| model | pp512 before | pp512 after | pp16384 before | pp16384 after |
| --- | --- | --- | --- | --- |
| UD-IQ1_S | 685 | 726-732 | 971 | 971 |
| APEX-I-Nano | 697-708 | (same dispatch) | 1013 | - |

Correctness: `test-backend-ops test -b ROCm0 -o MMB_QUANT,MUL_MAT_ID` covers `[*, 4]` and `[*, 48]`
rows for every quantized type and for F32 (added with this change): 1073/1073.

## Notes and further work (P1-6)

- At 512 tokens the inject shapes still run a 4-block grid, so they remain latency bound; splitting
  the waves differently would need the load loops parameterized by blockDim instead of the
  `MMB_NT` constant, and a 128-thread launch.
- The routers (`M = 512`) are unaffected: 11 ms per pass at pp512 and 159 ms at pp16384, about 1%.
  A narrower tile there trades occupancy for doubled weight re-reads and measured no clear win, so it
  was left alone.
