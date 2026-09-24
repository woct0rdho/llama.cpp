# qwen4exp optimization plan for gfx1151 (forward, batch 1, ROCm)

Scope: Qwen3.8-Flash-Next on gfx1151 (Radeon 8060S, 40 CU, wave32, 64 KB LDS/CU, LPDDR5-8000 256-bit).
Forward kernels only, single sequence (batch 1). ROCm only. Context up to 131072 in production.

Models and quant mix (MiB of weights, active per token unless marked):

| tensor group | IQ4_NL | APEX-I-Nano | UD-IQ1_S |
| --- | --- | --- | --- |
| experts (all 512) | IQ4_NL 64800 | IQ4_NL 21600, IQ1_M 7875, IQ1_S 5156, IQ2_XXS 3094, IQ2_XS 694 | IQ4_NL 21600, IQ1_S 10625, IQ2_XXS 5775 |
| experts (10 used) | IQ4_NL 1266 | IQ4_NL 422, IQ1_M 154, IQ1_S 101, IQ2_XXS 60, IQ2_XS 14 | IQ4_NL 422, IQ1_S 208, IQ2_XXS 113 |
| linear attn (gdn) | IQ4_NL 308 | Q5_K 371, Q6_K 7 | Q6_K 443 |
| full attn + indexer | IQ4_NL 1012, BF16 38, Q8_0 32 | Q5_K 1238, BF16 38, Q6_K 25 | Q5_K 1231, BF16 38, Q6_K 33 |
| hyper-connections | IQ4_NL 340 | Q8_0 641 | Q8_0 638 |
| ffn misc / ple | F32 241, IQ4_NL 127, Q8_0 33 | F32 241, Q6_K 149, Q8_0 80 | F32 241, Q8_0 113, Q5_K 101 |
| embed + lm head | Q6_K 645, IQ4_NL 341 | Q6_K 497, IQ4_XS 322, Q5_K 124 | Q4_K 682, Q5_K 124 |

Current state (pp512 / pp16384 / tg128, PM4 on, r=3/1/3, at `f30f4f9af`): IQ4_NL 960 / 1285 / 34.9,
APEX 745 / 1127 / 33.8, UD 785 / 1094 / 35.8. PPL at `-c 8192 -ub 8192 --chunks 1` on the docs corpus:
3.4125 / 3.6290 / 3.6320. Long context: IQ4_NL tg128 is 19.0 t/s at d=65536 against 17.5 t/s with the
sparse decode path disabled. Kernel-level numbers are in `hc_combine_norm_bf16.md`,
`qsa_sparse_decode.md` and `qsa_pp_tg_margins.md`.

## 1. Roofline (measured on this machine)

Machine ceilings measured with a purpose-built HIP microbenchmark, GPU clock sustained at 2.63-2.9 GHz:

| resource | measured | note |
| --- | --- | --- |
| DRAM read | 230 GB/s | 256 theoretical; this is the hard wall for TG |
| DRAM copy (r+w) | 205 GB/s | |
| L2 read (4 MB set) | 654 GB/s | |
| bf16/f16 WMMA -> f32 | 51.2 / 49.7 TFLOP/s | 95% of the 53.9 the ISA formula gives at 2.63 GHz |
| int8 WMMA -> i32 | 50.3 TFLOP/s | the prefill MMQ engine; i8 is not faster than bf16 here |
| int8 dot4 (dp4a) | 28.7 TFLOP/s | m=1 MMVQ path only |
| bf16 dot2 | 19.6 TFLOP/s | |
| fp32 FMA | 11.1 TFLOP/s | single issue; clang does not form VOPD pairs for plain FMA |
| VALU ops next to WMMA | costly per op | measured with 2 extra ops/WMMA: 51.2 -> 45.4 TFLOP/s (-11%); with 8: 40.2 (-21%) |

Workload (per token / per 16384-token pass, IQ4_NL geometry, same shapes in all three models):

| item | value |
| --- | --- |
| active params/token | 6.671 B (2.36 B routed experts, 3.49 B dense, 0.82 B lm head) |
| FLOPs | 13.34 GFLOP/token; 191.6 TFLOP/pass (lm head counted once per pass) |
| weight bytes | 4.25 GB/token; 70.9 GB/pass (all experts are read with 16384 tokens) |
| residual stream | hc=4 -> 10240 wide, F32, 671 MB per full-tensor touch at 16384 tokens |

Roofline results:
- PP16384: GEMM engine bound 5.5-6.3 s (191.6 TFLOP at the best measured engine rate 30.5-35 TFLOP/s), activation-traffic bound ~3.9 s (~0.9 TB at 230 GB/s, largely not overlappable with compute in separate kernels), serial bound ~8.3 s = 1970 t/s, overlapped bound ~6.3 s = 2600 t/s, stated WMMA-peak floor 3.74 s = 4400 t/s. Measured 14.6 s = 1121 t/s (57% of the serial bound). Intensity is 190 FLOP/byte, i.e. right at the machine knee (223 FLOP/byte), so both sides must move.
- TG128: 4.25 GB/token at 230 GB/s = 18.5 ms = 54 t/s hard bound. Measured 27.9 ms = 36 t/s (66%). The weight-reading kernels already run at ~200 GB/s, so the recoverable part is the ~9.4 ms/token that is not weight traffic (copies, elementwise, quantize, F32 router GEMMs, launch count).
- TG at long context: the dense path reads 12 layers x nk x 512 B x 2 = 1.6 GB/token of KV at 131072, plus
  0.4 GB of indexer K scan, on top of 4.25 GB of weights. The sparse decode path landed in `2f3a77aec` and
  takes over from about twelve cache cells per selected cell: it reads the 2051 selected cells (25 MB/token)
  instead of the cache. Measured tg128: 26.9 t/s at d=32768 against 26.6 dense, 19.0 at d=65536 against 17.5,
  and the kernel is flat in depth (148 us per row over twelve heads). What is left at depth is the indexer
  scan and the non-weight kernel time, not the attention.

Why the GEMM engine is at 26-30 TFLOP/s and not 51: a timing experiment with the IQ4_NL dequant replaced by a trivial store gained only 11-16% (dense 27.2 -> 31.4, routed 17.4 -> 19.3 TFLOP/s). So the dequant ALU is a minor cost; the loss is operand movement and scheduling (LDS issue, fragment reload per k-step, VGPR pressure/occupancy). This matches external evidence in ~/ComfyUI-FeatherOps/docs: in a comparable gfx1151 WMMA kernel LDS was 36.8% of issued instructions, LDS cannot dual-issue with VALU, and conversion VALU hides in LDS stall slots.

## 2. Where the time actually goes now

PP16384, IQ4_NL, one pass, profiled with graphs disabled so that every dispatch is visible. Kernel
time sums to 25.4 s against 12.65 s of wall time, because the hipBLASLt and copy work overlaps the
main stream; shares below are of the kernel sum.

| kernel | ms | share | launches |
| --- | ---: | ---: | ---: |
| mmb_routed_glu_kernel<64,128> (MoE gate/up) | 3494 | 13.7% | 94 |
| mmb_dense_kernel<128,256,64,64,0> | 3256 | 12.8% | 168 |
| qsa3_attn_kernel (sparse attention) | 1928 | 7.6% | 24 |
| mmb_routed_kernel<128,128> (MoE down) | 1867 | 7.3% | 94 |
| gated_delta_net_tiled_cuda | 1844 | 7.3% | 72 |
| mmb_dense_kernel<128,128,32,64,0> | 1770 | 7.0% | 498 |
| hc_combine_norm_f32 | 1721 | 6.8% | 190 |
| hc_gate_mix_kernel (gate GEMM + mix) | 1064 | 4.2% | 190 |
| mmb_dense_kernel<384,64,96,32,0> (HC down\|inject) | 946 | 3.7% | 190 |
| qsa3_rows_kernel (selection validate + sort) | 818 | 3.2% | 24 |
| gdn_conv_direct_kernel | 612 | 2.4% | 72 |
| mmb_f32split_kernel<128,128,32,64,true> | 576 | 2.3% | 118 |
| mmb_dense_kernel<128,256,64,64,2> (bf16 weights) | 556 | 2.2% | 24 |
| rms_rows_f32<true> | 427 | 1.7% | 72 |

Grouped: quantized GEMMs (mmb dense/routed/glu/tall/f32split) ~13 s, QSA ~3.2 s (attention 1.9, rows
0.8, merge and pack the rest), GDN ~2.5 s, hyper-connections ~2.8 s (combine+norm 1.7, gate+mix 1.1),
norms ~0.9 s, elementwise, top-k, moe reduction, rope, ple and copies the rest.

TG128, IQ4_NL, per token, graphs disabled (28.3 ms of kernel time in a 37.6 ms token; with PM4 graphs
on the token is 27.7 ms, so ~9 ms of it is launch overhead that the graph hides):

| kernel | ms/token | share | launches/token |
| --- | ---: | ---: | ---: |
| mul_mat_vec_q<IQ4_NL,1> plain | 12.2 | 43.2% | 580 |
| mul_mat_vec_q<IQ4_NL,1> split/ids | 5.4 | 19.2% | 97 |
| mul_mat_vec_q<Q6_K,1> (lm head, attn out) | 3.5 | 12.4% | 13 |
| mul_mat_vec_f (F32 router GEMMs) | 1.4 | 4.9% | 97 |
| quantize_q8_1 | 0.86 | 3.0% | 719 |
| k_get_rows_float_vec (indexer pool gather) | 0.58 | 2.0% | 36 |
| mul_mat_vec_q<Q8_0,1> | 0.40 | 1.4% | 28 |
| topk_moe_cuda | 0.39 | 1.4% | 48 |
| gated_delta_net_cuda | 0.35 | 1.2% | 36 |
| scale_f32 (HC control chain) | 0.34 | 1.2% | 379 |
| hc_combine_norm_f32 | 0.31 | 1.1% | 96 |
| mul_mat_vec_f<bf16> | 0.24 | 0.9% | 24 |
| k_bin_bcast<op_add> | 0.23 | 0.8% | 177 |

The weight kernels are 23.2 ms of the 28.3 ms kernel sum; everything else is 5.1 ms, of which the two
largest single items are the quantize pass (719 launches, one per expert per matrix, which P0-4
describes hoisting) and the indexer pool gather plus its norm/rope rebuild, which is query independent
work redone every token (`qsa_pp_tg_margins.md`).

The two mixed models run the same kernel families with their own quant mix, so the IQ4_NL table above
is representative of the shape of the pass and P0-1/P0-2 record what happened to the MMQ and f32split
shares (an earlier partial trace put MMQ Q5_K at 17.7% of APEX pp512 and mmb_f32split at 24.3% of UD
pp512; both were reduced by `e8ba63209` and by the F32 projection work).

Conclusion: for the two mixed models a large share of the pass goes through paths that the IQ4_NL-optimized code does not cover (MMQ Q5_K/IQ1_S/IQ2_XXS, hipBLASLt Q6_K, mmb_f32split for F32), and the f32split share suggests the F32 router GEMM deserves its own look.

## 3. Work items, largest margin first

### P0-1. Bring the quant types of the mixed models onto the fast prefill path

Status: done for the quick gates (`c3aca8f3e` Q5_K, `1f3a25943` IQ4_XS, `e8ba63209` Q6_K). The three
codebook types stay on MMQ and their further optimization belongs to P1-6.

Landed, MMB against MMQ at `test-backend-ops perf -o MMB_PERF` (TFLOP/s, routed `n_mats 512,
n_used 10, m 640, k 2560` at the given token count, dense `m 2560, k 2560`):

| type | routed n=512 | routed n=16384 | dense n=512 | dense n=16384 | gate |
| --- | --- | --- | --- | --- | --- |
| Q5_K | 4.60 vs 5.25 | 17.43 vs 14.85 | 26.25 vs 24.28 | 31.22 vs 24.96 | dense always, routed from 2048 |
| IQ4_XS | 5.88 (MMQ) | 17.62 vs 15.49 | 30.77 vs 25.67 | 27.40 vs 26.16 | dense always, routed from 2048 |
| Q6_K | 3.64 vs 4.35 | 17.62 vs 10.98 | 27.74 vs 13.80 | 30.90 vs 17.93 | dense from 4096, routed from 2048 |

End to end: APEX-I-Nano pp16384 939 -> 1013 t/s, UD-IQ1_S 931 -> 981, IQ4_NL flat. UD pp512 needed the
Q6_K dense threshold (see `mmb_q6_k.md`): always-on measured 658 against 691 on the old path, from
4096 tokens it is 681-690 with the pp16384 gain kept.

Not done, with the reason: IQ1_S, IQ2_XXS and IQ2_XS sit at 23.5/24.6/20.4 TFLOP/s on MMQ against
10.3/7.9/7.6 on the current generic MMB path, i.e. a new recipe has to be 2-3x better than the
generic one just to reach parity, and their bf16 dequant needs a per-8-weight codebook grid plus a
sign plane. Their remaining share is 175 MiB of active experts (APEX) and 321 MiB (UD).

### P0-1 follow-up: the codebook trio is deferred on purpose

An IQ1_S recipe was written and reverted (not landed). What it established, so the next attempt does
not repeat it:
- The recipe shape is right: values are `d * (2*((qh >> 12) & 7) + 1)` times `(grid_nibble + delta)`
  with `delta = -1 +/- 0.125` per 32-weight sub-block, one `iq1s_grid_gpu` lookup per eight weights,
  and one fmaf plus one `mmb_pack2` per weight. With the staged build it measured 7.47/13.73/16.81
  TFLOP/s routed (512/2048/16384) and 30.68/27.35 dense, i.e. it would clear MMQ's 6.96/14.62 and
  23.5/24.7 on three of the five shapes. Those numbers are invalid because correctness failed
  (`ERR = 3.3` on MMB_QUANT), but they say the *cost model* is fine and the bug is a mapping detail.
- The element mapping, scale, delta and pack were each verified in isolation: with an all-zero grid
  index the recipe matches ggml's own `dequantize_iq1_s` plus the `mmb_quant_slice` mapping over
  256 elements, 0/256 differences. What is not verified is the grid index and the nibble order,
  because that degenerate case makes every nibble zero.
- In-kernel bisect: my staging + the generic loader passes 6/6, so the staging loads are fine; the
  recipe's own LDS write is what fails. But note that this bisect only proves the *loads* do not
  break anything, since the generic loader reads the weights itself.
- Harness pitfall that cost the most time: a standalone HIP test that includes `ggml-common.h` picks
  a *different* `iq1s_grid_gpu` variant for host and device (the host saw entry 1829 = 0x068b2e60, the
  device 0x21102012), so host/device comparisons there are meaningless. Debug this type inside the
  real build (print the instantiated `WTYPE` and a marker in the branch, or diff the two LDS rows
  in-kernel), not in a copied harness.
- Value: IQ1_S is 208 MiB of UD's 743 MiB active expert weights and 101 MiB of APEX's 751 MiB, all on
  the *routed* path, so the model-level ceiling is about +1-2% pp16384 for those two models and
  nothing for pp512 (the routed gate keeps MMQ below 2048 tokens anyway). The same recipe family
  would cover IQ2_XXS (113/60 MiB) and IQ2_XS, which have an extra per-8 sign LUT and a 4-bit scale.
  Deferred: P0-3 and P0-4 have larger margins for the same effort.

Original analysis:

Evidence: MMB vs MMQ measured per type at the qwen4exp shapes: IQ4_NL and Q8_0 win on MMB (1.03-1.86x), IQ1_M wins big (2.0x/5.2x), but IQ1_S, IQ2_XXS, IQ2_XS, IQ4_XS, Q4_K, Q5_K, Q6_K measure 0.18-0.70x on MMB, so `mmb_quant_type()` keeps them on MMQ (Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1 dequant paths exist in mmb-quant.cuh but are not enabled either). Consequence: APEX runs 175 of 751 MiB of active expert weights plus ~1.7 GiB of dense weights on MMQ/hipBLASLt; UD runs 321 MiB of experts plus ~2.6 GiB of dense (Q5_K 1456, Q6_K 479, Q4_K head 682) the same way.

Work:
- Rewrite the dequant recipes that lose on MMB: the codebook types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_*) need the 256-entry grid lookup hoisted out of the per-weight path (per k-tile, decode once into LDS in the target block layout, as `mmb_load_quant_tile` already does for some types), and the 12-bit/6-bit packed layouts of IQ1_S/IQ2_XXS need a nibble-plan that v_perm_b32 can serve. K-quants (Q4_K/Q5_K/Q6_K) need the 6-bit scale/min unpack amortized over the 256-element superblock.
- Re-measure each type with the existing harness (`test-backend-ops perf -o MMB_PERF -p "type_a=<t>"`) and flip the gate only where MMB beats MMQ by more than a few percent on the exact shapes (m=512, m=16384 dense, routed m=640/n=16384).
- If a type cannot be won on MMB, tune its MMQ path instead (MMQ is int8 WMMA here, ceiling 50.3 TFLOP/s; measured Q6_K MMQ is 13.8-17.9 dense and Q5_K/Q4_K are 24-25, so there is room).
Target: APEX/UD pp16384 from 928/930 to at least IQ4_NL-class per-shape rates, i.e. +10-20% pass level; the per-type kernel benchmark is the gate.

### P0-2. Fix the F32 projections of all three models

Status: done (`19ff6cb28`). The F32 work split into three groups, and the tiny ones dominated.

The F32 MMB ops in one pass are the hyper-connection injects ([10240, 4], 96 per pass), the GDN
alpha/beta ([2560, 48], 72) and the routers ([2560, 512], 48) - 226 in total. The first two ran in a
128-row tile where 75-97% of every tile is empty, with a 4-block grid at pp512, and they were about
9% of the UD-IQ1_S pp512 pass. A 16-row tile with all eight waves split over tokens keeps the weight
traffic and cuts the matrix work by up to 8x. UD-IQ1_S pp512 685 -> 726-732 t/s (+6.8%), pp16384
unchanged at 971; the numerics are unchanged (same splits, same k order).

The router itself measures 5.65 TFLOP/s at `[2560, 512] x T=512` (235 us per call, 11 ms per pass,
1.4%) and 12.99 TFLOP/s at T=16384 (3.3 ms per call, 159 ms per pass, about 1%). It is left as is:
the remaining ideas (a bf16 pair shadow to skip the in-kernel split, a 64x64 tile for small T) are
each worth well under 1% of the pass, and a narrower tile doubles the weight re-reads. See
`mmb_tiny_m.md` for the tile that landed.

Original analysis:

Evidence: the router (`ffn_gate_inp.weight`, [2560, 512] F32) is 241 MiB in every model, read in full for every m-tile. On IQ4_NL pp16384 it costs 286 ms/pass (7.2 TFLOP/s effective) through `mmb_f32split`; in the UD pp512 trace the same kernel family is the single largest entry (24%). TG pays 1.65 ms/token for 132 launches of F32/bf16 GEMMs.

Work:
- Add a load-time shadow copy of F32 matmul weights in a WMMA-friendly format (bf16 pair split once, or Q8_0), sized 60-120 MiB for the whole model instead of 241 MiB, and keep it behind a flag.
- Fuse the router GEMM with the top-k/routing step (the MoE top-k already exists as its own kernel) so the 512-wide output never round-trips.
- Gate on PPL and on routing-decision agreement against the F32 baseline (the router is a selection, small weight perturbations are visible).
Target: -0.25 s/pass at pp16384 and -1.5 ms/token at TG.

### P0-2 follow-up: the small ops that are left are not worth fusing

Measured in a pp16384 pass (rocprofv3 kernel trace on IQ4_NL, two passes averaged):

- `topk_moe_cuda`, 48 launches, 11 ms (0.08%). The router output is already consumed by a single
  fused op (`GGML_OP_TOPK_MOE`), so "fuse the router GEMM with top-k" exists upstream. The
  round-trip it avoids is [512, T] F32, 33.5 MB written and read, about 0.3 ms per pass. Nothing
  there.
- `mmb_f32split_kernel`, 59 launches, 312 ms (2.2%), of which 48 are the router `[2560, 512]` F32
  GEMM and the rest the hc injects. The router's weight traffic is 4 m-tiles x 128 t-tiles x 1.31 MB
  = 671 MB per router per pass, i.e. 32 GB per pass and 140 ms at 230 GB/s, which is why it measures
  13 TFLOP/s and not the 28 the F32 path can reach. A BF16 *pair* shadow does not cut those bytes at
  all (two BF16 is the same four bytes); it only removes the in-kernel split. A single BF16 shadow
  halves them (about 70 ms, 0.5%) and would additionally make 4 x 655 KB of A-tiles L2 resident
  against a 4 MB L2, worth up to the full 140 ms (1%). Both need the routing-agreement gate the plan
  already asks for, because the router is a selection. Verdict: 0.5-1% is not worth an accuracy
  risk, so P0-2 stays closed.
- The hc injects in that group already run the tall-M tile (`mmb_dense_kernel<384, 64, 96, 32>`, 95
  launches, 516 ms).

### P0-3. Cut F32 activation traffic in PP (hc stream, norms, converts)

Evidence: 3.9 s/pass (27% of the pass) of elementwise/norm/hc/copy work running at ~230 GB/s, i.e. already at the DRAM bound, over ~0.9 TB: hc pre/post 225 GB, 10240-wide F32 norms 134 GB, f32->bf16 converts 146 GB, generic elementwise 85 GB, copies 50 GB. All of it is avoidable traffic: the same tensors are read and written several times per layer because each step is its own kernel.

Work:
- Fuse the RMS norm into the `dsv4_hc_pre` epilogue (the mix output is immediately normalized and scaled).
- Fuse the f32->bf16 conversion into the producer epilogue (norm or hc kernel) so the GEMM reads bf16 directly; drop `mmb_cvt_f32_bf16` for those edges.
- Fuse `dsv4_hc_post` with the residual add/scale that follows it (and with the next norm if it is cheap).
- Evaluate bf16 storage for the hc residual stream (halves 225 GB) as an explicit accuracy experiment.

Target: -1.3 to -2.0 s/pass (9-14%) with no FLOP change; accuracy gate is PPL.

### P0-3 status: done

All four items landed.

- Fusing the norm into the hyper-connection chain: done, though not the way the plan first put it.
  The combine and the norm after it are one kernel now (`hc_combine_norm_f32`), one block per
  (stream, token), with the row that the norm needs staying in registers. See
  `hc_combine_norm_bf16.md`.
- Fusing the activation conversion into the producer: done earlier (`c1bcf6ffa`), the norm writes
  the BF16 copy the GEMMs read, gated at 64M elements because below that the consumer reads the
  tensor from cache and the conversion is nearly free. `pp_bf16_activation_norm.md`.
- Fusing the combine with the residual add/scale: it already was one op; what was left is the norm
  that follows, which item 1 covers.
- BF16 storage for the residual stream: done, and it costs nothing measurable in PPL. The
  precision question was settled against the reference material rather than by guessing: the
  Transformers DeepSeek V4 boundary returns the mHC stream in the incoming dtype and only fixes the
  controls, Sinkhorn state and control accumulation; the no-unsloth plan lists the mHC stream tensor
  under BF16 activation sizes and says it avoids full FP32 hidden-stream copies; the NVFP4 stack
  that took everything to 4-bit did not walk the residual stream back up. Nothing found argues
  against it.

Also done in the same area: the gate projection is fused into the stream mix (`0e1fbb7e3`), so the
gate never reaches memory.

End to end at pp16384, all with `-b 16384 -ub 16384 -p 16384 -n 0 -r 1`:

| model | session start | now | delta |
| --- | --- | --- | --- |
| IQ4_NL | 1131.9 | 1282.4 | +13.3% |
| APEX-I-Nano | 1006.9 | 1112.4 | +10.5% |
| UD-IQ1_S | 976.1 | 1089.2 | +11.6% |

pp512 is +8 to +15%, tg128 is unchanged, `test-backend-ops` stays at 1073/1073, and PPL at
`-ub 8192` on the docs corpus is 3.4151 / 3.6304 / 3.6183.

What is left here: BF16 `block_out` into the combine (one more producer change, another 671 MB per
layer), and the 250 ms per pass of conversions below the 64M element gate.

### P0-4 status: deferred to P2-7 (small margin)

The item is real but the remaining margin is per-family and small (1-2%, 0.5-1%, 0.7%, see
`p0_4_tg_overhead.md`), and measuring it needs a per-family wall-clock ablation rather than the
profiler, which does not attribute time on this APU. Moved next to P2-7 so that long context (P1-5)
is not blocked behind it.

#### Original notes: 9.4 ms of 27.9 ms is not weight traffic

The premise still holds: 27.9 ms per token against 4.254 GB of weight bytes is 152 GB/s of a 230 GB/s
ceiling, so roughly 9 ms per token is not weight traffic. The time attribution in the evidence above
is not trustworthy, though, and neither is anything derived from it: on this APU `rocprofv3
--kernel-trace` timestamps are dispatch times, not execution times (0.23% "busy" over a 26 s span
with the per-token period showing up as an idle gap). The dispatch *counts* are fine, the durations
are not.

Tried and reverted: fusing the combine's three-node control chain (288 launches per token) into the
combine kernel. It fires, it is arithmetically identical, and it changes tg128 by less than the
run-to-run spread (0.4% quiet, up to 3% warm). Reverted rather than kept unmeasured.

See `p0_4_tg_overhead.md` for the detail and for the three candidates that are left (MoE activation
quantization 1-2%, GDN conv window shift 0.5-1%, m=1 gate+mix 0.7%). The next thing this item needs
is a timing method that works here - per-family wall-clock ablation, disable one family and measure
tg128 - not another trace.

### P1-5 status: sparse decode attention done, indexer scan left

The decode kernel landed (`2f3a77aec`): it gathers K/V from the cache through the selected ids, splits
the keys into chunks so that twelve heads are not the whole grid, and merges the partial softmax
states. Kernel time is flat in depth, 148 us per query row over twelve heads at 32768 to 131072 keys
against 295 / 577 / 1147 us for the dense path, and end to end at `-n 128 -d 65536` it is 19.02 t/s
against 17.53. The dispatch picks the dense path below twelve cache cells per selected cell, which is
where the two cross. See `qsa_sparse_decode.md`.

Left in this item:
- The selection sort in the union builder (`qsa3_rows_kernel`, 818 ms per pp16384 pass, 3.2% of kernel
  time, 24 launches): it validates and bitonic-sorts 2051 ids per query row because the union/merge path
  needs sorted rows. A one-query-per-workgroup attention would consume the ids unsorted and delete this
  together with the merge and the 4x union redundancy.
- The indexer scan at depth: the score GEMM over all blocks, the relu-sum, the top-k and the gather,
  plus 0.4 GB per token of cached indexer keys at 131072. This is now the largest depth-dependent
  cost, and it is what limits what is left of TG at long context.
- Quantised K/V for the sparse path (int8/fp8): halves the 25 MB selection read and the prefill union
  traffic. The decode kernel reads the cache directly, so this needs the cache type, not the kernel,
  to change first.
- The prefill path still packs all of K/V every pass, 413 GB of union traffic at 16384 tokens.

### P1-6. GEMM structural work (start it after GGTensile in the torch-ggml-ops repo is stable)

Evidence: with the dequant removed entirely, MMB reaches 31.4 (dense) / 19.3 (routed) TFLOP/s against a 51.2 WMMA peak, so ~40% is lost to operand movement and scheduling. FeatherOps measured LDS at 36.8% of instructions in a comparable kernel and confirmed LDS cannot dual-issue.

Ideas, in order of expected value:
- Register blocking: raise WMMA work per A/B fragment load (larger TM/TN per wave, reuse the activation fragment across n-tiles) so the per-k-step LDS traffic per WMMA drops.
- LDS layout: swizzle for conflict-free d16/b128 access, and consider keeping the weight tile resident across m-tiles when an expert's rows exceed one tile (fewer dequant and LDS stores).
- Occupancy: check VGPR budget of the MMB kernels against the FeatherOps envelope (191 VGPR / 32 KB LDS gave them full occupancy); MMB carries many tile registers.
- VOPD pairing: identify instruction pairs that the ISA allows to co-issue (the manual's packed-math table pairs WMMA with DOT2) and check whether the compiler forms them.
- If hand-tuning stalls, run a Tensile/TensileLite-style search for the exact MMB shapes, as done in ~/evotensile and ~/torch-ggml-ops (the latter already generates grouped/paired MMQ forward kernels for Q2_K-Q8_0 and IQ2_S/IQ2_XXS on this ISA).
- Optional, larger: a prepacked weight shadow for the dense tensors only (2.1 GiB -> 3.5 GiB fp8 or 7 GiB bf16) that removes the dequant from the dense GEMMs. Memory cost must be checked against the 128 GB budget and the 26.8 GiB PLE table.

### P2-7. Small margins

- LM head: 0.68 GB/token (16% of TG bytes). Keep it quantized; consider a two-stage head only if it can preserve exact logits.
- `cpy_scalar_transpose` (141 ms/pass), `top_k` (77 ms), `idx_relu_sum` (74 ms), moe_weighted_reduction (214 ms), gdn conv (248 ms), rms_rows (290 ms), and the remaining `k_bin_bcast` traffic.
- QSA `qsa3_rows_kernel` reads the visibility mask per entry although the ids are pre-filtered; the mask read is needed for the dense-top-k models, so it needs an op-level flag rather than removal.
- `mul_mat_vec_q` split variant tuning for the MoE at m=1.

## 4. Measurement protocol and gates

- Exclusive jobs only (`heavy-run --`), one job at a time; always `DEBUG_HIP_GRAPH_PM4=1`.
- End-to-end acceptance: pp512 and tg128 with 3 repeats, pp16384 with 1 repeat, all three models; report t/s per model.
- Kernel-level (preferred for iteration and required for long context): a standalone bench harness for the new kernels; `test-backend-ops perf -o MMB_PERF` and `-o MMQ_PERF` for the GEMM shapes at m=512 and m=16384; context 2048/16384/32768/131072 for attention, indexer, and GDN.
- Accuracy gates per model: PPL with `llama-perplexity -c 8192 -b 8192 -ub 8192 --chunks 2` must stay within run-to-run noise of the pre-change baseline; `test-backend-ops test` must stay green for every touched op; new paths behind the existing gate functions must beat the incumbent on the exact shape or stay disabled.
- No regression: the current IQ4_NL numbers (pp512 960, pp16384 1285, tg128 34.9-36.1 across runs) are the
  floor, and PPL 3.4125 / 3.6290 / 3.6320 at `-ub 8192`; APEX/UD must not regress either. Run-to-run spread
  is about 0.4% on tg128 in a quiet machine state and up to 3% when the APU has been busy, so a TG change
  below 1% is not evidence.
- Profile attribution must run with `GGML_CUDA_DISABLE_GRAPHS=1`: with PM4 graphs captured the profiler
  sees only the host launched dispatches (84 of 3566 kernels per token) and reports dispatch times, which
  is what made the earlier TG table read low.

## 5. Suggested order

Done: P0-1 (quant coverage, `c3aca8f3e` `1f3a25943` `e8ba63209`), P0-2 (F32 projections and the router),
P0-3 (hyper-connection fusions, `0e1fbb7e3` `d5359dc92` `19ca7a316`), and the sparse decode attention of
P1-5 (`2f3a77aec`).

Open, largest margin first:
- P1-5 remainder: the QSA selection sort (0.82 s/pass, 3.2%) and the union's 4x redundant instruction
  count both disappear with a one-query-per-workgroup prefill kernel; the indexer pool rebuild at depth
  (~400 MB/token at 131072). Numbers and the plan are in `qsa_pp_tg_margins.md`.
- P1-6 (GEMM structure): the quantized GEMMs are ~13 s of the pp16384 pass at 26-30 TFLOP/s of a 51
  TFLOP/s engine, and the TG weight kernels are 22.4 ms of a 28.3 ms kernel sum. This is the largest
  remaining item by absolute time.
- P2-7 (small margins): the TG overhead, now measured at 5.9 ms/token of non-weight kernel time
  plus ~9 ms/token of launch overhead that graphs already hide (28.3 ms of kernel time in a 37.6 ms token
  with graphs off, 27.7 ms with them on). Needs per-family wall-clock ablation,
  since the profiler mis-attributes when graphs are on.
