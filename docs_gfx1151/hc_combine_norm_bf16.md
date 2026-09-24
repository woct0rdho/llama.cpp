# HC combine + norm fused, BF16 residual stream

Change: `d5359dc92` (`ggml/src/ggml-cuda/hc-cn.cu`, `hc-cn.cuh`, `ggml-cuda.cu`)

## The two kernels it replaces

Per hyper-connection block, `GGML_OP_DSV4_HC_POST` computes
`res_new[e,c,t] = block_out[e,t] * post[c,t] + res[e,c,t]` over `[n_embd, hc, T]`, and the grouped
RMS norm that follows it (`RMS_NORM` + `MUL`, the fused `rms_norm_f32<1024, true, false>`) reads the
whole `res_new` back and writes `xn[e,c,t] = rsqrt(mean_e(res_new^2) + eps) * res_new[e,c,t] *
gamma[e,c]`. At pp16384 that is read 671 MB + write 671 MB for the combine plus read 671 MB + write
671 MB + 336 MB of BF16 for the norm, 3.7 GB per layer, 96 blocks per pass.

`hc_combine_norm_f32` puts one block on each `(stream c, token t)` pair. It reads the control weight
once, walks the 2560-wide row once to combine it (writing the new residual), reduces the row sum of
squares with `block_reduce`, and walks the row again to write the normed result, writing the BF16
slot copy on the way. The row that the norm needs never leaves the registers. With the stream in F32
the output is bit-identical to the two-kernel path, which makes the PPL check exact: 3.4178 either
way.

## BF16 residual stream

`hc-cn.cuh` carries `res_in_bf16`, `res_out_bf16` and `out_xn_bf16`. A marked residual holds BF16 in
its own (F32 sized) buffer, read and written in place; the next combine reads the same layout back.

Whether a residual may be BF16 is `ggml_cuda_hc_res16_ok()`, a pure function of the graph: it finds
the producing node, requires it to be a combine this fusion takes, and then requires every reader of
the residual - following reshape/view chains - to be either that combine's own norm or another
combine this fusion takes. A resident 4-byte-per-element stream that is only ever consumed by the
fused kernel costs 336 MB per layer instead of 671 MB on both the read and the write.

Making it a pure function matters twice: the producer and each consumer evaluate the same question
independently and must agree, and a captured HIP graph bakes the answer into the kernel arguments, so
it has to be stable across replays. Enabling it is worth another 1 to 3% on top of the F32 fusion and
costs nothing measurable in PPL (3.4151 / 3.6304 / 3.6183 for IQ4_NL / APEX-I-Nano / UD-IQ1_S at
`-ub 8192` against 3.4178 / 3.6297 / 3.6365 with F32). This matches the reference material: the
Transformers DeepSeek V4 boundary casts the mHC controls to the entry dtype and returns the collapsed
sequence in the incoming dtype, the no-unsloth plan lists the `[B,S,4,4096]` mHC stream under BF16
activation sizes with only the controls and Sinkhorn state precision fixed, and the NVFP4 inference
stack that flattened everything to 4-bit did not walk the residual stream back up.

## Two traps that took the debugging

The early write. The kernel writes the `MUL`'s tensor at the combine's position, several nodes
earlier than the graph says. The allocator reuses buffers once their contents are dead, so the
`MUL`'s buffer can hold something still live at the combine. Writing it there corrupted a tensor
nobody suspected, and the `RMS_NORM`/`MUL` running afterwards did not restore it, because they are
skipped. `ggml_backend_cuda_graph_optimize` now registers the allocation dependencies for the pair
with the same `add_alloc_deps` helper the other fusions use, and the dispatch still rechecks with
`ggml_cuda_check_fusion_memory_ranges` and declines if the allocator did not separate them.

The decision has to include the recheck. With the alias check only in the dispatch, the producer
could be refused (writing F32) while the consumer's structural decision still said BF16 - reading
BF16 out of F32 data. The check is part of `ggml_cuda_hc_res16_ok()` for that reason.

## Effect

pp16384, `-b 16384 -ub 16384 -p 16384 -n 0 -r 1`, against the gatemix-only state (`0e1fbb7e3`):

| model | gatemix only | fused combine+norm, BF16 stream | delta |
| --- | --- | --- | --- |
| IQ4_NL | 1218.4 | 1282.4 | +5.2% |
| APEX-I-Nano | 1086.4 | 1112.4 | +2.4% |
| UD-IQ1_S | 1058.3 | 1089.2 | +2.9% |

pp512: IQ4_NL 964.7, APEX-I-Nano 744.6, UD-IQ1_S 787.1 (+8% to +15% over the session baseline);
tg128 unchanged, since the fusion needs the same 512 rows the GEMMs do.

## What is left in this area

`blk_in_bf16` (the `block_out` side of the combine) has a placeholder in the kernel but is not wired:
it needs a producer that honours the mark, and the attention output projection and the MoE merge are
MMB GEMMs whose `Dh` slot is only reserved for the `[320 -> 10240]` gate shape. Serving BF16
`block_out` costs one more producer change and would halve another 671 MB per layer.
