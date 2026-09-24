# BF16 activation copies written by the RMS norm producer

Change: `c1bcf6ffa` (`ggml/src/ggml-cuda/norm.cu`, `ggml/src/ggml-cuda/mmb.cu`, `mmb.cuh`)

## What it does

The BF16 WMMA GEMMs (`mmb.cu`) read the activation as BF16. When the graph hands them an F32
tensor, `mmb_bf16_activation()` converted it in its own pass (`mmb_cvt_f32_bf16`), then cached the
result keyed on the tensor. The RMS norm is the producer of nearly all wide activations in this
model, so `ggml_cuda_op_rms_norm_fused` now registers slot 0 through
`ggml_cuda_mmb_cache_reserve()` and the norm kernel writes the BF16 value in its epilogue, next to
the F32 store. `mmb_bf16_activation()` checks the slots before the conversion cache, so the
consumer finds it and the conversion pass never runs.

Both round the same way: `mmb_f2bf` (RNE, `u += 0x7fff + ((u >> 16) & 1)`) is the scalar form of
`mmb_rne_bf16`, which `mmb_pack2` uses in the conversion kernel. The values in the slot are
therefore bit-identical to what the conversion produced, and the kernel writes them at
`dst_off + col`, the same flat index the conversion kernel derived from `mul_tensor->data`. The
F32 output is still written unconditionally, so a consumer that misses the slot (different `n`,
different graph instance, MMB off) silently falls back to the conversion path. Nothing depends on
the slot being filled.

## Why it is gated at 64M elements

The conversion only pays off when the tensor is much larger than L2 (4 MB on gfx1151). Below that
the consumer reads the activation from cache, where the conversion is nearly free and its stores
act as a prefetch for the GEMM that follows. Measured on IQ4_NL pp, tokens T:

| T | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| --- | --- | --- | --- | --- | --- | --- |
| conversion into the producer | -1.1% | -2.2% | -1.2% | -1.7% | +3.6% | +2.2% |

A profile at T=512 shows why the sign flips: the conversion saves 9.6 ms (370 -> 287 launches,
14.4 -> 4.9 ms) and the norm pays 2.2 ms for the extra stores, but `mmb_dense_kernel<128,128>`
takes 13.5 ms *longer* once the activation is no longer written immediately in front of it. The
reservation is therefore refused below `n = 64M` elements (128 MB of BF16), which is where the
crossover sits. `mmb_min_t()` (512 rows) stays as the low bound.

## Effect

pp16384, `llama-bench -b 16384 -ub 16384 -p 16384 -n 0 -r 1`:

| model | before | after | delta |
| --- | --- | --- | --- |
| IQ4_NL | 1131.9 | 1140.4 | +0.7% |
| APEX-I-Nano | 1006.9 | 1037.7 | +3.1% |
| UD-IQ1_S | 976.1 | 1012.5 | +3.7% |

pp8192 on IQ4_NL: 1144.4 -> 1187.0 (+3.7%). pp512, 1024, 2048, 4096 are unchanged within noise for
all three models (the gate keeps them on the conversion path); tg128 is unchanged because the gate
requires the same 512 rows the GEMMs do.

Accuracy: `llama-perplexity -c 8192 -b 8192 -ub 8192 --chunks 1` on a docs-derived corpus gives
IQ4_NL 3.4078 (3.4516 at `-ub 4096`, where the slot is inactive), APEX-I-Nano 3.6362, UD-IQ1_S
3.6280. `test-backend-ops test -o MMB_QUANT,MUL_MAT_ID` stays at 1073/1073.

## Remaining conversions

After this change the pp16384 IQ4_NL profile still shows 286 `mmb_cvt_f32_bf16` launches per pass,
250 ms, i.e. 1.7% of the pass. They convert activations below the gate: the [2560, T] attention and
FFN norms (42M elements) and the GDN and MoE intermediates. The same producer trick applies to them
if their consumer set allows it, but each producer needs its own epilogue change, and their
consumers are mixed (F32 and BF16), so the F32 output cannot be dropped.
