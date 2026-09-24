# BF16 weights: the bf16w path and the tall-M tile

Operator: BF16 weights in the dense MMB dispatch (`ggml_cuda_mmb_supported_mm` with `mmb_bf16w()` and `mmb_dense_kernel<..., WT=2>`), `ggml/src/ggml-cuda/mmb.cu` and `.cuh`.

Relevant because `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf` keeps its 194 hyper-connection tensors in BF16 (1.18 GiB): per layer `hc_attn_down`/`hc_ffn_down` `[10240, 320]` (k=10240, m=320) and `hc_attn_up`/`hc_ffn_up` `[320, 10240]` (k=320, m=10240), plus `output_hc_down`/`up` at the same shapes. The other three models hold the same tensors in IQ4_NL, Q8_0 or a F32-split form, so this is the first model that leans on the bf16w path.

## MMB against hipBLASLt

`test-backend-ops perf -o MMB_PERF`, TFLOP/s. Both columns from one binary, with a temporary `mmb_bf16w()` override so the second column is the hipBLASLt path with an F32-to-BF16 activation conversion:

| shape | MMB bf16w | hipBLASLt | winner |
| --- | --- | --- | --- |
| m=320 n=512 k=10240 | 19.19 | 10.59 | MMB 1.81x |
| m=10240 n=512 k=320 | 22.01 | 12.47 | MMB 1.77x |
| m=320 n=2048 k=10240 | 15.73 | 9.74 | MMB 1.62x |
| m=10240 n=2048 k=320 | 18.38 | 11.87 | MMB 1.55x |
| m=320 n=16384 k=10240 | 10.91 | 11.10 | tie (-1.7%) |
| m=10240 n=16384 k=320 | 18.89 | 12.67 | MMB 1.49x |

MMB wins five of six and ties the sixth, so the bf16w path stays on for BF16 weights with no gate change. The one tie is the deep-prefill narrow-M case.

## The tall-M tile does not transfer to BF16

The tall-M tile (`mmb_dense_kernel<384, 64, 96, 32, WT>`, `M <= 384 && K >= 4096 && T >= 2048`) exists for exactly the m=320 k=10240 shape, but it was only ever enabled for IQ4_NL, whose weight for that shape is 2.2 MiB. Extending the same tile to BF16 (WT=2) was measured and rejected:

| shape | generic 128-row tile | tall-M tile | delta |
| --- | --- | --- | --- |
| m=320 n=512 k=10240 | 19.06 | 19.25 | +1% |
| m=320 n=2048 k=10240 | 15.64 | 11.07 | -29% |
| m=320 n=16384 k=10240 | 10.84 | 7.57 | -30% |

The tall tile trades M-block passes over the activations for repeated K passes over the weight slab. That is a win when the weight is a few MiB and the activations dominate, and a loss at 16 bits, where the same slab is 3.6x larger and no longer fits the reuse that the generic tiling gets from L2. The change was reverted. `mmb_tall()` still applies to IQ4_NL only.

## Model cost

At pp16384 a single `[10240, 320]` projection takes 9.93 ms and a single `[320, 10240]` takes 5.68 ms, both BF16 on MMB. With two of each per layer over 48 layers that is 1.5 s, about 8.7% of the 17.25 s pass, for weights that would cost roughly a third of that in IQ4_NL. That is a property of the quant recipe rather than of the kernel: the path chosen is the fastest one measured for these shapes.

At one token these weights are pure matvec traffic: 1.27 GB per token over the four per-layer projections, which is 5.5 ms of DRAM time at 230 GB/s and a visible share of the 29 ms/token that the model's 34.36 t/s implies. They cannot use MMVQ (BF16 is not quantized), so `mmvf`/hipBLASLt are the only alternatives and neither is faster at these widths.
