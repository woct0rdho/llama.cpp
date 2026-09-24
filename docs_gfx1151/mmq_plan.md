# Strix Halo MMQ tuning plan

Upstream context:
- The original gfx1151 MMQ work is carried by PR #21344.
- Upstream PR #26199 split RDNA3 and RDNA3.5 MMQ configuration tables.
- Upstream PR #26141 added the low-shared-memory MMQ fallback guard.

Current retained settings:
- RDNA3.5 MMQ uses the upstream configuration table through a local tuned wrapper that sets `nthreads=128` and `I=64`.
- Dense dispatch searches through `J<=128`.
- RDNA3.5 MoE dispatch keeps the `J<=48` cap to avoid fragmentation on cold experts.

Implementation:
- Keep the upstream `mmq-config-rdna3-5.cuh` table intact and normalize its selected RDNA3.5 configs in `mmq.cuh`.
- Select the fallback mode using the gfx1151 `I=64` tile width.
- Cap only the RDNA3.5 MoE path at `J=48`; dense GEMMs retain `J=128` candidates.

Benchmark target:
- Model: `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Quality.gguf`
- Focus: single GPU prompt processing (`pp` tests), with the same ROCm and `ROCBLAS_USE_HIPBLASLT=1` settings used by the other plans.

Log:
- MMQ prompt baseline with the existing RDNA3.5 MoE `J<=48` cap: `pp512 1349.96 +/- 5.42`, `pp2048 1330.70 +/- 6.90`, `pp8192 1219.11 +/- 3.27`.
- Candidate K: lower the RDNA3.5 MoE cap to `J<=32`.
  - Result: `pp512 1326.55 +/- 6.01`, `pp2048 1302.91 +/- 1.27`, `pp8192 1197.26 +/- 4.94`.
  - Regresses PP by about 1.7-2.1%; rejected.
- Candidate L: raise the RDNA3.5 MoE cap to `J<=64`.
  - Result: `pp512 1293.66 +/- 3.58`, `pp2048 1274.80 +/- 11.36`, `pp8192 1172.82 +/- 2.51`.
  - Regresses PP by about 4.1-4.2%; rejected. The existing `J<=48` cap is retained.
- Candidate M: use the RDNA4 `I=128`, `nthreads=256` tile only for IQ4_XS, retaining the gfx1151 tile for other types.
  - Result: `pp512 1330.08 +/- 8.67`, `pp2048 1306.88 +/- 7.06`, `pp8192 1205.87 +/- 5.11`.
  - Regresses PP by about 1.1-1.8%; rejected.
- Candidate N: use the RDNA4 `I=128`, `nthreads=256` tile only for Q5_K.
  - Result: `pp512 1340.20 +/- 6.27`, `pp2048 1316.87 +/- 13.28`, `pp8192 1208.87 +/- 3.09`.
  - Regresses PP by about 0.7-1.0%; rejected.
- Candidate O: use the RDNA4 `I=128`, `nthreads=256` tile only for Q6_K.
  - Result: `pp512 1332.51 +/- 11.41`, `pp2048 1310.43 +/- 15.66`, `pp8192 1205.57 +/- 3.78`.
  - Regresses PP by about 1.1-1.5%; rejected.

MMQ conclusion: retain the gfx1151 `I=64`, `nthreads=128`, and MoE `J<=48` configuration for Q5_K, Q6_K, and IQ4_XS.

Final retained configuration validation after restoring all rejected MMQ and flash-attention experiments:
- `pp512 1346.85 +/- 6.70`, `pp2048 1327.95 +/- 8.52`, `tg128 58.87 +/- 0.09`, `tg512 59.08 +/- 0.02`.
- Relative to the steady samples of the fresh Q6_K=4 baseline, TG improves about 2.0%; PP is within about 0.2% of the restored MMQ baseline.
- `cmake --build build` completed successfully without `-j`.
- `ROCBLAS_USE_HIPBLASLT=1 build/bin/test-backend-ops test -b ROCm0 -o MUL_MAT_ID -j 1`: 790/790 tests passed.

## qwen4exp MMQ vs MMB per-type selection

Measured with `test-backend-ops perf -o MMB_PERF` on gfx1151 (TFLOP/s). Ceilings measured on this machine:
int8 WMMA 50.3, bf16 WMMA 51.2, f16 WMMA 49.7, int8 dot4 (dp4a) 28.7, bf16 dot2 19.6, fp32 FMA 11.1 TFLOP/s;
DRAM read 230 GB/s; LDS 64 KB/CU.

Engine note, verified by disassembly: on gfx1151 the prefill MMQ kernel is **not** a dp4a kernel. `RDNA3` is defined
for gfx11xx (`vendors/hip.h:219`), so `AMD_WMMA_AVAILABLE` is defined and `ggml_cuda_mmq_get_util_funcs()` takes the
`use_mma_data_layout()` branch, which skips the whole dp4a switch and uses the `_mma` vec dots
(`mmq-vec-dot.cuh`, `tile<16,8,int>` A/B, `tile<16,16,int>` C). Those lower to `v_wmma_i32_16x16x16_iu8`
(`mma.cuh` RDNA3 branch). Disassembly of the built template instances: `mmq-instance-q5_k.cu.o` has 204
`v_wmma_i32_16x16x16_iu8` and 0 `v_dot4`/`v_dot8`; `mmq-instance-iq4_nl.cu.o` has 276 and 0. The dp4a vec dots in
the same file are dead code on this target; the only dp4a user left is the m=1 MMVQ path (`ggml_cuda_dp4a`).

Protocol: one `heavy-run` job, `DEBUG_HIP_GRAPH_PM4=1`, 512-thread blocks, perf mode self-selects the run count. The
MMB column is a build with the per-type gate forced open (`MMB_ALL_TYPES=1`), i.e. every type measured through
`mmb_dense_kernel`; the MMQ column is the current dispatch (the gate sends IQ4_NL, Q8_0 and IQ1_M to MMB, so those
three rows show MMB in both columns).

Test shapes (`n` is the token count, `m` the weight's output width, `k` the reduction):

| label | shape | model use |
| --- | --- | --- |
| routed n=512 | n_mats=512, n_used=10, m=640, n=512, k=2560 | MoE gate/up at pp512 |
| routed n=16384 | n_mats=512, n_used=10, m=640, n=16384, k=2560 | MoE gate/up at pp16384 |
| dense n=512 | m=2560, n=512, k=2560 | dense projection at pp512 |
| dense n=16384 | m=2560, n=16384, k=2560 | dense projection at pp16384 |

MMQ (current dispatch per type):

| type | routed n=512 | routed n=16384 | dense n=512 | dense n=16384 |
| --- | --- | --- | --- | --- |
| IQ4_NL | 6.56 | 17.32 | 30.00 | 27.23 |
| Q8_0 | 3.59 | 17.20 | 32.49 | 30.99 |
| IQ1_M | 2.21 | 10.07 | 12.51 | 11.87 |
| IQ4_XS | 5.88 | 15.49 | 25.67 | 26.16 |
| Q5_K | 5.28 | 14.71 | 23.73 | 24.57 |
| Q4_K | 5.64 | 15.13 | 24.01 | 24.88 |
| Q6_K | 4.35 | 10.98 | 13.80 | 17.93 |
| IQ1_S | 6.96 | 14.62 | 23.54 | 24.73 |
| IQ2_XXS | 6.82 | 14.49 | 24.61 | 25.23 |
| IQ2_XS | 5.61 | 12.75 | 20.37 | 21.16 |

MMB (gate forced open, i.e. the generic scalar-dequant tile loader for every non-IQ4_NL type):

| type | routed n=512 | routed n=16384 | dense n=512 | dense n=16384 | MMQ/MMB dense 16384 |
| --- | --- | --- | --- | --- | --- |
| IQ4_NL | 6.65 | 17.41 | 30.40 | 27.11 | 1.00 |
| Q8_0 | 3.62 | 17.56 | 32.41 | 31.10 | 1.00 |
| IQ1_M | 2.28 | 10.25 | 12.51 | 11.90 | 1.00 |
| IQ4_XS | 1.04 | 5.60 | 5.61 | 7.98 | 3.28 |
| Q5_K | 1.50 | 9.59 | 11.15 | 15.47 | 1.59 |
| Q4_K | 2.15 | 11.38 | 15.80 | 14.70 | 1.69 |
| Q6_K | 2.21 | 11.00 | 15.44 | 13.29 | 1.35 |
| IQ1_S | 2.20 | 10.26 | 12.35 | 11.72 | 2.11 |
| IQ2_XXS | 1.56 | 7.92 | 8.64 | 8.19 | 3.08 |
| IQ2_XS | 1.48 | 7.58 | 8.21 | 7.88 | 2.69 |

Findings:
- MMQ is an int8-WMMA path on this target and reaches only 44-49% of the measured 50.3 TFLOP/s i8-WMMA ceiling
  (Q5_K 24.6, Q4_K 24.9, IQ4_XS 26.2, IQ1_S 24.7, IQ2_XXS 25.2 dense n=16384), so unlike the earlier dp4a reading it
  is far from an engine ceiling: up to about 2x sits in the MMQ structure itself (per-m-tile weight expansion,
  epilogue scale fixups, LDS/issue overhead). That is P1-6 work, not per-type gate work.
- The generic MMB path (`mmb_decode_slice` -> `dequantize_*<float>` with a per-element bf16 store) is 1.35-3.3x
  behind MMQ on every gated-out type. That is a recipe problem, not an engine problem: IQ4_NL, which has a
  hand-written `v_perm`-based recipe, reaches 27.1 dense against MMQ at about 28.
- Q6_K is the only type where MMB already wins one shape (dense n=512: 15.44 vs 13.80) and loses the other.
- Target for a rewritten recipe: beat MMQ by at least 1.2x on both dense shapes and the routed shape before the type
  is gated in, otherwise keep MMQ. Per-type work items, largest first: Q5_K, IQ1_S/IQ2_XXS/IQ2_XS (experts),
  Q6_K, Q4_K, IQ4_XS.

Dequant cost model (from measuring a build whose IQ4_NL dequant is replaced by a trivial store):
- A 16x16x16 bf16 WMMA occupies about 64 issue slots on one SIMD, and VALU/LDS share that issue stream, so
  dequant ALU removes matrix throughput directly. IQ4_NL costs about 8.6 VALU ops per weight, i.e. 640
  instructions per thread per 64-weight k-tile against about 4096 matrix clocks per wave = 15.6% -> measured
  27.2 -> 31.4 TFLOP/s dense when removed.
- A 40% cheaper pack (truncating instead of rounded float->bf16) gave +3.8% dense, so the sensitivity is
  roughly linear in dequant ops per weight.
- Corollary: the codebook types would need their 256-entry grid lookup amortized over the k-tile before an MMB
  recipe could beat the int8-WMMA MMQ path; the generic `dequantize_*<float>` helpers do it per weight.
