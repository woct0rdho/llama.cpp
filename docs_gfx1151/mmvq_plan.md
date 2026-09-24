# Strix Halo MMVQ tuning plan

Upstream context:
- Upstream maps RDNA3.5 to RDNA2 MMVQ parameters because the RDNA3_0 `nwarps=8` table regressed on Strix Halo.
- PR #20831 dynamically lowers runtime nwarps for narrow MoE matrices, but only for RDNA3_0 and RDNA4, so it does not affect gfx1151 as-is.
- PR #25707 added Q2_0 to the CUDA MMVQ and MMQ paths; the retained tuning below targets Q5_K, Q6_K, and IQ4_XS.

Current retained settings:
- RDNA3.5 MMVQ `max_nwarps`: Q5_K=2, Q6_K=4, IQ4_XS=2.
- Runtime clamping lowers the actual nwarps for narrow matrices.
- The RDNA3.5 default remains equivalent to RDNA2 unless a type is explicitly tuned.

Initial implementation idea:
- Add a dedicated `MMVQ_PARAMETERS_RDNA3_5` table id.
- Port PR #20831's max_nwarps/runtime-nwarps split so runtime can clamp narrow matrices safely.
- Try a conservative RDNA3.5 whitelist for `ncols_dst == 1` after baseline, then benchmark Qwen3.6 35B A3B TG.

Benchmark target:
- Model: `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Quality.gguf`
- Focus: single GPU, batch-1 token generation (`tg` tests).

Log:
- Baseline TG-only without `ROCBLAS_USE_HIPBLASLT=1`: `tg128 56.50 +/- 0.14`, `tg512 56.54 +/- 0.00`.
- Combined PP/TG needs `ROCBLAS_USE_HIPBLASLT=1`; otherwise the pip ROCm SDK finds `librocblas.so` but no `rocblas/library/TensileLibrary.dat`.
- Baseline with `ROCBLAS_USE_HIPBLASLT=1`: `pp512 1438.81 +/- 5.84`, `pp2048 1423.97 +/- 5.34`, `tg128 56.36 +/- 0.09`, `tg512 56.45 +/- 0.01`.
- Candidate A: dedicated RDNA3.5 MMVQ table with Q6_K `max_nwarps=2` and PR #20831 runtime clamp.
  - Result: `pp512 1435.94 +/- 9.50`, `pp2048 1419.18 +/- 5.02`, `tg128 56.92 +/- 0.07`, `tg512 56.98 +/- 0.02`.
  - TG improves about 0.9-1.0%; PP is about 0.2-0.3% lower, likely noise or unrelated BLAS path.
- Candidate B: raise RDNA3.5 Q6_K `max_nwarps` to 4 while keeping runtime clamp.
  - Result: `pp512 1436.44 +/- 11.12`, `pp2048 1420.76 +/- 9.34`, `tg128 57.19 +/- 0.06`, `tg512 57.29 +/- 0.03`.
  - TG improves about 1.5%; PP remains within noise.
- Candidate C: raise RDNA3.5 Q6_K `max_nwarps` to 8 while keeping runtime clamp.
  - Result: `pp512 1438.00 +/- 12.27`, `pp2048 1418.20 +/- 3.30`, `tg128 54.81 +/- 0.07`, `tg512 54.95 +/- 0.01`.
  - Regresses TG by about 2.8-2.9%; rejected.
- Final choice: Candidate B (`Q6_K max_nwarps=4`).
  - Final validation: `pp512 1431.17 +/- 13.49`, `pp2048 1419.29 +/- 9.59`, `tg128 57.20 +/- 0.07`, `tg512 57.28 +/- 0.02`.
  - Compared to baseline, TG improves about 1.5%; PP is lower by about 0.5% but within observed variance and mostly BLAS-driven.

Post-rebase tuning against upstream `bf2c86ddc`:
- Fresh Q6_K=4 baseline with `ROCBLAS_USE_HIPBLASLT=1`: `pp512 1350.41 +/- 8.02`, `pp2048 1338.84 +/- 5.05`, `tg128 57.58 +/- 0.44`, `tg512 57.91 +/- 0.02`.
  - The first `tg128` sample was cold (`56.79 t/s`); the remaining samples were `57.76-57.78 t/s`.
- rocWMMA is excluded because it is already known to be slower on this machine.
- Candidate D: add RDNA3.5 Q5_K `max_nwarps=2`, retaining Q6_K=4.
  - Result: `pp512 1343.15 +/- 4.67`, `pp2048 1332.61 +/- 6.48`, `tg128 58.14 +/- 0.09`, `tg512 58.33 +/- 0.05`.
  - Compared to the fresh baseline, TG improves about 0.7% on steady samples and 0.74% on `tg512`; PP changes by less than 0.5%.
- Candidate E: raise RDNA3.5 Q5_K `max_nwarps` from 2 to 4.
  - Result: `pp512 1353.21 +/- 3.74`, `pp2048 1342.68 +/- 5.44`, `tg128 58.12 +/- 0.06`, `tg512 58.28 +/- 0.02`.
  - Does not improve TG over Q5_K=2 and lowers `tg512` by about 0.09%; rejected.
- Candidate F: retain Q5_K=2 and add IQ4_XS `max_nwarps=2`.
  - Result: `pp512 1341.71 +/- 8.46`, `pp2048 1316.25 +/- 3.98`, `tg128 58.30 +/- 0.17`, `tg512 58.48 +/- 0.03`.
  - Improves `tg512` by about 0.26% over Q5_K=2 and about 1.0% over the fresh baseline; retained.
  - PP movement is treated as run-to-run noise because these single-column MMVQ settings do not control the large-batch MMQ path.
- Candidate G: lower Q6_K from 4 to 3 with Q5_K=2 and IQ4_XS=2 retained.
  - TG-only result: `tg128 57.84 +/- 0.16`, `tg512 58.04 +/- 0.02`.
  - Regresses `tg512` by about 0.76% relative to Candidate F; rejected.
- Candidate H: raise Q6_K from 4 to 5 with Q5_K=2 and IQ4_XS=2 retained.
  - TG-only result: `tg128 57.09 +/- 0.16`, `tg512 57.38 +/- 0.02`.
  - Regresses `tg512` by about 1.9% relative to Candidate F; rejected.
  - Q6_K=6 was not run because Q6_K=3, 5, and the earlier Q6_K=8 result all regress around the Q6_K=4 optimum.

Final MMVQ combination: Q5_K=2, Q6_K=4, IQ4_XS=2.
- TG-only validation: `tg128 58.52 +/- 0.16`, `tg512 58.76 +/- 0.02`.
- Compared to the fresh rebased baseline, reported TG improves about 1.6% on `tg128` and 1.5% on `tg512`.
- The focused ROCm validation after the later upstream Q2_0 additions passed `865/865` `MUL_MAT_ID` tests.

## qwen4exp (Qwen3.8-Flash-Next) TG follow-up

Benchmark target: `~/models/qwen4/Qwen3.8-Flash-Next-IQ4_NL.gguf` (and the APEX/UD variants), batch 1, `DEBUG_HIP_GRAPH_PM4=1`.

TG128 kernel breakdown with `rocprofv3 --kernel-trace` (27.9 ms per token, kernel sum 30.7 ms over two tokens):

| family | ms/token | launches/token |
| --- | --- | --- |
| MMVQ IQ4_NL dense | 12.25 | 576 |
| MMVQ IQ4_NL split (MoE) | 5.23 | 96 |
| MMVQ Q6_K (lm head) | 3.13 | 13 |
| matvec F32 (router) | 1.38 | 96 |
| device copies/fills (cache/state update) | 2.70 | ~1964 |
| elementwise | 2.05 | ~1180 |
| quantize_q8_1 | 0.88 | 713 |

The IQ4_NL matvecs read 3.2 GB/token at 261 GB/s, i.e. at (slightly above) the streaming DRAM read ceiling of
230 GB/s, so they only move with fewer bytes. A new perf harness (`-o MMVQ_PERF`, m=1) gives the per-type picture:

| type | lm head (m=248320, k=2560) | dense (m=10240, k=2560) |
| --- | --- | --- |
| Q4_K | 236 GB/s | 497 GB/s (L2-resident) |
| Q8_0 | 224 GB/s | 792 GB/s |
| IQ4_XS | 219 GB/s | 483 GB/s |
| IQ1_S | 187 GB/s | 226 GB/s |
| IQ2_XXS | 174 GB/s | 192 GB/s |
| Q6_K | 162 GB/s | - |

- Candidate A: raise `VDR_Q6_K_Q8_1_MMVQ` from 1 to 2 (two lanes cooperate per 256-weight block).
  - Result: lm head 162 -> 217 GB/s (+34%, 3.13 -> 2.35 ms/token), dense m=10240 regresses 40.35 -> 45.05 us.
  - Correctness: `test-backend-ops test -o MUL_MAT -p type_a=q6_K` fails with `MUL_MAT(m=16,n=1..8,k=256)` at
    ERR 0.83-1.33 vs a 0.0005 tolerance; with VDR=1 the same selection is clean (0 failures).
  - Conclusion: the vdr>1 MMVQ reduction is incorrect for small-k shapes; rejected and reverted. Fixing the
    reduction would be worth about 1 ms/token (3.5%) on the Q6_K lm head, so it stays on the list as a bug fix
    rather than a tuning knob.
- Not pursued: the dense IQ4_NL/Q8_0/IQ4_XS matvecs and the MoE matvecs (all at the DRAM wall).

Long context: the sparse QSA path is not used in decode (`qsa_prefill_supported` needs `q->ne[1] >= 128`), so a
decode step at 131072 context reads 1.6 GB of KV plus a 0.4 GB indexer scan on top of the 4.25 GB of weights.
