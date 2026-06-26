# Strix Halo MMVQ tuning plan

Goal: port the useful part of PR #20831 to gfx1151 / RDNA3.5 for single-GPU batch-1 token generation.

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
