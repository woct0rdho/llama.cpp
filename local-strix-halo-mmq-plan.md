# Strix Halo MMQ tuning plan

Goal: retain the useful gfx1151 MMQ tile and J-selection tuning from PR #21344 for Strix Halo while keeping the upstream RDNA3/RDNA3.5 configuration split easy to rebase.

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
