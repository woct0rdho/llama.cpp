# Strix Halo flash-attention tuning plan

Goal: retain the useful gfx1151 flash-attention tile override from PR #21344 without carrying rejected ncols=8 or rocWMMA experiments.

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
