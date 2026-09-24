# MMB IQ4_XS dequant

Operator: `mmb_dq_row_iq4xs` in `ggml/src/ggml-cuda/mmb.cu`. It turns one 64-element k-tile of an
IQ4_XS weight row (two 32-weight sub-blocks) into 64 bf16 values in LDS for the bf16 WMMA prefill
GEMM (MMB). Model independent; IQ4_XS appears in the APEX lm head and in other checkpoints.

## Format facts

`block_iq4_xs` is 136 bytes for 256 weights:

| offset | field | notes |
| --- | --- | --- |
| 0 | `d` (fp16) | super-block scale |
| 2 | `scales_h` (u16) | two high bits per sub-block, 2 bits * sub-block index |
| 4 | `scales_l[4]` | four low bits per sub-block, one byte holds two sub-blocks |
| 8 | `qs[128]` | 4-bit LUT indices |

- Eight sub-blocks of 32 weights. Sub-block `j` scale: `ls = ((scales_l[j/2] >> 4*(j%2)) & 0xF) |
  ((scales_h >> 2*j) & 3) << 4`, and the value is `kvalues_iq4nl[code] * d * (ls - 32)`.
- Within a sub-block, `qs[16j .. 16j+15]` holds the 32 codes: the low nibbles are the first 16
  weights, the high nibbles the last 16. So dword `w` of a sub-block covers weights `4w..4w+3` (low
  nibbles) and `16+4w..19+4w` (high nibbles).
- A 64-element tile is sub-blocks `2m` and `2m+1` where `m = ks % 4`; it needs `qs[32m .. 32m+31]`
  and both scales, which share one byte of `scales_l` (`scales_l[m]`, nibbles 0 and 1) plus the 2-bit
  fields `2*(2m)` and `2*(2m)+1` of `scales_h`.
- The LUT is the same `kvalues_iq4nl` table as IQ4_NL.

## Recipe

The recipe is the IQ4_NL byte-LUT recipe with a per-sub-block scale. The LUT lives in registers as
`kv + 128` packed one byte per entry (four dwords), `v_perm_b32` selects four entries per nibble pass
with a second perm plus mask for entries 8..15, and each weight becomes one fmaf:

    value = fmaf(kv + 128, dsc, -128 * dsc)   ->  fl(kv * dsc)

This is bit-exact against llama.cpp's `dequantize_iq4_xs` because `-128*dsc` is a power-of-two
multiple (exact) and the fma rounds the exact expression once.

Staging per tile (4 registers): `a2` = the u32 at +0 (`d`, `scales_h`), `a0.x` = the u32 at +4
(`scales_l`), `a3`/`a4` = `qs[32m .. 32m+31]`. Cost is about 7 VALU ops per weight for the LUT path
plus the two scale unpacks per tile (amortized).

## Measurements

`test-backend-ops perf -o MMB_PERF`, TFLOP/s. Routed = `n_mats 512, n_used 10, m 640, k 2560`;
dense = `m 2560, k 2560`; `n` is the token count. MMQ numbers are from the same harness before the
type was gated in.

| shape | MMB | MMQ | delta |
| --- | --- | --- | --- |
| routed n=512 | (MMQ, gated) 5.88 | 5.88 | 0 |
| routed n=2048 | 13.68 | not measured | - |
| routed n=16384 | 17.62 | 15.49 | +14% |
| dense n=512 | 30.77 | 25.67 | +20% |
| dense n=16384 | 27.40 | 26.16 | +4.7% |

The generic scalar MMB path was at 5.6/8.0 on the dense shapes, i.e. 4.6x behind MMQ - the reason
the type was gated out in the first place.

Gate: `mmb_quant_type()` includes IQ4_XS (dense always MMB); `mmb_quant_type_mmid()` keeps MMQ below
2048 tokens, mirroring Q5_K. The routed crossover between 512 and 16384 is not measured, so the
threshold is inherited rather than fitted.

## Integration pitfall (cost two recipes)

`mmb_tile_gemm` has two `if constexpr` chains over `WTYPE`: the register loads in `load_regs` and the
LDS write in `store_lds`. Their `Q5_K` arms start with the same text
(`else if constexpr (WTYPE == 32 + GGML_TYPE_Q5_K) {`), so a patch anchored on that string lands in
`load_regs`. A recipe branch placed there is unreachable: the LDS row is never written, the WMMA
consumes stale data, and the failure shows up as non-deterministic NaN *plus* implausibly fast
timings (43 TFLOP/s dense, faster than the dequant-free ceiling) because no dequant runs at all.
Anchor the store branch on the arm that contains `uint32_t * ar`, and when debugging this file print
the instantiated `WTYPE` from `mmb_tile_gemm` plus a marker inside the new branch - that is what
found it.
