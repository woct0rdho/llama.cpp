# MMB Q6_K dequant

Operator: `mmb_dq_row_q6k` in `ggml/src/ggml-cuda/mmb.cu`. It turns one 64-element k-tile of a Q6_K
weight row into 64 bf16 values in LDS for the bf16 WMMA prefill GEMM (MMB). Model independent:
Q6_K is `attn_output` in the qwen4exp models, the lm head of IQ4_NL and APEX, and the whole GDN of
UD-IQ1_S.

Status: landed (`e8ba63209`), correctness and end to end verified.

## Format facts

`block_q6_K` is 210 bytes for 256 weights:

| offset | field | notes |
| --- | --- | --- |
| 0 | `ql[128]` | low four bits |
| 128 | `qh[64]` | two high bits per weight, four 2-bit fields per byte |
| 192 | `scales[16]` (int8) | one scale per 16 weights |
| 208 | `d` (fp16) | super-block scale |

Two 128-element groups per block. Group `g` (`g = (ks/2) % 2` for tile `ks`) uses `ql[64g..64g+63]`,
`qh[32g..32g+31]` and `scales[8g..8g+7]`. Reference element map, for `l = 0..31`:

- `y[l]      = d * sc[is + 0] * ((ql[l]      & 0xF | ((qh[l] >> (4H + 0)) & 3) << 4) - 32)`
- `y[l + 32] = d * sc[is + 2] * ((ql[l + 32] & 0xF | ((qh[l] >> (4H + 2)) & 3) << 4) - 32)`

with `is = l/16` and `H = 0` taking the low nibbles (group elements 0..63) and `H = 1` the high
nibbles (elements 64..127). So a tile is one half of one group, four int8 scales per tile, and every
weight is one code extraction plus one fmaf on `(code - 32)`.

Verified bit-close against ggml's own `dequantize_q6_K` plus the `mmb_quant_slice` mapping
(`index = offset + n - begin`, written only when `0 <= index < 64`): 0/256 differences over the four
tiles of a random block (`/tmp/peak/q6k_test.cpp`).

## Recipe

`template <int H> mmb_dq_row_q6k(ql0..ql3, qh0, qh1, scw, d2, arow)`, one thread per row per tile.
Staging: `a0,a1,a3,a4` = `ql[0..63]`, `a5,a6` = `qh[0..31]`, `a7` = the group's eight scales (only
four are used per half), `a2` = the u32 at +208 (`d`).

Per dword (4 weights): nibble pick (1-2 ops), qh 2-bit field into bits 4-5 (3 ops), or (1), four
`v_cvt_f32_ubyteN` folding the byte extraction, four fmaf against `(dsc, -32*dsc)`, two `mmb_pack2`.
About 6.8 VALU ops per weight.

## Measurements

`test-backend-ops perf -o MMB_PERF`, TFLOP/s. Routed = `n_mats 512, n_used 10, m 640, k 2560`; dense
= `m 2560, k 2560`; `n` is the token count.

| shape | MMB | MMQ | generic MMB (before) | delta vs MMQ |
| --- | --- | --- | --- | --- |
| routed n=512 | 3.64 | 4.35 | 2.21 | -16% |
| routed n=2048 | 10.66 | not measured | 4.85 | - |
| routed n=16384 | 17.62 | 10.98 | 11.00 | +60% |
| dense n=512 | 27.74 | 13.80 | 15.44 | +101% |
| dense n=16384 | 30.90 | 17.93 | 13.29 | +72% |

## Gate: token-count dependent on both paths

Q6_K dense weights also have a bf16 shadow path (`mmb_dq_q6k_bf16_kernel`, built once per weight in
`graph_optimize`, mode 2 of `mmb_shadow_mode`). Its GEMM is the dequant-free WTYPE=2 kernel, and it
only pays off for wide token tiles. End to end on UD-IQ1_S:

| configuration | pp512 | pp16384 |
| --- | --- | --- |
| Q6_K on the old path (MMQ/hipBLASLt) | 691 | 971 |
| Q6_K on MMB for every token count | 658 | 987 |
| Q6_K on MMB from 4096 tokens (shipped) | 681-690 | 981 |

The 512-token shape is `[6144, 2560]` (the GDN `ssm_out`), whose 20x4 launch grid leaves the machine
under-occupied, and the one-off shadow build is not amortized. Hence
`mmb_quant_type_mm()`: Q6_K dense from 4096 tokens, everything else follows `mmb_quant_type()`;
`mmb_quant_type_mmid()` keeps the routed path on MMB from 2048 tokens. Both thresholds are measured,
the crossover between them is not.

APEX-I-Nano and UD-IQ1_S pp16384: 992/971 before this change, 1013/981 after.

## Integration pitfall (cost the first two attempts)

`mmb_tile_gemm` has two `if constexpr` chains over `WTYPE`: the register loads in `load_regs` and the
LDS write in `store_lds`. Their `Q5_K` arms start with the same text, so a patch anchored on that
string lands in `load_regs`. A recipe branch placed there is unreachable: the LDS row is never
written, the WMMA consumes stale data, and the failure shows as non-deterministic NaN *plus*
implausibly fast timings (43 TFLOP/s dense, above the dequant-free ceiling) because no dequant runs.
Anchor the store branch on the arm that contains `uint32_t * ar`, and print the instantiated `WTYPE`
from `mmb_tile_gemm` plus a marker inside the new branch when debugging. Both recipes in this series
were originally lost to this.
