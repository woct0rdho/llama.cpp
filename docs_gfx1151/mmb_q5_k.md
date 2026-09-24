# MMB Q5_K dequant

Operator: `mmb_dq_row_q5k` in `ggml/src/ggml-cuda/mmb.cu`. It turns one 64-element k-tile of a Q5_K
weight row into 64 bf16 values in LDS, feeding the bf16 WMMA prefill GEMM (MMB). This is model
independent: any architecture with Q5_K tensors (attention, experts, MTP heads) uses the same path.

## Format facts

`block_q5_K` is 176 bytes for 256 weights, laid out as:

| offset | field | notes |
| --- | --- | --- |
| 0 | `d` (fp16) | super-block scale |
| 2 | `dmin` (fp16) | super-block min scale |
| 4 | `scales[12]` | 8 six-bit scales and 8 six-bit mins |
| 16 | `qh[32]` | fifth bit of each weight, one bit per weight per tile |
| 48 | `qs[128]` | low four bits, packed nibbles |

- Eight sub-blocks of 32 weights. Sub-block `j` is scaled by `d * sc[j]` and offset by `dmin * m[j]`.
- `sc[j]`, `m[j]` unpacking (`get_scale_min_k4`): for `j < 4` they are `scales[j] & 63` and
  `scales[j+4] & 63`; for `j >= 4` they are `(scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)` and
  `(scales[j+4] >> 4) | ((scales[j] >> 6) << 4)`.
- Value of a weight: `(nibble | (qh_bit << 4)) * d * sc - dmin * m`, i.e. a 5-bit unsigned code.
- Sub-block `j` covers element range `32j .. 32j+31`, so a 64-element tile is exactly two
  sub-blocks: `2*IL` (elements 0..31, the low nibbles) and `2*IL+1` (elements 32..63, the high
  nibbles).
- `qh` byte `b` holds the fifth bit of tile-local element `b` in bit `2*IL` and of element `32+b` in
  bit `2*IL+1`, and `qs[tile_base + b]` holds the nibble for both. `qh`, `d`, `dmin` and `scales`
  are shared by the four tiles of the block; `qs` advances 32 bytes per tile.

## Addressing rule (easy to get wrong)

A 256-weight block holds four 64-element tiles, but the tiles do **not** sit at a 44-byte stride:
the header (`d`, `dmin`, `scales`, `qh` = 48 bytes) is shared and only `qs` advances. The tile
header is at `row_start + (ks/4)*176` and its nibbles at `+48 + 32*(ks%4)`. Using `ks*44` reads
into the middle of `qh` and produces garbage weights (verified the hard way: NaN outputs).

## Recipe

`template <int IL> mmb_dq_row_q5k(...)` with the eight registers staged by `load_regs`:
`a0` = `d, dmin, scales[12]`, `a1`/`a3` = `qh[0..15]`/`qh[16..31]`, `a4`/`a5` = this tile's 32 `qs`
bytes. `IL` is a template parameter, so the two scale/min pairs, the `qh` shift and every byte
offset are compile-time; the call site switches on `weight_ks & 3`.

Per 64 weights (one thread, one row):
- 8 dword iterations; each produces 4 low-nibble and 4 high-nibble weights:
  - `v0 = (ql & 0x0F0F0F0F) | (((qh >> 2*IL) & 0x01010101) << 4)` (4 ops),
  - `v1` with `2*IL+1` (4 ops),
  - 8 `v_cvt_f32_ubyteN` + 8 fmaf (the byte extraction folds into the conversion),
  - 4 `mmb_pack2` (one `v_perm_b32` after two RNE roundings),
  - 4 LDS stores.
- Two 6-bit scale/min pairs unpacked once per tile (about 20 ops, amortized to 0.3 ops/weight).

Cost is about 5.5 VALU ops per weight, against roughly 8.6 for the IQ4_NL LUT recipe, and the
matrix work per wave-tile is unchanged, so the tile is expected to land at or slightly above the
IQ4_NL MMB rate.

Pitfalls already hit here:
- Do not take the address of a register operand (`(const uint8_t *)&w0`); it forces the value to
  private memory. Unpack bytes with shifts, or templatize the index.
- Mask every byte read out of `scales[]`; the 12 bytes cross three dwords and the last four begin
  mid-dword, so an unmasked shift yields a 24-bit scale and NaN weights.
- `mmb_pack2`'s `v_perm_b32` selector is one byte per output byte (0..7 indexing the two rounded
  values), not one nibble per output byte.

## Measurements

`test-backend-ops perf -o MMB_PERF`, TFLOP/s. Routed = `n_mats 512, n_used 10, m 640, k 2560` with
`n` tokens; dense = `m 2560, k 2560` with `n` tokens. MMQ numbers are the same harness with the
gate sending Q5_K to MMQ.

| shape | MMB | MMQ | delta |
| --- | --- | --- | --- |
| routed n=512 | 4.60 | 5.25 | -12% |
| routed n=2048 | 12.64 | 12.34 | +2% |
| routed n=16384 | 17.43 | 14.85 | +17% |
| dense n=512 | 26.25 | 24.28 | +8% |
| dense n=16384 | 31.22 | 24.96 | +25% |

Model level (pp16384, one repeat): APEX-I-Nano 939 -> 984 t/s, UD-IQ1_S 931 -> 971 t/s; pp512
+1.3-1.9%; tg128 unchanged (the m=1 path is MMVQ and does not use MMB).

Correctness: `test-backend-ops test -b ROCm0 -o MMB_QUANT,MUL_MAT_ID` passes 1021/1021 with the
routed threshold temporarily lowered to 512 tokens so the routed kernel exercises the new recipe
too, and again with the shipped threshold.

## Gate

`mmb_quant_type()` decides the dense path, `mmb_quant_type_mmid(type, n_tokens)` the routed and glu
paths. A type may need both: Q5_K wins densely at every token count and on routed work only once
the token tiles cover the weight (2048 tokens here, i.e. about 40 rows per expert at top-10 of 512),
so the routed gate keeps MMQ below that. When adding a type, measure all five shapes before
touching the gate, and prefer a threshold backed by two measured sizes around the crossover.
