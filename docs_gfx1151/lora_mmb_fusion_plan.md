# Fusing the LoRA apply into the MMB matmul

Status: implemented for both halves - dense weights and routed (MoE) experts - plus a slot-batched `u` prepass for the expert half and an f16 WMMA correction for the routed geometry. This round landed as three commits (the prepass, the epilogue, and the compile-time pass) on top of the earlier fusion work. Prefill tax on the combined adapter went from 30.6 % to about 15.5 % (pp 1065 -> ~1300 t/s at 6329 tokens, session noise ~1 %). Decode is untouched and is now the largest single item left, see the separate PP and TG plans at the end.

No adapter is ever baked into a quantized weight: this is a graph rewrite plus an epilogue.

The separate-pass `GGML_OP_MUL_MAT_LORA` that this builds on is also in the tree and validated.

## Why a separate pass cannot get much better

Measured prefill tax at 6329 tokens, ub 8192, Qwen3.8 IQ4_NL (`~/tmp/llama-cpp/lora-server-perf.md`), before any fusion. Tax is `1 - pp_lora/pp_base`, i.e. the share of the slower run's time that the LoRA machinery costs:

| config | pp | tax |
| --- | --- | --- |
| no lora | 1523 t/s | - |
| dense weights only (288) | 1360 | 10.7 % |
| expert weights only (144) | 1167 | 23.8 % |
| all weights | 1070 | 29.8 % |

A separate apply pass must read `x`, then read and write the [M, N] output: 3 passes per weight, e.g. 195 MiB for a 2560x2560 weight at 16384 tokens. That floor is ~6 % for the dense half, and the MoE half cannot even reach its own floor without sorting (the factor traffic of an ungrouped loop order is `N * n_used * M * rank * 2` = 3200 MiB per ffn_down_exps weight). For an expert weight the product is `[M, n_used, T]` - ten times a dense output at the same token count - which is why the MoE half is where the win is.

Fusing into the matmul that already produces the output removes both: MMB writes `out` once, and the LoRA correction is added to the accumulator before that store.

## Where it goes

`mmb_store_tile()` (`ggml/src/ggml-cuda/mmb.cu`) is the single epilogue used by every MMB tile kernel (`mmb_tile_gemm`, `mmb_tile_gemm_glu`, the routed/id variants). Each lane there holds eight consecutive M rows of one output column:

```
n = lane >> 1;  half = lane & 1;
column = n_base + n;                     // output column, i.e. the token for a dense weight
rows   = m_base + half*8 + [0..8);       // eight consecutive M rows
v0, v1 = *(const float4 *)(stg + ...)    // the tile's accumulator values
```

so the correction is a rank-deep update per lane, before the store. There are two implementations of it, and which one runs depends on the geometry:

```
# scalar chain, for the wide tile and for weights with a w_s factor
if (bs) v[k] *= bs[column]                           // the graph's w_s factor, if this weight has one
for i in [0, rank):
    ui = lu[i + rank*column]                         // one float, reused for eight rows
    h8 = *(const uint4 *)(lb + (m_base + half*8)*rank)   // 8*rank halves, whole 16-byte reads
    v[k] += scale * ui * half_to_float(h8[k])

# f16 wmma, for the routed geometry (TM <= 2 and lr.eb != 0)
ac = { lora_b rows (lane & 15), 16 halves = rank values + zeros }   // registers, 2 loads
bc = { u column (lane & 15),   16 halves = rank f16 values }        // registers, rank cvt
acc[i][j] += wmma_f32_16x16x16_f16_w32(bc, ac, acc[i][j])           // 1 instruction per fragment
```

`lora_b` is F16 and the f16 wmma takes F16 bits as they are, which is why the wmma path is f16 and not bf16 (fed to the bf16 variant, a 0.01 factor becomes a BF16 denormal and the correction silently disappears).

Four layout facts, each of which cost a debugging round:
- `lora_b` is `[rank, M]` (or `[rank, M, n_expert]`) in ggml, so `(i, m)` sits at `i + m*rank` - the eight rows of one lane and all ranks are one contiguous span `8*rank` halves, read as whole `uint4`s, not a column per rank.
- `u` is a ggml tensor, so the rank axis is innermost: `(i, col)` is at `i + rank*col`. Indexing it as `i*n_col + col` (row-major, which is what the dense `u` buffer originally used) reads the wrong column. The result looks plausible and only perplexity catches it.
- the column is the global one, i.e. what `drow()` returns (the same value the store uses), not the tile-local `n_base + n`. For a dense weight that is the token. For an expert weight it is `n_used*token + slot`, which is exactly the column of `u = mul_mat_id(lora_a, x, ids)` (the reference lays it out as `slot + token*ne11`), so both halves use the same index.
- `ids` is `[n_used, T]`, so the expert of `(slot, token)` is `ids[token*si1 + slot]`, not `ids[slot*si1 + token]`. Getting that backwards reads out of bounds and faults the queue, it does not merely give wrong numbers.

## How it is wired

The fork already has a graph fusion framework (`ggml_cuda_try_fuse`, `ggml_can_fuse_subgraph_ext`, `ggml_node_has_n_uses`), so the apply is folded at the graph level.

Dense: `MUL_MAT` followed by `MUL_MAT_LORA` on its result becomes one node. `ggml_cuda_mul_mat_mmb_lora` owns its launch: it re-checks `mmb_quant_type_dense` / `ggml_cuda_mmb_supported_mm` itself and launches exactly one `mmb_dense_lora_kernel` instantiation, with the same weight representation (`mmb_shadow_lookup` for IQ4_NL/Q6_K) and the same tile geometry the unfused matmul would have picked, so the base product stays bit-identical and the only difference left is the correction. `u` is computed there by `ggml_cuda_lora_u`.

MoE: `mul_mat_id(b, scale(mul_mat_id(a, x)))` added on top of `mul_mat_id(w, x)`, five adjacent nodes. `build_lora_mm_id` emits the chain first (it does not depend on the product) and puts the LoRA term first in the `add` (addition is commutative in IEEE arithmetic). The fusion is anchored at the `u` matmul and replaces all four nodes after it, so `ggml_cuda_mul_mat_id_mmb_lora` owns the whole chain: it builds `u` itself with `ggml_cuda_lora_u_id`, then runs the routed matmul with the correction in its epilogue. Nothing else computes `u`, the `b`-matmul result never exists, and the `[M, n_used, T]` product is written once instead of three times. A weight whose graph multiplies the product by a shared-expert scale (`w_s`) before the add folds that factor in too - it is per (slot, token), i.e. per column, the index the epilogue already has.

The `u` prepass (`lora_u_id_kernel`, `ggml/src/ggml-cuda/lora.cu`) is where the expert half's remaining traffic was. ggml's `mul_mat_id` gathers `x[.,t]` once per used slot, i.e. `n_used` = 10 times per token, because each column of `u` is a separate gathered row. The prepass keeps one warp per token and produces all `n_used` slots of that token in one pass over `x`, so `x` is read once and the expert factors are read ten times instead - they are 10 MB per weight and stay in L2, while `x` is the whole activation and comes from DRAM. It has to respect that `u`'s input column is ggml's `slot + n_used*token`: for the down projection, whose activation is `[K, n_used, T]`, every used expert has its own column, so there the pass is per slot by necessity (and that activation is ten times bigger to begin with). Getting this wrong is a PPL move of +0.010, not a crash.

Gates: F16 contiguous factors, rank even in [2, 8], `K % MMB_BK == 0`, `M % 16 == 0`, `x` is `[K, 1, T]` or `[K, n_used, T]` (the down projection's per-slot SwiGLU output), and not `ggml_cuda_mmb_is_bf16_only(dst)` (the fused epilogue stores F32, so those graphs keep the separate chain). The wmma epilogue additionally requires `TM <= 2` (on the wide tile its rank variants cost that kernel its register budget) and `lr.eb != 0` (the routed geometry, where the operand build has been validated to 0.02 % PPL. The dense narrow path keeps the scalar chain until PP #2 lands).

Debug switches: `GGML_CUDA_NO_LORA_FUSE=1` disables both halves, `GGML_CUDA_LORA_FUSE_DEBUG=1` prints one line per fused weight and the reason for every rejected candidate, `GGML_CUDA_LORA_FUSE_BENCH=1` runs the kernel microbenchmark below. There is no way to get a silently-missing correction: the fused path owns its launch instead of adding a defaulted parameter to the 29 MMB launch sites, so a call site that nobody updated cannot compile.

## The four things that had to be fixed before it paid off

Scratch-memory spill in the epilogue. With a runtime `rank`, the operand staging array `uint4 hv[...]` is indexed dynamically, so the compiler drops it in scratch memory. The correction then cost ~15 ps per output element, 40x what its four FMAs cost, and ate the entire win. Making rank a template constant (`switch (lr.rank) { case 2/4/6/8: corr(integral_constant<int, r>{}) }`) puts the staging in registers and cuts the correction cost by 60 %.

Tile geometry. The first version forced 128x128. For the big-M weights the unfused matmul uses 128x256, which is 27-38 % faster there - so the fusion gave back on the base matmul exactly what it saved on the passes, and the whole-model gain was +0.6 %. Mirroring the geometry choice (including the `shadow && K >= 2560 && T >= 4096` clause) moved it to +3.7 % on the dense half.

Node order in the MoE half. The graph is emitted depth-first from its output, so building `u` before the product in `build_lora_mm_id` does not put it earlier in `cgraph` - only putting the LoRA term first in the `add` does. And every node of the chain that the fusion does not include still runs: anchored at the product it skips only the add (1292 t/s on the expert-only adapter), anchored at the `b`-matmul it skips the product too (1373), and anchoring at the `u` matmul skips the whole chain and builds `u` once per token (1414). Each step is a whole weight class of saved work, so the anchor is worth re-checking whenever the builder changes.

Rejecting the down projection for a non-reason. `ffn_down_exps` did not fuse for a while because its activation is `[K, n_used, T]` (one SwiGLU result per used slot) and the gate demanded `ne[1] == 1`. The routed kernel already gathers that layout correctly. Worth remembering when adding gates: a gate that is stricter than the kernel it guards silently costs a whole weight class.

## Results

Kernel microbenchmark (`GGML_CUDA_LORA_FUSE_BENCH=1`, synthetic IQ4_NL weight, rank 4, T=8192, ms per launch, 20 reps). The columns are the scalar chain before and after the rank templating, which is the historical progression. The current state has a third column per geometry, the f16 wmma:

| M | K | plain 128x128 | scalar 128x128 | delta | plain 128x256 | scalar 128x256 | delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10240 | 2560 | 18.70 | 19.20 | 6.8 % -> 2.6 -> 0.1 % | 13.53 | 14.39 | 8.9 % -> 6.3 % |
| 12288 | 2560 | 22.46 | 23.05 | 6.7 % -> 2.6 -> 0.1 % | 16.86 | 17.76 | 9.7 % -> 5.3 % |
| 6144 | 2560 | 11.31 | 11.58 | 6.7 % -> 2.4 -> 0.1 % | 8.20 | 8.72 | 8.5 % -> 6.4 % |
| 2560 | 6144 | 11.50 | 11.59 | 2.6 % -> 0.8 -> 0.2 % | 9.90 | 10.11 | 2.5 % -> 2.1 % |
| 640 | 2560 | 1.35 | 1.37 | -0.1 % -> 1.9 -> -0.6 % | 1.12 | 1.13 | 0.0 % -> 1.1 % |
| 2560 | 640 | 1.35 | 1.45 | 21.9 % -> 7.3 -> 3.0 % | 1.10 | 1.25 | 22.3 % -> 14.0 % |
| 512 | 2560 | 1.29 | 1.31 | 2.2 % -> 1.5 -> -0.5 % | 1.26 | 1.32 | 1.4 % -> 5.1 % |

Bold is the f16 wmma, which the bench only exercises with `lr.eb != 0` (the routed geometry) - the same switch the kernel uses, so the 128x256 column there is the scalar path and stays at 2.1-6.4 %. The correction costs 1-6 % of the matmul it rides on for K >= 2560, and it is the same absolute time in both geometries - it is proportional to the output elements, so it looks relatively worse on the fast tile and on thin-K weights. At K=640 an eight-fold increase in matmul speed would still leave a double-digit correction, because the correction is arithmetic per output element while the base matmul amortizes its MACs over K. That is exactly the shape of PP #1-#3 below.

Whole model, Qwen3.8 IQ4_NL, 6329-token prompt, ub 8192, 3 reps, 3rd rep:

| config | fused | fused off | tax (fused / off) |
| --- | --- | --- | --- |
| no lora | 1528 | - | - |
| dense only (288 weights) | 1430 | 1364 | 7.2 % / 10.7 % |
| experts only (144 weights) | 1414 | 1165 | 8.2 % / 24.1 % |
| all weights | 1302 | 1065 | 15.5 % / 30.6 % |

Session-to-session the no-lora baseline drifts by ~1 % (1523-1540 absolute), so the taxes are the comparable number and carry about ±0.5 points. The same three rows re-measured after the compile-time commit came out 1411 / 1412 / 1297-1311, inside that noise. All 288 dense weights and all 144 expert weights fuse (gate/up/down, 3 per layer x 48).

Decode is unaffected by the fusion, as expected: tg 33.4 fused and fused-off with the dense adapter. The adapter itself costs 36.6 -> 33.1 t/s at 1 token, and the cause is kernel count, not work: disabling the narrow-shape LoRA kernels (`GGML_CUDA_NO_MMID_LORA=1`) changes decode by 0.4 % and prefill by 0.13 %, so those kernels are near-free - the cost is the fallback chain's 4 kernels per weight per token (a, scale, b, add), roughly 1.7 k launches against a 30 ms token.

Correctness: `llama-perplexity -c 8192 --chunks 2` gives PPL 3.2274 with `hc-dense.gguf` (identical to fused-off: the dense half is bit-exact) and 3.2638 with `hc-exps.gguf` against 3.2650 fused-off. The routed epilogue's f16 operands are the only difference: the same fused path with the scalar epilogue measured 3.2631, i.e. 3.2631 -> 3.2638 for the f16 operands (0.02 %) and 3.2650 -> 3.2631 for the prepass's reduction order (0.06 %). Both adapters at once: 5.2070 fused vs 5.2196 fused-off. `tests/test-backend-ops` `MUL_MAT_LORA` is 17/17 (13 shapes incl. N=262144, plus 4 fused-graph cases). Those exercise the separate-pass op. The graph-level fusion does not fire under the test harness, so its regression test is the perplexity comparison above - see "Measurement gaps".

## Where the remaining tax is

The pass is ~15.5 % taxed. Per weight it is still the epilogue's per-element arithmetic plus one prepass. At 6329 tokens, K=2560 (shares overlap):

| component | share | why it is still there |
| --- | --- | --- |
| dense epilogue, 288 weights (128x256 tile) | 2-6 % of those kernels | these still run the scalar chain: the wmma path is gated off on the wide tile because its four rank variants overflow that kernel's register budget |
| expert epilogue, gate/up, 96 weights | 0.1-0.2 % of those kernels | shipped: f16 wmma with register-built operands |
| expert epilogue, down, 48 weights (the largest outputs) | 11-17 % of those kernels | excluded from the wmma path because `w_s` has to scale the product before the correction |
| dense `u` prepass | ~1.7 % | one launch per weight, each reading all of `x` (65 MiB at 6329 tokens) for a 4*K*T = 65 MMAC product |
| MoE `u` prepass | ~3.5 % | reads `x` once per weight but the expert factors ten times: ~180 GiB of L2 traffic per pass |

Two facts still explain all of it: the correction is instruction-bound rather than bandwidth-bound (a WMMA does 4096 MACs per instruction, a rank-long FMA chain does one), and it scales with `M * n_used * T`, so at K=640 it is inherently ~15 % whatever the tile does.

The prepass is L2-bound now, not DRAM-bound: its traffic is `n_used * rank * K * 2` bytes of expert factors per token (200 KiB at K=2560), and the factors are 10 MB per weight so they hit L2. Reading `x` once instead of ten times is what took this from ~6 % to ~3.5 %. Halving it further needs two tokens to share a factor read, which only happens when they picked the same expert - not worth chasing.

## Plan: prefill (PP)

Shipped this round: the MoE `u` prepass (-2.3 points, L2-bound, so less than the traffic estimate), the f16 WMMA epilogue for the routed geometry (correction 0.1-0.2 % there), and the `u` prefetch above the epilogue's staging barrier. Open, ordered by expected value:

| # | change | expected | note |
| --- | --- | --- | --- |
| 1 | Fold the `w_s` factor into the fragment so the 48 down-projection weights get the wmma path | -1 to -1.5 points | the accumulator layout already gives each lane its own column (`stg[(2e+cn)*16 + cm]`), so a lane can scale its own eight elements by `bs[col]` before adding the wmma result - "scale the product first" only rules out scaling *after* the add |
| 2 | Wide tile: source the wmma operands from the tile LDS per fragment | -2 to -3 points on the dense half | what failed before was hoisting `v16s ac/bc` out of the j-loop, which cost that kernel its register budget (plain path 14 -> 36 ms). The k-loop already reads operands from `As`/`Bs` with `r = lane & 15`, and a 1 KiB `lb` + 2 KiB `u` tile fit in the unused tail of the other buffer, so the operands need no registers at all |
| 3 | `v_dot2c_f32_f16` for whatever stays scalar | ~1 point | one instruction per two rank values, f32 accumulate. Halves the chain on the thin-K and `w_s` weights |
| 4 | Store `u` as f16 | -0.5 to -1 point | the wmma operands are f16 anyway, so this halves the prepass writes and the epilogue reads and removes the per-fragment conversion |
| 5 | Bake the adapter scale into `lora_b` at load | removes a kernel per weight | the scale rides on `u` today. Folding it into the F16 factors once is a rounding change of the kind already accepted |
| 6 | Group the dense prepass per input tensor | -1 point | 2-3 weights of a layer share `cur`. Needs a graph-level prepass node |
| 7 | Prepass on a side stream | hides ~1.7 points | `u` only needs `x`, which is ready a layer early. Worth it after 4 and 6 shrink the number |
| 8 | In-kernel `u` (k-loop accumulation) instead of the prepass | probably not | zero extra traffic, but `M/BM` redundancy (5x gate/up, 20x down) puts it at ~100-200 ms of compute against the prepass's ~130 ms, and it perturbs the hot loop |

If 1-4 land, the pass should be at roughly 9-10 % tax, i.e. within a few percent of the no-LoRA prefill number.

## Plan: decode (TG)

36.6 -> 33.1 t/s with the adapter, and the measurement above says that is kernel count, not work. The fallback chain runs 4 kernels per weight per token. The fused op that already solves this for prefill exists.

| # | change | expected |
| --- | --- | --- |
| 1 | Emit the fused apply op at 1 token too (`build_lora_mm` gates it on `ne[1] >= 256`) and give `mul_mat_lora_apply` a one-token path (warp per row, `rank` FMAs) | recovers most of the 9 %: 4 launches per weight become 1 |
| 2 | Fuse the correction into `mmvq_mmid` for the id path | the same for the 144 expert weights x n_used |
| 3 | Bake the scale into `lora_b` (PP #5) | one launch less per weight per token even before #1 |
| 4 | Profile the decode step with rocprofv3 before building #1/#2 | confirms the launch count is the whole story. `~/ComfyUI-FeatherOps/docs/gfx1151_reference.md` has the gfx1151 profiling notes |

## Not worth doing

- More sorting of expert-factor traffic: 200 KiB per token per weight is inherent to rank x n_used x K, and sorting tokens by expert costs more than the ~3.5 % it could save.
- Merging `lb @ la` into per-expert delta weights: that materialises M x K per expert, i.e. as large as the experts themselves.
- Rows-per-instruction packing beyond `dot2`: after PP #1-#3 the remaining scalar cases are the cheap ones.

## Build time

`mmb.cu` took 269 s to compile, all of it in one HIP object (each link is 0.13 s and every other file together is under a second). The compile-time commit brought it to 104 s by removing instantiations that can never run (the fused dense kernel now dispatches over the three representations it can be handed, the epilogue is one body instead of five, and the dense/hyper-connection dispatchers name their set instead of covering all 28 quant types). `-O1` gives 72 s, `-O2` 100 s, so flags alone do not reach a minute. The next step, deferred on purpose because a 1000-line move would dominate a rebase, is the layout `mmq.cuh` uses: templates in a shared header plus thin `.cu` files per kernel family (`mmb-glu.cu`, `mmb-routed.cu`, `mmb-dense.cu`) that re-`#define` their entry and expose one launcher, so the three families compile in parallel (~30-45 s expected). Builds up to 2 minutes are acceptable for now.

## Relationship to the narrow-shape apply kernels

The fork already carried a set of narrow-shape apply kernels (`mul_mat_lora_a/b_kernel`, `ggml_cuda_mul_mat_lora`, `ggml_cuda_mul_mat_id_lora`) that accelerate the *generic chain's* two small matmuls (rank <= 8 out, K = rank in), picked from `ggml_cuda_mul_mat` / `mul_mat_id` via `lora_shapes_ok`. Measured with `GGML_CUDA_NO_MMID_LORA=1`: prefill 1311 vs 1309 t/s (superseded there, since above 256 tokens the fused op replaces the whole chain) and decode 30.63 vs 30.50 (0.4 %, i.e. the only place it still runs). They are also gated on `GGML_CUDA_CC_IS_RDNA3_5`, the same gate as `mmb_enabled()`, so they are not a general-machine optimization today. Making them one is a change to that predicate plus benchmarks elsewhere. Keep them either way: they cost nothing at prefill and they are the fallback wherever the fusion declines.

## Measurement gaps

The fusion cannot be exercised by `test-backend-ops`: the harness's graphs do not match the patterns (the dense test builds the two nodes but the fusion never fires there), so today the only regression test for the fused path is whole-model perplexity equality. That is too slow to iterate with - every A/B in this document cost two model loads - and too coarse for the id path, where one run validates 144 weights at once.

Missing, in order of usefulness:
- A routed-path kernel benchmark next to the dense one. `GGML_CUDA_LORA_FUSE_BENCH=1` measures `mmb_dense_lora_kernel` in six modes now (plain, scalar and wmma per geometry, with `lr.eb` as the switch that enables the wmma path), but the routed kernel has no such harness, so the numbers for the geometry that actually carries the expert half come from the model. Shapes that matter: M=640/2560, K=2560/640, n_used=10, T=4096/8192, both tile classes (BN=128 and BN=32).
- A decode loop that reports kernel counts and per-kernel time. The TG plan above is built on the inference that 4 launches per weight per token are the whole cost. One rocprofv3 trace of a decode step would confirm it, and the same harness would then measure whether PP #5 and TG #1 actually remove launches.
- A direct test of the epilogue against a naive reference at a few shapes - compute `base + scale*lb@u` with a trivial kernel and compare, which would have caught the `u` layout bug in seconds instead of four model runs.
- A way to make the harness fire the graph fusion, or an explicit fused op (`GGML_OP_MUL_MAT_LORA` already exists for the separate pass, so the honest fix is to give the fused path its own op instead of a pattern match). That would also remove the `build_lora_mm_id` operand-order dependency, which exists only so the pattern match can see the correction chain first, and it would make the fused path testable at a fixed shape instead of a whole model.

The one behavioural gap in the current tree is trellis: the compile-time commit declined trellis types in the fused dense LoRA entry (they keep the separate-pass chain, i.e. the pre-fusion behaviour) and there is no trellis model here to run, so that fallback is unexercised.
