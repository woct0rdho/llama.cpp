# Q2_0: MMB against MMQ, and MMVQ

Operator: `GGML_TYPE_Q2_0` (`QK2_0 = 64`, `block_q2_0 = { ggml_half d; uint8_t qs[16]; }`, 18 bytes per block, 2.25 bits per weight) in `mmb_quant_type` (`ggml/src/ggml-cuda/mmb-quant.cuh`), `ggml_cuda_should_use_mmq` (`ggml/src/ggml-cuda/mmq.cu`) and the MMVQ dispatch (`ggml/src/ggml-cuda/mmvq.cu`).

Introduced for `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf`, where it is 51% of the tensor bytes: all 144 expert tensors (`ffn_gate_exps`/`ffn_up_exps` `[2560, 640, 512]` and `ffn_down_exps` `[640, 2560, 512]`, 32400 MiB) plus a handful of attention projections (202 tensors, 31.7 GiB of 61.85 GiB).

## State before tuning

`ggml_cuda_should_use_mmq` already listed Q2_0, so MMQ handled every prefill size and MMVQ handled one token. The MMB side was also already wired: `mmb_dispatch_quant` has a `GGML_TYPE_Q2_0` case and `mmb-quant.cuh` carries a `dequantize_q2_0` slice path with `QK = 64`. What was missing was the gate: `mmb_quant_type` did not list the type, so MMB never ran it. Adding it there is what makes an A/B possible at all.

## MMB against MMQ

`test-backend-ops perf -o MMB_PERF`, TFLOP/s, ROCm backend only. Routed is the model's expert shape (`n_mats 512, n_used 10, m 640, k 2560`), dense is the attention width (`m 2560, k 2560`). Both columns come from the same binary, with a temporary gate override for the measurement.

| case | MMQ | MMB | winner |
| --- | --- | --- | --- |
| routed t=512 | 7.17 | 4.61 | MMQ 1.56x |
| routed t=2048 | 13.50 | 8.86 | MMQ 1.52x |
| routed t=4096 | 14.20 | 9.76 | MMQ 1.45x |
| routed t=8192 | 14.66 | 14.16 | MMQ 1.04x |
| routed t=16384 | 15.24 | 16.05 | MMB 1.05x |
| dense t=512 | 26.15 | 22.91 | MMQ 1.14x |
| dense t=16384 | 26.85 | 21.39 | MMQ 1.26x |

Decision: Q2_0 stays on MMQ, in the same group as the codebook quants. MMB wins only at the single deepest routed point, and only by 5%, while it loses 4% at t=8192 and 20-26% on the dense shapes at both ends. A token threshold that captured the one win would have to sit above 8192 - but `mmb_quant_type` is shared by the routed and dense paths, so it would also pay the dense regression unless a second per-type rule was added to `mmb_quant_type_mmid`, for one shape at +5% that is inside the crossover noise. `test-backend-ops perf` runs 29-30 iterations for that case, which is the thinnest evidence in the table.

Why MMQ wins: the 64-element block has to be unpacked per slice in MMB (`lane` walks 32 positions at a time over `2.25` bits per weight), and that unpack cost is not amortized at these K widths, while `MMQ_DP4A_TXS_Q8_0` reads the packed 2-bit weights natively with the int8 WMMA engine. Making MMB win here means a faster Q2_0 bit-unpack in `mmb_decode_slice`, which is kernel restructuring work (P1-6), not a gate change.

## MMVQ at one token

`test-backend-ops perf -o MMVQ_PERF`, us per call, with TFLOP/s in brackets:

| shape | Q2_0 | IQ4_NL | Q3_K | Q4_K |
| --- | --- | --- | --- | --- |
| lm head m=248320 k=2560 | 839 us [1.52] | 1513 us [0.84] | 1178 us [1.08] | 1512 us [0.84] |
| dense m=10240 k=2560 | 22.0 us [2.38] | 21.6 us [2.42] | 32.1 us [1.64] | 29.8 us [1.76] |
| expert gate/up m=1280 k=2560 | 4.77 us [1.37] | 4.34 us [1.51] | | |
| expert down m=2560 k=640 | 4.48 us [0.73] | 5.56 us [0.59] | | |

Q2_0 is the fastest type measured at the lm head (1.8x IQ4_NL at 839 us for 179 MB of weights, which is 213 GB/s, at the DRAM roofline) and at the expert down matvec, at parity on the dense projection, and 10% behind IQ4_NL on the expert gate/up. Both expert shapes are launch bound at 4-5 us whatever the type reads, so no MMVQ change was made. The paths taken are `vec_dot_q2_0_q8_1` with `VDR_Q2_0_Q8_1_MMVQ` and `mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>`.

## What it costs the model

The gate decision leaves 15.24 TFLOP/s on the table where an IQ4_XS-like MMB path reaches 17.62 TFLOP/s at the same routed shape (measured in `mmb_iq4_xs.md`), and the model runs 144 expert GEMMs of 537 GFLOP per 16384-token pass. That is 5.07 s against 4.39 s, so about 0.7 s, or 4% of the 17.25 s pass. It is a bounded, known gap that a faster MMB dequant would recover, and it does not change the gate decision today because the current MMB dequant is slower than MMQ at every shape but one.

Model level, `llama-bench` with PM4 graphs on: pp512 746.34 +/- 13.65 t/s, tg128 34.36 +/- 0.14 t/s, pp16384 949.84 t/s, PPL 3.6730 on the docs corpus at `-c 8192 -b 8192 -ub 8192 --chunks 1` (IQ4_NL 3.4149 and APEX-I-Nano 3.6257 on the same corpus and settings).
