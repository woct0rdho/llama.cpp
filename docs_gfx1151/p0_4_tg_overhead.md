# P0-4: TG per-token overhead - what is measurable and what is not

## The premise is solid

TG128 is 27.9 ms per token with 4.254 GB of weight bytes per token, i.e. 152 GB/s against a measured
DRAM read ceiling of 230 GB/s. The weight kernels are therefore not the whole story; about 9 ms per
token is something else. That accounting only needs bytes and time, so it holds regardless of how the
kernels are attributed.

## The profiler does not attribute time on this machine

`rocprofv3 --kernel-trace` returns dispatch records whose `Start_Timestamp`/`End_Timestamp` do not
reflect execution: a 128-token decode produces 10716 dispatches over a 26.2 s span with 61 ms of
total recorded duration, i.e. 0.23% "busy", and 145 idle gaps that include the 27.9 ms per-token
period. The dispatch counts are useful (84 per token, which is far below the real kernel count), the
durations are not, and the percentages built on them are misleading: they rank kernels by how long
the *dispatch* took, not by how long the GPU ran them.

The plan's original P0-4 evidence was collected the same way, so its 2.05 ms / 0.88 ms / 1.65 ms
figures are launch-cost estimates rather than execution times and should be read as an upper bound on
the host side only. Anything built on them needs to be re-measured end to end.

## What the numbers look like end to end

The only reliable signal is the accepted one: pp512 and tg128 with 3 repeats, pp16384 with 1 repeat,
all three models, one exclusive job at a time. TG128 run-to-run spread on this machine is about
0.4% in a quiet state and up to 3% when the APU has been busy, so a TG change below roughly 1% is not
evidence of anything.

## Tried and reverted: the combine's control chain

`build_hc_combine` builds `w = 2*sigmoid(inject/hc)` as three separate nodes (SCALE, UNARY/SIGMOID,
SCALE) on an `[hc, T]` tensor - four floats per token at TG - and that is 288 launches per token at
96 combines per pass. The kernel now computes the weight itself, matched structurally: the three
nodes have to be immediately in front of the combine, which they are, and the combine has to be the
only reader of each.

Measured, it is neutral: tg128 36.07 / 33.25 / 35.84 against 36.19 / 33.82 / 35.88, inside the
run-to-run spread, with identical PPL (3.4126 / 3.6258 / 3.6291 against 3.4131 / 3.6290 / 3.6320).
So the three tiny kernels cost less than 0.5% of TG: the launches are already overlapped with the
GEMMs that surround them. The change was reverted rather than kept unmeasured, since the matcher is
about forty lines of structure that buys nothing that can be demonstrated.

That is the useful negative result of this item: the per-token launch overhead is not where the 9 ms
is, and the way to find where it is, is not to count kernels.

## What is left, with honest estimates

- MoE activation quantization. Each token is quantized to q8_1 once per expert that uses it,
  ten times over, because the expert kernels read their own slice of the sorted activation buffer.
  The work is 7x the necessary bytes but the same number of launches; deduplicating it means
  quantizing before the sort and gathering q8_1 blocks instead of F32 rows. Estimated 1-2%.
- The GDN conv window shift. Three 12-byte rows copied per layer through `cpy_scalar_transpose`
  with a 122880-byte extent. The fix is a window pointer or an offset in the conv kernel, not a
  faster copy. Estimated 0.5-1%.
- The gate+mix path at m=1. The fused gate+mix kernel needs 512 rows, so TG runs the gate GEMM,
  the PRE op and nothing else - two kernels where one would do. An m=1 epilogue on the matvec is the
  way, about 0.7%.
- The LM head is 0.68 GB per token, 16% of the weight bytes, and stays quantized.

None of these is large enough to justify guessing at it, which is why this item stops here: the
measured state is committed, the profiler cannot attribute the remaining 9 ms, and the next step is a
timing method that works on this APU - a wall-clock ablation per kernel family (disable one family,
measure TG) rather than a trace.
