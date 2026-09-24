# Sparse decode attention (QSA) at long context

Change: `2f3a77aec` (`ggml/src/ggml-cuda/qsa-prefill.cu`, `fattn.cu`, `tests/test-backend-ops.cpp`)

## The problem

The QSA sparse kernel needs `q->ne[1] >= 128`, so decode took the dense flash attention path and
read the whole cache: at 131072 context that is 1.6 GB of K/V per token over the twelve full
attention layers, where the selection names 2051 cells (25 MB).

## Why the prefill path could not be reused

`qsa-prefill` packs K and V into a grouped layout first. That is right for prefill, where one K tile
serves four query rows and the pack is amortised, but at one row per pass the pack alone would read
every key, which is the entire cost the sparse path is supposed to avoid. So the decode kernel
gathers K and V straight from the cache through the ids.

Two properties of that shape drive the implementation:
- One query row gives twelve workgroups (one per head), far too few to reach DRAM.
- The V pass is then a sum over keys with one two byte load each, so it is latency bound.

Both are answered by splitting the keys into chunks of 512 and merging the partial softmax states in
a second small kernel: 60 workgroups instead of 12, and the loads of four keys in flight per thread.
The measured result is flat in depth, which is what the sparse path has to be.

The mask is an additive bias in logit space with -inf as the visibility flag, not a visibility flag
alone; the prefill kernel adds it (`qsa-prefill.cu:340`) and so does this one.

## Kernel numbers

`test-backend-ops perf -o QSA_DECODE`, one query row, twelve heads, gfx1151:

| cache keys | sparse | dense |
| ---: | ---: | ---: |
| 4096 | (not taken) | 32.8 us |
| 16384 | (not taken) | 107.5 us |
| 32768 | 147.3 us | 294.6 us |
| 65536 | 148.0 us | 1147.4 us |
| 131072 | 147.9 us | (grows further) |
| 131072, 16 rows | 2161.9 us | 4914.9 us |

The dense cost grows with the cache and this one does not, so the crossover is a ratio: about twelve
cache cells per selected cell. `qsa_decode_supported` uses exactly that and returns false below it,
which keeps the 16384 key shape on the dense kernel (108 us against 147 us).

## End to end

`llama-bench -n 128 -d 65536`, IQ4_NL: 19.02 t/s against 17.53 t/s with the decode path disabled
(+8.5%). The saving grows with depth because the kernel time does not: at 131072 keys the dense path
costs 12 ms per token more than this one.

pp512 (960/745/785), pp16384 (1285/1127/1094), tg128 (34.9/33.8/35.8) and PPL
(3.4125/3.6290/3.6320) are unchanged, since prefill and short-context decode keep their old paths.

## Not done here

- The indexer scan (full score GEMM, relu-sum, top-k, gather) and its 0.4 GB of cached indexer keys
  per token at 131072, which is the next depth-dependent cost.
- Quantised K/V for the sparse path, which would halve both the 25 MB selection read and the prefill
  union traffic.
- The prefill path still packs all of K/V every pass: 413 GB of union traffic at 16384 tokens, read
  once per group of four query rows.
