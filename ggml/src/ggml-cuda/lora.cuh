#pragma once

#include "common.cuh"

// LoRA factors arrive at mul_mat/mul_mat_id as F16 weights with a rank-sized matmul dimension
// (4 for the adapters in use here). Every other path rejects those shapes, so they take the
// fallback that builds the expert permutation on the host and syncs the stream per matmul,
// which also prevents CUDA graphs from covering the graph. These kernels do the same work in
// one launch per matmul, with no host interaction.
bool ggml_cuda_should_use_mmid_lora(const ggml_tensor * dst, int cc);
bool ggml_cuda_should_use_mm_lora(const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_id_lora(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_mul_mat_lora(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// out[M, N] = base[M, N] + scale * lora_b[rank, M] @ (lora_a[K, rank] @ x[K, N])
//
// Dense case: one launch computes the two matmuls, the scale and the add, so x, base and out are
// each touched once instead of the [M, N] product being written and re-read by a scale and an add.
// lora_a and lora_b are F16, x, base and out are F32, and the rank must be at most 8.
void ggml_cuda_mul_mat_lora_apply(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// u[rank, n_tok] = lora_a @ x, consumed by the fused MMB epilogue (ggml_cuda_mul_mat_mmb_lora)
void ggml_cuda_lora_u(const void * la, const float * x, float * u, int64_t n_k, int64_t n_tok, int n_rank, cudaStream_t stream);
