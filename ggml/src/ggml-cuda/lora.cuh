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
