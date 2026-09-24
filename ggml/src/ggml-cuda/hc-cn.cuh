#pragma once
#include "common.cuh"

// block layout of the fused combine+norm kernel, also used by the matcher
#define HC_CN_BLOCK 1024
#define HC_CN_CHUNK 3

// The hyper-connection combine writes the residual it updates and the grouped RMS norm that follows
// immediately reads it back. One block per (stream, token) does both: it keeps the combined row in
// registers, reduces it for the norm, and writes the normed row.
struct ggml_cuda_hc_combine_norm_args {
    const ggml_tensor * post;       // [hc, T]     F32, already scaled and sigmoided
    const ggml_tensor * residual;   // [n_embd, hc, T]
    const ggml_tensor * block_out;  // [n_embd, T]
    const ggml_tensor * gamma;      // [n_embd, hc]
    ggml_tensor *       out_res;    // [n_embd, hc, T]
    ggml_tensor *       out_xn;     // [n_embd, hc, T]
    float               eps;
    // 16-bit residual stream: a marked tensor holds BF16 in its own buffer, read and written in place
    const uint16_t *    res_in_bf16  = nullptr;
    uint16_t *          res_out_bf16 = nullptr;
    uint16_t *          out_xn_bf16  = nullptr;
};

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & args);
