#include "lora.cuh"

// mul_mat_id computes dst[i0, i1, i2] = sum_k src0[k, i0, ids[i1, i2]] * src1[k, i1, i2],
// mul_mat is the same without ids. A LoRA factor is F16 with a rank-sized dimension of 4,
// so it is narrow either in the output dim of lora_a ([K, rank, n_experts] / [K, rank]) or
// in the contraction dim of lora_b ([rank, M, n_experts] / [rank, M]). The wide counterpart
// stays long, so the work is bound by reading the long side once rather than by the
// rank-sized multiplication itself.
//
// Both entry points use the same kernels: per_expert selects whether src1 has one row per
// token or one row per (expert, token), has_ids selects the expert gather.

// lora_a: src0 is [K, rank, ...] and src1 holds a row per token (or per expert and token).
// One block per token keeps the token's activations in shared memory, so all experts of a
// token reuse them. With per-expert inputs there is nothing to share and the activations are
// read from global memory instead.
template <int n_threads, bool per_expert, bool has_ids>
__launch_bounds__(n_threads) static __global__ void mul_mat_lora_a_kernel(
        const __half  * __restrict__ src0, const float * __restrict__ src1,
        const int32_t * __restrict__ ids,  float * __restrict__ dst,
        const int64_t n_k, const int64_t n_rank, const int64_t n_cols,
        const size_t nb01, const size_t nb02,
        const size_t nb10, const size_t nb_c, const size_t nb_t,
        const size_t nb20, const size_t nb21,
        const size_t nb_c_out, const size_t nb_t_out) {
    const int64_t i2 = blockIdx.x;

    // one output per (rank, expert used) pair, split over n_tpo threads for the K reduction
    const int64_t n_out = n_rank * n_cols;
    const int     n_tpo = n_threads / (int) n_out;

    extern __shared__ float smem[];
    float * s_x   = smem;                                  // [n_k], unused with per-expert inputs
    float * s_red = smem + (per_expert ? 0 : n_k);         // [n_out * n_tpo]

    if (!per_expert) {
        for (int64_t k = threadIdx.x; k < n_k; k += n_threads) {
            s_x[k] = *(const float *)((const char *) src1 + k*nb10 + i2*nb_t);
        }
        __syncthreads();
    }

    const int64_t i_out = threadIdx.x / n_tpo;
    const int     i_tpo = threadIdx.x % n_tpo;

    if (i_out < n_out) {
        const int64_t i0 = i_out % n_rank;
        const int64_t i1 = i_out / n_rank;
        const int32_t iexpert = has_ids ? *(const int32_t *)((const char *) ids + i1*nb20 + i2*nb21) : 0;

        const __half2 * s0_row = (const __half2 *)((const char *) src0 + i0*nb01 + iexpert*nb02);
        const char    * x_row  = (const char *) src1 + i1*nb_c + i2*nb_t;

        float acc = 0.0f;
        for (int64_t i = i_tpo; i < n_k/2; i += n_tpo) {
            const float2 v = __half22float2(s0_row[i]);

            float x0, x1;
            if (per_expert) {
                x0 = *(const float *)(x_row + (2*i + 0)*nb10);
                x1 = *(const float *)(x_row + (2*i + 1)*nb10);
            } else {
                x0 = s_x[2*i + 0];
                x1 = s_x[2*i + 1];
            }
            acc = fmaf(v.x, x0, acc);
            acc = fmaf(v.y, x1, acc);
        }
        s_red[i_out*n_tpo + i_tpo] = acc;
    }
    __syncthreads();

    if (threadIdx.x < n_out) {
        const int64_t i_out = threadIdx.x;

        float acc = 0.0f;
        for (int i = 0; i < n_tpo; ++i) {
            acc += s_red[i_out*n_tpo + i];
        }

        const int64_t i0 = i_out % n_rank;
        const int64_t i1 = i_out / n_rank;
        *(float *)((char *) dst + i0*sizeof(float) + i1*nb_c_out + i2*nb_t_out) = acc;
    }
}

// lora_b: src0 is [rank, M, ...], src1 is the result of lora_a.
// The contraction is the rank itself, so one block per (expert used, token) with the rank
// kept in registers is enough; the kernel is bound by writing dst.
template <int n_threads, bool has_ids>
__launch_bounds__(n_threads) static __global__ void mul_mat_lora_b_kernel(
        const __half  * __restrict__ src0, const float * __restrict__ src1,
        const int32_t * __restrict__ ids,  float * __restrict__ dst,
        const int64_t n_rank, const int64_t n_out,
        const size_t nb01, const size_t nb02,
        const size_t nb10, const size_t nb_c, const size_t nb_t,
        const size_t nb20, const size_t nb21,
        const size_t nb_c_out, const size_t nb_t_out) {
    const int64_t i1 = blockIdx.y;
    const int64_t i2 = blockIdx.x;

    const int32_t iexpert = has_ids ? *(const int32_t *)((const char *) ids + i1*nb20 + i2*nb21) : 0;

    float s_x[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        s_x[r] = r < n_rank ? *(const float *)((const char *) src1 + r*nb10 + i1*nb_c + i2*nb_t) : 0.0f;
    }

    const char * s0_col = (const char *) src0 + iexpert*nb02;
    char       * d_row  = (char *) dst + i1*nb_c_out + i2*nb_t_out;

    for (int64_t i0 = threadIdx.x; i0 < n_out; i0 += n_threads) {
        const __half * s0_row = (const __half *)(s0_col + i0*nb01);

        float acc = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < n_rank) {
                acc = fmaf(__half2float(s0_row[r]), s_x[r], acc);
            }
        }
        *(float *)(d_row + i0*sizeof(float)) = acc;
    }
}


template <int n_threads>
static void launch_lora_a(
        cudaStream_t stream,
        const __half * src0, const float * src1, const int32_t * ids, float * dst,
        int64_t n_tokens, int64_t n_k, int64_t n_rank, int64_t n_cols,
        size_t nb01, size_t nb02, size_t nb10, size_t nb_c, size_t nb_t,
        size_t nb20, size_t nb21, size_t nb_c_out, size_t nb_t_out,
        bool per_expert, bool has_ids) {
    const int64_t n_out = n_rank * n_cols;
    const int     n_tpo = n_threads / (int) n_out;
    const size_t  smem  = (per_expert ? 0 : n_k*sizeof(float)) + size_t(n_out)*n_tpo*sizeof(float);
    const dim3    blocks((unsigned int) n_tokens);

    if (per_expert) {
        mul_mat_lora_a_kernel<n_threads, true,  true ><<<blocks, n_threads, smem, stream>>>(
                src0, src1, ids, dst, n_k, n_rank, n_cols, nb01, nb02, nb10, nb_c, nb_t, nb20, nb21, nb_c_out, nb_t_out);
    } else if (has_ids) {
        mul_mat_lora_a_kernel<n_threads, false, true ><<<blocks, n_threads, smem, stream>>>(
                src0, src1, ids, dst, n_k, n_rank, n_cols, nb01, nb02, nb10, nb_c, nb_t, nb20, nb21, nb_c_out, nb_t_out);
    } else {
        mul_mat_lora_a_kernel<n_threads, false, false><<<blocks, n_threads, smem, stream>>>(
                src0, src1, ids, dst, n_k, n_rank, n_cols, nb01, nb02, nb10, nb_c, nb_t, nb20, nb21, nb_c_out, nb_t_out);
    }
}

template <int n_threads>
static void launch_lora_b(
        cudaStream_t stream,
        const __half * src0, const float * src1, const int32_t * ids, float * dst,
        int64_t n_tokens, int64_t n_out, int64_t n_rank, int64_t n_cols,
        size_t nb01, size_t nb02, size_t nb10, size_t nb_c, size_t nb_t,
        size_t nb20, size_t nb21, size_t nb_c_out, size_t nb_t_out,
        bool has_ids) {
    const dim3 blocks((unsigned int) n_tokens, (unsigned int) n_cols);

    if (has_ids) {
        mul_mat_lora_b_kernel<n_threads, true ><<<blocks, n_threads, 0, stream>>>(
                src0, src1, ids, dst, n_rank, n_out, nb01, nb02, nb10, nb_c, nb_t, nb20, nb21, nb_c_out, nb_t_out);
    } else {
        mul_mat_lora_b_kernel<n_threads, false><<<blocks, n_threads, 0, stream>>>(
                src0, src1, ids, dst, n_rank, n_out, nb01, nb02, nb10, nb_c, nb_t, nb20, nb21, nb_c_out, nb_t_out);
    }
}

// shapes and strides the kernels handle: F16 weights, F32 activations, one rank-sized matmul
// dimension, and dense rank-sized dims
static bool lora_shapes_ok(const ggml_tensor * dst, bool has_ids) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (has_ids && dst->src[2]->type != GGML_TYPE_I32) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(GGML_TYPE_F16) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[3] != 1 || (has_ids && dst->src[2]->nb[0] != sizeof(int32_t))) {
        return false;
    }

    const bool narrow_out = src0->ne[1] <= 8 && src0->ne[0] >= 64 && src0->ne[0] <= 4096 && src0->ne[0] % 2 == 0;
    const bool narrow_in  = src0->ne[0] <= 8 && src0->ne[1] >= 64;

    // n_out outputs are spread over the block, so keep the count inside the block size
    if (narrow_out && src0->ne[1] * dst->ne[1] <= 128) {
        return true;
    }
    return narrow_in;
}

bool ggml_cuda_should_use_mmid_lora(const ggml_tensor * dst, int cc) {
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }

    static const bool no_lora = getenv("GGML_CUDA_NO_MMID_LORA") != nullptr;
    if (no_lora) {
        return false;
    }

    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    if (ids->ne[2] != 1 || ids->ne[3] != 1 || !ggml_is_contiguous(src1)) {
        return false;
    }
    // src1 is either one row per token or one row per (expert, token)
    if (src1->ne[1] != 1 && src1->ne[1] != ids->ne[0]) {
        return false;
    }

    return lora_shapes_ok(dst, /*has_ids =*/ true);
}

bool ggml_cuda_should_use_mm_lora(const ggml_tensor * dst, int cc) {
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }

    static const bool no_lora = getenv("GGML_CUDA_NO_MMID_LORA") != nullptr;
    if (no_lora) {
        return false;
    }

    // the mul_mat_id grouped fallback calls this with stack-built slices that carry no sources
    if (dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    // the dense factors have src1 laid out as [K, n_tokens]
    if (dst->src[1]->ne[2] != 1) {
        return false;
    }

    return lora_shapes_ok(dst, /*has_ids =*/ false);
}

void ggml_cuda_mul_mat_id_lora(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(ne20 == dst->ne[1]);  // one ids entry per (expert used, token) pair
    GGML_ASSERT(ne21 == dst->ne[2]);

    constexpr int n_threads = 256;
    cudaStream_t stream = ctx.stream();

    if (ne01 <= 8) {
        launch_lora_a<n_threads>(stream,
                (const __half *) src0->data, (const float *) src1->data, (const int32_t *) src2->data, (float *) dst->data,
                ne2, ne00, ne01, ne20,
                nb01, nb02, nb10, ne11 != 1 ? nb11 : 0, nb12,
                nb20, nb21, nb1, nb2,
                /*per_expert =*/ ne11 != 1, /*has_ids =*/ true);
    } else {
        launch_lora_b<n_threads>(stream,
                (const __half *) src0->data, (const float *) src1->data, (const int32_t *) src2->data, (float *) dst->data,
                ne2, ne01, ne00, ne20,
                nb01, nb02, nb10, nb11, nb12,
                nb20, nb21, nb1, nb2,
                /*has_ids =*/ true);
    }

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_lora(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    constexpr int n_threads = 256;
    cudaStream_t stream = ctx.stream();

    if (ne01 <= 8) {
        launch_lora_a<n_threads>(stream,
                (const __half *) src0->data, (const float *) src1->data, nullptr, (float *) dst->data,
                ne1, ne00, ne01, 1,
                nb01, nb02, nb10, 0, nb11,
                0, 0, 0, nb1,
                /*per_expert =*/ false, /*has_ids =*/ false);
    } else {
        launch_lora_b<n_threads>(stream,
                (const __half *) src0->data, (const float *) src1->data, nullptr, (float *) dst->data,
                ne1, ne01, ne00, 1,
                nb01, nb02, nb10, 0, nb11,
                0, 0, 0, nb1,
                /*has_ids =*/ false);
    }

    CUDA_CHECK(cudaGetLastError());
}
