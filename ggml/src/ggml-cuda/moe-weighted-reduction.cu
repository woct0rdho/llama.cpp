#include "moe-weighted-reduction.cuh"
#include "mmb.cuh"

#include <cstdint>

static __device__ __forceinline__ float moe_bf16_to_f32(const uint16_t h) {
    return __uint_as_float(((uint32_t) h) << 16);
}

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

// Same arithmetic, four columns per thread. The scalar kernel moves 10 KiB per block over
// n_tokens*n_embd/256 blocks and reaches about half of this machine's bandwidth; widening the
// access gives each thread four loads in flight per expert instead of one.
// The multiply order is spelled out to match the scalar kernel exactly.
static __global__ void moe_weighted_reduction_f32_vec4(const float4 * __restrict__ experts,
                                                       const float * __restrict__ expert_scale,
                                                       const float * __restrict__ weights,
                                                       float4 * __restrict__ dst,
                                                       const int64_t n_embd4,
                                                       const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd4) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float    first_w     = weights[first_row];
    const float4   first_v     = experts[first_row * n_embd4 + col];

    float4 sum;
    sum.x = (first_v.x * first_scale) * first_w;
    sum.y = (first_v.y * first_scale) * first_w;
    sum.z = (first_v.z * first_scale) * first_w;
    sum.w = (first_v.w * first_scale) * first_w;

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float    w     = weights[row];
        const float4   v     = experts[row * n_embd4 + col];
        sum.x += (v.x * scale) * w;
        sum.y += (v.y * scale) * w;
        sum.z += (v.z * scale) * w;
        sum.w += (v.w * scale) * w;
    }

    dst[token * n_embd4 + col] = sum;
}

// BF16 expert outputs: the packed down-GEMM can write its result as bf16 in place, which halves the bytes
// this reduction reads. Same arithmetic and column order as the float4 kernel.
static __global__ void moe_weighted_reduction_bf16_vec4(const uint16_t * __restrict__ experts,
                                                        const float * __restrict__ expert_scale,
                                                        const float * __restrict__ weights,
                                                        float4 * __restrict__ dst,
                                                        const int64_t n_embd4,
                                                        const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd4) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float    first_w     = weights[first_row];
    const ushort4  first_v     = *reinterpret_cast<const ushort4 *>(experts + first_row * n_embd4 * 4 + col * 4);

    float4 sum;
    sum.x = (moe_bf16_to_f32(first_v.x) * first_scale) * first_w;
    sum.y = (moe_bf16_to_f32(first_v.y) * first_scale) * first_w;
    sum.z = (moe_bf16_to_f32(first_v.z) * first_scale) * first_w;
    sum.w = (moe_bf16_to_f32(first_v.w) * first_scale) * first_w;

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float    w     = weights[row];
        const ushort4  v     = *reinterpret_cast<const ushort4 *>(experts + row * n_embd4 * 4 + col * 4);
        sum.x += (moe_bf16_to_f32(v.x) * scale) * w;
        sum.y += (moe_bf16_to_f32(v.y) * scale) * w;
        sum.z += (moe_bf16_to_f32(v.z) * scale) * w;
        sum.w += (moe_bf16_to_f32(v.w) * scale) * w;
    }

    dst[token * n_embd4 + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;

    const bool aligned = n_embd % 4 == 0 &&
        ((uintptr_t) experts % sizeof(float4)) == 0 && ((uintptr_t) dst % sizeof(float4)) == 0;

    if (aligned) {
        const int64_t n_embd4 = n_embd / 4;
        const dim3 blocks(n_tokens, (n_embd4 + threads - 1) / threads, 1);
        moe_weighted_reduction_f32_vec4<<<blocks, threads, 0, stream>>>(
            (const float4 *) experts, expert_scale, weights, (float4 *) dst, n_embd4, n_expert_used);
        return;
    }

    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    if (ggml_cuda_mmb_down16() && ggml_cuda_mmb_is_bf16_only(experts)) {
        // the packed down-GEMM wrote the expert outputs as bf16 in place; read them as such
        GGML_ASSERT(n_embd % 4 == 0);
        constexpr int threads = 256;
        const int64_t n_embd4 = n_embd / 4;
        const dim3 blocks(n_tokens, (n_embd4 + threads - 1) / threads, 1);
        moe_weighted_reduction_bf16_vec4<<<blocks, threads, 0, stream>>>(
            (const uint16_t *) experts->data,
            expert_scale ? (const float *) expert_scale->data : nullptr,
            (const float *) weights->data,
            (float4 *) dst->data, n_embd4, (int) n_expert_used);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
