#include "hc-cn.cuh"
#include "mmb.cuh"

static __device__ __forceinline__ float hc_bf2f32(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
static __device__ __forceinline__ uint16_t hc_f2bf32(const float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

// One block per (stream, token). The row is combined, kept in registers for the RMS reduction, and the
// normed row is written, so the residual is read once and the intermediate row never reaches memory.
// Same expressions as ggml_cuda_op_dsv4_hc_post plus rms_norm_f32<..., true>.
static __global__ void __launch_bounds__(HC_CN_BLOCK, 1) hc_combine_norm_f32(
        const float *    post,
        const float *    residual,
        const uint16_t * res_in_bf16,
        const float *    block_out,
        const float *    gamma,
        float *          out_res,
        uint16_t *       res_out_bf16,
        float *          out_xn,
        uint16_t *       out_xn_bf16,
        const int        n_embd,
        const float      eps) {
    __shared__ float s_sum[32];

    const int c   = blockIdx.x;
    const int t   = blockIdx.y;
    const int hc  = gridDim.x;
    const int tid = threadIdx.x;

    const float w = post[(int64_t) t * hc + c];

    const int64_t   row   = (int64_t) t * hc + c;
    const float *   res   = residual + row * n_embd;
    const uint16_t * res16 = res_in_bf16 ? res_in_bf16 + row * n_embd : nullptr;
    float *         dst   = out_res + row * n_embd;
    uint16_t *      dst16 = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    const float *   blk   = block_out + (int64_t) t * n_embd;
    const float *   g     = gamma + (int64_t) c * n_embd;
    float *         xn    = out_xn + row * n_embd;
    uint16_t *      xh    = out_xn_bf16 ? out_xn_bf16 + row * n_embd : nullptr;

    float xs[HC_CN_CHUNK];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < HC_CN_CHUNK; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        xs[k] = 0.0f;
        if (col < n_embd) {
            const float xi = blk[col] * w + (res16 ? hc_bf2f32(res16[col]) : res[col]);
            if (dst16) {
                dst16[col] = hc_f2bf32(xi);
            } else {
                dst[col] = xi;
            }
            xs[k] = xi;
            tmp  += xi * xi;
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK>(tmp, s_sum);

    const float scale = rsqrtf(tmp / n_embd + eps);

#pragma unroll
    for (int k = 0; k < HC_CN_CHUNK; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        if (col < n_embd) {
            const float v = scale * xs[k] * g[col];
            xn[col] = v;
            if (xh) {
                xh[col] = hc_f2bf32(v);
            }
        }
    }
}

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & args) {
    const int64_t n_embd = args.out_res->ne[0];
    const int64_t hc     = args.out_res->ne[1];
    const int64_t n_tok  = args.out_res->ne[2];

    GGML_ASSERT(hc > 0 && hc <= 32);
    GGML_ASSERT(n_embd > 0 && n_embd <= HC_CN_CHUNK * HC_CN_BLOCK);
    GGML_ASSERT(ggml_is_contiguous(args.post) && ggml_is_contiguous(args.block_out) &&
                ggml_is_contiguous(args.gamma) && ggml_is_contiguous(args.out_res) && ggml_is_contiguous(args.out_xn));

    const dim3 grid((unsigned) hc, (unsigned) n_tok);
    const ggml_cuda_kernel_launch_params launch_params(grid, HC_CN_BLOCK, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_norm_f32, launch_params,
            (const float *) args.post->data,
            (const float *) args.residual->data,
            args.res_in_bf16,
            (const float *) args.block_out->data,
            (const float *) args.gamma->data,
            (float *) args.out_res->data,
            args.res_out_bf16,
            (float *) args.out_xn->data,
            args.out_xn_bf16,
            (int) n_embd, args.eps);
}
