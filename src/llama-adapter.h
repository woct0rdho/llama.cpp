#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;

//
// llama_adapter_lora concatenation
//

// Sum of the deltas of n adapters as a single adapter, by concatenating the factors along the rank
// axis:
//
//   sum_i scale_i * B_i @ A_i = [scale_1 B_1 | ... | scale_n B_n] @ [A_1 ; ... ; A_n]
//
// The concatenated adapter holds one entry per weight that any of the inputs defines, with a rank equal
// to the sum of the ranks of the inputs that define it, so no weight is zero padded and no weight
// is visited more than once by the graph. Adapters that define different weights are fine: a
// weight only present in some of them gets the ranks of those only.
//
// llama.cpp scales a weight by `adapter_scale * alpha / b->ne[0]`, so the concatenated adapter stores B_i
// multiplied by `scale_i * rank_t / alpha` with `alpha = max_t rank_t`. That keeps every weight at
// its own `scale_i` and the concatenated adapter linear in its own adapter_scale.
//
// Returns nullptr when the inputs cannot be concatenated (no overlap is not an error, but a single
// adapter or an empty list is not worth concatenating); the caller then keeps the adapters separate.
// The result is owned by the caller and is not registered with the model.
llama_adapter_lora * llama_adapter_lora_concat(
        const llama_model * model,
        const llama_adapter_lora * const * adapters,
        const float * scales,
        size_t n_adapters);
