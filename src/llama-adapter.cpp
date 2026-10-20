#include "llama-adapter.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include <map>
#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <sstream>
#include <stdexcept>

// vec

ggml_tensor * llama_adapter_cvec::tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const {
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = ggml_add(ctx, cur, layer_dir);
    }

    return cur;
}

bool llama_adapter_cvec::init(const llama_model & model) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(tensors.empty());
    GGML_ASSERT(ctxs.empty());
    GGML_ASSERT(bufs.empty());

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer()*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // make tensors
    tensors.reserve(hparams.n_layer());
    tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < hparams.n_layer(); il++) {
        ggml_backend_buffer_type_t buft = model.select_buft(il);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    return true;
}

bool llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    const auto & hparams = model.hparams;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        layer_start = -1;
        layer_end   = -1;
        return true;
    }

    if (n_embd != (int) hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return false;
    }

    if (tensors.empty()) {
        if (!init(model)) {
            return false;
        }
    }

    layer_start = il_start;
    layer_end   = il_end;

    for (size_t il = 1; il < hparams.n_layer(); il++) {
        assert(tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(tensors[il], data + off, 0, n_embd * ggml_element_size(tensors[il]));
        }
    }

    return true;
}

// lora

llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    const std::string name(w->name);

    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

static void llama_adapter_lora_init_impl(llama_model & model, FILE * file, llama_adapter_lora & adapter) {
    ggml_context * ctx_init;
    gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_init,
    };

    gguf_context_ptr ctx_gguf { gguf_init_from_file_ptr(file, meta_gguf_params) };
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load lora adapter from file");
    }

    ggml_context_ptr ctx { ctx_init };

    // must come after gguf_init_from_file_ptr, the llama_file constructor moves the file position
    llama_file gguf_file(file);

    // check metadata
    {
        const gguf_context * gguf_ctx = ctx_gguf.get();

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values.\n", __func__);

        // get metadata as string
        for (int i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            gguf_type type = gguf_get_kv_type(gguf_ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(gguf_ctx, i)), gguf_get_arr_n(gguf_ctx, i))
                : gguf_type_name(type);
            const char * name = gguf_get_key(gguf_ctx, i);
            const std::string value = gguf_kv_to_str(gguf_ctx, i);

            if (type != GGUF_TYPE_ARRAY) {
                adapter.gguf_kv.emplace(name, value);
            }

            const size_t MAX_VALUE_LEN = 40;
            std::string print_value = value.size() > MAX_VALUE_LEN ? format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str()) : value;
            replace_all(print_value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), print_value.c_str());
        }

        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? "" : std::string(gguf_get_val_str(gguf_ctx, id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? 0.0f : gguf_get_val_f32(gguf_ctx, id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            // throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));

        // parse alora invocation sequence vector
        const auto & key = llm_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS);
        const int kid = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (kid >= 0) {
            if (gguf_get_kv_type(ctx_gguf.get(), kid) != GGUF_TYPE_ARRAY) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            const auto arr_type = gguf_get_arr_type(ctx_gguf.get(), kid);
            if (arr_type != GGUF_TYPE_UINT32) {
                throw std::runtime_error("invalid gguf element type for " + key);
            }
            const size_t seq_len = gguf_get_arr_n(ctx_gguf.get(), kid);
            const void * data = gguf_get_arr_data(ctx_gguf.get(), kid);
            adapter.alora_invocation_tokens.resize(seq_len);
            std::copy(
                (const llama_token *)data,
                (const llama_token *)data + seq_len,
                adapter.alora_invocation_tokens.begin());
        }
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

    // contexts for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * buft_ctx = ggml_init(params);
            if (!buft_ctx) {
                return nullptr;
            }
            ctx_map[buft] = buft_ctx;
            adapter.ctxs.emplace_back(buft_ctx);
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (str_endswith(name, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    }

    // get extra buffer types of the CPU
    // TODO: a more general solution for non-CPU extra buft should be implemented in the future
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12593#pullrequestreview-2718659948
    std::vector<ggml_backend_buffer_type_t> buft_extra;
    {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);

        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_extra.emplace_back(*extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & w = it.second;
        bool is_token_embd = str_endswith(name, "token_embd.weight");

        if (!w.a || !w.b) {
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }

        // device buft and device ctx
        const auto * model_tensor = model.get_tensor(name.c_str());
        if (!model_tensor) {
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
        }
        // the LoRA delta of a Hadamard-folded weight reads the transformed input, so the result is wrong
        if (model.hdmd.weight_blocks.count(name) || model.hdmd.inverse_blocks.count(name)) {
            throw std::runtime_error("LoRA tensor '" + name + "' targets a prism.hadamard folded weight, which is not supported");
        }

        auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);

        // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
        for (auto & ex : buft_extra) {
            if (ex == buft) {
                LLAMA_LOG_WARN("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error(format("%s: no CPU backend found", __func__));
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);

                break;
            }
        }

        LLAMA_LOG_DEBUG("%s: lora for '%s' -> '%s'\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

        ggml_context * dev_ctx = ctx_for_buft(buft);
        // validate tensor shape
        if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor->ne[0] != w.b->ne[1] || model_tensor->ne[1] != w.a->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
        } else {
            if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[1] != w.b->ne[0]) {
                throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
            }
        }

        // save tensor to adapter
        ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
        ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
        ggml_set_name(tensor_a, w.a->name);
        ggml_set_name(tensor_b, w.b->name);
        adapter.ab_map[name] = llama_adapter_lora_weight(tensor_a, tensor_b);
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            ggml_backend_buffer_type_t buft = it.first;
            ggml_context * ctx_dev = it.second;
            ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft) };
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            adapter.bufs.emplace_back(std::move(buf));
        }
    }

    // set tensor data
    {
        std::vector<uint8_t> read_buf;
        auto set_tensor = [&](ggml_tensor * orig, ggml_tensor * dev) {
            const size_t offs = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), gguf_find_tensor(ctx_gguf.get(), orig->name));
            const size_t size = ggml_nbytes(orig);
            if (offs + size < offs || offs + size > gguf_file.size()) {
                throw std::runtime_error(format("LoRA tensor '%s' data is not within the file bounds, file is corrupted or incomplete", orig->name));
            }
            read_buf.resize(size);
            gguf_file.seek(offs, SEEK_SET);
            gguf_file.read_raw(read_buf.data(), size);
            ggml_backend_tensor_set(dev, read_buf.data(), 0, size);
        };
        for (auto & it : adapter.ab_map) {
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
        }
    }

    // register adapter with model
    model.loras.insert(&adapter);

    LLAMA_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, adapter.ab_map.size()*2);
}

llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    FILE * file = ggml_fopen(path_lora, "rb");
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path_lora, strerror(errno));
        return nullptr;
    }

    llama_adapter_lora * adapter = llama_adapter_lora_init_from_file_ptr(model, file);
    fclose(file);

    return adapter;
}

llama_adapter_lora * llama_adapter_lora_init_from_file_ptr(llama_model * model, FILE * file) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }

    llama_adapter_lora * adapter = new llama_adapter_lora(model);

    try {
        llama_adapter_lora_init_impl(*model, file, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());

        delete adapter;
    }

    return nullptr;
}

//
// llama_adapter_lora concatenation
//

// write n elements of a source chunk into a destination chunk, scaled by c
static void llama_adapter_lora_concat_chunk(
        uint8_t * dst, const uint8_t * src, ggml_type type_src, ggml_type type_dst, int64_t n, float c) {
    if (c == 1.0f && type_src == type_dst) {
        memcpy(dst, src, n*ggml_type_size(type_src));
        return;
    }

    GGML_ASSERT((type_src == GGML_TYPE_F16 || type_src == GGML_TYPE_F32) &&
                (type_dst == GGML_TYPE_F16 || type_dst == GGML_TYPE_F32));

    for (int64_t i = 0; i < n; ++i) {
        const float v = c*(type_src == GGML_TYPE_F32 ?
            ((const float *) src)[i] : ggml_fp16_to_fp32(((const ggml_fp16_t *) src)[i]));

        if (type_dst == GGML_TYPE_F32) {
            ((float *) dst)[i] = v;
        } else {
            ((ggml_fp16_t *) dst)[i] = ggml_fp32_to_fp16(v);
        }
    }
}

// concatenate the factors of `n_adapters` adapters along their rank axis
llama_adapter_lora * llama_adapter_lora_concat(
        const llama_model * model,
        const llama_adapter_lora * const * adapters,
        const float * scales,
        size_t n_adapters) {
    if (!model || !adapters || !scales || n_adapters < 2) {
        return nullptr;
    }

    // sources of every weight, in a deterministic order
    std::map<std::string, std::vector<std::pair<size_t, const llama_adapter_lora_weight *>>> wmap;

    for (size_t i = 0; i < n_adapters; ++i) {
        if (!adapters[i]) {
            return nullptr;
        }
        if (!adapters[i]->alora_invocation_tokens.empty()) {
            // the scale of an activated aLoRA changes per request
            LLAMA_LOG_INFO("%s: not concatenating, adapter %zu is an aLoRA\n", __func__, i);
            return nullptr;
        }
        for (const auto & it : adapters[i]->ab_map) {
            if (!it.second.a || !it.second.b) {
                return nullptr;
            }
            wmap[it.first].push_back({i, &it.second});
        }
    }

    if (wmap.empty()) {
        return nullptr;
    }

    // the concatenated rank of a weight is the sum of the ranks of its sources, so no weight is padded;
    // the largest of those ranks becomes the adapter alpha, see the header for the scaling
    std::map<std::string, int64_t> rank_t;
    int64_t alpha = 0;

    for (const auto & it : wmap) {
        int64_t r = 0;
        for (const auto & s : it.second) {
            r += s.second->b->ne[0];
        }
        if (r <= 0) {
            return nullptr;
        }
        rank_t[it.first] = r;
        alpha = std::max(alpha, r);
    }

    // the result is owned by the caller and is never registered with the model, so the model is only
    // ever read through this pointer
    auto * adapter = new llama_adapter_lora(const_cast<llama_model *>(model));
    adapter->alpha = (float) alpha;

    double size_mib = 0.0;

    const int64_t t_start = ggml_time_us();
    int64_t t_alloc = t_start;

    try {
        std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
        auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
            auto it = ctx_map.find(buft);
            if (it != ctx_map.end()) {
                return it->second;
            }
            ggml_init_params params = {
                /*.mem_size   =*/ wmap.size()*2*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to allocate a LoRA concat context");
            }
            ctx_map[buft] = ctx;
            adapter->ctxs.emplace_back(ctx);
            return ctx;
        };

        // the concatenated factors: the same shape as the sources, with the rank axis widened
        for (const auto & it : wmap) {
            const std::string & name = it.first;
            const auto & src = it.second;

            const ggml_tensor * a0 = src[0].second->a;
            const ggml_tensor * b0 = src[0].second->b;

            ggml_type type_a = GGML_TYPE_F16;
            ggml_type type_b = GGML_TYPE_F16;
            for (const auto & s : src) {
                if (s.second->a->type == GGML_TYPE_F32) {
                    type_a = GGML_TYPE_F32;
                }
                if (s.second->b->type == GGML_TYPE_F32) {
                    type_b = GGML_TYPE_F32;
                }
            }

            // lora_a is [k, rank] / [k, rank, n_expert], lora_b is [rank, m] / [rank, m, n_expert]
            const int64_t ne_a[GGML_MAX_DIMS] = { a0->ne[0], rank_t[name], a0->ne[2], a0->ne[3] };
            const int64_t ne_b[GGML_MAX_DIMS] = { rank_t[name], b0->ne[1], b0->ne[2], b0->ne[3] };

            ggml_context * ctx_a = ctx_for_buft(ggml_backend_buffer_get_type(a0->buffer));
            ggml_context * ctx_b = ctx_for_buft(ggml_backend_buffer_get_type(b0->buffer));

            ggml_tensor * a = ggml_new_tensor(ctx_a, type_a, ggml_n_dims(a0), ne_a);
            ggml_tensor * b = ggml_new_tensor(ctx_b, type_b, ggml_n_dims(b0), ne_b);

            const std::string suffix_a = name + ".lora_a";
            const std::string suffix_b = name + ".lora_b";
            ggml_set_name(a, suffix_a.c_str());
            ggml_set_name(b, suffix_b.c_str());

            adapter->ab_map[name] = llama_adapter_lora_weight(a, b);

            size_mib += (ggml_nbytes(a) + ggml_nbytes(b))/1024.0/1024.0;
        }

        // allocate the concatenated tensors
        for (auto & it : ctx_map) {
            ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(it.second, it.first) };
            if (!buf) {
                throw std::runtime_error("failed to allocate the buffer of a concatenated LoRA adapter");
            }
            adapter->bufs.emplace_back(std::move(buf));
        }

        t_alloc = ggml_time_us();

        // fill: assemble each concatenated factor in host memory, then upload it in one go
        std::vector<uint8_t> host_src;
        std::vector<uint8_t> host_a;
        std::vector<uint8_t> host_b;

        for (const auto & it : wmap) {
            const std::string & name = it.first;
            const auto & src = it.second;

            llama_adapter_lora_weight & dst = adapter->ab_map[name];

            const int64_t r_all = rank_t[name];

            // every element below is overwritten by one of the sources, so no zero fill
            if (host_a.size() < ggml_nbytes(dst.a)) {
                host_a.resize(ggml_nbytes(dst.a));
            }
            if (host_b.size() < ggml_nbytes(dst.b)) {
                host_b.resize(ggml_nbytes(dst.b));
            }

            int64_t r_off = 0;
            for (const auto & s : src) {
                const float c = scales[s.first]*(float) r_all/(float) alpha;

                const ggml_tensor * a_src = s.second->a;
                const ggml_tensor * b_src = s.second->b;

                const int64_t r_src = b_src->ne[0];

                // lora_a: the rank axis is ne[1], so a source contributes runs of ne[0] elements
                {
                    const int64_t k  = a_src->ne[0];
                    const int64_t n2 = a_src->ne[2];
                    const int64_t n3 = a_src->ne[3];
                    const size_t  sz_src = ggml_type_size(a_src->type);
                    const size_t  sz_dst = ggml_type_size(dst.a->type);

                    if (host_src.size() < ggml_nbytes(a_src)) {
                        host_src.resize(ggml_nbytes(a_src));
                    }
                    ggml_backend_tensor_get(a_src, host_src.data(), 0, ggml_nbytes(a_src));

                    for (int64_t p = 0; p < n2*n3; ++p) {
                        for (int64_t j = 0; j < r_src; ++j) {
                            // the scale is folded into lora_b only, so lora_a is a plain concatenation
                            llama_adapter_lora_concat_chunk(
                                    host_a.data() + (p*r_all + r_off + j)*k*sz_dst,
                                    host_src.data() + (p*r_src + j)*k*sz_src,
                                    a_src->type, dst.a->type, k, 1.0f);
                        }
                    }
                }

                // lora_b: the rank axis is ne[0], so a source contributes runs of r_src elements
                {
                    const int64_t m  = b_src->ne[1];
                    const int64_t n2 = b_src->ne[2];
                    const int64_t n3 = b_src->ne[3];
                    const size_t  sz_src = ggml_type_size(b_src->type);
                    const size_t  sz_dst = ggml_type_size(dst.b->type);

                    if (host_src.size() < ggml_nbytes(b_src)) {
                        host_src.resize(ggml_nbytes(b_src));
                    }
                    ggml_backend_tensor_get(b_src, host_src.data(), 0, ggml_nbytes(b_src));

                    for (int64_t p = 0; p < n2*n3; ++p) {
                        for (int64_t i1 = 0; i1 < m; ++i1) {
                            llama_adapter_lora_concat_chunk(
                                    host_b.data() + ((p*m + i1)*r_all + r_off)*sz_dst,
                                    host_src.data() + (p*m + i1)*r_src*sz_src,
                                    b_src->type, dst.b->type, r_src, c);
                        }
                    }
                }

                r_off += r_src;
            }

            GGML_ASSERT(r_off == r_all);

            ggml_backend_tensor_set(dst.a, host_a.data(), 0, ggml_nbytes(dst.a));
            ggml_backend_tensor_set(dst.b, host_b.data(), 0, ggml_nbytes(dst.b));
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to concatenate lora adapters: %s\n", __func__, err.what());

        delete adapter;
        return nullptr;
    }

    // debug: dump the concatenated factors so they can be compared against an offline concatenation
    if (const char * dump_dir = getenv("LLAMA_LORA_CONCAT_DUMP")) {
        for (const auto & it : adapter->ab_map) {
            const ggml_tensor * ts[2] = { it.second.a, it.second.b };
            const char * sfx[2] = { ".lora_a", ".lora_b" };
            for (int i = 0; i < 2; ++i) {
                std::vector<uint8_t> buf(ggml_nbytes(ts[i]));
                ggml_backend_tensor_get(ts[i], buf.data(), 0, buf.size());
                const std::string path = std::string(dump_dir) + "/" + it.first + sfx[i] + ".bin";
                if (FILE * f = fopen(path.c_str(), "wb")) {
                    fwrite(buf.data(), 1, buf.size(), f);
                    fclose(f);
                }
            }
        }
        LLAMA_LOG_INFO("%s: dumped the concatenation to '%s'\n", __func__, dump_dir);
    }

    int64_t rank_min = INT64_MAX;
    for (const auto & it : rank_t) {
        rank_min = std::min(rank_min, it.second);
    }

    LLAMA_LOG_INFO("%s: concatenated %zu adapters into %zu weights, alpha = %.1f, rank %" PRId64 "..%" PRId64
            ", %.0f MiB, alloc %.2f s, copy %.2f s\n", __func__, n_adapters, adapter->ab_map.size(), adapter->alpha,
            rank_min, alpha, size_mib, (t_alloc - t_start)/1e6, (ggml_time_us() - t_alloc)/1e6);

    return adapter;
}

int32_t llama_adapter_meta_val_str(const llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size) {
    const auto & it = adapter->gguf_kv.find(key);
    if (it == adapter->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_adapter_meta_count(const llama_adapter_lora * adapter) {
    return (int)adapter->gguf_kv.size();
}

int32_t llama_adapter_meta_key_by_index(const llama_adapter_lora * adapter, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_adapter_meta_val_str_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

void llama_adapter_lora_free(llama_adapter_lora * adapter) {
    if (adapter == nullptr) {
        return;
    }

    if (adapter->model != nullptr) {
        adapter->model->loras.erase(adapter);
        adapter->model = nullptr;
    }

    delete adapter;
}

uint64_t llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter) {
    if (!adapter) {
        return 0;
    }
    return adapter->alora_invocation_tokens.size();
}

const llama_token * llama_adapter_get_alora_invocation_tokens(const llama_adapter_lora * adapter) {
    GGML_ASSERT(adapter);
    return adapter->alora_invocation_tokens.data();
}
