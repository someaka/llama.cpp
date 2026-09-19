// tools/server/server-hidden-states.h — fork-owned /hidden-states endpoint logic.
// Pure code motion out of server-context.cpp (post_hidden_states lambda body,
// server_context_impl::send_hidden_states, HS_MAX_POOL_NONE_FLOATS): the bodies
// are unchanged except that member references are qualified through `self`.
//
// Why free templates: server_context_impl, server_slot and server_res_generator
// are cpp-local types in server-context.cpp, so the moved bodies cannot live in a
// plain TU — they are templates over the owning object, instantiated at their
// existing call sites (init_routes lambda / update_slots decode path). This keeps
// the fork's endpoint surface in fork-owned files next to upstream's own
// per-endpoint split (server-chat.cpp, server-models.cpp, server-tools.cpp).
#pragma once
#include "masked-mean.h"  // shared skip_mean kernel (tools/hs-extract-common/)
#include "server-common.h"
#include "server-http.h"  // server_http_req (handler signature)
#include "server-task.h"

#include <memory>
#include <string>
#include <vector>

// /hidden-states pool=none response cap: max total floats (all layers)
// returned by a single request. 25M floats = 100 MB raw, ~0.3 GB as %.9g JSON.
constexpr size_t HS_MAX_POOL_NONE_FLOATS = 25'000'000;

// /hidden-states HTTP handler (was the body of the post_hidden_states lambda
// in server_routes::init_routes). `self` is the server_routes object.
template <typename Self>
auto handle_hidden_states(Self & self, const server_http_req & req) {
    auto res = self.create_response();

    // Check if hidden-states endpoint is disabled via --no-hidden-states
    if (self.params.no_hidden_states) {
        res->error(format_error_response("Hidden states endpoint disabled by --no-hidden-states flag", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    // Check if hidden states are supported
    const int32_t n_layer = llama_model_n_layer(self.ctx_server.model_tgt);
    if (n_layer <= 0) {
        res->error(format_error_response("Hidden states extraction not supported by this model", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // The capability registry is the single source of truth for which
    // architectures emit per-layer capture. Refuse here with the arch
    // name so the client learns the request cannot succeed on this model.
    if (!llama_model_supports_hidden_states(self.ctx_server.model_tgt)) {
        const std::string arch_name = llama_model_arch_name(self.ctx_server.model_tgt);
        res->error(format_error_response(
            std::string("hidden-state extraction not implemented for architecture '")
            + arch_name + "'",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // Parse layers parameter
    std::vector<int> layers;
    bool all_layers = false;

    // Layers follow the hidden_states convention: 0 = embeddings,
    // i = state entering block i, n_layer = final block output.
    // Valid range is [0, n_layer] inclusive (n_layer + 1 slots).
    if (body.contains("layers")) {
        const json & layers_json = body["layers"];
        if (layers_json.is_string() && layers_json.get<std::string>() == "all") {
            all_layers = true;
        } else if (layers_json.is_array()) {
            if (layers_json.empty()) {
                res->error(format_error_response("layers array must not be empty", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            for (const auto & layer : layers_json) {
                if (!layer.is_number_integer()) {
                    res->error(format_error_response("layers array must contain integers", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                // Parse as int64 (common_json has no unsigned predicate).
                // uint64 values >= 2^63 narrow negative under two's
                // complement and are rejected by the < 0 check below;
                // no uint64 in [2^63, 2^64) can narrow into the valid
                // range [0, n_layer], so the narrowing is never a false
                // accept. Valid range is [0, n_layer] inclusive.
                const int64_t layer_req = layer.get<int64_t>();
                if (layer_req < 0 || layer_req > (int64_t) n_layer) {
                    res->error(format_error_response(
                        "layer " + std::to_string(layer_req) + " out of range [0, " + std::to_string(n_layer) + "]",
                        ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                const int layer_num = (int) layer_req;
                // Duplicate ids would silently collapse in the object-
                // keyed response; reject like the CLI tools do.
                if (std::find(layers.begin(), layers.end(), layer_num) != layers.end()) {
                    res->error(format_error_response(
                        "duplicate layer index " + std::to_string(layer_num) + " in layers array",
                        ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                layers.push_back(layer_num);
            }
        } else {
            res->error(format_error_response("layers must be an array or string 'all'", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        // Default to all layers if not specified
        all_layers = true;
    }

    // Parse normalize parameter
    bool normalize = false;
    if (body.contains("normalize")) {
        if (!body["normalize"].is_boolean()) {
            res->error(format_error_response("normalize must be a boolean", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        normalize = body["normalize"].get<bool>();
    }

    // Parse pool parameter (default: "last" = last token, "skip_mean" = masked-mean)
    std::string pool = "last";
    if (body.contains("pool")) {
        if (!body["pool"].is_string()) {
            res->error(format_error_response("pool must be a string ('last', 'skip_mean', or 'none')", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        pool = body["pool"].get<std::string>();
        if (pool != "last" && pool != "skip_mean" && pool != "none") {
            res->error(format_error_response("unknown pool value '" + pool + "' (must be 'last', 'skip_mean', or 'none')", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // normalize is only defined for pooled vectors: pool=none returns
    // per-token rows, and a scalar norm over the flattened block would
    // scale every row by one wrong factor. Every other invalid parameter
    // combination on this endpoint fails loud; a silently-ignored flag is
    // the same trap the empty-input pins exist to prevent.
    if (normalize && pool == "none") {
        res->error(format_error_response(
            "normalize is only supported for pooled output (pool 'last' or 'skip_mean'); "
            "it is undefined for pool 'none' — drop the flag or change the pool",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Parse skip_offset parameter (default: 0 = pool over all tokens, used with pool="skip_mean")
    int32_t skip_offset = 0;
    if (body.contains("skip_offset")) {
        if (!body["skip_offset"].is_number_integer()) {
            res->error(format_error_response("skip_offset must be an integer", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        // Parse as int64 (common_json has no unsigned predicate).
        // uint64 values >= 2^63 narrow negative under two's complement
        // and are rejected by the < 0 check below; no uint64 in
        // [2^63, 2^64) can narrow into [0, INT32_MAX], so the narrowing
        // is never a false accept.
        const int64_t skip_req = body["skip_offset"].get<int64_t>();
        if (skip_req < 0 || skip_req > 2147483647LL) {
            res->error(format_error_response("skip_offset " + std::to_string(skip_req) +
                                             " out of range [0, 2147483647]", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        skip_offset = (int32_t)skip_req;
    }
    // skip_offset is consumed only by pool="skip_mean"; accepting it in any
    // other mode would silently ignore a caller-specified flag — the same
    // trap the empty-input and normalize+pool=none pins prevent (80a9fe37e).
    // Both CLIs reject --token-skip in non-consuming modes; hold the server
    // to the same bar. An explicit skip_offset:0 is rejected too: echoing
    // the default is still echoing a flag the endpoint would ignore.
    if (body.contains("skip_offset") && pool != "skip_mean") {
        res->error(format_error_response(
            "skip_offset " + std::to_string(skip_offset) +
            " is only valid with pool=skip_mean (requested pool=" + pool + ")",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Get input text
    json prompt;
    if (body.contains("input")) {
        prompt = body["input"];
    } else {
        res->error(format_error_response("input field is required", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // HS-1 (P0) pin: an empty-string element in an input array is invalid
    // input regardless of vocabulary — on BOS-add vocabs it would silently
    // become a bare-[BOS] 1-token prompt (a request the caller almost
    // certainly did not mean), and on no-BOS vocabs it tokenizes to zero
    // tokens, which used to abort the server. Reject by content, not by
    // token count, so the contract does not depend on the model.
    // The scalar form is the same hazard: a scalar empty string reaches
    // tokenize_input_prompts identically and slips through on BOS-add
    // vocabs (tour dry-run finding P5, 2026-09-17), so reject it here too.
    if (prompt.is_string() && prompt.get<std::string>().empty()) {
        res->error(format_error_response(
            "input is an empty string — cannot extract hidden states",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (prompt.is_array()) {
        for (size_t i = 0; i < prompt.size(); i++) {
            if (prompt[i].is_string() && prompt[i].get<std::string>().empty()) {
                res->error(format_error_response(
                    "input[" + std::to_string(i) + "] is an empty string — cannot extract hidden states",
                    ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
    }

    // Tokenize the input using the same pattern as embeddings. A malformed
    // input (bare `[]`, non-string junk) throws inside tokenize_input_prompts;
    // surface that as 400 rather than an unhandled 500.
    std::vector<server_tokens> tokenized_prompts;
    try {
        tokenized_prompts = tokenize_input_prompts(self.ctx_server.vocab, self.ctx_server.mctx, prompt, true, true, self.ctx_server.init_opt);
    } catch (const std::exception & e) {
        res->error(format_error_response(std::string("Failed to tokenize input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (tokenized_prompts.empty() || tokenized_prompts[0].empty()) {
        res->error(format_error_response("Failed to tokenize input", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    // HS-1 (P0): an empty ANY-element (empty string on a no-BOS arch, or a
    // raw `[]` token array — json_is_array_of_numbers accepts it vacuously)
    // would create a 0-token HIDDEN_STATES task; update_slots answers such
    // a task with a cmpl_final result and the endpoint's result-type assert
    // below GGML_ABORTs the whole server. Reject up front, naming the index.
    for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        if (tokenized_prompts[i].empty()) {
            res->error(format_error_response(
                "input[" + std::to_string(i) + "] tokenizes to zero tokens — cannot extract hidden states",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // skip_offset >= token count is knowable now; reject before spending
    // the decode (the pool=skip_mean path would reject it post-decode).
    if (pool == "skip_mean" && skip_offset > 0) {
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            if ((size_t) skip_offset >= (size_t) tokenized_prompts[i].size()) {
                res->error(format_error_response(
                    "skip_offset (" + std::to_string(skip_offset) + ") >= n_tokens (" +
                    std::to_string(tokenized_prompts[i].size()) + ") for input[" + std::to_string(i) +
                    "]: prompt too short",
                    ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
    }

    // Create and queue the task
    std::vector<std::string> res_texts;
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_HIDDEN_STATES);
            task.id = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);
            // Disable prompt caching for hidden-states extraction. Cached tokens
            // skip the forward pass, but t_hidden_layers[] is only populated for
            // decoded tokens. With cache_prompt=true, repeated requests for the
            // same text return stale/partial hidden states because the matched
            // tokens are not re-decoded. Hidden-states requires a fresh full
            // forward pass every time to produce correct, deterministic results.
            task.params.cache_prompt = false;
            task.params.hidden_layers = layers;
            task.params.hidden_all_layers = all_layers;
            task.params.hidden_normalize = normalize;
            task.params.hidden_pool = pool;
            task.params.hidden_skip_offset = skip_offset;
            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // Wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // Collect results
    if (all_results.is_terminated) {
        return res;
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & result : all_results.results) {
            auto * hs_result = dynamic_cast<server_task_result_hidden_states*>(result.get());
            // HS-1 hardening: a foreign result type here is a bug, not a
            // fatal condition — answer 500 and keep the server alive
            // instead of tripping the (always-active) GGML_ABORT.
            if (hs_result == nullptr) {
                // HS-1 hardening: a foreign result type used to abort the
                // server; answer 500 and stay alive instead.
                res->error(format_error_response(
                    "internal error: unexpected result type for hidden-states task",
                    ERROR_TYPE_SERVER));
                return res;
            }
            // %.9g text form: nlohmann's double writer re-expands ~0.3% of
            // values past 9 significant digits, so the wire numbers are
            // emitted raw instead of through the json serializer.
            res_texts.push_back(hs_result->to_json_text());
        }
    }

    // Format response (values are raw %.9g JSON numbers)
    std::string root_text = "[";
    for (size_t i = 0; i < res_texts.size(); i++) {
        if (i > 0) {
            root_text += ",";
        }
        // move: each res_texts[i] can approach the response cap; copying
        // would hold the full payload twice at peak
        root_text += std::move(res_texts[i]);
    }
    root_text += "]";
    res->status = 200;
    res->data = std::move(root_text);
    return res;
}

// Capture + queue the hidden-states result for a finished decode (was
// server_context_impl::send_hidden_states). `self` is the server_context_impl,
// `slot` the server_slot whose task just decoded.
template <typename Self, typename Slot>
void send_hidden_states(Self & self, const Slot & slot) {
    // Every return path below must leave capture off: the toggle is
    // context-lifetime state on a shared context, and a leaked enable
    // would make the server capture every decode for every request.
    struct hs_toggle_reset {
        llama_context * c;
        // Disable-only: the setter can only refuse an *enable*
        // (its refusal guards all sit behind `value &&`), so this
        // destructor legitimately ignores the return — there is no
        // refusal path to a false call.
        ~hs_toggle_reset() { llama_set_extract_hidden_states(c, false); }
    } hs_reset{slot.ctx_tgt};

    auto res = std::make_unique<server_task_result_hidden_states>();
    res->id       = slot.task->id;
    res->index    = slot.task->index;
    res->n_tokens = slot.task->n_tokens();

    const int32_t n_embd = llama_model_n_embd_out(self.model_tgt);
    const int32_t n_layer = llama_model_n_layer(self.model_tgt);

    // determine which layers to extract
    std::vector<int> layers;
    if (slot.task->params.hidden_all_layers) {
        // Full hidden_states ladder: 0 (embeddings) .. n_layer (final
        // block output) inclusive -- n_layer + 1 slots, matching the
        // [0, n_layer] range accepted for explicit layer lists.
        layers.resize(n_layer + 1);
        for (int32_t i = 0; i <= n_layer; i++) {
            layers[i] = i;
        }
    } else {
        layers = slot.task->params.hidden_layers;
    }

    const int32_t n_hs_tokens = llama_get_hidden_state_n_tokens(slot.ctx_tgt);

    if (n_hs_tokens == 0) {
        auto err = std::make_unique<server_task_result_error>();
        err->id    = slot.task->id;
        err->index = slot.task->index;
        err->err_type = ERROR_TYPE_SERVER;
        err->err_msg = "no hidden states available (decode may have failed)";
        self.queue_results.send(std::move(err));
        return;
    }

    // HS-2 belt-and-braces: the capture resets per llama_decode(), so a
    // batch that was somehow split across decode calls would leave only
    // the tail chunk in the buffer while n_tokens still claims the full
    // prompt — silent data corruption. Refuse rather than return partial
    // data (the launch-time n_batch guard makes this unreachable today;
    // this check keeps it that way if the retry path ever changes).
    if (n_hs_tokens != slot.task->n_tokens()) {
        auto err = std::make_unique<server_task_result_error>();
        err->id    = slot.task->id;
        err->index = slot.task->index;
        err->err_type = ERROR_TYPE_SERVER;
        err->err_msg = string_format(
            "hidden-states capture incomplete (%d of %d tokens) — refusing to return partial data",
            n_hs_tokens, slot.task->n_tokens());
        self.queue_results.send(std::move(err));
        return;
    }

    // Response size guard for pool=none: returns n_tokens * n_embd * n_layers
    // floats as JSON. Without a cap, a single request with long input + all
    // layers can exhaust server memory (DoS vector). Cap: HS_MAX_POOL_NONE_FLOATS
    // (file scope, above).
    if (slot.task->params.hidden_pool == "none") {
        size_t total_all_layers = (size_t)n_hs_tokens * (size_t)n_embd * (size_t)layers.size();
        if (total_all_layers > HS_MAX_POOL_NONE_FLOATS) {
            auto err = std::make_unique<server_task_result_error>();
            err->id   = slot.task->id;
            err->index = slot.task->index;
            err->err_type = ERROR_TYPE_INVALID_REQUEST;
            err->err_msg = "pool=none response too large: " +
                           std::to_string(total_all_layers) + " floats across " +
                           std::to_string(layers.size()) + " layers (limit: " +
                           std::to_string(HS_MAX_POOL_NONE_FLOATS) +
                           "). Reduce input length or number of layers.";
            self.queue_results.send(std::move(err));
            return;
        }
    }

    // Layer indices follow the hidden_states convention:
    // 0 = embeddings, i = state entering block i, n_layer = final block
    // output. Valid range is [0, n_layer] inclusive.
    const int32_t n_hs_slots = n_layer + 1;
    for (int layer : layers) {
        if (layer < 0 || layer > n_layer) {
            auto err = std::make_unique<server_task_result_error>();
            err->id   = slot.task->id;
            err->index = slot.task->index;
            err->err_type = ERROR_TYPE_INVALID_REQUEST;
            err->err_msg = "layer " + std::to_string(layer) +
                           " out of range [0, " + std::to_string(n_hs_slots - 1) + "]";
            self.queue_results.send(std::move(err));
            return;
        }
    }

    // Fetch all layer pointers in one call (single synchronize + single
    // validation pass) instead of one llama_get_hidden_state() per layer.
    // layer_ids is the validated copy of `layers` (int32_t, the API width)
    // and is used for all indexing below; `layers` is not read again.
    std::vector<int32_t> layer_ids(layers.begin(), layers.end());
    std::vector<float *> layer_ptrs(layer_ids.size(), nullptr);
    if (llama_get_hidden_states_batch(slot.ctx_tgt, layer_ids.data(), (int32_t) layer_ids.size(), layer_ptrs.data()) != 0) {
        auto err = std::make_unique<server_task_result_error>();
        err->id   = slot.task->id;
        err->index = slot.task->index;
        err->err_type = ERROR_TYPE_SERVER;
        err->err_msg = "failed to get hidden states for the requested layers";
        self.queue_results.send(std::move(err));
        return;
    }

    for (size_t li = 0; li < layer_ids.size(); li++) {
        const int layer = layers[li];
        float * hs = layer_ptrs[li];
        if (hs == nullptr) {
            auto err = std::make_unique<server_task_result_error>();
            err->id   = slot.task->id;
            err->index = slot.task->index;
            err->err_type = ERROR_TYPE_SERVER;
            err->err_msg = "failed to get hidden state for layer " + std::to_string(layer);
            self.queue_results.send(std::move(err));
            return;
        }

        std::vector<float> vec;

        if (slot.task->params.hidden_pool == "none") {
            // Per-token: return all n_hs_tokens vectors (flattened)
            // Size guard is hoisted before the layer loop (above).
            size_t total = (size_t)n_hs_tokens * (size_t)n_embd;
            vec.assign(hs, hs + total);
        } else if (slot.task->params.hidden_pool == "skip_mean") {
            // Masked-mean pooling: mean over [skip_offset, n_hs_tokens).
            // start >= 0 is guaranteed by request-parse range validation
            // (no assert here: this endpoint answers 400s, never aborts).
            const int32_t start = slot.task->params.hidden_skip_offset;
            if (start >= n_hs_tokens) {
                auto err = std::make_unique<server_task_result_error>();
                err->id   = slot.task->id;
                err->index = slot.task->index;
                err->err_type = ERROR_TYPE_INVALID_REQUEST;
                err->err_msg = "skip_offset (" + std::to_string(start) +
                               ") >= n_tokens (" + std::to_string(n_hs_tokens) + "): prompt too short";
                self.queue_results.send(std::move(err));
                return;
            }
            vec.resize(n_embd, 0.0f);
            // Shared kernel (tools/hs-extract-common/masked-mean.h):
            // the same accumulation the batch extractor runs, so a fix
            // in either lands in both. size_t stride math lives there.
            const int64_t n_acc = hs_compute_single_range_mean(
                hs, n_hs_tokens, n_embd, start, n_hs_tokens, vec.data());
            // The pre-decode range check above already rejected
            // start >= n_hs_tokens, so the kernel cannot fail here;
            // if it ever does, that is a contract violation — abort
            // loud rather than return a silently wrong mean.
            GGML_ASSERT(n_acc > 0);
        } else {
            // Default: last token's hidden state
            vec.assign(hs + (size_t)(n_hs_tokens - 1) * n_embd,
                       hs + (size_t)n_hs_tokens * n_embd);
        }

        if (slot.task->params.hidden_normalize) {
            // pool=none + normalize is rejected at request parse (400),
            // so this branch only ever sees pooled vectors here.
            float norm = 0.0f;
            for (float v : vec) norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 0.0f) {
                for (float & v : vec) v /= norm;
            }
        }

        res->hidden_states[layer] = std::move(vec);
    }

    // capture is disabled by hs_toggle_reset on scope exit

    SLT_DBG(slot, "%s", "sending hidden states\n");
    self.queue_results.send(std::move(res));
}