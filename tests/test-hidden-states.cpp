#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// RAII wrappers (LlamaBackend, LlamaModel, LlamaContext, LlamaBatch) are in
// common/llama-raii.h (shared with hs-extract tools and examples).
#include "llama-raii.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    LlamaBackend backend;

    // Load model
    llama_model_params mparams = llama_model_default_params();
    LlamaModel model(llama_model_load_from_file(argv[1], mparams));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", argv[1]);
        return 1;
    }

    // Create context with hidden state extraction enabled
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.extract_hidden_states = true;

    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd = llama_model_n_embd_out(model);
    const int n_layer = llama_model_n_layer(model);

    printf("Model loaded: n_vocab=%d, n_embd=%d, n_layer=%d\n", n_vocab, n_embd, n_layer);

    // Tokenize a simple prompt
    const char * prompt = "Hello world";
    int n_tokens = llama_tokenize(llama_model_get_vocab(model), prompt, strlen(prompt), NULL, 0, true, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens; // negative means buffer too small, get actual count
    }
    if (n_tokens <= 0) {
        fprintf(stderr, "Tokenization error: %d\n", n_tokens);
        return 1;
    }

    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(llama_model_get_vocab(model), prompt, strlen(prompt), tokens.data(), n_tokens, true, true);

    printf("Tokenized '%s' -> %d tokens\n", prompt, n_tokens);

    // Decode using the ext batch API (capacity = context n_batch)
    LlamaBatch batch_wrapper;
    batch_wrapper.init(ctx);
    llama_batch_ext * batch = batch_wrapper.batch;
    for (int i = 0; i < n_tokens; i++) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        if (idx < 0) {
            fprintf(stderr, "batch add failed at token %d (idx=%d)\n", i, idx);
            return 1;
        }
        const llama_pos pos = i;
        llama_batch_ext_set_pos(batch, idx, &pos);
        if (i == n_tokens - 1) {
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
    }

    int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    if (ret != 0) {
        fprintf(stderr, "llama_process failed with %d\n", ret);
        return 1;
    }

    printf("Decode succeeded.\n");

    llama_synchronize(ctx);

    // Test hidden state extraction for layer 0
    int32_t n_hidden_tokens = llama_get_hidden_state_n_tokens(ctx);
    printf("n_hidden_tokens = %d\n", (int)n_hidden_tokens);

    if (n_hidden_tokens <= 0) {
        fprintf(stderr, "FAIL: n_hidden_tokens is %d (expected > 0)\n", (int)n_hidden_tokens);
        return 1;
    }

    int non_zero = 0;
    int total = 0;

    // Check first 3 layers
    int check_layers = n_layer < 3 ? n_layer : 3;
    for (int il = 0; il < check_layers; il++) {
        float * hs = llama_get_hidden_state(ctx, il);
        if (!hs) {
            fprintf(stderr, "FAIL: llama_get_hidden_state(ctx, %d) returned NULL\n", il);
            return 1;
        }

        // Check for non-zero values
        for (int i = 0; i < n_hidden_tokens * n_embd; i++) {
            total++;
            if (fabsf(hs[i]) > 1e-6f) {
                non_zero++;
            }
        }
    }

    // Ladder top slot: n_layer (final block output) must be readable,
    // non-NULL and finite. This is the fork's headline extension over
    // upstream's 0..n_layer-1 range; it went untested before.
    {
        float * hs = llama_get_hidden_state(ctx, n_layer);
        if (!hs) {
            fprintf(stderr, "FAIL: llama_get_hidden_state(ctx, n_layer) returned NULL (top slot)\n");
            return 1;
        }
        int top_non_zero = 0;
        for (int i = 0; i < n_hidden_tokens * n_embd; i++) {
            if (!std::isfinite(hs[i])) {
                fprintf(stderr, "FAIL: top slot value [%d] is not finite\n", i);
                return 1;
            }
            if (fabsf(hs[i]) > 1e-6f) top_non_zero++;
        }
        printf("Top slot (layer %d): non-zero %d / %d\n", n_layer, top_non_zero, n_hidden_tokens * n_embd);
        if (top_non_zero == 0) {
            fprintf(stderr, "FAIL: top slot is all zeros\n");
            return 1;
        }
        total    += n_hidden_tokens * n_embd;
        non_zero += top_non_zero;
    }

    // Out-of-range boundary: n_layer + 1 must be rejected (NULL), and the
    // batch getter must refuse it (-1) WITHOUT writing any output pointer
    // (its contract validates all indices before writing any entry).
    {
        if (llama_get_hidden_state(ctx, n_layer + 1) != NULL) {
            fprintf(stderr, "FAIL: llama_get_hidden_state(ctx, n_layer+1) should return NULL\n");
            return 1;
        }
        float * ptrs[2] = { nullptr, nullptr };
        int32_t layers[2] = { 0, n_layer + 1 };
        // Use a sentinel so we can detect any output-pointer write.
        float * sentinel = (float *) 0x1;
        ptrs[0] = sentinel;
        ptrs[1] = sentinel;
        if (llama_get_hidden_states_batch(ctx, layers, 2, ptrs) != -1) {
            fprintf(stderr, "FAIL: llama_get_hidden_states_batch must reject layer n_layer+1 with -1\n");
            return 1;
        }
        if (ptrs[0] != sentinel || ptrs[1] != sentinel) {
            fprintf(stderr, "FAIL: batch getter wrote an output pointer while rejecting\n");
            return 1;
        }
        printf("Boundary check: layer n_layer+1 rejected (single: NULL, batch: -1, no partial write)\n");
    }

    // HS-6b: failed-decode invalidation. The capture resets at decode entry
    // (before any early exit), so a FAILED decode must leave the getters
    // empty — never the previous batch's states. Fill the KV cache past
    // n_ctx (512) with 3 x 200-token batches on one sequence: the third
    // decode cannot fit and must fail.
    {
        // Start the filler sequence from a clean slate (the prompt decode
        // already used positions 0..n_tokens-1 on seq 0; positions must be
        // consecutive per sequence, so restart the sequence at 0).
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        LlamaBatch fbatch_wrapper;
        fbatch_wrapper.init(ctx);
        llama_batch_ext * fbatch = fbatch_wrapper.batch;
        for (int b = 0; b < 3; b++) {
            llama_batch_ext_clear(fbatch);
            for (int i = 0; i < 200; i++) {
                const int32_t idx = llama_batch_ext_add_token(fbatch, 0, tokens[i % n_tokens]);
                if (idx < 0) {
                    fprintf(stderr, "FAIL: filler batch add failed (b=%d, i=%d, idx=%d)\n", b, i, idx);
                    return 1;
                }
                const llama_pos pos = b * 200 + i;
                llama_batch_ext_set_pos(fbatch, idx, &pos);
                if (i == 199) {
                    llama_batch_ext_set_output_logits(fbatch, idx, true);  // mark an output row to mirror typical usage; not required by the ext API
                }
            }
            int fret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, fbatch);
            if (b < 2) {
                if (fret != 0) {
                    fprintf(stderr, "FAIL: filler decode %d unexpectedly failed with %d\n", b + 1, fret);
                    return 1;
                }
                llama_synchronize(ctx);
            } else {
                if (fret == 0) {
                    fprintf(stderr, "FAIL: over-capacity decode unexpectedly succeeded (test premise broken)\n");
                    return 1;
                }
                // The failed decode must have invalidated the capture.
                if (llama_get_hidden_state_n_tokens(ctx) != 0) {
                    fprintf(stderr, "FAIL: after a failed decode, capture still reports %d tokens (stale data exposure)\n",
                            llama_get_hidden_state_n_tokens(ctx));
                    return 1;
                }
                if (llama_get_hidden_state(ctx, 0) != NULL) {
                    fprintf(stderr, "FAIL: after a failed decode, layer-0 capture is readable (stale data exposure)\n");
                    return 1;
                }
                printf("Failed-decode invalidation: getters empty after failed decode (no stale exposure)\n");

                // Recovery: a subsequent good decode re-populates the capture.
                llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);  // fresh positions (fillers used 0..599)
                llama_batch_ext_clear(batch);
                for (int i = 0; i < n_tokens; i++) {
                    const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
                    if (idx < 0) {
                        fprintf(stderr, "FAIL: recovery batch add failed at token %d\n", i);
                        return 1;
                    }
                    const llama_pos pos = i;
                    llama_batch_ext_set_pos(batch, idx, &pos);
                    if (i == n_tokens - 1) {
                        llama_batch_ext_set_output_logits(batch, idx, true);
                    }
                }
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
                    fprintf(stderr, "FAIL: recovery decode failed\n");
                    return 1;
                }
                llama_synchronize(ctx);
                if (llama_get_hidden_state_n_tokens(ctx) != n_tokens || llama_get_hidden_state(ctx, 0) == NULL) {
                    fprintf(stderr, "FAIL: capture did not recover after a good decode\n");
                    return 1;
                }
                printf("Recovery: capture live again after good decode (%d tokens)\n", n_tokens);
                break;
            }
        }
    }

    // Toggle-off contract (fork API): disabling extraction clears the
    // capture state; getters return NULL/0 until a re-enabled decode
    // repopulates them. Then the requested-layer mask: only selected slots
    // are captured, unrequested getters return NULL, and selected values are
    // byte-identical to the full-ladder capture.
    {
        auto refill = [&]() {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            llama_batch_ext_clear(batch);
            for (int i = 0; i < n_tokens; i++) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
                if (idx < 0) {
                    fprintf(stderr, "FAIL: batch add failed at token %d\n", i);
                    return false;
                }
                const llama_pos pos = i;
                llama_batch_ext_set_pos(batch, idx, &pos);
                if (i == n_tokens - 1) {
                    llama_batch_ext_set_output_logits(batch, idx, true);
                }
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
                fprintf(stderr, "FAIL: decode failed\n");
                return false;
            }
            llama_synchronize(ctx);
            return true;
        };

        if (llama_set_extract_hidden_states(ctx, false) != 0) {
            fprintf(stderr, "FAIL: toggle-off returned non-zero\n");
            return 1;
        }
        if (llama_get_hidden_state_n_tokens(ctx) != 0 ||
            llama_get_hidden_state(ctx, 0) != NULL ||
            llama_get_hidden_state(ctx, n_layer) != NULL) {
            fprintf(stderr, "FAIL: toggle-off left stale capture state\n");
            return 1;
        }
        printf("PASS: Toggle-off clears capture state (getters NULL/0).\n");

        if (llama_set_extract_hidden_states(ctx, true) != 0) {
            fprintf(stderr, "FAIL: re-enable returned non-zero\n");
            return 1;
        }
        if (!refill()) return 1;
        if (llama_get_hidden_state_n_tokens(ctx) != n_tokens || llama_get_hidden_state(ctx, 0) == NULL) {
            fprintf(stderr, "FAIL: capture not live after re-enable\n");
            return 1;
        }
        printf("PASS: Re-enable restores capture (%d tokens).\n", n_tokens);

        // Full-ladder reference for slots 0 and 2 (fixture has >= 3 slots).
        std::vector<float> ref0(n_embd), ref2(n_embd);
        memcpy(ref0.data(), llama_get_hidden_state(ctx, 0), (size_t) n_embd * sizeof(float));
        memcpy(ref2.data(), llama_get_hidden_state(ctx, 2), (size_t) n_embd * sizeof(float));

        const int32_t sel[2] = {0, 2};
        if (llama_set_hidden_state_layers(ctx, sel, 2) != 0) {
            fprintf(stderr, "FAIL: layer selection returned non-zero\n");
            return 1;
        }
        if (!refill()) return 1;
        float * m0 = llama_get_hidden_state(ctx, 0);
        float * m2 = llama_get_hidden_state(ctx, 2);
        if (llama_get_hidden_state_n_tokens(ctx) != n_tokens || m0 == NULL || m2 == NULL ||
            llama_get_hidden_state(ctx, 1) != NULL || llama_get_hidden_state(ctx, n_layer) != NULL) {
            fprintf(stderr, "FAIL: mask did not restrict capture to slots {0, 2}\n");
            return 1;
        }
        if (memcmp(ref0.data(), m0, (size_t) n_embd * sizeof(float)) != 0 ||
            memcmp(ref2.data(), m2, (size_t) n_embd * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: masked capture values differ from full-ladder capture\n");
            return 1;
        }
        printf("PASS: Requested-layer mask captures only {0, 2}, values byte-identical.\n");

        // Invalid selection is refused with -1, context unchanged.
        const int32_t bad[1] = {n_layer + 1};
        if (llama_set_hidden_state_layers(ctx, bad, 1) != -1) {
            fprintf(stderr, "FAIL: out-of-range layer selection was not refused\n");
            return 1;
        }
        if (llama_get_hidden_state(ctx, 0) == NULL) {
            fprintf(stderr, "FAIL: refused selection changed capture state\n");
            return 1;
        }
        // Clearing the mask restores the full ladder.
        if (llama_set_hidden_state_layers(ctx, NULL, 0) != 0) {
            fprintf(stderr, "FAIL: clearing the layer selection returned non-zero\n");
            return 1;
        }
        if (!refill()) return 1;
        if (llama_get_hidden_state(ctx, 1) == NULL || llama_get_hidden_state(ctx, n_layer) == NULL) {
            fprintf(stderr, "FAIL: clearing the mask did not restore the full ladder\n");
            return 1;
        }
        printf("PASS: Mask refusal (-1) and clear-to-full-ladder contract.\n");
    }

    // HS-6a: multi-ubatch accumulation. Same prompt decoded with
    // n_ubatch < n_tokens must produce byte-identical captures to the
    // single-ubatch context above (the per-ubatch accumulation path with
    // n_tokens_prev offsets).
    {
        llama_context_params cparams2 = llama_context_default_params();
        cparams2.n_ctx = 512;
        cparams2.n_batch = 512;
        cparams2.n_ubatch = 8;  // forces ceil(n_tokens/8) ubatches for any prompt
        cparams2.extract_hidden_states = true;
        LlamaContext ctx2(llama_init_from_model(model, cparams2));
        if (!ctx2) {
            fprintf(stderr, "Failed to create multi-ubatch context\n");
            return 1;
        }
        LlamaBatch batch2_wrapper;
        batch2_wrapper.init(ctx2);
        llama_batch_ext * batch2 = batch2_wrapper.batch;
        for (int i = 0; i < n_tokens; i++) {
            const int32_t idx = llama_batch_ext_add_token(batch2, 0, tokens[i]);
            if (idx < 0) {
                fprintf(stderr, "FAIL: multi-ubatch batch add failed at token %d\n", i);
                return 1;
            }
            const llama_pos pos = i;
            llama_batch_ext_set_pos(batch2, idx, &pos);
            if (i == n_tokens - 1) {
                llama_batch_ext_set_output_logits(batch2, idx, true);
            }
        }
        if (llama_process(ctx2, LLAMA_PROCESS_TYPE_DECODE, batch2) != 0) {
            fprintf(stderr, "FAIL: multi-ubatch decode failed\n");
            return 1;
        }
        llama_synchronize(ctx2);
        if (llama_get_hidden_state_n_tokens(ctx2) != n_tokens) {
            fprintf(stderr, "FAIL: multi-ubatch capture reports %d tokens, expected %d\n",
                    llama_get_hidden_state_n_tokens(ctx2), n_tokens);
            return 1;
        }
        const float * ref0 = llama_get_hidden_state(ctx, 0);
        const float * ub0  = llama_get_hidden_state(ctx2, 0);
        if (!ref0 || !ub0) {
            fprintf(stderr, "FAIL: multi-ubatch comparison getter returned NULL\n");
            return 1;
        }
        if (memcmp(ref0, ub0, (size_t)n_tokens * n_embd * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: multi-ubatch layer-0 capture differs from single-ubatch reference\n");
            return 1;
        }
        const float * refN = llama_get_hidden_state(ctx, n_layer);
        const float * ubN  = llama_get_hidden_state(ctx2, n_layer);
        if (!refN || !ubN || memcmp(refN, ubN, (size_t)n_tokens * n_embd * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: multi-ubatch top-layer capture differs from single-ubatch reference\n");
            return 1;
        }
        printf("Multi-ubatch accumulation: layer 0 and top slot byte-identical across ubatch configs\n");
    }

    // API coverage: llama_model_supports_hidden_states + llama_model_arch_name
    // (the server pre-checks via these; no test called them directly until now).
    // This model carries a registry arch (extraction context was created above),
    // so supports() must be true and arch_name() non-null and non-empty.
    {
        const bool supported = llama_model_supports_hidden_states(model);
        const char * arch_name = llama_model_arch_name(model);
        if (!supported) {
            fprintf(stderr, "FAIL: llama_model_supports_hidden_states must be true for a registry arch with an extraction context\n");
            return 1;
        }
        if (arch_name == nullptr || arch_name[0] == '\0') {
            fprintf(stderr, "FAIL: llama_model_arch_name returned %s\n",
                    arch_name == nullptr ? "NULL" : "empty string");
            return 1;
        }
        printf("Supports hidden states: %s (arch: %s)\n", supported ? "true" : "false", arch_name);
    }

    // Refusal contract (optional second argument: a loadable GGUF whose arch
    // is NOT in the capability registry, e.g. models/hs-refusal-fixture.gguf
    // generated by tests/gen-refusal-fixture.py). Creation with extraction on
    // must fail cleanly; the runtime toggle must return -1 and leave the
    // context functional.
    if (argc > 2) {
        llama_model_params rp = llama_model_default_params();
        rp.n_gpu_layers = 0;
        LlamaModel rmodel(llama_model_load_from_file(argv[2], rp));
        if (!rmodel) {
            fprintf(stderr, "FAIL: refusal fixture '%s' failed to load\n", argv[2]);
            return 1;
        }
        llama_context_params rc = llama_context_default_params();
        rc.n_ctx = 64;
        rc.n_batch = 64;
        rc.n_ubatch = 64;
        rc.extract_hidden_states = true;
        llama_context * rctx_bad = llama_init_from_model(rmodel, rc);
        if (rctx_bad != NULL) {
            fprintf(stderr, "FAIL: creation with extraction on a non-registry arch was not refused\n");
            llama_free(rctx_bad);
            return 1;
        }
        rc.extract_hidden_states = false;
        LlamaContext rctx(llama_init_from_model(rmodel, rc));
        if (!rctx) {
            fprintf(stderr, "FAIL: plain context on the refusal fixture failed\n");
            return 1;
        }
        if (llama_set_extract_hidden_states(rctx, true) != -1) {
            fprintf(stderr, "FAIL: runtime toggle on a non-registry arch did not return -1\n");
            return 1;
        }
        if (llama_get_hidden_state_n_tokens(rctx) != 0 || llama_get_hidden_state(rctx, 0) != NULL) {
            fprintf(stderr, "FAIL: refused toggle left capture state behind\n");
            return 1;
        }
        printf("PASS: Non-registry arch refused at creation and at toggle (-1), context unchanged.\n");
    }

    // Show some values from layer 0
    printf("Layer 0 first 10 values: ");
    float * hs0 = llama_get_hidden_state(ctx, 0);
    if (!hs0) { printf("(unavailable)\n"); return 1; }
    for (int i = 0; i < 10 && i < n_embd; i++) {
        printf("%.6f ", hs0[i]);
    }
    printf("\n");

    if (total > 0) {
        printf("Non-zero / total (first %d layers): %d / %d (%.1f%%)\n", check_layers, non_zero, total, 100.0f * non_zero / total);
    } else {
        printf("Non-zero / total (first %d layers): no data (total=0)\n", check_layers);
    }

    // RAII destructors handle all cleanup automatically

    if (non_zero > 0) {
        printf("PASS: Hidden states contain non-zero values.\n");
        return 0;
    }
    fprintf(stderr, "FAIL: All hidden state values are zero.\n");
    return 1;
}
