#include "llama.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// C smoke test for hidden states extraction API
// Usage: test-hidden-states-c <model_path>
int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return 1;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 64;
    ctx_params.extract_hidden_states = true;

    struct llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Tokenize "Hello world"
    const char * prompt = "Hello world";
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token tokens[16];
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, 16, true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "Failed to tokenize\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Build batch
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    batch.logits[n_tokens - 1] = 1;
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Check hidden states
    int32_t n_hidden = llama_get_hidden_state_n_tokens(ctx);
    printf("n_hidden_tokens = %d\n", n_hidden);

    if (n_hidden <= 0) {
        fprintf(stderr, "ERROR: no hidden tokens extracted\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    int n_layer = llama_model_n_layer(model);
    printf("n_layer = %d\n", n_layer);

    int ok = 1;
    for (int il = 0; il < n_layer; il++) {
        float * hs = llama_get_hidden_state(ctx, il);
        if (!hs) {
            fprintf(stderr, "ERROR: layer %d returned NULL\n", il);
            ok = 0;
            break;
        }
        // Check first value is not NaN
        if (hs[0] != hs[0]) {  // NaN check
            fprintf(stderr, "ERROR: layer %d has NaN\n", il);
            ok = 0;
            break;
        }
    }

    // Last-token access with the single getter: pointer arithmetic on the
    // documented layout (layer base + (n_tokens-1) * n_embd_out).
    {
        float * hs0 = llama_get_hidden_state(ctx, 0);
        if (hs0 == NULL) {
            fprintf(stderr, "ERROR: layer 0 returned NULL before last-token check\n");
            ok = 0;
        } else {
            int32_t n_embd = llama_model_n_embd(model);
            float * hs_last = hs0 + (size_t)(n_hidden - 1) * n_embd;
            int bad = 0;
            for (int32_t d = 0; d < n_embd; d++) {
                if (hs_last[d] != hs_last[d]) {  // NaN check
                    bad = 1;
                    break;
                }
            }
            if (bad) {
                fprintf(stderr, "ERROR: last-token row for layer 0 contains NaN\n");
                ok = 0;
            }
        }
    }

    // Test out-of-range layer indices are rejected
    float * hs_bad_layer = llama_get_hidden_state(ctx, -1);
    if (hs_bad_layer != NULL) {
        fprintf(stderr, "ERROR: negative layer should return NULL\n");
        ok = 0;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (ok) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL\n");
    return 1;
}
