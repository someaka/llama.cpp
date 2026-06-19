#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        fprintf(stderr, "%s: failed to load model '%s'\n", __func__, argv[1]);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.extract_hidden_states = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    const char * prompt = "Hello, world!";
    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(model, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "%s: failed to tokenize\n", __func__);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    tokens.resize(n_tokens);

    printf("Prompt: %s\n", prompt);
    printf("Tokens: %d\n", n_tokens);

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokens.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = tokens.size();

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: failed to decode\n", __func__);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_batch_free(batch);

    int n_layers = llama_model_n_layer(model);
    int32_t n_hidden_tokens = llama_get_hidden_state_n_tokens(ctx);

    printf("Hidden states: %d tokens, %d layers\n", n_hidden_tokens, n_layers);

    for (int il = 0; il < n_layers && il < 5; il++) {
        float * hidden = llama_get_hidden_state(ctx, il);
        if (hidden) {
            printf("Layer %d: [", il);
            for (int i = 0; i < 5; i++) {
                printf("%.4f%s", hidden[i], i < 4 ? ", " : "");
            }
            printf("...]\n");
        } else {
            printf("Layer %d: NULL\n", il);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
