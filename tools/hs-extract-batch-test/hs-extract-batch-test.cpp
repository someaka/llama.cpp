#include "common.h"
#include "llama.h"
#include "build-info.h"
#include "llama-raii.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Diagnostic tool: test if batch construction method affects hidden state values
// Compares llama_batch_get_one() (CLI path) vs explicit batch.add() (server path)

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt_text = "hello";
    int n_gpu_layers = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "error: --model requires an argument\n"); return 1; }
            model_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) { fprintf(stderr, "error: --prompt requires an argument\n"); return 1; }
            prompt_text = argv[i];
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) { fprintf(stderr, "error: --n-gpu-layers requires an argument\n"); return 1; }
            char *endptr;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0') { fprintf(stderr, "error: --n-gpu-layers value must be a number\n"); return 1; }
            n_gpu_layers = (int) val;
        }
    }

    if (!model_path) {
        fprintf(stderr, "usage: %s -m model.gguf [-p prompt] [-ngl N]\n", argv[0]);
        return 1;
    }

    LlamaBackend backend;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    LlamaModel model(llama_model_load_from_file(model_path, model_params));
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path);
        return 1;
    }

    const int n_layers = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd_out(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    ctx_params.extract_hidden_states = true;
    ctx_params.no_perf = true;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::string prompt(prompt_text);
    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, add_bos, true);

    fprintf(stderr, "Tokenized '%s' into %zu tokens\n", prompt_text, tokens.size());

    // Test 1: CLI-style batch (llama_batch_get_one)
    fprintf(stderr, "\n=== Test 1: CLI-style batch (llama_batch_get_one) ===\n");
    {
        LlamaContext ctx(llama_init_from_model(model, ctx_params));
        if (!ctx) {
            fprintf(stderr, "error: failed to create context\n");
            return 1;
        }

        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
        int32_t ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "error: llama_decode failed with code %d\n", ret);
            return 1;
        }

        llama_synchronize(ctx);

        int layer = (n_layers > 10) ? 10 : 0;
        float * hs = llama_get_hidden_state(ctx, layer);
        if (!hs) {
            fprintf(stderr, "error: llama_get_hidden_state returned NULL\n");
            return 1;
        }

        fprintf(stderr, "Layer %d, first 5 values: ", layer);
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, "%.6f ", hs[i]);
        }
        fprintf(stderr, "\n");
    }

    // Test 2: Server-style batch (explicit positions, seq_id=0, logits=false except last)
    fprintf(stderr, "\n=== Test 2: Server-style batch (explicit pos, seq_id=0) ===\n");
    {
        LlamaContext ctx(llama_init_from_model(model, ctx_params));
        if (!ctx) {
            fprintf(stderr, "error: failed to create context\n");
            return 1;
        }

        llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            common_batch_add(batch, tokens[i], (llama_pos) i, {0}, (i == tokens.size() - 1));
        }

        int32_t ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "error: llama_decode failed with code %d\n", ret);
            return 1;
        }

        llama_synchronize(ctx);

        int layer = (n_layers > 10) ? 10 : 0;
        float * hs = llama_get_hidden_state(ctx, layer);
        if (!hs) {
            fprintf(stderr, "error: llama_get_hidden_state returned NULL\n");
            return 1;
        }

        fprintf(stderr, "Layer %d, first 5 values: ", layer);
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, "%.6f ", hs[i]);
        }
        fprintf(stderr, "\n");
    }

    // Test 3: Server-style with seq_id=1 (simulate slot 1)
    fprintf(stderr, "\n=== Test 3: Server-style batch (seq_id=1) ===\n");
    {
        LlamaContext ctx(llama_init_from_model(model, ctx_params));
        if (!ctx) {
            fprintf(stderr, "error: failed to create context\n");
            return 1;
        }

        llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            common_batch_add(batch, tokens[i], (llama_pos) i, {1}, (i == tokens.size() - 1));
        }

        int32_t ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "error: llama_decode failed with code %d\n", ret);
            return 1;
        }

        llama_synchronize(ctx);

        int layer = (n_layers > 10) ? 10 : 0;
        float * hs = llama_get_hidden_state(ctx, layer);
        if (!hs) {
            fprintf(stderr, "error: llama_get_hidden_state returned NULL\n");
            return 1;
        }

        fprintf(stderr, "Layer %d, first 5 values: ", layer);
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, "%.6f ", hs[i]);
        }
        fprintf(stderr, "\n");
    }

    // Test 4: Server-style with seq_id=0, all tokens output (logits=true for all)
    fprintf(stderr, "\n=== Test 4: Server-style batch (seq_id=0, all tokens output) ===\n");
    {
        LlamaContext ctx(llama_init_from_model(model, ctx_params));
        if (!ctx) {
            fprintf(stderr, "error: failed to create context\n");
            return 1;
        }

        llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            common_batch_add(batch, tokens[i], (llama_pos) i, {0}, true);
        }

        int32_t ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "error: llama_decode failed with code %d\n", ret);
            return 1;
        }

        llama_synchronize(ctx);

        int layer = (n_layers > 10) ? 10 : 0;
        float * hs = llama_get_hidden_state(ctx, layer);
        if (!hs) {
            fprintf(stderr, "error: llama_get_hidden_state returned NULL\n");
            return 1;
        }

        fprintf(stderr, "Layer %d, first 5 values: ", layer);
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, "%.6f ", hs[i]);
        }
        fprintf(stderr, "\n");
    }

    return 0;
}
