// Logits-equivalence probe: with the post-final-norm pruning fix, logits
// must be identical whether or not extract_hidden_states is enabled.
// Same model, same prompt, greedy next-token: compare argmax + top-8 logit
// values across extraction on/off.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt     = argv[2];

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(64);
    int n = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), toks.size(), true, false);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), toks.size(), true, false); }
    toks.resize(n);

    int argmaxes[2];

    for (int mode = 0; mode < 2; ++mode) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 128;
        cparams.n_batch = 128;
        cparams.n_ubatch = 128;
        cparams.extract_hidden_states = (mode == 1);

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fprintf(stderr, "ctx failed mode=%d\n", mode); return 1; }

        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            batch.token[i]    = toks[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (i == n - 1);
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed mode=%d\n", mode); llama_batch_free(batch); return 1; }

        // get_logits_ith takes the BATCH TOKEN index, not the output row
        const float * logits = llama_get_logits_ith(ctx, n - 1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        argmaxes[mode] = best;

        // capture the top-8 logit values (sorted) for a value-level check
        std::vector<float> lg(logits, logits + n_vocab);
        std::vector<size_t> idx(n_vocab);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        // simple selection of top 8
        std::partial_sort(idx.begin(), idx.begin()+8, idx.end(), [&](size_t a, size_t b){ return lg[a] > lg[b]; });
        printf("mode=%s argmax=%d\n", mode ? "EXTRACT" : "plain ", best);
        fflush(stdout);
        for (int k = 0; k < 8; ++k) printf("  top%d: tok=%zu logit=%.6f\n", k+1, idx[k], lg[idx[k]]);

        llama_batch_free(batch);
        llama_free(ctx);
    }

    bool same = argmaxes[0] == argmaxes[1];
    printf("ARGMAX_MATCH=%s\n", same ? "YES" : "NO");
    llama_model_free(model);
    llama_backend_free();
    return same ? 0 : 2;
}
