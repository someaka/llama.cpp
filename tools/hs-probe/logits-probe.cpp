// Logits-equivalence probe: with the post-final-norm pruning fix, logits
// must be identical whether or not extract_hidden_states is enabled.
// Same model, same prompt, greedy next-token. Compared across extraction
// on/off: (1) the argmax token id, and (2) the top-8 logit value sets
// (sorted, compared with a 1e-6 tolerance and the measured max delta
// reported — same-binary same-GPU decode is deterministic: measured delta
// is 0, so any drift beyond 1e-6 is a regression). Exit 0 on match,
// 2 on mismatch (a failed decode is reported as a probe-failure MISMATCH
// and also exits 2), 1 on setup failure (usage, model load, prompt too
// long, context creation, unknown argument).
//
// Two extraction paths are covered:
//   mode 1: creation-time  — context created with extract_hidden_states=true.
//   mode 2: runtime toggle — context created plain, decodes once, then
//           llama_set_extract_hidden_states(true) and decodes again. This is
//           the path the /hidden-states server actually exercises (warmup
//           first, toggle on per request), previously unprobed.
// Modes 1 and 2 are each compared against mode 0. The mode-2 comparison also
// re-uses the same context, so it additionally guards the toggle itself
// against perturbing logits.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "llama.h"

namespace {

struct ProbeResult {
    int argmax = -1;
    std::vector<float> top8;
};

ProbeResult probe_once(llama_context * ctx, const llama_vocab * vocab,
                       const std::vector<llama_token> & toks) {
    const int n = (int) toks.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n - 1);
    }
    batch.n_tokens = n;
    fprintf(stderr, "probe: decode begin (n=%d)\n", n);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "decode failed (n=%d)\n", n);
        llama_batch_free(batch);
        return {};
    }
    fprintf(stderr, "probe: decode ok, reading logits\n");

    // get_logits_ith takes the BATCH TOKEN index, not the output row
    const float * logits = llama_get_logits_ith(ctx, n - 1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    ProbeResult r;
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
    r.argmax = best;

    // capture the top-8 logit values (sorted) for a value-level check
    std::vector<float> lg(logits, logits + n_vocab);
    std::vector<size_t> idx(n_vocab);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    // simple selection of top 8
    std::partial_sort(idx.begin(), idx.begin()+8, idx.end(), [&](size_t a, size_t b){ return lg[a] > lg[b]; });
    for (int k = 0; k < 8; ++k) r.top8.push_back(lg[idx[k]]);
    printf("argmax=%d\n", best);
    fflush(stdout);
    for (int k = 0; k < 8; ++k) fprintf(stderr, "  top%d: tok=%zu logit=%.6f\n", k+1, idx[k], lg[idx[k]]);

    llama_batch_free(batch);
    return r;
}

// exact top-8 comparison with a 1e-6 tolerance fallback; prints the verdict.
// Size-safe: an empty (failed-decode) result never indexes the vector, and
// mismatches on size or sentinel argmax report the cause instead of UB.
bool compare(const char * tag, const ProbeResult & plain, const ProbeResult & extract) {
    if (plain.top8.size() != 8 || extract.top8.size() != 8 ||
        plain.argmax < 0 || extract.argmax < 0) {
        printf("%s: MISMATCH (probe failure: plain argmax=%d top8=%zu, extract argmax=%d top8=%zu)\n",
               tag, plain.argmax, plain.top8.size(), extract.argmax, extract.top8.size());
        fflush(stdout);
        return false;
    }
    bool same = plain.argmax == extract.argmax;
    const bool argmax_ok = same;
    float max_delta = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const float d = std::fabs(plain.top8[k] - extract.top8[k]);
        if (d > max_delta) max_delta = d;
        if (d > 1e-6f) same = false;
    }
    printf("%s: %s", tag, same ? "MATCH" : "MISMATCH");
    if (!same) printf(" (top-8 value drift%s)", argmax_ok ? "" : "; argmax also differs");
    printf("\n");
    printf("%s_MAX_DELTA=%.9g\n", tag, max_delta);
    fflush(stdout);
    return same;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <prompt> [--runtime-toggle]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * prompt     = argv[2];
    bool runtime_toggle = false;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--runtime-toggle") {
            runtime_toggle = true;
        } else {
            // A misspelled flag would silently skip a mode and report MATCH
            // from less coverage — a probe must fail loudly on unknown input.
            fprintf(stderr, "Error: unknown argument '%s' (supported: --runtime-toggle)\n", argv[i]);
            return 1;
        }
    }

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
    if (n > 120) { fprintf(stderr, "prompt too long: %d tokens (probe limit 120, sized for headroom under the probe's 128-slot context)\n", n); return 1; }

    bool all_ok = true;

    // mode 0: plain context (reference)
    llama_context_params cparams0 = llama_context_default_params();
    cparams0.n_ctx = 128;
    cparams0.n_batch = 128;
    cparams0.n_ubatch = 128;
    llama_context * ctx0 = llama_init_from_model(model, cparams0);
    if (!ctx0) { fprintf(stderr, "ctx failed (plain)\n"); return 1; }
    printf("mode=plain  prompt_tokens=%d\n", n);
    const ProbeResult plain = probe_once(ctx0, vocab, toks);
    llama_free(ctx0);

    // mode 1: created with extraction on (creation-time init path)
    llama_context_params cparams1 = cparams0;
    cparams1.extract_hidden_states = true;
    llama_context * ctx1 = llama_init_from_model(model, cparams1);
    if (!ctx1) { fprintf(stderr, "ctx failed (creation-time extract)\n"); return 1; }
    printf("mode=EXTRACT (creation-time)\n");
    const ProbeResult created = probe_once(ctx1, vocab, toks);
    llama_free(ctx1);
    all_ok = compare("CREATION_TIME", plain, created) && all_ok;

    // mode 2 (optional): runtime toggle on a plain context — the server path.
    // llama_memory_clear mirrors server usage: each request decodes fresh, so
    // the warmup pass's KV positions must not constrain the toggled decode
    // (position continuity would otherwise reject it, ret = -1).
    if (runtime_toggle) {
        llama_context * ctx2 = llama_init_from_model(model, cparams0);
        if (!ctx2) { fprintf(stderr, "ctx failed (runtime toggle)\n"); return 1; }
        const ProbeResult warmup = probe_once(ctx2, vocab, toks);
        all_ok = compare("WARMUP", plain, warmup) && all_ok;
        llama_memory_clear(llama_get_memory(ctx2), /*data=*/true);
        llama_set_extract_hidden_states(ctx2, true);
        printf("mode=EXTRACT (runtime toggle)\n");
        const ProbeResult toggled = probe_once(ctx2, vocab, toks);
        fprintf(stderr, "probe: toggle probe returned argmax=%d top8=%zu\n", toggled.argmax, toggled.top8.size());
        llama_free(ctx2);
        all_ok = compare("RUNTIME_TOGGLE", plain, toggled) && all_ok;
    }

    llama_model_free(model);
    llama_backend_free();
    return all_ok ? 0 : 2;
}
