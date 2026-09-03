// Test whether warmup + dynamic extract_hidden_states toggle changes values.
// Modes 0/1 reproduce the CLI and server init paths and print values (debug).
// Mode 1 mirrors the server /hidden-states init contract: extraction off and
// embeddings off at creation, BOS+EOS warmup decode, toggle on, perf reset,
// memory clear. n_ubatch = n_ctx in every mode. Scope note: n_threads is
// left at the backend default (the server sets cpuparams.n_threads) —
// scheduling only; mode 2's bitwise comparison is the proof that no
// result-affecting parameter differs.
// Mode 2 (default) runs BOTH paths in one process and compares the captured
// hidden states bitwise — a single-mode run can only print, never verify.
// Exit codes: 0 = identical (mode 2) or ran (modes 0/1), 1 = setup error,
// 2 = mode-2 comparison mismatch.

#include "common.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace {

// Run one decode of `tokens` under the requested init style and return the
// full layer-10 hidden-state row (n_embd floats). Exits on any failure:
// a test binary must never fall through a failed decode into a "pass".
std::vector<float> run_and_capture(llama_model * model, const llama_vocab * vocab,
                                   const std::vector<llama_token> & tokens,
                                   bool server_style) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx  = 2048;
    cparams.n_batch = 2048;
    // One ubatch per prompt: the capture path parity rule (fork D2/C1). A
    // split ubatch changes which kernel shapes produce the capture.
    cparams.n_ubatch = cparams.n_ctx;
    // Server init keeps embeddings off for every request type (common_params
    // default); logits are a decode-time batch flag in both paths.
    cparams.embeddings = false;

    if (server_style) {
        // Server style: extraction off at creation, warmup decode, then the
        // per-request toggle + memory clear before the real decode.
        cparams.extract_hidden_states = false;
    } else {
        // CLI style: extraction on from context creation.
        cparams.extract_hidden_states = true;
    }
    cparams.no_perf = true;
    // Logits are controlled at decode time (llama_batch logits flags), not at
    // context creation; both styles leave them off by the same route.

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to init context (%s)\n", server_style ? "server" : "cli");
        exit(1);
    }

    if (server_style) {
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);
        std::vector<llama_token> warmup_tokens;
        if (bos != LLAMA_TOKEN_NULL) {
            warmup_tokens.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            warmup_tokens.push_back(eos);
        }
        if (!warmup_tokens.empty()) {
            llama_batch wb = llama_batch_get_one(warmup_tokens.data(), warmup_tokens.size());
            if (llama_decode(ctx, wb) != 0) {
                fprintf(stderr, "Warmup decode failed\n");
                exit(1);
            }
            llama_synchronize(ctx);
        }
        llama_set_extract_hidden_states(ctx, true);
        llama_memory_clear(llama_get_memory(ctx), true);
        // Server reset: perf counters (common_init_from_params warmup) and
        // memory (per-request prompt_clear), so the decode starts at pos 0.
        llama_perf_context_reset(ctx);
    }

    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Decode failed (%s)\n", server_style ? "server" : "cli");
        exit(1);
    }
    llama_synchronize(ctx);

    const int layer = 10;
    float * hs = llama_get_hidden_state(ctx, layer);
    if (!hs) {
        fprintf(stderr, "No hidden states available (%s); n_hidden_tokens = %d\n",
                server_style ? "server" : "cli", llama_get_hidden_state_n_tokens(ctx));
        exit(1);
    }

    const int n_embd = llama_model_n_embd(model);
    std::vector<float> out(hs, hs + n_embd);
    llama_free(ctx);
    return out;
}

void print_row(const char * label, const std::vector<float> & v) {
    printf("%s - Layer 10, first 5 values:\n", label);
    for (int i = 0; i < 5 && i < (int) v.size(); i++) {
        printf("  [%d]: %.8f\n", i, v[i]);
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model> <prompt> [mode]\n", argv[0]);
        fprintf(stderr, "  mode=0: no warmup, extract_hidden_states=true from start (CLI style; prints)\n");
        fprintf(stderr, "  mode=1: warmup first, then toggle extract_hidden_states (server style; prints)\n");
        fprintf(stderr, "  mode=2: run both paths and compare bitwise (default; exits 2 on mismatch)\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * prompt     = argv[2];
    const int    mode       = argc > 3 ? atoi(argv[3]) : 2;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 100;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    tokens.resize(strlen(prompt) + 16);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);
    if (n_tokens <= 0) {
        fprintf(stderr, "Tokenization failed\n");
        return 1;
    }
    tokens.resize(n_tokens);

    if (mode == 0) {
        print_row("Mode 0 (CLI)", run_and_capture(model, vocab, tokens, false));
    } else if (mode == 1) {
        print_row("Mode 1 (Server-style)", run_and_capture(model, vocab, tokens, true));
    } else {
        std::vector<float> cli    = run_and_capture(model, vocab, tokens, false);
        std::vector<float> server = run_and_capture(model, vocab, tokens, true);
        print_row("Mode 0 (CLI)", cli);
        print_row("Mode 1 (Server-style)", server);

        if (cli.size() != server.size()) {
            fprintf(stderr, "COMPARE: MISMATCH (size %zu vs %zu)\n", cli.size(), server.size());
            return 2;
        }
        if (memcmp(cli.data(), server.data(), cli.size() * sizeof(float)) != 0) {
            for (size_t i = 0; i < cli.size(); i++) {
                if (cli[i] != server[i]) {
                    fprintf(stderr, "COMPARE: MISMATCH at word %zu: %.8f vs %.8f\n", i, cli[i], server[i]);
                    break;
                }
            }
            return 2;
        }
        printf("COMPARE: IDENTICAL (%zu floats, bitwise)\n", cli.size());
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
