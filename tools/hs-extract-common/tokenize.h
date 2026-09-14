// Shared bounded tokenizer for the hs-extract tools.
//
// Single source of truth (B-audit D3): both tools previously kept their own
// tokenizer - hs-extract called common_tokenize() (no per-token validation),
// hs-extract-batch hand-rolled a validating variant, and parity between the
// two was maintained by hand-copying checks. This header is the one
// implementation:
//   - Token output is identical to common_tokenize(vocab, text, add_bos,
//     /*special=*/true) on the happy path (same underlying llama_tokenize
//     call and flags).
//   - Every returned token is validated against the vocab bound; an
//     out-of-range id would be an out-of-bounds embedding lookup (no
//     validation downstream), so it hard-errors instead.
//   - Returns an empty vector on failure instead of throwing - callers
//     treat empty as their error contract (producer thread / raw path).
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "llama.h"

inline std::vector<llama_token> hs_tokenize_bounded(
    const llama_vocab* vocab,
    const char* text,
    size_t text_len,
    bool add_bos
) {
    int n = -llama_tokenize(vocab, text, text_len, nullptr, 0, add_bos, true);
    if (n <= 0) return {};
    std::vector<llama_token> toks((size_t)n);
    int n_tokens = llama_tokenize(vocab, text, text_len, toks.data(), (int)toks.size(), add_bos, true);
    if (n_tokens <= 0) return {};
    toks.resize((size_t)n_tokens);

    // Validate token bounds (prevent crashes from invalid token IDs)
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token tok : toks) {
        if (tok < 0 || tok >= n_vocab) {
            fprintf(stderr, "Error: tokenizer returned out-of-bounds token %d (vocab size: %d)\n", tok, n_vocab);
            return {};
        }
    }
    return toks;
}
