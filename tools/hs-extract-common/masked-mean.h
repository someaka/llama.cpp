// Masked-mean accumulation kernel shared by the hidden-states tools.
//
// Single source of truth for range-meaning a [n_tokens, n_embd] row-major
// float block: the batch extractor (multi-range assignments) and the
// server's skip_mean pooling (single range) run the same accumulation and
// the same validation semantics, so a fix in either lands in both.
//
// Error contract: range violations are hard errors returned as -1 with a
// message on stderr — no silent clamping, no graceful degradation. The
// caller decides policy (batch tool: fatal; server: 400).
#pragma once

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

// HS_SIMD: vectorization hint for the hot mean-accumulation loops. Under
// OpenMP it becomes `omp simd`; elsewhere it compiles to nothing.
#ifdef _OPENMP
#define HS_SIMD _Pragma("omp simd")
#else
#define HS_SIMD
#endif

// Accumulates rows in [start, end) over each range into out (which must be
// zeroed by the caller), then scales by 1/count. Returns the number of
// accumulated tokens, or -1 on an invalid / out-of-bounds range.
inline int64_t hs_compute_masked_mean(
    const float * data,
    int n_tokens,
    int n_embd,
    const std::vector<std::pair<int, int>> & ranges,
    float * out
) {
    int64_t count = 0;
    for (auto [start, end] : ranges) {
        // All range violations are hard errors. The caller must produce
        // correct ranges — no silent clamping, no graceful degradation.
        if (start < 0 || end < 0 || start >= end) {
            fprintf(stderr, "Error: invalid masked-mean range [%d, %d)  -  negative or empty range\n", start, end);
            return -1;
        }
        if (end > n_tokens || start > n_tokens) {
            fprintf(stderr, "Error: masked-mean range [%d, %d) exceeds n_tokens=%d  -  "
                            "the assignment's token range does not fit this prompt's tokenization "
                            "(ranges are raw token indices, post-tokenization, BOS included per the "
                            "tokenizer; see tools/hs-extract-batch/README.md, Input Format)\n",
                    start, end, n_tokens);
            return -1;
        }

        for (int t = start; t < end; t++) {
            const float * row = data + (size_t)t * (size_t)n_embd;
            HS_SIMD
            for (int d = 0; d < n_embd; d++) {
                out[d] += row[d];
            }
        }
        count += (end - start);
    }

    if (count > 0) {
        float inv = 1.0f / (float)count;
        HS_SIMD
        for (int d = 0; d < n_embd; d++) {
            out[d] *= inv;
        }
    }
    return count;
}

// Single contiguous [start, end) span: a specialization of
// hs_compute_masked_mean with one range. Delegates for identical
// validation and accumulation semantics.
inline int hs_compute_single_range_mean(
    const float * data,
    int n_tokens,
    int n_embd,
    int start,
    int end,
    float * out
) {
    return (int)hs_compute_masked_mean(data, n_tokens, n_embd, {{start, end}}, out);
}
