// Internal header for hs-extract-batch: masked-mean kernel entry points.
// The kernel lives in the shared tools/hs-extract-common/masked-mean.h so
// the server's skip_mean pooling and this tool accumulate identically;
// this header keeps the tool's historical names for its own TUs.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "masked-mean.h"  // tools/hs-extract-common/

// Historically #define aliases to the shared hs_* kernels; the macros leaked
// into every consumer TU. Typed inline wrappers keep the tool's historical
// names without macro pollution.
static inline int64_t compute_masked_mean(const float * data, int n_tokens, int n_embd,
                                          const std::vector<std::pair<int, int>> & ranges,
                                          float * out) {
    return hs_compute_masked_mean(data, n_tokens, n_embd, ranges, out);
}

static inline int compute_single_range_mean(const float * data, int n_tokens, int n_embd,
                                            int start, int end, float * out) {
    return hs_compute_single_range_mean(data, n_tokens, n_embd, start, end, out);
}
