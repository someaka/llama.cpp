// Internal header for hs-extract-batch: masked-mean kernel prototypes.
// The kernel bodies live in hs-extract-batch.cpp; this header exists so the
// self-test translation unit can call them.
#pragma once

#include <utility>
#include <vector>

// Mean of hidden-state data over arbitrary [start, end) token ranges.
// Returns the token count included in the mean, or -1 on a hard range error.
int64_t compute_masked_mean(
    const float* data,
    int n_tokens,
    int n_embd,
    const std::vector<std::pair<int,int>>& ranges,
    float* out
);

// Mean over one contiguous [start, end) token range. Returns the token count,
// or -1 on a hard range error.
int compute_single_range_mean(
    const float* data,
    int n_tokens,
    int n_embd,
    int start,
    int end,
    float* out
);
