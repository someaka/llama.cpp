// Internal header for hs-extract-batch: accumulator data structures and the
// flat-key encoding shared by the accumulation, checkpoint, output-writing,
// and self-test translation units.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

static constexpr int32_t OUTPUT_MAGIC      = 0x43524432;  // binary accumulator format v2

// Per (group, mask, layer): running sum and count for accumulating means.
// Flat key: (group_id << 32) | (mask_id << 16) | layer_idx for single hash lookup.
struct AccumulatedVector {
    std::vector<float> sum;  // size n_embd, initialized lazily on first access
    int count = 0;
};

// Flat accumulator map: single-level hash for better cache locality than nested maps.
using AccumulatorMap = std::unordered_map<uint64_t, AccumulatedVector>;

// Helper to construct flat key from components.
// Layout: bits 0-15 = layer_idx, bits 16-31 = mask_id, bits 32-63 = group_id
inline uint64_t make_accum_key(int32_t group_id, int32_t mask_id, int32_t layer_idx) {
    return ((uint64_t)(uint32_t)group_id << 32) | ((uint64_t)(uint16_t)mask_id << 16) | (uint64_t)(uint16_t)layer_idx;
}

// Helper to extract components from flat key.
inline void decode_accum_key(uint64_t key, int32_t& group_id, int32_t& mask_id, int32_t& layer_idx) {
    layer_idx = (int32_t)(key & 0xFFFF);
    mask_id   = (int32_t)((key >> 16) & 0xFFFF);
    group_id  = (int32_t)((key >> 32) & 0xFFFFFFFF);
}
