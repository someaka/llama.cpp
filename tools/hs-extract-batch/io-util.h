// Internal header for hs-extract-batch: durable/atomic I/O helpers
// (checkpoint read/write, atomic temp+fsync+rename, progress reporting).
// Checkpoint implementations currently live in hs-extract-batch.cpp.
#pragma once

#include <cstdint>

#include "hs-accum.h"

// Write checkpoint: version + n_iterated + accumulator state (binary
// accumulator format). The checkpoint file is output_path + ".checkpoint".
bool write_checkpoint(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd,
    int32_t n_iterated
);

// Read checkpoint: restore accumulator state and return n_iterated (skip
// count). Returns false if checkpoint doesn't exist or is corrupt.
bool read_checkpoint(
    const char* output_path,
    AccumulatorMap& accumulators,
    int32_t& n_iterated,
    int32_t expected_n_embd
);
