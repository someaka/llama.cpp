// Internal header for hs-extract-batch: named size limits shared across
// translation units, the assignments.bin data model and readers, and the
// per-record sidecar write macro. Extracted verbatim from hs-extract-batch.cpp
// (pure code motion).
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// -- Named Constants (replaces magic numbers throughout) -------------------

static constexpr int32_t MIN_CTX_SIZE       = 512;     // minimum context size
static constexpr double   CHARS_PER_TOKEN   = 3.5;     // tokenizer estimate for auto-ctx
static constexpr int32_t MAX_PROMPTS        = 10000000; // max prompts in file
static constexpr int32_t MAX_N_EMBD         = 65536;   // max embedding dimension
static constexpr int32_t MAX_GROUPS         = 10000;   // max label groups
static constexpr int32_t MAX_LAYERS         = 10000;   // max layers in checkpoint
static constexpr int32_t MAX_MASKS          = 65536;   // max masks in checkpoint
static constexpr int32_t MAX_ASSIGNMENTS    = 100000;  // max assignments per prompt
static constexpr int32_t MAX_RANGES         = 100000;  // max ranges per assignment
static constexpr int32_t MAX_GROUP_NAME_LEN = 1000;    // max group name length
static constexpr int32_t ALL_GPU_LAYERS     = 99;      // load all layers on GPU
static constexpr int      PROGRESS_INTERVAL  = 100;     // prompts between progress reports
static constexpr int32_t MAX_BATCH_SIZE     = 256;     // max batch size for multi mode

// -- Argument Parsing ---------------------------------------------------

static constexpr int32_t ASSIGNMENTS_MAGIC = 0x43524431;  // "CRD1"

// Repeat-penalty window (last_n) for generation-mode sampling: tokens further
// back than this are not penalized. Fixed, not CLI-tunable; named instead of
// a bare literal at the call site.

// One (group, mask) assignment for a prompt.
struct Assignment {
    int32_t group_id;
    int32_t mask_id;
    int32_t mask_type;    // 0 = simple_skip, 1 = explicit_ranges
    int32_t skip;         // valid when mask_type == 0
    std::vector<std::pair<int,int>> ranges;  // valid when mask_type == 1
};

// Group name table from assignments.bin header.
struct GroupTable {
    int32_t n_groups = 0;
    std::vector<std::string> names;  // indexed by group_id
};

enum class AssignmentReadStatus {
    ok,
    eof,
    error,
};

struct AssignmentReadResult {
    AssignmentReadStatus status = AssignmentReadStatus::error;
    std::vector<Assignment> assignments;
};

// Wraps fwrite for per-record sidecar writes. On failure: print error, signal
// producer to stop, join the producer thread, remove the per-record temp file
// (same cleanup contract as STOP_PRODUCER_AND_JOIN — a leftover
// .records.bin.tmp would be picked up by a subsequent run's parser), and
// return 1 from the calling function. All captured state is passed
// explicitly; no implicit scope capture.
#define RECORDS_WRITE(ptr, size, count, fp, pfq_ref, thread_ref)               \
    do {                                                                        \
        if (fwrite((ptr), (size), (count), (fp)) != (size_t)(count)) {         \
            fprintf(stderr, "Error: per-record write failed at %s:%d\n",        \
                    __FILE__, __LINE__);                                        \
            { std::lock_guard<std::mutex> _lk((pfq_ref).mtx); (pfq_ref).producer_done = true; } \
            (pfq_ref).cv_space.notify_all();                                    \
            (thread_ref).join();                                                \
            if (!(records_temp_path).empty()) std::remove((records_temp_path).c_str()); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

/**
 * Read assignments.bin header: magic, n_prompts, n_embd, group name table.
 * Leaves file position at the start of per-prompt data.
 * Returns true on success.
 */
bool read_assignments_header(
    FILE* f, int32_t& n_prompts, int32_t& n_embd_expected, GroupTable& groups
);

/**
 * Read one prompt's assignments from assignments.bin (sequential read).
 * Must be called in prompt order, matching prompts.txt line order.
 * Returns explicit status so EOF, valid zero-assignment prompts, and truncated
 * records cannot be confused.
 */
AssignmentReadResult read_prompt_assignments(FILE* f);
