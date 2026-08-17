// assignments.bin readers for hs-extract-batch: header parsing and the
// per-prompt sequential assignment reader. Extracted verbatim from
// hs-extract-batch.cpp (pure code motion).

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "assignments-io.h"

/**
 * Read assignments.bin header: magic, n_prompts, n_embd, group name table.
 * Leaves file position at the start of per-prompt data.
 * Returns true on success.
 */
bool read_assignments_header(
    FILE* f, int32_t& n_prompts, int32_t& n_embd_expected, GroupTable& groups
) {
    int32_t magic = 0;
    if (fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != ASSIGNMENTS_MAGIC) {
        fprintf(stderr, "Error: invalid assignments.bin magic (got 0x%08X, expected 0x%08X)\n",
                magic, ASSIGNMENTS_MAGIC);
        return false;
    }
    if (fread(&n_prompts, sizeof(int32_t), 1, f) != 1) return false;
    if (fread(&n_embd_expected, sizeof(int32_t), 1, f) != 1) return false;

    // Validate n_prompts and n_embd to prevent UB on corrupt file
    if (n_prompts <= 0 || n_prompts > MAX_PROMPTS) {
        fprintf(stderr, "Error: n_prompts %d out of range [1, %d] - corrupt assignments.bin\n", n_prompts, MAX_PROMPTS);
        return false;
    }
    if (n_embd_expected < 0 || n_embd_expected > MAX_N_EMBD) {
        fprintf(stderr, "Error: n_embd %d out of range [0, %d] - corrupt assignments.bin\n", n_embd_expected, MAX_N_EMBD);
        return false;
    }

    if (fread(&groups.n_groups, sizeof(int32_t), 1, f) != 1) return false;

    // Validate n_groups to prevent OOM on corrupt file
    if (groups.n_groups < 0 || groups.n_groups > MAX_GROUPS) {
        fprintf(stderr, "Error: n_groups %d out of range [0, %d] - corrupt assignments.bin\n", groups.n_groups, MAX_GROUPS);
        return false;
    }
    groups.names.resize(groups.n_groups);
    for (int32_t i = 0; i < groups.n_groups; i++) {
        int32_t name_len = 0;
        if (fread(&name_len, sizeof(int32_t), 1, f) != 1) return false;
        // Validate name_len to prevent OOM on corrupt file
        if (name_len < 0 || name_len > MAX_GROUP_NAME_LEN) {
            fprintf(stderr, "Error: group name length %d out of range [0, %d] - corrupt assignments.bin\n", name_len, MAX_GROUP_NAME_LEN);
            return false;
        }
        std::vector<char> buf(name_len > 0 ? name_len : 1, 0);
        if (name_len > 0 && fread(buf.data(), 1, name_len, f) != (size_t)name_len) return false;
        groups.names[i] = std::string(buf.data(), name_len);
    }

    fprintf(stderr, "Assignments: %d prompts, %d groups, n_embd_expected=%d\n",
            n_prompts, groups.n_groups, n_embd_expected);
    return true;
}

/**

/** * Read one prompt's assignments from assignments.bin (sequential read).
 * Must be called in prompt order, matching prompts.txt line order.
 * Returns explicit status so EOF, valid zero-assignment prompts, and truncated
 * records cannot be confused.
 */
AssignmentReadResult read_prompt_assignments(FILE* f) {
    int32_t n_assignments = 0;
    if (fread(&n_assignments, sizeof(int32_t), 1, f) != 1) {
        if (feof(f)) return {AssignmentReadStatus::eof, {}};
        fprintf(stderr, "Error: failed to read n_assignments at offset %ld\n", ftell(f));
        return {AssignmentReadStatus::error, {}};
    }

    if (n_assignments < 0 || n_assignments > MAX_ASSIGNMENTS) {
        fprintf(stderr, "Error: n_assignments %d out of range [0, %d]\n", n_assignments, MAX_ASSIGNMENTS);
        return {AssignmentReadStatus::error, {}};
    }

    std::vector<Assignment> assignments(n_assignments);
    for (int32_t i = 0; i < n_assignments; i++) {
        Assignment& a = assignments[i];
        if (fread(&a.group_id, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
        if (fread(&a.mask_id, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
        // group_id/mask_id feed make_accum_key(), which truncates them to
        // 16 bits (and group_id to 32 bits). Out-of-range values would
        // silently alias into a different group/mask's statistics and
        // produce checkpoints the tool's own reader rejects on --resume.
        // Reject here, at the parse site, so every consumer (generation
        // path, comprehension path, checkpoint writer) is covered.
        if (a.group_id < 0 || a.group_id > 0xFFFF) {
            fprintf(stderr, "Error: group_id %d exceeds 16-bit range  -  corrupt assignments.bin\n", a.group_id);
            return {AssignmentReadStatus::error, {}};
        }
        if (a.mask_id < 0 || a.mask_id > 0xFFFF) {
            fprintf(stderr, "Error: mask_id %d exceeds 16-bit range  -  corrupt assignments.bin\n", a.mask_id);
            return {AssignmentReadStatus::error, {}};
        }
        if (fread(&a.mask_type, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};

        if (a.mask_type == 0) {
            if (fread(&a.skip, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
        } else if (a.mask_type == 1) {
            int32_t n_ranges = 0;
            if (fread(&n_ranges, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
            if (n_ranges < 0 || n_ranges > MAX_RANGES) {
                fprintf(stderr, "Error: n_ranges %d out of range [0, %d]\n", n_ranges, MAX_RANGES);
                return {AssignmentReadStatus::error, {}};
            }
            a.ranges.resize(n_ranges);
            for (int32_t r = 0; r < n_ranges; r++) {
                int32_t start = 0, end = 0;
                if (fread(&start, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
                if (fread(&end, sizeof(int32_t), 1, f) != 1) return {AssignmentReadStatus::error, {}};
                if (start < 0 || end < 0) {
                    fprintf(stderr, "Error: assignment range [%d, %d) has negative bound  -  corrupt assignments.bin\n", start, end);
                    return {AssignmentReadStatus::error, {}};
                }
                if (start > end) {
                    fprintf(stderr, "Error: assignment range [%d, %d) has start > end  -  corrupt assignments.bin\n", start, end);
                    return {AssignmentReadStatus::error, {}};
                }
                if (start == end) {
                    // compute_masked_mean treats start >= end as a hard error, so
                    // an empty range would abort the entire run deep in the kernel.
                    // Reject it here at read time with a clear diagnostic instead.
                    fprintf(stderr, "Error: assignment range [%d, %d) is empty (start == end)  -  "
                                    "degenerate token span in assignments.bin\n", start, end);
                    return {AssignmentReadStatus::error, {}};
                }
                a.ranges[r] = {start, end};
            }
        } else {
            fprintf(stderr, "Error: unknown mask_type %d\n", a.mask_type);
            return {AssignmentReadStatus::error, {}};
        }
    }
    return {AssignmentReadStatus::ok, std::move(assignments)};
}
