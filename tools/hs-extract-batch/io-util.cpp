// Durable/atomic I/O implementations for hs-extract-batch: batch output
// writer, checkpoint write/read, and fsync helpers (temp+fsync+rename
// finalization contract shared by all writers).


#include "io-util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "hs-accum.h"
#include "assignments-io.h"

/**
 * Write accumulator state to an open FILE* in binary accumulator format.
 * Shared core used by both write_batch_output() and write_checkpoint().
 * Returns false on write error.
 */
static bool _write_accumulator_to_file(
    const AccumulatorMap& accumulators,
    FILE* out,
    int32_t n_embd,
    bool write_sum = false
) {
    // Collect and sort flat keys for deterministic output.
    // Key layout: group_id (bits 32-63) | mask_id (bits 16-31) | layer_idx (bits 0-15)
    // Sorting the full 64-bit key yields: group_id ASC, mask_id ASC, layer_idx ASC.
    std::vector<uint64_t> flat_keys;
    flat_keys.reserve(accumulators.size());
    for (const auto& [key, _] : accumulators) flat_keys.push_back(key);
    std::sort(flat_keys.begin(), flat_keys.end());

    // Compute n_layers (max layer_idx + 1)
    int32_t max_layer = 0;
    for (uint64_t key : flat_keys) {
        int32_t group_id, mask_id, layer_idx;
        decode_accum_key(key, group_id, mask_id, layer_idx);
        if (layer_idx > max_layer) max_layer = layer_idx;
    }

    // Build unique (group_id, mask_id) pairs, each with its sorted layer indices.
    // Since flat_keys is sorted, sequential iteration naturally groups identical
    // (group, mask) pairs contiguously and layers arrive in ascending order.
    struct GroupMask {
        int32_t group_id;
        int32_t mask_id;
        std::vector<int32_t> layer_indices;
    };
    std::vector<GroupMask> gm_pairs;

    for (uint64_t key : flat_keys) {
        int32_t group_id, mask_id, layer_idx;
        decode_accum_key(key, group_id, mask_id, layer_idx);

        bool new_pair = gm_pairs.empty()
            || gm_pairs.back().group_id != group_id
            || gm_pairs.back().mask_id   != mask_id;
        if (new_pair) {
            gm_pairs.push_back({group_id, mask_id, {}});
        }
        gm_pairs.back().layer_indices.push_back(layer_idx);
    }

    // Count distinct groups (first element of each new group in gm_pairs)
    int32_t n_groups = 0;
    {
        int32_t prev_group = -1;
        for (const auto& gm : gm_pairs) {
            if (gm.group_id != prev_group) { n_groups++; prev_group = gm.group_id; }
        }
    }

    // Write header
    int32_t magic = OUTPUT_MAGIC;
    if (!checked_write(&magic, sizeof(int32_t), 1, out)) return false;
    if (!checked_write(&n_groups, sizeof(int32_t), 1, out)) return false;
    int32_t n_layers = max_layer + 1;
    if (!checked_write(&n_layers, sizeof(int32_t), 1, out)) return false;
    if (!checked_write(&n_embd, sizeof(int32_t), 1, out)) return false;

    // Write per-group data: iterate over groups, and within each group
    // over its (group, mask) pairs. Each mask block is emitted exactly once.
    std::vector<float> mean(n_embd);
    size_t gm_idx = 0;

    while (gm_idx < gm_pairs.size()) {
        int32_t group_id = gm_pairs[gm_idx].group_id;

        // Count how many mask pairs belong to this group
        int32_t n_masks = 0;
        size_t mask_start = gm_idx;
        while (gm_idx < gm_pairs.size() && gm_pairs[gm_idx].group_id == group_id) {
            n_masks++;
            gm_idx++;
        }

        if (!checked_write(&group_id, sizeof(int32_t), 1, out)) return false;
        if (!checked_write(&n_masks, sizeof(int32_t), 1, out)) return false;

        for (size_t mi = mask_start; mi < mask_start + (size_t)n_masks; mi++) {
            const auto& gm = gm_pairs[mi];
            if (!checked_write(&gm.mask_id, sizeof(int32_t), 1, out)) return false;

            int32_t n_layers_data = (int32_t)gm.layer_indices.size();
            if (!checked_write(&n_layers_data, sizeof(int32_t), 1, out)) return false;

            for (int32_t li : gm.layer_indices) {
                uint64_t layer_key = make_accum_key(group_id, gm.mask_id, li);
                const auto& av = accumulators.at(layer_key);
                if (!checked_write(&li, sizeof(int32_t), 1, out)) return false;
                if (!checked_write(&av.count, sizeof(int32_t), 1, out)) return false;

                if (write_sum) {
                    // Checkpoint format (v2+): write raw sum directly to avoid
                    // precision loss from mean=sum/count then sum=mean*count roundtrip.
                    if (av.count > 0 && !av.sum.empty()) {
                        if (!checked_write(av.sum.data(), sizeof(float), n_embd, out)) return false;
                    } else {
                        std::fill(mean.begin(), mean.end(), 0.0f);
                        if (!checked_write(mean.data(), sizeof(float), n_embd, out)) return false;
                    }
                } else {
                    // Output format: write mean = sum / count for downstream consumers.
                    if (av.count > 0 && !av.sum.empty()) {
                        float inv = 1.0f / (float)av.count;
                        for (int d = 0; d < n_embd; d++) mean[d] = av.sum[d] * inv;
                    } else {
                        std::fill(mean.begin(), mean.end(), 0.0f);
                    }
                    if (!checked_write(mean.data(), sizeof(float), n_embd, out)) return false;
                }
            }
        }
    }
    return true;
}

/**
 * Write accumulated means to output.bin (binary accumulator format).
 */
bool write_batch_output(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd
) {
    // Write to a temp file then atomically rename, so a crash/disk-full during
    // the final write can never leave a truncated output.bin in place (the
    // checkpoint code below uses the same pattern). A truncated final output
    // would be silently misread by the Python parser on the next run.
    std::string temp_path = std::string(output_path) + ".tmp";
    FilePtr out(fopen(temp_path.c_str(), "wb"));
    if (!out) {
        fprintf(stderr, "Error: cannot open output file %s\n", temp_path.c_str());
        return false;
    }
    bool ok = _write_accumulator_to_file(accumulators, out, n_embd);
    if (!ok) {
        out.reset();
        std::remove(temp_path.c_str());  // no orphaned .tmp on write failure
        return false;
    }
    if (!out.sync()) {  // flush + fsync: check for failures before rename
        fprintf(stderr, "Error: sync failed for %s (disk full or I/O error)\n", temp_path.c_str());
        out.reset();
        std::remove(temp_path.c_str());
        return false;
    }
    out.reset();  // close file before rename
    if (rename(temp_path.c_str(), output_path) != 0) {
        fprintf(stderr, "Error: cannot rename %s to %s\n", temp_path.c_str(), output_path);
        std::remove(temp_path.c_str());  // no orphaned .tmp on rename failure
        return false;
    }
    if (!fsync_parent_dir(output_path)) {
        return false;
    }
    fprintf(stderr, "Output: written to %s\n", output_path);
    return true;
}
// -- Checkpoint / Resume ------------------------------------------------

static constexpr int32_t CHECKPOINT_VERSION = 4;
static constexpr int32_t CHECKPOINT_VERSION_V3 = 3; // legacy: no content check
static constexpr int32_t CHECKPOINT_VERSION_V2 = 2; // legacy: no run fingerprint
static constexpr int32_t CHECKPOINT_VERSION_V1 = 1;  // legacy: stored mean, restored via sum=mean*count

// FNV-1a 64 of the line_number-th non-empty line (1-based), matching
// scan_prompts_file's line accounting. Returns false when the line does not
// exist or the file cannot be read.
static bool hash_prompt_line(const char* prompts_file, int32_t line_number, uint64_t& out) {
    std::ifstream fin(prompts_file);
    if (!fin) {
        fprintf(stderr, "Error: cannot open %s for the resume content check\n", prompts_file);
        return false;
    }
    std::string line;
    int32_t n = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (++n == line_number) {
            out = hs_fnv1a64(line);
            return true;
        }
    }
    if (fin.bad()) {
        fprintf(stderr, "Error: I/O error reading prompts file for the resume content check\n");
        return false;
    }
    fprintf(stderr, "Error: prompts file has %d non-empty lines but the checkpoint skip count is %d - "
                    "the prompts file shrank since the checkpoint\n", n, line_number);
    return false;
}

/**
 * Write checkpoint: version + n_iterated + accumulator state (binary accumulator format).
 * The checkpoint file is output_path + ".checkpoint".
 */
bool write_checkpoint(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd,
    int32_t n_iterated,
    const checkpoint_fingerprint& fp
) {
    std::string ckpt_path = std::string(output_path) + ".checkpoint";
    std::string temp_path = ckpt_path + ".tmp";

    // Write to temporary file first
    FilePtr f(fopen(temp_path.c_str(), "wb"));
    if (!f) {
        fprintf(stderr, "Error: cannot write checkpoint to %s\n", temp_path.c_str());
        return false;
    }
    if (!checked_write(&CHECKPOINT_VERSION, sizeof(int32_t), 1, f)) {
        f.reset();
        std::remove(temp_path.c_str());
        return false;
    }
    if (!checked_write(&n_iterated, sizeof(int32_t), 1, f)) {
        f.reset();
        std::remove(temp_path.c_str());
        return false;
    }
    // v3: run fingerprint so --resume can refuse checkpoints written under
    // different extraction settings (see checkpoint_fingerprint in io-util.h).
    {
        int32_t n_fp_layers = (int32_t)fp.layers.size();
        if (!checked_write(&n_fp_layers, sizeof(int32_t), 1, f) ||
            !checked_write(&fp.generate_mode, sizeof(bool), 1, f) ||
            !checked_write(&fp.generate_tokens, sizeof(int32_t), 1, f) ||
            !checked_write(&fp.token_skip, sizeof(int32_t), 1, f) ||
            (n_fp_layers > 0 &&
             !checked_write(fp.layers.data(), sizeof(int32_t), (size_t)n_fp_layers, f))) {
            f.reset();
            std::remove(temp_path.c_str());
            return false;
        }
    }
    // v4: prompts-file content identity (hash of the skip_count-th non-empty
    // line + expected prompt count) so --resume cannot silently continue a
    // checkpoint over different prompt content.
    if (!checked_write(&fp.content_fnv64, sizeof(uint64_t), 1, f) ||
        !checked_write(&fp.n_prompts, sizeof(int32_t), 1, f)) {
        f.reset();
        std::remove(temp_path.c_str());
        return false;
    }
    bool ok = _write_accumulator_to_file(accumulators, f, n_embd, /*write_sum=*/true);
    if (!ok) {
        f.reset();
        std::remove(temp_path.c_str());  // no orphaned .tmp on write failure
        return false;
    }
    if (!f.sync()) {  // flush + fsync: check for failures before rename
        fprintf(stderr, "Error: sync failed for checkpoint %s\n", temp_path.c_str());
        f.reset();
        std::remove(temp_path.c_str());
        return false;
    }
    f.reset();  // close file before rename
    // Atomic rename: temp -> final
    if (rename(temp_path.c_str(), ckpt_path.c_str()) != 0) {
        fprintf(stderr, "Error: cannot rename %s to %s\n", temp_path.c_str(), ckpt_path.c_str());
        std::remove(temp_path.c_str());  // no orphaned .tmp on rename failure
        return false;
    }
    if (!fsync_parent_dir(ckpt_path.c_str())) {
        return false;
    }
    fprintf(stderr, "Checkpoint saved: %d prompts -> %s\n", n_iterated, ckpt_path.c_str());
    return true;
}

/**
 * Read checkpoint: restore accumulator state and return n_iterated (skip count).
 * Returns false if checkpoint doesn't exist or is corrupt.
 */
bool read_checkpoint(
    const char* output_path,
    AccumulatorMap& accumulators,
    int32_t& n_iterated,
    int32_t expected_n_embd,
    const checkpoint_fingerprint& expected,
    const char* prompts_file
) {
    std::string ckpt_path = std::string(output_path) + ".checkpoint";
    FilePtr f(fopen(ckpt_path.c_str(), "rb"));
    if (!f) return false;  // no checkpoint -- fresh start

    // Read and validate checkpoint version
    int32_t version = 0;
    if (fread(&version, sizeof(int32_t), 1, f) != 1) return false;
    if (version != CHECKPOINT_VERSION && version != CHECKPOINT_VERSION_V3 &&
        version != CHECKPOINT_VERSION_V2 && version != CHECKPOINT_VERSION_V1) {
        fprintf(stderr, "Error: checkpoint version mismatch (got %d, expected %d) - incompatible checkpoint format\n",
                version, CHECKPOINT_VERSION);
        return false;
    }
    // v1 checkpoints stored mean (sum/count); v2+ stores raw sum (no precision loss).
    // The read loop below branches on version to restore sums correctly.
    const bool is_v1 = (version == CHECKPOINT_VERSION_V1);

    // Read n_iterated
    if (fread(&n_iterated, sizeof(int32_t), 1, f) != 1) return false;

    // Validate n_iterated to prevent negative loop counts
    if (n_iterated < 0) {
        fprintf(stderr, "Error: checkpoint contains invalid n_iterated=%d (would cause infinite loop)\n", n_iterated);
        return false;
    }

    // v3: run fingerprint. Must match the current invocation exactly; a
    // mismatch means the checkpoint was written under different extraction
    // settings and resuming would silently merge incompatible accumulators.
    if (version >= CHECKPOINT_VERSION_V3) {
        int32_t n_fp_layers = 0;
        bool generate_mode = false;
        int32_t generate_tokens = 0, token_skip = 0;
        if (fread(&n_fp_layers, sizeof(int32_t), 1, f) != 1) return false;
        if (fread(&generate_mode, sizeof(bool), 1, f) != 1) return false;
        if (fread(&generate_tokens, sizeof(int32_t), 1, f) != 1) return false;
        if (fread(&token_skip, sizeof(int32_t), 1, f) != 1) return false;
        if (n_fp_layers < 0 || n_fp_layers > MAX_LAYERS) {
            fprintf(stderr, "Error: checkpoint fingerprint layer count %d out of range [0, %d] - corrupt checkpoint\n",
                    n_fp_layers, MAX_LAYERS);
            return false;
        }
        std::vector<int32_t> fp_layers((size_t)n_fp_layers);
        if (n_fp_layers > 0 && fread(fp_layers.data(), sizeof(int32_t), (size_t)n_fp_layers, f) != (size_t)n_fp_layers) {
            return false;
        }
        if (generate_mode != expected.generate_mode || generate_tokens != expected.generate_tokens ||
            token_skip != expected.token_skip || fp_layers != expected.layers) {
            fprintf(stderr, "Error: checkpoint run fingerprint mismatch - checkpoint was written with different "
                            "extraction settings (mode=%s generate_tokens=%d token_skip=%d n_layers=%d) than the "
                            "current invocation (mode=%s generate_tokens=%d token_skip=%d n_layers=%zu). "
                            "Discard the checkpoint (remove %s) or rerun with the original settings.\n",
                    generate_mode ? "generate" : "comprehension", generate_tokens, token_skip, n_fp_layers,
                    expected.generate_mode ? "generate" : "comprehension", expected.generate_tokens,
                    expected.token_skip, expected.layers.size(), ckpt_path.c_str());
            return false;
        }
        if (version < CHECKPOINT_VERSION) {
            fprintf(stderr, "Warning: checkpoint predates content check\n");
        }
    } else {
        fprintf(stderr, "Warning: checkpoint version %d carries no run fingerprint - cannot verify that extraction "
                        "settings (mode/--generate/--token-skip/--layers) match this invocation. Proceeding; if the "
                        "checkpoint came from a different run configuration the output will silently mix settings.\n",
                version);
    }

    // v4: prompts-file content identity. The stored hash covers the
    // skip_count-th non-empty line -- the next prompt a resume would skip
    // over -- and the stored count binds the run to the full dataset size.
    if (version >= CHECKPOINT_VERSION) {
        uint64_t stored_fnv = 0;
        int32_t stored_n_prompts = 0;
        if (fread(&stored_fnv, sizeof(uint64_t), 1, f) != 1) return false;
        if (fread(&stored_n_prompts, sizeof(int32_t), 1, f) != 1) return false;
        if (stored_n_prompts != expected.n_prompts) {
            fprintf(stderr, "Error: checkpoint prompt count mismatch - checkpoint was written over %d prompts "
                            "but the current invocation has %d. Discard the checkpoint (remove %s) or restore "
                            "the original prompts file.\n",
                    stored_n_prompts, expected.n_prompts, ckpt_path.c_str());
            return false;
        }
        if (n_iterated > 0 && stored_fnv != 0 && prompts_file && prompts_file[0] != '\0') {
            uint64_t current_fnv = 0;
            if (!hash_prompt_line(prompts_file, n_iterated, current_fnv)) {
                return false;
            }
            if (current_fnv != stored_fnv) {
                fprintf(stderr, "Error: prompts file content mismatch at resume - line %d hashed to 0x%016llx but "
                                "the checkpoint (written over line %d) recorded 0x%016llx. The prompts file changed "
                                "since the checkpoint; the accumulator holds results for different content. "
                                "Discard the checkpoint (remove %s) or restore the original prompts file.\n",
                        n_iterated, (unsigned long long)current_fnv, n_iterated,
                        (unsigned long long)stored_fnv, ckpt_path.c_str());
                return false;
            }
        }
    }

    // Read accumulator state (binary accumulator format)
    int32_t magic = 0;
    if (fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != OUTPUT_MAGIC) {
        return false;
    }
    int32_t n_groups = 0, n_layers = 0, n_embd = 0;
    if (fread(&n_groups, sizeof(int32_t), 1, f) != 1) return false;
    if (fread(&n_layers, sizeof(int32_t), 1, f) != 1) return false;
    if (fread(&n_embd, sizeof(int32_t), 1, f) != 1) return false;

    // Validate checkpoint dimensions to prevent OOM from corrupt data
    if (n_groups < 0 || n_groups > MAX_GROUPS) {
        fprintf(stderr, "Error: checkpoint n_groups=%d out of range [0, %d] - corrupt checkpoint\n", n_groups, MAX_GROUPS);
        return false;
    }
    if (n_layers < 0 || n_layers > MAX_LAYERS) {
        fprintf(stderr, "Error: checkpoint n_layers=%d out of range [0, %d] - corrupt checkpoint\n", n_layers, MAX_LAYERS);
        return false;
    }

    // Validate n_embd matches the model being used
    if (n_embd != expected_n_embd) {
        fprintf(stderr, "Error: checkpoint n_embd=%d does not match model n_embd=%d - checkpoint is from a different model\n",
                n_embd, expected_n_embd);
        return false;
    }

    // All-or-nothing restore: parse into a local map and publish only after
    // the final record validates, so a corrupt record at offset N cannot
    // leave a partially-mutated caller map behind.
    AccumulatorMap restored;

    for (int32_t g = 0; g < n_groups; g++) {
        int32_t group_id = 0, n_masks = 0;
        if (fread(&group_id, sizeof(int32_t), 1, f) != 1) return false;
        if (fread(&n_masks, sizeof(int32_t), 1, f) != 1) return false;

        // Validate group_id fits in 16-bit range (matches the parse-site bounds check in read_prompt_assignments)
        if (group_id < 0 || group_id > 0xFFFF) {
            fprintf(stderr, "Error: checkpoint contains group_id=%d out of range [0, 65535] - corrupt checkpoint\n", group_id);
            return false;
        }

        // Validate n_masks to prevent unbounded loop on corrupt checkpoint
        if (n_masks < 0 || n_masks > MAX_MASKS) {
            fprintf(stderr, "Error: checkpoint contains n_masks=%d out of range [0, %d] - corrupt checkpoint\n", n_masks, MAX_MASKS);
            return false;
        }

        for (int32_t m = 0; m < n_masks; m++) {
            int32_t mask_id = 0, n_layers_data = 0;
            if (fread(&mask_id, sizeof(int32_t), 1, f) != 1) return false;
            if (fread(&n_layers_data, sizeof(int32_t), 1, f) != 1) return false;

            if (mask_id < 0 || mask_id > 0xFFFF) {
                fprintf(stderr, "Error: checkpoint contains mask_id=%d out of range [0, 65535] - corrupt checkpoint\n", mask_id);
                return false;
            }

            // Validate n_layers_data to prevent unbounded loop on corrupt checkpoint
            if (n_layers_data < 0 || n_layers_data > MAX_LAYERS) {
                fprintf(stderr, "Error: checkpoint contains n_layers_data=%d out of range [0, %d] - corrupt checkpoint\n", n_layers_data, MAX_LAYERS);
                return false;
            }

            for (int32_t l = 0; l < n_layers_data; l++) {
                int32_t layer_idx = 0, count = 0;
                if (fread(&layer_idx, sizeof(int32_t), 1, f) != 1) return false;
                if (fread(&count, sizeof(int32_t), 1, f) != 1) return false;

                // Validate layer_idx and count to prevent silent corruption
                if (layer_idx < 0 || layer_idx > 0xFFFF) {
                    fprintf(stderr, "Error: checkpoint contains layer_idx=%d out of range [0, 65535] - corrupt checkpoint\n", layer_idx);
                    return false;
                }
                if (count < 0) {
                    fprintf(stderr, "Error: checkpoint contains count=%d (negative) - corrupt checkpoint\n", count);
                    return false;
                }
                // Upper-bound count: a single (group,mask,layer) accumulator can
                // receive at most one increment per (prompt, assignment) pair, so
                // its count cannot exceed n_iterated prompts * MAX_ASSIGNMENTS.
                // An unbounded count here is later used as a DIVISOR (write: 1/count)
                // and a v1 MULTIPLIER (sum *= count), so a corrupt huge value would
                // silently produce garbage means. Use int64 to avoid overflow in the
                // bound computation.
                if ((int64_t)count > (int64_t)n_iterated * MAX_ASSIGNMENTS) {
                    fprintf(stderr, "Error: checkpoint contains count=%d exceeding max %lld (n_iterated=%d * MAX_ASSIGNMENTS=%d) - corrupt checkpoint\n",
                            count, (long long)((int64_t)n_iterated * MAX_ASSIGNMENTS), n_iterated, MAX_ASSIGNMENTS);
                    return false;
                }

                // Use flat key for single-level map lookup
                uint64_t key = make_accum_key(group_id, mask_id, layer_idx);
                auto& av = restored[key];
                av.count = count;
                av.sum.resize(n_embd);
                if (is_v1) {
                    // v1 format: stored mean = sum/count. Read mean, then multiply
                    // by count to restore sum. Introduces ~1 ULP error per dimension.
                    if (fread(av.sum.data(), sizeof(float), n_embd, f) != (size_t)n_embd) {
                        return false;
                    }
                    if (count > 0) {
                        for (int d = 0; d < n_embd; d++) av.sum[d] *= (float)count;
                    }
                } else {
                    // v2+ format: stores raw sum directly (no precision loss).
                    if (fread(av.sum.data(), sizeof(float), n_embd, f) != (size_t)n_embd) {
                        return false;
                    }
                }
            }
        }
    }

    fprintf(stderr, "Checkpoint restored: %d prompts already processed\n", n_iterated);
    accumulators = std::move(restored);
    return true;
}

