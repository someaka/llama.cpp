/*
 * CrimsonRed batch hidden-state extraction CLI.
 *
 * Three modes:
 *   Batch:     hs-extract-batch <model> <prompts.txt> [layers] <output.bin> --batch --assignments <file> [flags]
 *   Raw:       hs-extract-batch <model> <prompts.txt> [layers] <output.bin> --raw [flags]
 *   Self-test: hs-extract-batch --self-test
 *
 * Production mode is --batch: reads prompts.txt + assignments.bin, streams
 * one prompt at a time, computes masked means per assignment, accumulates
 * per group/mask/layer, writes output.bin with final means.
 *
 * --raw is a debug/parity mode that dumps per-prompt binary data.
 *
 * Flags:
 *   --mean           In --raw: output token means instead of full per-token output
 *   --token-skip N   Skip first N tokens when computing mean (default: 0)
 *
 * Batch-only flags:
 *   --assignments F  Path to assignments.bin (required)
 *   --ctx-size N     Override auto context sizing (default: auto)
 *   --checkpoint-every N  Write checkpoint every N prompts (default: 10000)
 *   --resume         Resume from last checkpoint
 */

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <iostream>
#include <thread>
#include <algorithm>
#include <cmath>
#include <utility>
#include <unordered_map>

// ── Argument Parsing ───────────────────────────────────────────────────

struct Args {
    const char* model_path = nullptr;
    const char* prompts_file = nullptr;
    std::string layers_str = "all";
    const char* output_path = nullptr;
    bool mean_mode = false;      // kept for --raw mean output
    int token_skip = 0;
    bool self_test = false;
    bool batch_mode = false;     // --batch: accumulate mode (production)
    bool raw_mode = false;       // --raw: per-prompt dump (debug/parity)
    const char* assignments_file = nullptr;   // --assignments: assignments.bin
    int ctx_size = 0;            // --ctx-size: override auto (0=auto)
    int checkpoint_every = 10000; // --checkpoint-every N
    bool resume = false;          // --resume: continue from checkpoint
};

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  Batch (production): %s <model> <prompts.txt> [layers] <output.bin> --batch --assignments <file> [flags]\n", prog);
    fprintf(stderr, "  Raw (debug):        %s <model> <prompts.txt> [layers] <output.bin> --raw [flags]\n", prog);
    fprintf(stderr, "  Self-test:          %s --self-test\n", prog);
    fprintf(stderr, "\nModes:\n");
    fprintf(stderr, "  --batch          Accumulate masked means per group/mask/layer → output.bin (production)\n");
    fprintf(stderr, "  --raw            Per-prompt dump for debugging and parity testing\n");
    fprintf(stderr, "  --self-test      Run synthetic tests on compute_masked_mean(), no model needed\n");
    fprintf(stderr, "\nFlags:\n");
    fprintf(stderr, "  --mean           In --raw mode: output token means instead of full per-token data\n");
    fprintf(stderr, "  --token-skip N   Skip first N tokens for mean computation (default: 0)\n");
    fprintf(stderr, "  --assignments F  Path to assignments.bin (required with --batch)\n");
    fprintf(stderr, "  --ctx-size N     Override auto context sizing (default: auto from prompts)\n");
    fprintf(stderr, "  --checkpoint-every N  Write checkpoint every N prompts (default: 10000)\n");
    fprintf(stderr, "  --resume         Resume from last checkpoint\n");
    fprintf(stderr, "\nLayers: comma-separated list (e.g. '0,5,10') or 'all' (default)\n");
    fprintf(stderr, "Output: file path (required)\n");
}

static Args parse_args(int argc, char** argv) {
    Args args;
    std::vector<const char*> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mean") {
            args.mean_mode = true;
        } else if (arg == "--token-skip") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --token-skip requires a value\n");
                exit(1);
            }
            args.token_skip = atoi(argv[++i]);
        } else if (arg == "--self-test") {
            args.self_test = true;
        } else if (arg == "--batch") {
            args.batch_mode = true;
        } else if (arg == "--raw") {
            args.raw_mode = true;
        } else if (arg == "--assignments") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --assignments requires a value\n");
                exit(1);
            }
            args.assignments_file = argv[++i];
        } else if (arg == "--ctx-size") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --ctx-size requires a value\n");
                exit(1);
            }
            args.ctx_size = atoi(argv[++i]);
        } else if (arg == "--checkpoint-every") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --checkpoint-every requires a value\n");
                exit(1);
            }
            args.checkpoint_every = atoi(argv[++i]);
        } else if (arg == "--resume") {
            args.resume = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            positional.push_back(argv[i]);
        }
    }

    // --self-test needs no model or prompts — skip positional validation
    if (args.self_test) {
        return args;
    }

    // --batch mode has its own positional layout: [model] [prompts] [layers] [output]
    if (args.batch_mode) {
        if (positional.empty()) {
            fprintf(stderr, "Error: --batch requires a model path\n");
            print_usage(argv[0]);
            exit(1);
        }
        args.model_path = positional[0];
        if (positional.size() >= 2) args.prompts_file = positional[1];
        if (positional.size() >= 3) args.layers_str = positional[2];
        if (positional.size() >= 4) args.output_path = positional[3];

        if (!args.prompts_file) {
            fprintf(stderr, "Error: --batch requires a prompts file\n");
            print_usage(argv[0]);
            exit(1);
        }
        if (!args.assignments_file) {
            fprintf(stderr, "Error: --batch requires --assignments <file>\n");
            print_usage(argv[0]);
            exit(1);
        }
        if (!args.output_path) {
            fprintf(stderr, "Error: --batch requires an output file path\n");
            print_usage(argv[0]);
            exit(1);
        }
        return args;
    }

    // --raw mode: [model] [prompts] [layers] [output]
    if (args.raw_mode) {
        if (positional.empty()) {
            fprintf(stderr, "Error: --raw requires a model path\n");
            print_usage(argv[0]);
            exit(1);
        }
        args.model_path = positional[0];
        if (positional.size() >= 2) args.prompts_file = positional[1];
        if (positional.size() >= 3) args.layers_str = positional[2];
        if (positional.size() >= 4) args.output_path = positional[3];

        if (!args.prompts_file) {
            fprintf(stderr, "Error: --raw requires a prompts file\n");
            print_usage(argv[0]);
            exit(1);
        }
        if (!args.output_path) {
            fprintf(stderr, "Error: --raw requires an output file path\n");
            print_usage(argv[0]);
            exit(1);
        }
        return args;
    }

    // No mode specified — show usage and exit
    fprintf(stderr, "Error: must specify --batch, --raw, or --self-test\n\n");
    print_usage(argv[0]);
    exit(1);
}

// ── Batch Mode Constants & Data Structures ─────────────────────────────

static constexpr int32_t ASSIGNMENTS_MAGIC = 0x43524431;  // "CRD1"
static constexpr int32_t OUTPUT_MAGIC      = 0x43524432;  // "CRD2"

// Per (group, mask, layer): running sum and count for accumulating means.
struct AccumulatedVector {
    std::vector<float> sum;  // size n_embd, initialized lazily on first access
    int count = 0;
};

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

// ── Layer Parsing ──────────────────────────────────────────────────────

static std::vector<int> parse_layers(const std::string& s, int n_layer) {
    std::vector<int> out;
    if (s == "all") {
        for (int i = 0; i < n_layer; i++) out.push_back(i);
        return out;
    }
    const char* p = s.c_str();
    while (*p) {
        out.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return out;
}

// ── Masked Mean Computation ────────────────────────────────────────────

/**
 * Compute mean of hidden state data over specified token ranges.
 *
 * Generalizes the single-contiguous-range mean (token_skip → n_tokens) to
 * arbitrary collections of [start, end) token index pairs. Used by both the
 * refactored one-shot/persistent mean mode (single range) and the future
 * batch-accumulate mode (arbitrary ranges for deflection masking).
 *
 * @param data      Pointer to hidden state data, shape (n_tokens, n_embd).
 * @param n_tokens  Number of tokens in the sequence.
 * @param n_embd    Hidden dimension size.
 * @param ranges    Vector of (start, end) token index pairs.
 * @param out       Output buffer, size n_embd. Must be zeroed by caller.
 * @return          Number of tokens included in the mean (0 = empty mask).
 */
static int compute_masked_mean(
    const float* data,
    int n_tokens,
    int n_embd,
    const std::vector<std::pair<int,int>>& ranges,
    float* out
) {
    int count = 0;
    for (auto [start, end] : ranges) {
        // Clamp to valid range [0, n_tokens]
        start = std::max(0, std::min(start, n_tokens));
        end   = std::max(0, std::min(end, n_tokens));
        if (start >= end) continue;

        for (int t = start; t < end; t++) {
            const float* row = data + t * n_embd;
            for (int d = 0; d < n_embd; d++) {
                out[d] += row[d];
            }
        }
        count += (end - start);
    }

    if (count > 0) {
        float inv = 1.0f / (float)count;
        for (int d = 0; d < n_embd; d++) {
            out[d] *= inv;
        }
    }
    return count;
}

// ── Prompt Processing ──────────────────────────────────────────────────

/**
 * Process a single prompt: decode, synchronize, extract hidden states, write to file.
 *
 * In full mode: writes n_tokens * hidden_size floats per layer.
 * In mean mode: writes hidden_size floats per layer (mean from token_skip to n_tokens).
 *
 * Skipped prompts (empty tokens, decode failure) write a marker block [idx, 0, 0].
 *
 * Returns true if processing should continue, false if a fatal error occurred
 * (e.g. disk full) and the caller should stop.
 */
static bool process_prompt(
    llama_context* ctx,
    const std::vector<llama_token>& tokens,
    const std::vector<int32_t>& target_layers,
    int32_t n_embd,
    size_t prompt_idx,
    FILE* out,
    bool mean_mode,
    int token_skip
) {
    int n_tokens = (int)tokens.size();

    // Empty tokenization — write marker block
    if (n_tokens <= 0) {
        fprintf(stderr, "Warning: prompt %zu tokenized to empty, skipping\n", prompt_idx);
        int32_t empty[3] = {(int32_t)prompt_idx, 0, 0};
        fwrite(empty, sizeof(int32_t), 3, out);
        return true;
    }

    // Clear KV cache before processing this prompt
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // Create batch — seq_id=0 (KV cache is cleared per prompt, so no collision)
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    // Decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "Warning: decode failed for prompt %zu (ret=%d), skipping\n", prompt_idx, ret);
        llama_batch_free(batch);
        int32_t empty[3] = {(int32_t)prompt_idx, 0, 0};
        fwrite(empty, sizeof(int32_t), 3, out);
        return true;
    }

    // CRITICAL: synchronize before reading hidden states.
    // llama_decode() submits the compute graph asynchronously on CUDA.
    // Without sync, llama_get_hidden_state() can read a partially-written
    // tensor, triggering GGML_ASSERT (tensor read out of bounds).
    llama_synchronize(ctx);

    // Extract hidden state metadata
    int n_hidden_tokens = llama_get_hidden_state_n_tokens(ctx);

    // Write per-prompt header: [prompt_idx][n_tokens][n_layers][layer_indices...]
    std::vector<int32_t> header(3 + target_layers.size());
    header[0] = (int32_t)prompt_idx;
    header[1] = n_hidden_tokens;
    header[2] = (int32_t)target_layers.size();
    for (size_t i = 0; i < target_layers.size(); i++) {
        header[3 + i] = target_layers[i];
    }

    if (fwrite(header.data(), sizeof(int32_t), 3 + target_layers.size(), out) != 3 + target_layers.size()) {
        fprintf(stderr, "Error: failed to write header for prompt %zu\n", prompt_idx);
        llama_batch_free(batch);
        return false;  // disk full or I/O error — stop
    }

    // Write hidden state data for each layer
    bool write_ok = true;
    for (int32_t layer : target_layers) {
        float* data = llama_get_hidden_state(ctx, layer);
        if (!data) {
            fprintf(stderr, "Error: failed to get hidden state for layer %d\n", layer);
            write_ok = false;
            break;
        }

        if (mean_mode) {
            // Compute mean over [token_skip, n_hidden_tokens) using compute_masked_mean.
            // This is the single-contiguous-range case; arbitrary token ranges
            // are used by batch-accumulate mode (Phase 2+). Clamping and the
            // all-tokens-skipped (zero output) case are handled inside the function.
            std::vector<float> mean(n_embd, 0.0f);
            std::vector<std::pair<int,int>> ranges;
            ranges.push_back({token_skip, n_hidden_tokens});
            compute_masked_mean(data, n_hidden_tokens, n_embd, ranges, mean.data());

            if (fwrite(mean.data(), sizeof(float), n_embd, out) != (size_t)n_embd) {
                fprintf(stderr, "Error: failed to write mean data for layer %d\n", layer);
                write_ok = false;
                break;
            }
        } else {
            // Full mode: write all tokens × hidden_size
            size_t data_size = n_hidden_tokens * n_embd;
            if (fwrite(data, sizeof(float), data_size, out) != data_size) {
                fprintf(stderr, "Error: failed to write hidden state data for layer %d\n", layer);
                write_ok = false;
                break;
            }
        }
    }

    llama_batch_free(batch);

    if (!write_ok) {
        return false;  // disk full or I/O error — stop
    }

    return true;
}

// ── Tokenization ───────────────────────────────────────────────────────

/**
 * Tokenize a prompt. Returns token vector (empty on failure).
 */
static std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n <= 0) return {};
    std::vector<llama_token> toks(n);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), (int)toks.size(), true, true);
    if (n_tokens > 0) {
        toks.resize(n_tokens);
    } else {
        toks.clear();
    }
    return toks;
}

// ── Batch Mode Helpers ─────────────────────────────────────────────────

/**
 * Pre-scan prompts.txt to estimate max token count and auto-size n_ctx.
 * Uses chars / 3.5 as token estimate. Returns max(estimated_max + 64, 512).
 */
static int32_t auto_size_ctx(const char* prompts_file) {
    std::ifstream fin(prompts_file);
    if (!fin) {
        fprintf(stderr, "Warning: cannot open %s for ctx pre-scan, defaulting to 512\n", prompts_file);
        return 512;
    }

    size_t max_len = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.size() > max_len) max_len = line.size();
    }

    int32_t estimated = (int32_t)(max_len / 3.5) + 64;
    int32_t n_ctx = estimated > 512 ? estimated : 512;
    fprintf(stderr, "Auto-ctx: max line %zu chars → estimated %d tokens → n_ctx=%d\n",
            max_len, (int)(max_len / 3.5), n_ctx);
    return n_ctx;
}

/**
 * Read assignments.bin header: magic, n_prompts, n_embd, group name table.
 * Leaves file position at the start of per-prompt data.
 * Returns true on success.
 */
static bool read_assignments_header(
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

    if (fread(&groups.n_groups, sizeof(int32_t), 1, f) != 1) return false;
    groups.names.resize(groups.n_groups);
    for (int32_t i = 0; i < groups.n_groups; i++) {
        int32_t name_len = 0;
        if (fread(&name_len, sizeof(int32_t), 1, f) != 1) return false;
        std::vector<char> buf(name_len > 0 ? name_len : 1, 0);
        if (name_len > 0 && fread(buf.data(), 1, name_len, f) != (size_t)name_len) return false;
        groups.names[i] = std::string(buf.data(), name_len);
    }

    fprintf(stderr, "Assignments: %d prompts, %d groups, n_embd_expected=%d\n",
            n_prompts, groups.n_groups, n_embd_expected);
    return true;
}

/**
 * Read one prompt's assignments from assignments.bin (sequential read).
 * Must be called in prompt order, matching prompts.txt line order.
 * Returns empty vector on error (caller should check feof for expected EOF).
 */
static std::vector<Assignment> read_prompt_assignments(FILE* f) {
    int32_t n_assignments = 0;
    if (fread(&n_assignments, sizeof(int32_t), 1, f) != 1) {
        if (feof(f)) return {};  // expected EOF
        fprintf(stderr, "Error: failed to read n_assignments\n");
        return {};
    }

    std::vector<Assignment> assignments(n_assignments);
    for (int32_t i = 0; i < n_assignments; i++) {
        Assignment& a = assignments[i];
        if (fread(&a.group_id, sizeof(int32_t), 1, f) != 1) return {};
        if (fread(&a.mask_id, sizeof(int32_t), 1, f) != 1) return {};
        if (fread(&a.mask_type, sizeof(int32_t), 1, f) != 1) return {};

        if (a.mask_type == 0) {
            if (fread(&a.skip, sizeof(int32_t), 1, f) != 1) return {};
        } else if (a.mask_type == 1) {
            int32_t n_ranges = 0;
            if (fread(&n_ranges, sizeof(int32_t), 1, f) != 1) return {};
            a.ranges.resize(n_ranges);
            for (int32_t r = 0; r < n_ranges; r++) {
                int32_t start = 0, end = 0;
                if (fread(&start, sizeof(int32_t), 1, f) != 1) return {};
                if (fread(&end, sizeof(int32_t), 1, f) != 1) return {};
                a.ranges[r] = {start, end};
            }
        } else {
            fprintf(stderr, "Error: unknown mask_type %d\n", a.mask_type);
            return {};
        }
    }
    return assignments;
}

// ── Raw Mode (debug/parity) ────────────────────────────────────────────

/**
 * Raw mode: per-prompt binary dump for debugging and parity testing.
 *
 * Streams prompts.txt, decodes each, writes per-prompt hidden state data.
 * Same output format as the old one-shot mode:
 *   [n_prompts: int32]
 *   per prompt: [prompt_idx][n_tokens][n_layers][layer_indices][data]
 *
 * With --mean: writes mean vectors (token_skip→n_tokens) per layer.
 * Without --mean: writes full per-token data (n_tokens × hidden_size per layer).
 */
static int run_raw(const Args& args) {
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(args.model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model %s\n", args.model_path);
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_layers = llama_model_n_layer(model);

    fprintf(stderr, "Model loaded: n_ctx_train=%d, n_embd=%d, n_layers=%d\n",
            n_ctx_train, n_embd, n_layers);

    // Resolve layers
    auto raw_layers = parse_layers(args.layers_str, n_layers);
    std::vector<int32_t> target_layers;
    for (int l : raw_layers) {
        if (l < 0) l = n_layers + l;
        if (l >= 0 && l < n_layers) {
            target_layers.push_back(l);
        } else {
            fprintf(stderr, "Warning: layer %d out of range [0, %d), skipping\n", l, n_layers);
        }
    }

    // Auto-size context or use --ctx-size
    int32_t n_ctx;
    if (args.ctx_size > 0) {
        n_ctx = args.ctx_size;
    } else {
        n_ctx = auto_size_ctx(args.prompts_file);
    }
    if (n_ctx > n_ctx_train) n_ctx = n_ctx_train;
    fprintf(stderr, "Using n_ctx=%d\n", n_ctx);

    // Create context FIRST (Qwen3.5 Gated Delta Net OOM fix)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = 512;
    cparams.extract_hidden_states = true;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Open output file
    FILE* out = fopen(args.output_path, "wb");
    if (!out) {
        fprintf(stderr, "Error: cannot open output file %s\n", args.output_path);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    fprintf(stderr, "Writing output to %s\n", args.output_path);

    // Stream prompts.txt — count lines for header first
    // Two-pass: first count lines, then process
    int32_t n_prompts_total = 0;
    {
        std::ifstream count_fin(args.prompts_file);
        std::string line;
        while (std::getline(count_fin, line)) {
            if (!line.empty()) n_prompts_total++;
        }
    }

    // Global header
    fwrite(&n_prompts_total, sizeof(int32_t), 1, out);

    // Process prompts (streaming)
    std::ifstream fin(args.prompts_file);
    auto start_time = std::chrono::steady_clock::now();
    int prompt_idx = 0;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        auto tokens = tokenize(vocab, line);

        if (!process_prompt(ctx, tokens, target_layers, n_embd, prompt_idx, out,
                            args.mean_mode, args.token_skip)) {
            break;  // fatal I/O error
        }

        prompt_idx++;

        if (prompt_idx % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            float pps = prompt_idx * 1000.0f / (elapsed > 0 ? elapsed : 1);
            fprintf(stderr, "Processed %d/%d prompts (%.2f prompts/sec)\n",
                    prompt_idx, n_prompts_total, pps);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    fprintf(stderr, "Done. Processed %d prompts in %.2f seconds (%.2f prompts/sec)\n",
            prompt_idx, total_ms / 1000.0,
            prompt_idx * 1000.0 / (total_ms > 0 ? total_ms : 1));

    fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ── Batch-Accumulate Output Writer ─────────────────────────────────────

/**
 * Write accumulator state to an open FILE* in CRD2 format.
 * Shared core used by both write_batch_output() and write_checkpoint().
 * Returns false on write error.
 */
static bool _write_accumulator_to_file(
    const std::unordered_map<int, std::unordered_map<int, AccumulatedVector>>& accumulators,
    FILE* out,
    int32_t n_embd
) {
    // Collect and sort keys for deterministic output.
    std::vector<int> keys;
    keys.reserve(accumulators.size());
    for (const auto& [key, _] : accumulators) keys.push_back(key);
    std::sort(keys.begin(), keys.end());

    // Compute n_layers (max layer_idx + 1)
    int32_t max_layer = 0;
    for (const auto& [_, layer_map] : accumulators) {
        for (const auto& [layer_idx, _] : layer_map) {
            if (layer_idx > max_layer) max_layer = layer_idx;
        }
    }

    // Count distinct groups
    int32_t n_groups = 0;
    int prev_group = -1;
    for (int key : keys) {
        int gid = key >> 16;
        if (gid != prev_group) { n_groups++; prev_group = gid; }
    }

    // Write header
    int32_t magic = OUTPUT_MAGIC;
    fwrite(&magic, sizeof(int32_t), 1, out);
    fwrite(&n_groups, sizeof(int32_t), 1, out);
    int32_t n_layers = max_layer + 1;
    fwrite(&n_layers, sizeof(int32_t), 1, out);
    fwrite(&n_embd, sizeof(int32_t), 1, out);

    // Write per-group data
    prev_group = -1;
    std::vector<float> mean(n_embd);

    for (int key : keys) {
        int group_id = key >> 16;
        int mask_id = key & 0xFFFF;

        if (group_id != prev_group) {
            prev_group = group_id;
            int32_t n_masks = 0;
            for (int k : keys) {
                if ((k >> 16) == group_id) n_masks++;
            }
            fwrite(&group_id, sizeof(int32_t), 1, out);
            fwrite(&n_masks, sizeof(int32_t), 1, out);
        }

        fwrite(&mask_id, sizeof(int32_t), 1, out);

        const auto& layer_map = accumulators.at(key);
        int32_t n_layers_data = (int32_t)layer_map.size();
        fwrite(&n_layers_data, sizeof(int32_t), 1, out);

        std::vector<int> layer_indices;
        layer_indices.reserve(layer_map.size());
        for (const auto& [li, _] : layer_map) layer_indices.push_back(li);
        std::sort(layer_indices.begin(), layer_indices.end());

        for (int li : layer_indices) {
            const auto& av = layer_map.at(li);
            fwrite(&li, sizeof(int32_t), 1, out);
            fwrite(&av.count, sizeof(int32_t), 1, out);

            if (av.count > 0 && !av.sum.empty()) {
                float inv = 1.0f / (float)av.count;
                for (int d = 0; d < n_embd; d++) mean[d] = av.sum[d] * inv;
            } else {
                std::fill(mean.begin(), mean.end(), 0.0f);
            }
            if (fwrite(mean.data(), sizeof(float), n_embd, out) != (size_t)n_embd) {
                fprintf(stderr, "Error: failed to write mean data for group %d mask %d layer %d\n",
                        group_id, mask_id, li);
                return false;
            }
        }
    }
    return true;
}

/**
 * Write accumulated means to output.bin (CRD2 format).
 */
static bool write_batch_output(
    const std::unordered_map<int, std::unordered_map<int, AccumulatedVector>>& accumulators,
    const char* output_path,
    int32_t n_embd
) {
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "Error: cannot open output file %s\n", output_path);
        return false;
    }
    bool ok = _write_accumulator_to_file(accumulators, out, n_embd);
    fclose(out);
    if (ok) {
        fprintf(stderr, "Output: written to %s\n", output_path);
    }
    return ok;
}

// ── Checkpoint / Resume ────────────────────────────────────────────────

/**
 * Write checkpoint: n_iterated + accumulator state (CRD2 format).
 * The checkpoint file is output_path + ".checkpoint".
 */
static bool write_checkpoint(
    const std::unordered_map<int, std::unordered_map<int, AccumulatedVector>>& accumulators,
    const char* output_path,
    int32_t n_embd,
    int32_t n_iterated
) {
    std::string ckpt_path = std::string(output_path) + ".checkpoint";
    FILE* f = fopen(ckpt_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Warning: cannot write checkpoint to %s\n", ckpt_path.c_str());
        return false;
    }
    fwrite(&n_iterated, sizeof(int32_t), 1, f);
    bool ok = _write_accumulator_to_file(accumulators, f, n_embd);
    fclose(f);
    if (ok) {
        fprintf(stderr, "Checkpoint saved: %d prompts → %s\n", n_iterated, ckpt_path.c_str());
    }
    return ok;
}

/**
 * Read checkpoint: restore accumulator state and return n_iterated (skip count).
 * Returns false if checkpoint doesn't exist or is corrupt.
 */
static bool read_checkpoint(
    const char* output_path,
    std::unordered_map<int, std::unordered_map<int, AccumulatedVector>>& accumulators,
    int32_t& n_iterated
) {
    std::string ckpt_path = std::string(output_path) + ".checkpoint";
    FILE* f = fopen(ckpt_path.c_str(), "rb");
    if (!f) return false;  // no checkpoint — fresh start

    // Read n_iterated
    if (fread(&n_iterated, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }

    // Read accumulator state (CRD2 format)
    int32_t magic = 0;
    if (fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != OUTPUT_MAGIC) {
        fclose(f);
        return false;
    }
    int32_t n_groups = 0, n_layers = 0, n_embd = 0;
    if (fread(&n_groups, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
    if (fread(&n_layers, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
    if (fread(&n_embd, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }

    for (int32_t g = 0; g < n_groups; g++) {
        int32_t group_id = 0, n_masks = 0;
        if (fread(&group_id, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
        if (fread(&n_masks, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }

        for (int32_t m = 0; m < n_masks; m++) {
            int32_t mask_id = 0, n_layers_data = 0;
            if (fread(&mask_id, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
            if (fread(&n_layers_data, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }

            int key = (group_id << 16) | mask_id;

            for (int32_t l = 0; l < n_layers_data; l++) {
                int32_t layer_idx = 0, count = 0;
                if (fread(&layer_idx, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
                if (fread(&count, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }

                auto& av = accumulators[key][layer_idx];
                av.count = count;
                av.sum.resize(n_embd);
                // Read mean values, then multiply by count to restore sums
                if (fread(av.sum.data(), sizeof(float), n_embd, f) != (size_t)n_embd) {
                    fclose(f);
                    return false;
                }
                if (count > 0) {
                    for (int d = 0; d < n_embd; d++) av.sum[d] *= (float)count;
                }
            }
        }
    }

    fclose(f);
    fprintf(stderr, "Checkpoint restored: %d prompts already processed\n", n_iterated);
    return true;
}

// ── Batch-Accumulate Mode ──────────────────────────────────────────────

/**
 * Batch-accumulate mode: the production extraction path.
 *
 * Reads prompts.txt (streamed one line at a time) and assignments.bin (sequential).
 * For each prompt: tokenize → decode → compute masked means per assignment → accumulate.
 * At the end: write output.bin with per-group/mask/layer mean vectors.
 *
 * Memory is O(groups × masks × layers × n_embd), independent of prompt count.
 */
static int run_batch(const Args& args) {
    // ── Setup ──
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(args.model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model %s\n", args.model_path);
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_layers = llama_model_n_layer(model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);

    fprintf(stderr, "Model loaded: n_ctx_train=%d, n_embd=%d, n_layers=%d\n",
            n_ctx_train, n_embd, n_layers);

    // Resolve layers
    auto raw_layers = parse_layers(args.layers_str, n_layers);
    std::vector<int32_t> target_layers;
    for (int l : raw_layers) {
        if (l < 0) l = n_layers + l;
        if (l >= 0 && l < n_layers) {
            target_layers.push_back(l);
        } else {
            fprintf(stderr, "Warning: layer %d out of range [0, %d), skipping\n", l, n_layers);
        }
    }
    if (target_layers.empty()) {
        fprintf(stderr, "Error: no valid target layers\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Auto-size context or use --ctx-size
    int32_t n_ctx;
    if (args.ctx_size > 0) {
        n_ctx = args.ctx_size;
    } else {
        n_ctx = auto_size_ctx(args.prompts_file);
    }
    if (n_ctx > n_ctx_train) n_ctx = n_ctx_train;
    fprintf(stderr, "Using n_ctx=%d\n", n_ctx);

    // Create context FIRST (Qwen3.5 Gated Delta Net OOM fix — Pitfall #23)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = 512;
    cparams.extract_hidden_states = true;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ── Open input files ──
    std::ifstream prompts_fin(args.prompts_file);
    if (!prompts_fin) {
        fprintf(stderr, "Error: cannot open prompts file %s\n", args.prompts_file);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    FILE* assign_fin = fopen(args.assignments_file, "rb");
    if (!assign_fin) {
        fprintf(stderr, "Error: cannot open assignments file %s\n", args.assignments_file);
        prompts_fin.close();
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    // Read assignments header
    int32_t n_prompts_expected = 0;
    int32_t n_embd_expected = 0;
    GroupTable groups;
    if (!read_assignments_header(assign_fin, n_prompts_expected, n_embd_expected, groups)) {
        fclose(assign_fin); prompts_fin.close();
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    // Validate n_embd if specified
    if (n_embd_expected > 0 && n_embd_expected != n_embd) {
        fprintf(stderr, "Error: assignments.bin expects n_embd=%d but model has n_embd=%d\n",
                n_embd_expected, n_embd);
        fclose(assign_fin); prompts_fin.close();
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    // ── Hot loop ──
    // Accumulator: key = (group_id << 16) | mask_id → layer_idx → AccumulatedVector
    std::unordered_map<int, std::unordered_map<int, AccumulatedVector>> accumulators;
    std::vector<float> mean_buf(n_embd, 0.0f);  // reused buffer for masked mean

    // Resume from checkpoint if --resume is set
    int skip_count = 0;
    if (args.resume) {
        read_checkpoint(args.output_path, accumulators, skip_count);
    }

    int prompt_idx = 0;
    int n_processed = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::string line;
    while (std::getline(prompts_fin, line)) {
        // Read assignments for this prompt (sequential, must match prompts.txt order)
        auto assignments = read_prompt_assignments(assign_fin);
        if (assignments.empty() && feof(assign_fin)) {
            fprintf(stderr, "Warning: assignments.bin exhausted at prompt %d/%d\n",
                    prompt_idx, n_prompts_expected);
            break;
        }

        // Skip already-processed prompts when resuming from checkpoint
        if (prompt_idx < skip_count) {
            prompt_idx++;
            continue;
        }

        // Tokenize
        auto tokens = tokenize(vocab, line);

        if (tokens.empty()) {
            fprintf(stderr, "Warning: prompt %d tokenized to empty, skipping\n", prompt_idx);
            prompt_idx++;
            continue;
        }

        // Truncate if exceeds context
        if ((int)tokens.size() > n_ctx) {
            fprintf(stderr, "Warning: prompt %d has %zu tokens, truncating to %d\n",
                    prompt_idx, tokens.size(), n_ctx);
            tokens.resize(n_ctx);
        }

        // Clear KV cache
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) llama_memory_clear(mem, true);

        // Create batch
        int n_tokens = (int)tokens.size();
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }
        batch.n_tokens = n_tokens;

        // Decode
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "Warning: decode failed for prompt %d (ret=%d), skipping\n", prompt_idx, ret);
            llama_batch_free(batch);
            prompt_idx++;
            continue;
        }

        // CRITICAL: synchronize before reading hidden states (CUDA async race — Pitfall #17)
        llama_synchronize(ctx);

        int n_hidden = llama_get_hidden_state_n_tokens(ctx);

        // For each assignment: compute masked mean per layer and accumulate
        for (const auto& assign : assignments) {
            std::vector<std::pair<int,int>> ranges;
            if (assign.mask_type == 0) {
                ranges.push_back({assign.skip, n_hidden});
            } else {
                ranges = assign.ranges;
            }

            int key = (assign.group_id << 16) | assign.mask_id;

            for (int32_t layer_idx : target_layers) {
                float* data = llama_get_hidden_state(ctx, layer_idx);
                if (!data) {
                    fprintf(stderr, "Error: null hidden state for layer %d\n", layer_idx);
                    continue;
                }

                std::fill(mean_buf.begin(), mean_buf.end(), 0.0f);
                compute_masked_mean(data, n_hidden, n_embd, ranges, mean_buf.data());

                auto& av = accumulators[key][layer_idx];
                if (av.sum.empty()) av.sum.resize(n_embd, 0.0f);
                for (int d = 0; d < n_embd; d++) av.sum[d] += mean_buf[d];
                av.count++;
            }
        }

        llama_batch_free(batch);
        prompt_idx++;
        n_processed++;

        // Periodic checkpoint
        if (args.checkpoint_every > 0 && n_processed % args.checkpoint_every == 0) {
            write_checkpoint(accumulators, args.output_path, n_embd, prompt_idx);
        }

        // Progress reporting
        if (n_processed % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            float pps = n_processed * 1000.0f / (elapsed > 0 ? elapsed : 1);
            fprintf(stderr, "Processed %d/%d prompts (%.2f prompts/sec)\n",
                    n_processed, n_prompts_expected, pps);
        }
    }

    // ── Finalize ──
    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    fprintf(stderr, "Done. Processed %d prompts in %.2f seconds (%.2f prompts/sec)\n",
            n_processed, total_ms / 1000.0,
            n_processed * 1000.0 / (total_ms > 0 ? total_ms : 1));

    // Write output
    if (!write_batch_output(accumulators, args.output_path, n_embd)) {
        fclose(assign_fin); prompts_fin.close();
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    // Delete checkpoint on successful completion
    std::string ckpt_path = std::string(args.output_path) + ".checkpoint";
    remove(ckpt_path.c_str());

    // Cleanup
    fclose(assign_fin);
    prompts_fin.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}

// ── Self-Test Mode ─────────────────────────────────────────────────────

/**
 * Run synthetic known-value tests on compute_masked_mean() with no model loaded.
 * Exits 0 on pass, 1 on fail.
 */
static int run_self_test() {
    fprintf(stderr, "Running compute_masked_mean self-tests...\n\n");

    int passed = 0;
    const int total = 5;
    bool all_ok = true;

    // All tests use data layout: 3 tokens × 2 dims, row-major
    // data[0..5] = {1, 2, 3, 4, 5, 6}
    //   token0 = [1, 2], token1 = [3, 4], token2 = [5, 6]
    float data1[6] = {1, 2, 3, 4, 5, 6};

    // Test 1: single contiguous range = mean over all tokens
    {
        std::vector<std::pair<int,int>> ranges = {{0, 3}};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (1+3+5)/3 = 3.0   dim1: (2+4+6)/3 = 4.0
        bool ok = (count == 3)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 1 (single contiguous range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 2: skip first token
    {
        std::vector<std::pair<int,int>> ranges = {{1, 3}};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (3+5)/2 = 4.0   dim1: (4+6)/2 = 5.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 4.0f) < 1e-6f)
               && (std::abs(out[1] - 5.0f) < 1e-6f);
        fprintf(stderr, "  Test 2 (skip first token): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 3: non-contiguous ranges (deflection pattern: tokens 0 and 2, skip 1)
    {
        std::vector<std::pair<int,int>> ranges = {{0, 1}, {2, 3}};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (1+5)/2 = 3.0   dim1: (2+6)/2 = 4.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 3 (non-contiguous ranges): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 4: empty mask (zero ranges)
    {
        std::vector<std::pair<int,int>> ranges = {};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        // count should be 0, out stays zeroed
        bool ok = (count == 0)
               && (std::abs(out[0]) < 1e-6f)
               && (std::abs(out[1]) < 1e-6f);
        fprintf(stderr, "  Test 4 (empty mask): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 5: clamping (end > n_tokens)
    {
        std::vector<std::pair<int,int>> ranges = {{0, 100}};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        // end clamped to 3 → same result as test 1
        bool ok = (count == 3)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 5 (clamping end > n_tokens): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    fprintf(stderr, "\n%d/%d tests passed\n", passed, total);

    if (all_ok) {
        fprintf(stderr, "All self-tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "SELF-TEST FAILED\n");
        return 1;
    }
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.self_test) {
        return run_self_test();
    }

    if (args.batch_mode) {
        return run_batch(args);
    }

    if (args.raw_mode) {
        return run_raw(args);
    }

    // Should not reach here — parse_args exits with usage if no mode set
    fprintf(stderr, "Error: must specify --batch, --raw, or --self-test\n");
    return 1;
}
