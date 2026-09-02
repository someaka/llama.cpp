/*
 * Batch hidden-state extraction CLI.
 *
 * Four modes:
 *   Batch:      hs-extract-batch <model> <prompts.txt> [layers] <output.bin> --batch --assignments <file> [flags]
 *               (with --generate N: generation-based extraction, see generate_assignment())
 *   Raw:        hs-extract-batch <model> <prompts.txt> [layers] <output.bin> --raw [flags]
 *   Self-test:  hs-extract-batch --self-test
 *   Generate:   hs-extract-batch ... --batch --generate N [sampling flags]
 *
 * Production mode is --batch: reads prompts.txt + assignments.bin, streams
 * one prompt at a time, computes masked means per assignment, accumulates
 * per group/mask/layer, writes output.bin with final means. With --generate N
 * it autoregressively generates N tokens per prompt and accumulates the
 * hidden states of the generated tokens instead (generate_assignment()).
 *
 * --raw is a debug/parity mode that dumps per-prompt binary data.
 *
 * Flags (see print_usage for the full list):
 *   --mean           In --raw: output token means instead of full per-token output
 *   --token-skip N   Skip first N tokens when a mean is computed (default: 0)
 *   --generate N     Batch mode: autoregressive generation extraction
 *                    (pair with --temperature/--top-k/--top-p/--repeat-penalty)
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
#include <cerrno>
#include <climits>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <utility>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#ifndef _WIN32
#include <unistd.h>  // fsync for durability before rename (POSIX)
#endif
#include <queue>
#include <atomic>

// HS_SIMD: vectorization hint for the hot mean-accumulation loops. Under
// OpenMP it is an omp simd pragma; without OpenMP it expands to nothing
// (correct, just not auto-vectorized by pragma).
#if defined(_OPENMP)
#define HS_SIMD _Pragma("omp simd")
#else
#define HS_SIMD
#endif

// -- Checked Write Macro ------------------------------------------------

// RAII wrappers (LlamaModel, LlamaContext, LlamaBackend, LlamaBatch) are in
// common/llama-raii.h (shared with hs-extract, tests, examples).
#include "llama-raii.h"

#include "hs-accum.h"
#include "hs-kernels.h"
#include "io-util.h"
#include "self-test.h"
#include "assignments-io.h"
// Shared layer-list parser (same file as hs-extract uses)
#include "layer-parse.h"

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
    bool checkpoint_explicit = false; // whether --checkpoint-every was passed
    bool resume = false;          // --resume: continue from checkpoint
    int n_gpu_layers = ALL_GPU_LAYERS; // --n-gpu-layers / -ngl: GPU offload (ALL_GPU_LAYERS = all)
    bool save_per_record = false;  // --save-per-record: write per-record vectors sidecar
    int batch_size = 1;           // --batch-size N: pack N prompts per decode (default: 1)
    bool profile = false;         // --profile: print per-phase timing breakdown
    bool no_bos = false;          // --no-bos: force BOS off (default: follow tokenizer's add_bos_token)
    int generate_tokens = 0;      // --generate N: autoregressive generation mode (extract from generated tokens)
    float temperature = 0.0f;     // --temperature: sampling temperature (0 = greedy argmax, the default)
    float repeat_penalty = 1.0f;  // --repeat-penalty: penalty for repeated tokens (1.0 = disabled)
    int top_k = 0;                // --top-k: top-k sampling (0 = disabled)
    float top_p = 1.0f;           // --top-p: nucleus sampling threshold (1.0 = disabled)
};

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  Batch (production): %s <model> <prompts.txt> [layers] <output.bin> --batch --assignments <file> [flags]\n", prog);
    fprintf(stderr, "  Raw (debug):        %s <model> <prompts.txt> [layers] <output.bin> --raw [flags]\n", prog);
    fprintf(stderr, "  Self-test:          %s --self-test\n", prog);
    fprintf(stderr, "\nModes:\n");
    fprintf(stderr, "  --batch          Accumulate masked means per group/mask/layer -> output.bin (production)\n");
    fprintf(stderr, "  --raw            Per-prompt dump for debugging and parity testing\n");
    fprintf(stderr, "  --self-test      Run synthetic tests on compute_masked_mean(), no model needed\n");
    fprintf(stderr, "\nFlags:\n");
    fprintf(stderr, "  --mean           In --raw mode: output token means instead of full per-token data\n");
    fprintf(stderr, "  --assignments F  Path to assignments.bin (required with --batch)\n");
    fprintf(stderr, "  --ctx-size N     Override auto context sizing (default: auto from prompts)\n");
    fprintf(stderr, "  --checkpoint-every N  Write checkpoint every N prompts (default: 10000)\n");
    fprintf(stderr, "  --resume         Resume from last checkpoint\n");
    fprintf(stderr, "  --n-gpu-layers N / -ngl N  Layers to offload to GPU (default: 99 = all)\n");
    fprintf(stderr, "  --save-per-record  Also write per-record vectors to <output>.records.bin\n");
    fprintf(stderr, "  --batch-size N   Only 1 is supported; the flag is parsed so old driver scripts\n                   keep working, and values > 1 are rejected at runtime\n");
    fprintf(stderr, "                  (multi-prompt batching is not implemented). Default: 1.\n");
    fprintf(stderr, "  --profile       Print per-phase timing breakdown (KV clear, decode, sync, extract, mean, accumulate)\n");
    fprintf(stderr, "  --no-bos        Force BOS off (default: follow the tokenizer's add_bos_token)\n");
    fprintf(stderr, "  --generate N    Generation-based extraction: generate N tokens after each prompt,\n");
    fprintf(stderr, "                  extract hidden states from GENERATED tokens only.\n");
    fprintf(stderr, "                  Without this flag, extraction is comprehension-based (forward pass only).\n");
    fprintf(stderr, "  --token-skip N  Skip the first N tokens when a mean is computed: with --raw --mean,\n");
    fprintf(stderr, "                  skip input tokens; with --batch --generate, skip generated tokens.\n");
    fprintf(stderr, "                  Rejected in batch comprehension mode and in --raw full dumps. Default: 0.\n");
    fprintf(stderr, "  --temperature F Sampling temperature for generation mode (0 = greedy argmax, default).\n");
    fprintf(stderr, "                  temperature > 0 reduces cross-backend divergence from greedy decoding.\n");
    fprintf(stderr, "  --top-k K       Top-k sampling (generation mode only, default: 0 = disabled).\n");
    fprintf(stderr, "  --top-p F       Nucleus sampling threshold (generation mode only, default: 1.0 = disabled).\n");
    fprintf(stderr, "  --repeat-penalty F  Repeat penalty (generation mode only, default: 1.0 = disabled).\n");
    fprintf(stderr, "\nLayers: comma-separated list (e.g. '0,5,10') or 'all' (default)\n");
    fprintf(stderr, "        Layer indices follow the hidden_states convention: 0 = token embeddings,\n");
    fprintf(stderr, "        i = state entering block i (= HF hidden_states[i]), N = final block output.\n");
    fprintf(stderr, "        Legacy fork numbering (post-block-i residual) = upstream index - 1.\n");
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
            i++;
            char* endptr = nullptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || errno == ERANGE || val < 0 || val > INT_MAX) {
                fprintf(stderr, "Error: --token-skip requires a non-negative integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.token_skip = (int)val;
        } else if (arg == "--self-test") {
            args.self_test = true;
        } else if (arg == "--batch") {
            args.batch_mode = true;
        } else if (arg == "--raw") {
            args.raw_mode = true;
        } else if (arg == "--no-bos") {
            args.no_bos = true;
        } else if (arg == "--generate") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --generate requires a value (number of tokens to generate)\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || errno == ERANGE || val <= 0 || val > INT_MAX) {
                fprintf(stderr, "Error: --generate requires a positive integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.generate_tokens = (int)val;
        } else if (arg == "--temperature") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --temperature requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            args.temperature = strtof(argv[i], &endptr);
            if (!std::isfinite(args.temperature) || *endptr != '\0' || endptr == argv[i] || args.temperature < 0.0f) {
                fprintf(stderr, "Error: --temperature requires a non-negative float, got '%s'\n", argv[i]);
                exit(1);
            }
        } else if (arg == "--repeat-penalty") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --repeat-penalty requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            args.repeat_penalty = strtof(argv[i], &endptr);
            if (!std::isfinite(args.repeat_penalty) || *endptr != '\0' || endptr == argv[i] || args.repeat_penalty < 1.0f) {
                fprintf(stderr, "Error: --repeat-penalty requires a float >= 1.0, got '%s'\n", argv[i]);
                exit(1);
            }
        } else if (arg == "--top-k") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --top-k requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || errno == ERANGE || val < 0 || val > INT_MAX) {
                fprintf(stderr, "Error: --top-k requires a non-negative integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.top_k = (int)val;
        } else if (arg == "--top-p") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --top-p requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            args.top_p = strtof(argv[i], &endptr);
            if (!std::isfinite(args.top_p) || *endptr != '\0' || endptr == argv[i] || args.top_p <= 0.0f || args.top_p > 1.0f) {
                fprintf(stderr, "Error: --top-p requires a float in (0, 1], got '%s'\n", argv[i]);
                exit(1);
            }
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
            i++;
            char* endptr = nullptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || errno == ERANGE || val < 0 || val > INT_MAX) {
                fprintf(stderr, "Error: --ctx-size requires a non-negative integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.ctx_size = (int)val;
        } else if (arg == "--checkpoint-every") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --checkpoint-every requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || errno == ERANGE || val <= 0 || val > INT_MAX) {
                fprintf(stderr, "Error: --checkpoint-every requires a positive integer, got '%s'\n", argv[i]);
                exit(1);
            }
            args.checkpoint_every = (int)val;
            args.checkpoint_explicit = true;
        } else if (arg == "--resume") {
            args.resume = true;
        } else if (arg == "--save-per-record") {
            args.save_per_record = true;
        } else if (arg == "--profile") {
            args.profile = true;
        } else if (arg == "--batch-size") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --batch-size requires a value\n");
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || val < 1 || val > MAX_BATCH_SIZE) {
                fprintf(stderr, "Error: --batch-size requires an integer in [1, %d], got '%s'\n", MAX_BATCH_SIZE, argv[i]);
                exit(1);
            }
            args.batch_size = (int)val;
        } else if (arg == "--n-gpu-layers" || arg == "-ngl") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires a value\n", arg.c_str());
                exit(1);
            }
            i++;
            char* endptr = nullptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0' || endptr == argv[i] || errno == ERANGE || val < 0 || val > INT_MAX) {
                fprintf(stderr, "Error: %s requires a non-negative integer, got '%s'\n", arg.c_str(), argv[i]);
                exit(1);
            }
            args.n_gpu_layers = (int)val;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else if (!arg.empty() && arg[0] == '-') {
            fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            exit(1);
        } else {
            positional.push_back(argv[i]);
        }
    }

    // --self-test needs no model or prompts -- skip positional validation
    if (args.self_test) {
        return args;
    }

    // --mean is only consumed in --raw mode; run_batch ignores it. Reject the
    // combination so the user gets a clear error instead of silently receiving
    // batch output without mean pooling.
    if (args.batch_mode && args.mean_mode) {
        fprintf(stderr, "Error: --mean is not supported with --batch (it only applies to --raw mode)\n");
        exit(1);
    }

    // --token-skip >= --generate silently produces empty output: the gate
    // `gen_step >= token_skip` never passes, every layer's gen_count stays 0,
    // every layer is skipped, and output.bin gets a header with no data while
    // the run exits 0. Reject it loudly rather than waste a 239K-prompt run.
    if (args.generate_tokens > 0 && args.token_skip >= args.generate_tokens) {
        fprintf(stderr,
                "Error: --token-skip (%d) must be < --generate (%d); otherwise every "
                "generated token is skipped and the output is empty.\n",
                args.token_skip, args.generate_tokens);
        exit(1);
    }

    // --token-skip is consumed ONLY in generation mode (and in --raw mode,
    // which is a different invocation). In batch comprehension mode the
    // token ranges come exclusively from the assignments file; the flag
    // would be silently ignored. Reject the combination so the flag
    // contract in --help is not a lie.
    if (args.batch_mode && args.generate_tokens <= 0 && args.token_skip > 0) {
        fprintf(stderr,
                "Error: --token-skip applies only to --raw mode or --batch --generate mode; "
                "in batch comprehension mode the token ranges come from the assignments file. "
                "Encode the skip as a mask_type=0 assignment instead.\n");
        exit(1);
    }

    // --batch and --raw together: the batch block below returns first and
    // the raw invocation would be silently discarded.
    if (args.batch_mode && args.raw_mode) {
        fprintf(stderr, "Error: --batch and --raw are mutually exclusive\n");
        exit(1);
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
        // --token-skip is consumed only by the --raw --mean path (mean over
        // [skip, n_tokens)); full dumps carry every token, so the flag
        // would be silently ignored.
        if (args.token_skip > 0 && !args.mean_mode) {
            fprintf(stderr, "Error: --token-skip applies only to --raw --mean or --batch --generate mode; "
                            "--raw full dumps contain every token. Drop the flag or pass --mean.\n");
            exit(1);
        }
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
        // The per-record sidecar is written by the batch-mode accumulation
        // path only; raw mode writes its own per-prompt output format.
        if (args.save_per_record) {
            fprintf(stderr, "Error: --save-per-record requires batch mode; "
                    "raw mode writes per-prompt vectors to the output file directly\n");
            print_usage(argv[0]);
            exit(1);
        }
        // Flags consumed only by run_batch would be silently ignored in raw
        // mode: --generate N performs a plain comprehension dump and exits 0,
        // sampling/resume/checkpoint flags vanish without a trace. Reject
        // them so every accepted invocation means what it says.
        if (args.generate_tokens > 0) {
            fprintf(stderr, "Error: --generate requires --batch (raw mode is comprehension-only)\n");
            exit(1);
        }
        if (args.temperature != 0.0f || args.top_k != 0 || args.top_p != 1.0f || args.repeat_penalty != 1.0f) {
            fprintf(stderr, "Error: sampling flags (--temperature/--top-k/--top-p/--repeat-penalty) "
                    "require --batch --generate; raw mode is comprehension-only\n");
            exit(1);
        }
        // checkpoint_every defaults to 10000 (batch mode); reject only an
        // EXPLICIT --resume/--checkpoint-every in raw mode, not the default.
        if (args.resume || args.checkpoint_explicit) {
            fprintf(stderr, "Error: --resume/--checkpoint-every apply to batch mode only\n");
            exit(1);
        }
        return args;
    }


    // No mode specified -- show usage and exit
    fprintf(stderr, "Error: must specify --batch, --raw, or --self-test\n\n");
    print_usage(argv[0]);
    exit(1);
}



// -- Masked Mean Computation --------------------------------------------

// Repeat-penalty window for generation mode's sampler chain.
static constexpr int REPEAT_PENALTY_LAST_N = 64;

/**
 * Compute mean of hidden state data over specified token ranges.
 *
 * Generalizes the single-contiguous-range mean (token_skip -> n_tokens) to
 * arbitrary collections of [start, end) token index pairs. Used by both the
 * refactored one-shot/persistent mean mode (single range) and the future
 * batch-accumulate mode (arbitrary token ranges).
 *
 * @param data      Pointer to hidden state data, shape (n_tokens, n_embd).
 * @param n_tokens  Number of tokens in the sequence.
 * @param n_embd    Hidden dimension size.
 * @param ranges    Vector of (start, end) token index pairs.
 * @param out       Output buffer, size n_embd. Must be zeroed by caller.
 * @return          Number of tokens included in the mean (0 = empty mask).
 */
int64_t compute_masked_mean(
    const float* data,
    int n_tokens,
    int n_embd,
    const std::vector<std::pair<int,int>>& ranges,
    float* out
) {
    int64_t count = 0;
    for (auto [start, end] : ranges) {
        // All range violations are hard errors. The Python side must produce
        // correct ranges  -  no silent clamping, no graceful degradation.
        if (start < 0 || end < 0 || start >= end) {
            fprintf(stderr, "Error: invalid masked-mean range [%d, %d)  -  negative or empty range\n", start, end);
            return -1;
        }
        if (end > n_tokens || start > n_tokens) {
            fprintf(stderr, "Error: masked-mean range [%d, %d) exceeds n_tokens=%d  -  "
                            "token span computation is incorrect (check BOS_OFFSET in dialogue_utils.py)\n",
                    start, end, n_tokens);
            return -1;
        }

        for (int t = start; t < end; t++) {
            const float* row = data + (size_t)t * (size_t)n_embd;
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

int compute_single_range_mean(
    const float* data,
    int n_tokens,
    int n_embd,
    int start,
    int end,
    float* out
) {
    // Single contiguous [start, end) span: a specialization of
    // compute_masked_mean with one range. Delegates for identical
    // validation and accumulation semantics.
    return (int)compute_masked_mean(data, n_tokens, n_embd, {{start, end}}, out);
}

// -- Prompt Processing --------------------------------------------------

/**
 * Process a single prompt: decode, synchronize, extract hidden states, write to file.
 *
 * In full mode: writes n_tokens * hidden_size floats per layer.
 * In mean mode: writes hidden_size floats per layer (mean from token_skip to n_tokens).
 *
 * Returns true on success, false on fatal error (decode failure, I/O error).
 */
static bool process_prompt(
    llama_context* ctx,
    llama_batch& batch,
    float* mean_buf,
    std::vector<int32_t>& header,
    const std::vector<llama_token>& tokens,
    const std::vector<int32_t>& target_layers,
    int32_t n_embd,
    size_t prompt_idx,
    FILE* out,
    bool mean_mode,
    int token_skip
) {
    int n_tokens = (int)tokens.size();

    if (n_tokens <= 0) {
        fprintf(stderr, "Error: prompt %zu tokenized to empty\n", prompt_idx);
        return false;
    }

    // Clear KV cache before processing this prompt
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // Fill pre-allocated batch buffer (reused across prompts to avoid heap fragmentation)
    // seq_id=0 (KV cache is cleared per prompt, so no collision)
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;  // No logits needed  -  we only extract hidden states
    }
    batch.n_tokens = n_tokens;

    // Decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "Error: decode failed for prompt %zu (ret=%d)\n", prompt_idx, ret);
        return false;
    }

    // The getters synchronize the context before returning data
    // (see llama_get_hidden_state); no explicit sync is needed here.
    int n_hidden_tokens = llama_get_hidden_state_n_tokens(ctx);
    if (n_hidden_tokens <= 0) {
        fprintf(stderr, "Error: decode produced no hidden state tokens for prompt %zu\n", prompt_idx);
        return false;
    }

    // Write per-prompt header: [prompt_idx][n_tokens][n_layers][layer_indices...]
    header.resize(3 + target_layers.size());
    header[0] = (int32_t)prompt_idx;
    header[1] = n_hidden_tokens;
    header[2] = (int32_t)target_layers.size();
    for (size_t i = 0; i < target_layers.size(); i++) {
        header[3 + i] = target_layers[i];
    }

    if (!checked_write(header.data(), sizeof(int32_t), 3 + target_layers.size(), out)) return false;

    // Write hidden state data for each layer
    for (int32_t layer : target_layers) {
        float* data = llama_get_hidden_state(ctx, layer);
        if (!data) {
            fprintf(stderr, "Error: null hidden state for layer %d\n", layer);
            return false;
        }

        if (mean_mode) {
            std::fill(mean_buf, mean_buf + n_embd, 0.0f);
            if (compute_single_range_mean(data, n_hidden_tokens, n_embd, token_skip, n_hidden_tokens, mean_buf) < 0) {
                fprintf(stderr, "Error: compute_single_range_mean failed for prompt %zu\n", prompt_idx);
                return false;
            }

            if (!checked_write(mean_buf, sizeof(float), n_embd, out)) return false;
        } else {
            // Full mode: write all tokens x hidden_size
            size_t data_size = (size_t)n_hidden_tokens * (size_t)n_embd;
            if (!checked_write(data, sizeof(float), data_size, out)) return false;
        }
    }

    return true;
}

// -- Tokenization -------------------------------------------------------

/**
 * Tokenize a prompt. Returns token vector (empty on failure).
 *
 * Deliberately NOT common_tokenize(): this variant (a) validates every token
 * against the vocab bound and hard-errors on violation (no-silent-errors
 * rule), (b) returns an empty vector on failure instead of throwing -- the
 * producer thread and run_raw check emptiness as their error contract, and an
 * exception from here would change that contract. Token output is identical
 * to common_tokenize(vocab, text, add_bos, true) on the happy path.
 *
 * add_bos: callers pass the unified default -- BOS follows the vocab's own
 * add_bos_token declaration, AND --no-bos forces it off (matching hs-extract's
 * llama_vocab_get_add_bos(vocab) && !no_bos). Before 2026-09-01 this tool
 * instead added BOS unconditionally unless --no-bos: on no-BOS-default models
 * (e.g. Qwen3.5) a manual invocation silently injected BOS, shifting every
 * token position and polluting the skip window -- a silent-corruption trap
 * the Python driver dodged only by passing --no-bos itself. The default is
 * now tokenizer-faithful; --no-bos remains as a force-off override.
 */
static std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool follow_vocab_default = true, bool no_bos = false) {
    const bool add_bos = follow_vocab_default && llama_vocab_get_add_bos(vocab) && !no_bos;
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
    if (n <= 0) return {};
    std::vector<llama_token> toks(n);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), (int)toks.size(), add_bos, true);
    if (n_tokens > 0) {
        toks.resize(n_tokens);
    } else {
        toks.clear();
        return toks;
    }

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

// -- Batch Mode Helpers -------------------------------------------------

/**
 * Single pre-scan of prompts.txt. Collects everything the pre-flight passes
 * need: non-empty line count (header/validation) and max line length
 * (auto-ctx estimate). One I/O pass serves run_raw, run_batch, and
 * load_model_session.
 */
struct PromptsScan {
    int32_t n_nonempty = 0;
    size_t max_len = 0;
};

static bool scan_prompts_file(const char* prompts_file, PromptsScan& scan) {
    std::ifstream fin(prompts_file);
    if (!fin) {
        fprintf(stderr, "Error: cannot open %s for pre-scan\n", prompts_file);
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty()) scan.n_nonempty++;
        if (line.size() > scan.max_len) scan.max_len = line.size();
    }
    // getline returns false for both EOF and I/O error; a bad stream would
    // under-count and under-size everything downstream.
    if (fin.bad()) {
        fprintf(stderr, "Error: I/O error reading prompts file during pre-scan (bad stream)\n");
        return false;
    }
    return true;
}

/**
 * Auto-size n_ctx from a completed scan. Uses chars / 3.5 as token estimate.
 * Returns max(estimated_max + 64, 512).
 */
static int32_t ctx_from_scan(const PromptsScan& scan) {
    int32_t estimated = (int32_t)(scan.max_len / CHARS_PER_TOKEN) + 64;
    int32_t n_ctx = estimated > MIN_CTX_SIZE ? estimated : MIN_CTX_SIZE;
    fprintf(stderr, "Auto-ctx: max line %zu chars -> estimated %d tokens -> n_ctx=%d\n",
            scan.max_len, (int)(scan.max_len / CHARS_PER_TOKEN), n_ctx);
    return n_ctx;
}

// -- Raw Mode (debug/parity) --------------------------------------------

/**
 * Raw mode: per-prompt binary dump for debugging and parity testing.
 *
 * Streams prompts.txt, decodes each, writes per-prompt hidden state data.
 * Same output format as the old one-shot mode:
 *   [n_prompts: int32]
 *   per prompt: [prompt_idx][n_tokens][n_layers][layer_indices][data]
 *
 * With --mean: writes mean vectors (token_skip->n_tokens) per layer.
 * Without --mean: writes full per-token data (n_tokens x hidden_size per layer).
 */
// Shared model session for both run modes: owns the llama objects in the
// required teardown order (ctx dies before model, model before backend).
struct ModelSession {
    LlamaBackend backend;
    LlamaModel model;
    LlamaContext ctx;
    const llama_vocab* vocab = nullptr;
    int32_t n_embd = 0;
    int32_t n_layers = 0;
    int32_t n_ctx_train = 0;
    int32_t n_ctx = 0;
    std::vector<int32_t> target_layers;
};

// Common prologue shared by run_raw and run_batch: load the model, validate
// dimensions, resolve target layers, size the context, create the extraction
// context. Returns false (with a stderr message) on failure.
static bool load_model_session(const Args& args, const PromptsScan& scan, ModelSession& s) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    s.model = LlamaModel(llama_model_load_from_file(args.model_path, mparams));
    if (!s.model) {
        fprintf(stderr, "Error: failed to load model %s\n", args.model_path);
        return false;
    }

    s.vocab = llama_model_get_vocab(s.model);
    s.n_embd = llama_model_n_embd_out(s.model);
    s.n_layers = llama_model_n_layer(s.model);
    s.n_ctx_train = llama_model_n_ctx_train(s.model);

    if (s.n_embd <= 0 || s.n_layers <= 0) {
        fprintf(stderr, "Error: invalid model dimensions (n_embd=%d, n_layers=%d)\n", s.n_embd, s.n_layers);
        return false;
    }

    fprintf(stderr, "Model loaded: n_ctx_train=%d, n_embd=%d, n_layers=%d, n_gpu_layers=%d\n",
            s.n_ctx_train, s.n_embd, s.n_layers, args.n_gpu_layers);

    // Resolve layers (hidden_states indices: 0..n_layers inclusive); negative
    // indices resolve Python-style and duplicates are rejected in the parser.
    s.target_layers = hs_parse_layer_list(args.layers_str.c_str(), s.n_layers + 1);
    if (s.target_layers.empty()) {
        fprintf(stderr, "Error: no valid target layers\n");
        return false;
    }

    // Auto-size context or use --ctx-size
    if (args.ctx_size > 0) {
        s.n_ctx = args.ctx_size;
    } else {
        s.n_ctx = ctx_from_scan(scan);
    }
    if (s.n_ctx > s.n_ctx_train) s.n_ctx = s.n_ctx_train;
    fprintf(stderr, "Using n_ctx=%d\n", s.n_ctx);

    // Create the context BEFORE opening output files or reading assignments.
    // Qwen3.5 Gated Delta Net requires the context to exist before model
    // graph optimization, and creating it first prevents an OOM when the
    // graph optimizer allocates CUDA buffers that would conflict with
    // subsequent allocations. n_ubatch = n_ctx keeps the full prompt in one
    // ubatch pass (stable CUDA graph capture).
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = s.n_ctx;
    cparams.n_batch = s.n_ctx;
    cparams.n_ubatch = s.n_ctx;
    cparams.extract_hidden_states = true;

    s.ctx = LlamaContext(llama_init_from_model(s.model, cparams));
    if (!s.ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        return false;
    }
    return true;
}

static int run_raw(const Args& args) {
    PromptsScan scan;
    if (!scan_prompts_file(args.prompts_file, scan)) return 1;
    // Q-P1-10 (2026-08-29 quality pass): an empty prompts file would finalize
    // an output containing just the int32 0 header and exit 0 — silently
    // producing a "complete" empty dump. Reject instead.
    if (scan.n_nonempty == 0) {
        fprintf(stderr, "Error: prompts file '%s' contains no non-empty prompts\n", args.prompts_file);
        return 1;
    }
    ModelSession ms;
    if (!load_model_session(args, scan, ms)) return 1;
    LlamaContext& ctx = ms.ctx;
    const llama_vocab*& vocab = ms.vocab;
    int32_t& n_embd = ms.n_embd;
    int32_t& n_ctx = ms.n_ctx;
    std::vector<int32_t>& target_layers = ms.target_layers;

    // Open output file. Write to a temp path and atomically rename at the end;
    // a crash must never leave a truncated dump at the final path, because
    // downstream parsers would misread it.
    std::string tmp_path = std::string(args.output_path) + ".tmp";
    FilePtr out(fopen(tmp_path.c_str(), "wb"));
    if (!out) {
        fprintf(stderr, "Error: cannot open output file %s\n", tmp_path.c_str());
        return 1;
    }
    // Shared error cleanup: close the temp file and remove it so a failed
    // run never leaves a stray .tmp that a later run could mistake for
    // progress. Every error path below returns 1 after calling this.
    auto fail_cleanup = [&]() {
        out.reset();
        std::remove(tmp_path.c_str());
    };
    fprintf(stderr, "Writing output to %s\n", args.output_path);

    // Prompt count from the single pre-scan; the TOCTOU guard below still
    // compares it against the processed count from the streaming pass.
    int32_t n_prompts_total = scan.n_nonempty;

    // Global header
    if (fwrite(&n_prompts_total, sizeof(int32_t), 1, out) != 1) {
        fprintf(stderr, "Error: failed to write raw output header\n");
        fail_cleanup();
        return 1;
    }

    // Process prompts (streaming)
    std::ifstream fin(args.prompts_file);
    auto start_time = std::chrono::steady_clock::now();
    int prompt_idx = 0;

    // Pre-allocate batch and mean buffer once (avoid per-prompt heap fragmentation)
    LlamaBatch batch_wrapper;
    batch_wrapper.init(n_ctx, 0, 1);
    std::vector<float> mean_buf(n_embd, 0.0f);
    std::vector<int32_t> header;  // per-prompt output header (reused, written in process_prompt)

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            // Skip empty lines -- matches the counting pass (lines 902-906).
            // The header n_prompts_total already excludes these.
            continue;
        }

        auto tokens = tokenize(vocab, line, /*follow_vocab_default=*/true, /*no_bos=*/args.no_bos);
        if (tokens.empty()) {
            fprintf(stderr, "Error: prompt %d tokenized to empty\n", prompt_idx);
            fail_cleanup();
            return 1;
        }
        if ((int)tokens.size() > n_ctx) {
            fprintf(stderr, "Error: prompt %d has %zu tokens, exceeds n_ctx=%d\n",
                    prompt_idx, tokens.size(), n_ctx);
            fail_cleanup();
            return 1;
        }

        if (!process_prompt(ctx, batch_wrapper.batch, mean_buf.data(), header,
                            tokens, target_layers, n_embd, prompt_idx, out,
                            args.mean_mode, args.token_skip)) {
            fprintf(stderr, "Error: process_prompt failed at prompt %d\n", prompt_idx);
            fail_cleanup();
            return 1;  // fatal I/O error
        }

        prompt_idx++;

        if (prompt_idx % PROGRESS_INTERVAL == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            float pps = prompt_idx * 1000.0f / (elapsed > 0 ? elapsed : 1);
            fprintf(stderr, "Processed %d/%d prompts (%.2f prompts/sec)\n",
                    prompt_idx, n_prompts_total, pps);
        }
    }

    // getline returns false for both EOF and I/O error (bad stream). An
    // unchecked I/O error here would finalize a silently truncated dump via
    // the atomic rename below, and the header (written from the count pass)
    // would certify it as complete.
    if (fin.bad()) {
        fprintf(stderr, "Error: I/O error reading prompts file mid-run (bad stream) at prompt %d\n", prompt_idx);
        fail_cleanup();
        return 1;
    }
    // Position-based completeness check (TOCTOU guard, mirrors run_batch):
    // every non-empty line counted in the first pass must have been
    // processed in the second pass. A prompts file modified between the two
    // passes must abort before the dump is finalized, never rename a short
    // file certified complete by the header.
    if (prompt_idx != n_prompts_total) {
        fprintf(stderr, "Error: processed %d prompts but counted %d  -  prompts file changed between passes\n",
                prompt_idx, n_prompts_total);
        fail_cleanup();
        return 1;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    fprintf(stderr, "Done. Processed %d prompts in %.2f seconds (%.2f prompts/sec)\n",
            prompt_idx, total_ms / 1000.0,
            prompt_idx * 1000.0 / (total_ms > 0 ? total_ms : 1));

    // Atomic finalize: sync to durable storage, close, then rename into place.
    // The final path appears only when the write is complete (rename is atomic
    // on POSIX), so a kill mid-write can never leave a truncated raw dump.
    if (!out.sync()) {   // fflush + fsync(fileno): check for disk-full / I/O errors
        fprintf(stderr, "Error: sync failed for %s (disk full or I/O error)\n", tmp_path.c_str());
        fail_cleanup();
        return 1;
    }
    out.reset();  // fclose before rename
    if (rename(tmp_path.c_str(), args.output_path) != 0) {
        fprintf(stderr, "Error: cannot rename %s to %s\n", tmp_path.c_str(), args.output_path);
        std::remove(tmp_path.c_str());
        return 1;
    }
    if (!fsync_parent_dir(args.output_path)) {  // make the rename itself durable
        return 1;
    }

    return 0;
}

// -- Async prefetch pipeline (TU scope) ----------------------------------
//
// Producer-consumer: the producer thread does file I/O + tokenization for
// prompt N+1 while the GPU processes prompt N. The consumer (main thread)
// owns the llama_context exclusively -- only it calls decode/extract.
//
// Backpressure: the queue below is capped at MAX_PREFETCH items to bound
// memory (~2KB typical per item, up to ~50KB for long prompts; 64 items ->
// max ~3MB queued), preventing OOM on 200K-prompt runs where tokenization
// outpaces GPU decode.

struct PrefetchedPrompt {
    int prompt_idx = -1;
    std::vector<llama_token> tokens;
    std::vector<Assignment> assignments;
    bool skip = false;  // tokenization failed or exceeds ctx
    bool error = false; // set by producer on assignment read failure
    uint64_t line_fnv = 0;  // FNV-1a 64 of this prompt's raw line
};

struct PrefetchQueue {
    std::queue<PrefetchedPrompt> queue;
    std::mutex mtx;
    std::condition_variable cv;          // consumer waits: queue non-empty or producer done
    std::condition_variable cv_space;    // producer waits: queue has space
    std::atomic<bool> producer_done{false};
    std::atomic<int> n_produced{0};
};

// Backpressure cap: see the pipeline comment above.
static constexpr size_t MAX_PREFETCH = 64;

// Checked fwrite for the per-record sidecar.
// Write one field of a per-record row. On failure returns false ONLY --
// producer teardown (join + temp removal) is owned by the single
// stop_producer_and_join call at the caller's error site; joining here
// would double-join a non-joinable thread (std::system_error -> terminate).
// `what` names the field for the diagnostic (no __FILE__:__LINE__ here: it
// would always print this helper's own location, the trap io-util.h's
// checked_write avoids).
static inline bool records_write(
    const void* ptr, size_t size, size_t count, FILE* fp, const char* what
) {
    if (fwrite(ptr, size, count, fp) != count) {
        fprintf(stderr, "Error: per-record write failed (%s)\n", what);
        return false;
    }
    return true;
}


// Signal the producer to stop, join it, and remove the per-record temp file
// (a leftover .records.bin.tmp would be picked up by a subsequent run's
// parser). Inline function, explicit args -- no implicit scope capture.
static inline void stop_producer_and_join(
    PrefetchQueue& pfq,
    std::thread& producer_thread,
    const std::string& records_temp_path
) {
    {
        std::lock_guard<std::mutex> lk(pfq.mtx);
        pfq.producer_done = true;
    }
    pfq.cv_space.notify_all();
    producer_thread.join();
    if (!records_temp_path.empty()) std::remove(records_temp_path.c_str());
}

// -- Generation-based extraction -----------------------------------------
//
// If --generate N is set, autoregressively generate N tokens and extract
// hidden states from the GENERATED tokens only. This contrasts with the
// default comprehension-based mode, which reads hidden states from the
// INPUT tokens of a single forward pass: here the model's own output
// tokens are the span whose representations are captured.
// Returns 0 on success (caller runs the shared tail); 1 on error (caller
// stops the producer and returns 1).

struct GenContext {
    LlamaContext& ctx;
    llama_batch& batch;
    const Args& args;
    const std::vector<Assignment>& assignments;
    int prompt_idx;
    int n_tokens;
    int32_t n_embd;
    int32_t n_ctx;
    const std::vector<int32_t>& target_layers;
    std::vector<float*>& layer_ptrs;
    std::vector<float>& mean_buf;
    AccumulatorMap& accumulators;
};

static int generate_assignment(GenContext& g) {
    LlamaContext& ctx = g.ctx;
    llama_batch& batch = g.batch;
    const Args& args = g.args;
    const auto& assignments = g.assignments;
    const int prompt_idx = g.prompt_idx;
    const int n_tokens = g.n_tokens;
    const int32_t n_embd = g.n_embd;
    const int32_t n_ctx = g.n_ctx;
    const std::vector<int32_t>& target_layers = g.target_layers;
    std::vector<float*>& layer_ptrs = g.layer_ptrs;
    std::vector<float>& mean_buf = g.mean_buf;
    AccumulatorMap& accumulators = g.accumulators;
    int ret = 0;

    // --generate + --save-per-record is unsupported: the per-record
    // sidecar records are written by the comprehension block below, which
    // this path skips via `continue`. Fail loud rather than emit a
    // header-only sidecar that silently produces zero .pt files.
    if (args.save_per_record) {
        fprintf(stderr, "Error: --generate is incompatible with --save-per-record "
                "(per-record records are not written in generation mode)\n");
        return 1;
    }
    // Generation mode means over ALL generated tokens. Assignments
    // with skip>0 or ranges (mask_type==1) have no defined semantics here
    // since there is no pre-written content span to mask. Fail loud.
    for (const auto& a : assignments) {
        if (a.skip > 0 || a.mask_type == 1) {
            fprintf(stderr, "Error: --generate does not support assignment skip/ranges "
                    "(group=%d mask=%d has skip=%d mask_type=%d). Use skip=0, mask_type=0 "
                    "for generation-based extraction.\n",
                    a.group_id, a.mask_id, a.skip, a.mask_type);
                    return 1;
        }
    }
    // Accumulate hidden states from generated tokens into per-layer buffers.
    // We reuse mean_buf for per-token accumulation and a separate gen_accum
    // buffer for the running sum. Function-scope static: one allocation,
    // reused across calls; NOT thread-safe (run_batch is single-consumer).
    static std::vector<float> gen_accum;
    static std::vector<int64_t> gen_count;
    const size_t n_layers_total = target_layers.size();
    // Sanity-bound the accumulation buffer before allocating. n_layers_total
    // and n_embd come from the loaded model (target_layers is validated
    // against the model's layer count; n_embd is the model's embedding dim),
    // so the product is normally small (e.g. 42 layers x 2560 embd = 107K
    // floats). Guard against a corrupt/absurd configuration that would request
    // a multi-GB allocation: resize() throws std::bad_alloc on failure and
    // this block has no surrounding try, so an unbounded resize would call
    // std::terminate. 256M floats (1 GB) is far above any real model's
    // layer*embd product.
    constexpr size_t MAX_GEN_ACCUM_FLOATS = 256ull * 1024 * 1024;  // 1 GB
    if (n_layers_total > 0 && (size_t)n_embd > MAX_GEN_ACCUM_FLOATS / n_layers_total) {
        fprintf(stderr, "Error: generation accumulator size (%zu layers x %d embd) "
                "exceeds 1GB cap - configuration is corrupt or absurdly large\n",
                n_layers_total, n_embd);
        return 1;
    }
    gen_accum.resize(n_layers_total * n_embd, 0.0f);
    gen_count.assign(n_layers_total, 0);
    std::fill(gen_accum.begin(), gen_accum.end(), 0.0f);

    int cur_pos = n_tokens;  // next position after prompt
    llama_token next_token = LLAMA_TOKEN_NULL;

    // Build sampler chain for generation. Default (temperature=0) is greedy,
    // which is deterministic but divergent across CUDA/Vulkan backends because
    // small logit differences flip the argmax. temperature > 0 adds stochasticity
    // that, combined with repeat_penalty, stabilizes cross-backend trajectories.
    // The chain is built PER PROMPT (deliberate): llama_sampler_init_dist(0)
    // seeds a fresh RNG per prompt so sampled trajectories are reproducible
    // per-prompt. Hoisting the chain out of the loop would change RNG
    // sequencing across prompts. LlamaSampler (llama-raii.h) owns the free.
    // Named gen_vocab to avoid shadowing the session-level vocab.
    const llama_vocab * gen_vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(gen_vocab);
    LlamaSampler sampler_owner;
    // Chain is built when temperature > 0 OR when any secondary
    // sampling parameter is non-default: --repeat-penalty/--top-k/
    // --top-p must not be silently dropped at temperature=0 (the
    // documented greedy default). A temp(0) tail converts the chain
    // to greedy argmax AFTER penalties/top-k/top-p are applied, so
    // "greedy with repeat penalty" behaves as documented. Pure
    // defaults (all zero) keep the fast manual-argmax path.
    const bool want_sampling_chain = args.temperature > 0.0f
        || args.repeat_penalty > 1.0f
        || args.top_k > 0
        || args.top_p < 1.0f;
    if (want_sampling_chain) {
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sampler_owner = LlamaSampler(llama_sampler_chain_init(sparams));
        if (args.repeat_penalty > 1.0f) {
            // upstream API change (2026-08): llama_sampler_init_penalties now
            // takes n_vocab first to bound the repeat-scan range.
            llama_sampler_chain_add(sampler_owner, llama_sampler_init_penalties(n_vocab, REPEAT_PENALTY_LAST_N, args.repeat_penalty, 0.0f, 0.0f));
        }
        if (args.top_k > 0) {
            llama_sampler_chain_add(sampler_owner, llama_sampler_init_top_k(args.top_k));
        }
        if (args.top_p < 1.0f) {
            llama_sampler_chain_add(sampler_owner, llama_sampler_init_top_p(args.top_p, 1));
        }
        llama_sampler_chain_add(sampler_owner, llama_sampler_init_temp(args.temperature));
        llama_sampler_chain_add(sampler_owner, llama_sampler_init_dist(0));  // seed=0 for determinism
    }
    llama_sampler * sampler = sampler_owner;  // raw view used by the loop below

    for (int gen_step = 0; gen_step < args.generate_tokens; gen_step++) {
        // KV-cache bounds check: cur_pos must stay < n_ctx or llama_decode
        // writes/reads OOB KV state, the exact fault class that hard-locks
        // the GPU. Without this, a long prompt + large --generate (or a model
        // that never emits EOS) overruns the context window.
        if (cur_pos >= n_ctx) {
            fprintf(stderr, "Warning: generation reached n_ctx=%d at step %d for prompt %d: stopping\n",
                    n_ctx, gen_step, prompt_idx);
            break;
        }
        // Sample first token from prefill's last position
        if (gen_step == 0) {
            if (sampler) {
                next_token = llama_sampler_sample(sampler, ctx, n_tokens - 1);
            } else {
                // Greedy fallback (temperature=0): argmax
                float* token_logits = llama_get_logits_ith(ctx, n_tokens - 1);
                if (!token_logits) {
                    fprintf(stderr, "Error: no logits available for generation sampling\n");
                    return 1;
                }
                next_token = 0;
                float best_val = token_logits[0];
                for (int v = 1; v < n_vocab; v++) {
                    if (token_logits[v] > best_val) {
                        best_val = token_logits[v];
                        next_token = v;
                    }
                }
                // If all logits are NaN, every comparison is false,
                // so next_token stays 0 and NaN propagates into the
                // hidden-state accumulation silently. Detect and abort.
                if (std::isnan(best_val)) {
                    fprintf(stderr, "Error: NaN logits at generation step %d for prompt %d "
                                    "(numerical instability or corrupt model)\n", gen_step, prompt_idx);
                    return 1;
                }
            }
        }

        // Check for EOS
        if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx)), next_token)) {
            break;
        }

        // Decode single generated token with logits enabled for next step
        batch.n_tokens = 1;
        batch.token[0] = next_token;
        batch.pos[0] = cur_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = (gen_step < args.generate_tokens - 1) ? 1 : 0;

        ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "Error: generation decode failed at step %d for prompt %d (ret=%d)\n",
                    gen_step, prompt_idx, ret);
                    return 1;
        }
        cur_pos++;

        // Extract hidden states from this generated token
        if (llama_get_hidden_states_batch(ctx, target_layers.data(), target_layers.size(), layer_ptrs.data()) != 0) {
            fprintf(stderr, "Error: hidden state extraction failed during generation step %d\n", gen_step);
                    return 1;
        }
        // Each single-token decode refills the capture buffer from scratch, so
        // it must hold exactly one row (the generated token) per layer. Reading
        // row 0 without this check would silently take the wrong token if the
        // capture contract ever changed.
        {
            int32_t n_hidden = llama_get_hidden_state_n_tokens(ctx);
            if (n_hidden != 1) {
                fprintf(stderr, "Error: capture buffer holds %d tokens after single-token decode (expected 1) - "
                                "generation accumulation would read the wrong row\n", n_hidden);
                return 1;
            }
        }

        // Accumulate: each layer has 1 token x n_embd floats.
        // Skip the first args.token_skip generated tokens (matching
        // comprehension-mode behavior): the initial generated tokens
        // are "warm-up" content that dilutes the concept signal and
        // amplifies cross-backend divergence (the first ~50 tokens of
        // greedy decoding diverge most across CUDA/Vulkan backends).
        if (gen_step >= args.token_skip) {
            for (size_t li = 0; li < n_layers_total; li++) {
                float* data = layer_ptrs[li];
                if (data) {
                    float* accum_ptr = &gen_accum[li * n_embd];
                    HS_SIMD
                    for (int d = 0; d < n_embd; d++) {
                        accum_ptr[d] += data[d];
                    }
                    gen_count[li]++;
                }
            }
        }

        // Sample next token from this step's logits (for next iteration)
        if (gen_step < args.generate_tokens - 1) {
            if (sampler) {
                next_token = llama_sampler_sample(sampler, ctx, 0);
            } else {
                // Greedy fallback
                float* gen_logits = llama_get_logits_ith(ctx, 0);
                if (gen_logits) {
                    next_token = 0;
                    float best_val = gen_logits[0];
                    for (int v = 1; v < n_vocab; v++) {
                        if (gen_logits[v] > best_val) {
                            best_val = gen_logits[v];
                            next_token = v;
                        }
                    }
                    // NaN guard (same as the token_logits argmax above).
                    // If every logit is NaN, no comparison is true and
                    // next_token stays 0, propagating NaN into accumulation.
                    if (std::isnan(best_val)) {
                        fprintf(stderr, "Error: NaN logits at generation step %d for prompt %d "
                                        "(numerical instability or corrupt model)\n", gen_step, prompt_idx);
                        return 1;
                    }
                }
            }
        }
    }

    // Compute means from generated tokens and accumulate into the assignment buffers
    // (same path as comprehension mode, but using gen_accum instead of masked mean).
    // Q-P1-4 (2026-08-29 quality pass): the mean gen_accum[li]/n_gen is
    // per-LAYER, not per-assignment — hoisted out of the assignment loop
    // (A*L divides collapsed to L).
    for (size_t li = 0; li < target_layers.size(); li++) {
        int64_t n_gen = gen_count[li];
        if (n_gen == 0) continue;  // layer had no valid hidden states
        float* accum_ptr = &gen_accum[li * n_embd];
        float inv = 1.0f / (float)n_gen;
        HS_SIMD
        for (int d = 0; d < n_embd; d++) {
            mean_buf[d] = accum_ptr[d] * inv;
        }
        // Accumulate into the group/mask/layer accumulator.
        // Use target_layers[li] (the REAL layer number), matching the
        // comprehension accumulation loop. Using the loop index `li`
        // would silently mislabel layers in the binary accumulator output.
        // Per-layer gen_count[li] may differ (null layer_ptrs[li] skips
        // that layer's accumulation), so counts can differ across layers.
        for (const auto& assign : assignments) {
            uint64_t acc_key = make_accum_key(assign.group_id, assign.mask_id, target_layers[li]);
            auto& acc = accumulators[acc_key];
            if (acc.sum.empty()) {
                acc.sum.assign(mean_buf.data(), mean_buf.data() + n_embd);
                acc.count = 1;
            } else {
                HS_SIMD
                for (int d = 0; d < n_embd; d++) {
                    acc.sum[d] += mean_buf[d];
                }
                acc.count++;
            }
        }
    }


    return 0;
}

// Build the checkpoint run fingerprint. Layer list is sorted so the identity
// is order-independent (--layers 0,17,35 and 35,17,0 resume interchangeably).
// line_fnv is the FNV-1a 64 of the last-processed prompts line (0 before any
// prompt completes); n_prompts binds the run to the dataset size.
static checkpoint_fingerprint make_fingerprint(const Args& args, const std::vector<int32_t>& layers,
                                               uint64_t line_fnv, int32_t n_prompts) {
    checkpoint_fingerprint fp;
    fp.generate_mode   = args.generate_tokens > 0;
    fp.generate_tokens = args.generate_tokens;
    fp.token_skip      = args.token_skip;
    fp.layers          = layers;
    std::sort(fp.layers.begin(), fp.layers.end());
    fp.content_fnv64 = line_fnv;
    fp.n_prompts     = n_prompts;
    return fp;
}

// -- Batch-Accumulate Mode ----------------------------------------------

/**
 * Batch-accumulate mode: the production extraction path.
 *
 * Reads prompts.txt (streamed one line at a time) and assignments.bin (sequential).
 * For each prompt: tokenize -> decode -> compute masked means per assignment -> accumulate.
 * At the end: write output.bin with per-group/mask/layer mean vectors.
 *
 * Memory is O(groups x masks x layers x n_embd), independent of prompt count.
 */
static int run_batch(const Args& args) {
    // -- Setup (RAII: destructors handle cleanup on all return paths) --
    PromptsScan scan;
    if (!scan_prompts_file(args.prompts_file, scan)) return 1;
    ModelSession ms;
    if (!load_model_session(args, scan, ms)) return 1;
    LlamaContext& ctx = ms.ctx;
    const llama_vocab*& vocab = ms.vocab;
    int32_t& n_embd = ms.n_embd;
    int32_t& n_ctx = ms.n_ctx;
    std::vector<int32_t>& target_layers = ms.target_layers;

    // --batch-size > 1 is rejected: multi-prompt batching is not implemented
    // (shared KV-cache semantics would corrupt extraction). The parser still
    // range-checks the value so the error message can quote it.
    if (args.batch_size > 1) {
        fprintf(stderr, "Error: --batch-size %d is not supported. Multi-prompt batching is "
                        "not implemented (Gemma-4 shared-KV-cache corruption). Use --batch-size 1 "
                        "or omit the flag.\n", args.batch_size);
        return 1;
    }

    // -- Open input files --
    std::ifstream prompts_fin(args.prompts_file);
    if (!prompts_fin) {
        fprintf(stderr, "Error: cannot open prompts file %s\n", args.prompts_file);
        return 1;
    }

    FilePtr assign_fin(fopen(args.assignments_file, "rb"));
    if (!assign_fin) {
        fprintf(stderr, "Error: cannot open assignments file %s\n", args.assignments_file);
        return 1;
    }

    // Read assignments header
    int32_t n_prompts_expected = 0;
    int32_t n_embd_expected = 0;
    GroupTable groups;  // read but not used yet  -  reserved for future group-name output
    if (!read_assignments_header(assign_fin, n_prompts_expected, n_embd_expected, groups)) {
        return 1;
    }

    // Validate prompt count: assignments.bin and prompts.txt must agree
    // (count from the single pre-scan). Without this, a mismatch between
    // files from different pipeline runs causes silent desync (EOF or
    // leftover assignments) with a generic error message instead of a clear
    // upfront diagnostic.
    if (scan.n_nonempty != n_prompts_expected) {
        fprintf(stderr, "Error: prompt count mismatch - assignments.bin expects %d prompts "
                        "but prompts.txt has %d non-empty lines\n",
                n_prompts_expected, scan.n_nonempty);
        return 1;
    }

    // Validate n_embd if specified
    if (n_embd_expected > 0 && n_embd_expected != n_embd) {
        fprintf(stderr, "Error: assignments.bin expects n_embd=%d but model has n_embd=%d\n",
                n_embd_expected, n_embd);
        return 1;
    }

    // -- Hot loop --
    // Accumulator: flat key = make_accum_key(group_id, mask_id, layer_idx) -> AccumulatedVector
    AccumulatorMap accumulators;
    std::vector<float> mean_buf(n_embd, 0.0f);  // reused buffer for masked mean
    std::vector<std::pair<int,int>> ranges_buf;   // reused buffer for token ranges
    std::vector<float*> layer_ptrs(target_layers.size());  // Phase 2: reused buffer for layer pointers

    // Pre-allocate a reusable batch buffer sized to n_ctx.
    // Allocating/freeing a llama_batch on every prompt iteration caused
    // heap fragmentation over 90K+ iterations, eventually triggering glibc
    // "corrupted size vs. prev_size" aborts. By allocating once and reusing,
    // we eliminate all per-prompt heap churn from the batch structure.
    LlamaBatch batch_wrapper;
    batch_wrapper.init(n_ctx, 0, 1);
    llama_batch& batch = batch_wrapper.batch;

    // --resume + --save-per-record would silently truncate the sidecar:
    // resumed-over prompts (< skip_count) are skipped WITHOUT emitting their
    // per-record rows, and the STR1 header carries no record count, so the
    // downstream parser (file-size-derived total) cannot detect the gap.
    // Fail loud up front instead. (To resume a save-per-record run, re-run
    // it without --resume.)
    if (args.save_per_record && args.resume) {
        fprintf(stderr, "Error: --resume is incompatible with --save-per-record "
                "(resumed-over prompts would be missing from the sidecar)\n");
        return 1;
    }

    // Resume from checkpoint if --resume is set. The checkpoint carries a run
    // fingerprint (v3): mode, --generate length, --token-skip, and the target
    // layer list. A checkpoint written under different settings is rejected
    // loudly rather than silently merged. v4 adds the prompts-file content
    // identity (last-processed line hash + prompt count); read_checkpoint
    // re-hashes the current prompts file and rejects a mismatch.
    int skip_count = 0;
    if (args.resume) {
        checkpoint_fingerprint fp = make_fingerprint(args, target_layers, 0, n_prompts_expected);
        if (!read_checkpoint(args.output_path, accumulators, skip_count, n_embd, fp, args.prompts_file)) {
            fprintf(stderr, "Error: --resume requested but no valid checkpoint found at %s.checkpoint\n",
                    args.output_path);
            return 1;
        }
        // A checkpoint from a different (larger) dataset would skip every
        // prompt and still finalize a valid-looking output. Reject instead.
        if (skip_count > n_prompts_expected) {
            fprintf(stderr, "Error: checkpoint skip count %d exceeds dataset size %d - checkpoint is from a "
                            "different (larger) run. Discard the checkpoint (remove %s.checkpoint).\n",
                    skip_count, n_prompts_expected, args.output_path);
            return 1;
        }
        if (skip_count == n_prompts_expected) {
            fprintf(stderr, "Warning: checkpoint reports all %d prompts already processed - nothing to do.\n",
                    skip_count);
        }
    }

    int prompt_idx = 0;
    int n_processed = 0;
    // FNV-1a 64 of the most recently processed prompts line, updated by the
    // consumer and captured by the producer for checkpoint fingerprinting.
    uint64_t line_fnv = 0;
    auto start_time = std::chrono::steady_clock::now();

    // Profiling accumulators
    double prof_total_kv = 0, prof_total_decode = 0, prof_total_extract = 0, prof_total_mean = 0;
    int prof_count = 0;

    // Per-record sidecar file (optional)
    // Written to a temp path and atomically renamed to the final .records.bin
    // only on full success, so a mid-run crash can never leave a truncated
    // sidecar that the Python parser would misread on the next run.
    std::string records_path;
    std::string records_temp_path;
    FilePtr records_closer;
    if (args.save_per_record) {
        records_path = std::string(args.output_path) + ".records.bin";
        records_temp_path = records_path + ".tmp";
        records_closer.fp = fopen(records_temp_path.c_str(), "wb");
        if (!records_closer) {
            fprintf(stderr, "Error: cannot open per-record file %s\n", records_temp_path.c_str());
            return 1;
        }
        FILE* records_fp = records_closer.fp;
        int32_t record_magic = 0x53545231;  // per-record sidecar format
        if (fwrite(&record_magic, sizeof(int32_t), 1, records_fp) != 1 ||
            fwrite(&n_embd, sizeof(int32_t), 1, records_fp) != 1) {
            fprintf(stderr, "Error: per-record header write failed\n");
            // Remove the partially-written temp file.
            records_closer.reset();
            std::remove(records_temp_path.c_str());
            return 1;
        }
        int32_t n_target_layers = (int32_t)target_layers.size();
        if (fwrite(&n_target_layers, sizeof(int32_t), 1, records_fp) != 1) {
            fprintf(stderr, "Error: per-record header write failed\n");
            // Remove the partially-written temp file.
            records_closer.reset();
            std::remove(records_temp_path.c_str());
            return 1;
        }
        fprintf(stderr, "Per-record output: %s (n_embd=%d, n_layers=%d)\n",
                records_path.c_str(), n_embd, n_target_layers);
    }


    PrefetchQueue pfq;

    // Producer thread: reads prompts, tokenizes, reads assignments
    auto producer = [&]() {
      try {
        std::string p_line;
        while (std::getline(prompts_fin, p_line)) {
            // Skip empty lines to match validation's non-empty count.
            // Without this, an empty line in prompts.txt reads an assignment
            // record meant for the next real prompt, silently desyncing all
            // subsequent prompt/assignment pairings.
            if (p_line.empty()) continue;
            auto assignment_read = read_prompt_assignments(assign_fin);
            if (assignment_read.status != AssignmentReadStatus::ok) {
                // Q-P2-3: precise producer-side diagnostic (status, not a
                // generic consumer-side "assignment read failed").
                fprintf(stderr, "Error: assignment read status=%d for prompt %d (assignments stream desync or truncated)\n",
                        (int)assignment_read.status, (int)pfq.n_produced.load());
                PrefetchedPrompt p;
                p.error = true;
                {
                    std::lock_guard<std::mutex> lk(pfq.mtx);
                    pfq.queue.push(std::move(p));
                }
                pfq.cv.notify_one();
                return;
            }

            int p_idx = pfq.n_produced.fetch_add(1);

            PrefetchedPrompt pp;
            pp.prompt_idx = p_idx;
            pp.line_fnv = hs_fnv1a64(p_line);
            // On --resume, prompts below skip_count are discarded by the
            // consumer without decoding; their tokens are never used.
            // Skip the tokenization (pure waste on resume: a 150K-prompt
            // resume burns 150K tokenizations). The line itself must still
            // be consumed to keep the stream in sync with assignments.
            if (p_idx >= skip_count) {
                pp.tokens = tokenize(vocab, p_line, /*follow_vocab_default=*/true, /*no_bos=*/args.no_bos);
            }
            pp.assignments = std::move(assignment_read.assignments);
            // Empty tokens on a skipped prompt are expected (not tokenized);
            // the consumer's skip branch runs before the empty-tokens error.
            pp.skip = (p_idx < skip_count) ? false : (pp.tokens.empty() || ((int)pp.tokens.size() > n_ctx));

            {
                std::unique_lock<std::mutex> lk(pfq.mtx);
                // Backpressure: wait if queue is full before pushing.
                pfq.cv_space.wait(lk, [&]{ return pfq.queue.size() < MAX_PREFETCH || pfq.producer_done.load(); });
                if (pfq.producer_done.load()) return;  // consumer signaled abort
                pfq.queue.push(std::move(pp));
            }
            pfq.cv.notify_one();
        }
        // Detect I/O errors on the prompts stream. std::getline returns
        // false for both EOF and I/O errors; without this check, a disk failure
        // mid-file is indistinguishable from normal end-of-file, silently
        // truncating the run and exiting 0.
        if (prompts_fin.bad()) {
            fprintf(stderr, "Error: I/O error reading prompts file (bad stream)\n");
            PrefetchedPrompt ep;
            ep.error = true;
            {
                std::lock_guard<std::mutex> lk(pfq.mtx);
                pfq.producer_done = true;
                try { pfq.queue.push(std::move(ep)); } catch (...) {}
            }
            pfq.cv.notify_all();
            pfq.cv_space.notify_all();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(pfq.mtx);
            pfq.producer_done = true;
        }
        pfq.cv.notify_all();
      } catch (const std::exception& e) {
        // An uncaught exception in the producer thread (e.g. std::bad_alloc from
        // tokenization, or a stream exception) would call std::terminate and abort
        // the whole process, leaving the consumer on the main thread hanging with
        // no diagnostic. Instead, surface a clean error to the consumer via an
        // error item so it exits gracefully (the consumer already handles pp.error).
        fprintf(stderr, "Error: producer thread exception: %s\n", e.what());
        // Set producer_done FIRST (no allocation needed), so the consumer
        // is guaranteed to wake even if the queue.push throws bad_alloc again.
        {
            std::lock_guard<std::mutex> lk(pfq.mtx);
            pfq.producer_done = true;
            try {
                PrefetchedPrompt ep;
                ep.error = true;
                pfq.queue.push(std::move(ep));
            } catch (...) { /* consumer will see producer_done + empty queue */ }
        }
        pfq.cv.notify_all();
        pfq.cv_space.notify_all();
      } catch (...) {
        fprintf(stderr, "Error: producer thread unknown exception\n");
        PrefetchedPrompt ep;
        ep.error = true;
        {
            std::lock_guard<std::mutex> lk(pfq.mtx);
            pfq.queue.push(std::move(ep));
            pfq.producer_done = true;
        }
        pfq.cv.notify_all();
        pfq.cv_space.notify_all();
      }
    };

    std::thread producer_thread(producer);

    // Shared per-prompt tail: bump counters, checkpoint on interval, print
    // progress on interval. Used by both the generation path and the
    // comprehension path so their bookkeeping stays identical. Returns false
    // if the checkpoint write failed (caller must stop with the producer
    // joined and return 1).
    auto after_prompt_done = [&]() -> bool {
        prompt_idx++;
        n_processed++;

        if (args.checkpoint_every > 0 && n_processed % args.checkpoint_every == 0) {
            if (!write_checkpoint(accumulators, args.output_path, n_embd, prompt_idx,
                                  make_fingerprint(args, target_layers, line_fnv, n_prompts_expected))) {
                fprintf(stderr, "Error: checkpoint write failed  -  aborting to prevent data loss\n");
                return false;
            }
        }

        if (n_processed % PROGRESS_INTERVAL == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            float pps = n_processed * 1000.0f / (elapsed > 0 ? elapsed : 1);
            if (skip_count > 0) {
                fprintf(stderr, "Position %d/%d (%d new since resume, %.2f prompts/sec)\n",
                        prompt_idx, n_prompts_expected, n_processed, pps);
            } else {
                fprintf(stderr, "Processed %d/%d prompts (%.2f prompts/sec)\n",
                        prompt_idx, n_prompts_expected, pps);
            }
        }
        return true;
    };

    // Consumer (main thread): GPU decode + extract + accumulate
    while (true) {
        PrefetchedPrompt pp;
        {
            std::unique_lock<std::mutex> lk(pfq.mtx);
            pfq.cv.wait(lk, [&]{ return !pfq.queue.empty() || pfq.producer_done; });
            if (pfq.queue.empty() && pfq.producer_done) break;
            pp = std::move(pfq.queue.front());
            pfq.queue.pop();
            // Notify producer that queue has space (backpressure release).
            pfq.cv_space.notify_one();
        }

        if (pp.error) {
            // Producer-side failure (assignment read, prompts-stream I/O, or
            // tokenization); the producer printed the precise diagnostic.
            fprintf(stderr, "Error: producer failed on prompt %d\n", pp.prompt_idx);
            stop_producer_and_join(pfq, producer_thread, records_temp_path);
            return 1;
        }

        // Skip already-processed prompts when resuming
        if (pp.prompt_idx < skip_count) {
            prompt_idx++;
            continue;
        }

        if (pp.skip) {
            if (pp.tokens.empty()) {
                fprintf(stderr, "Error: prompt %d tokenized to empty\n", pp.prompt_idx);
            } else {
                fprintf(stderr, "Error: prompt %d has %zu tokens, exceeds n_ctx=%d\n",
                        pp.prompt_idx, pp.tokens.size(), n_ctx);
            }
            stop_producer_and_join(pfq, producer_thread, records_temp_path);
            return 1;
        }

        const auto& assignments = pp.assignments;
        const auto& tokens = pp.tokens;

        // Profiling timers
        auto t_kv_start = std::chrono::steady_clock::now();

        // Clear KV cache
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) llama_memory_clear(mem, true);

        auto t_kv_end = std::chrono::steady_clock::now();

        // Fill pre-allocated batch buffer
        int n_tokens = (int)tokens.size();
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (args.generate_tokens > 0 && i == n_tokens - 1) ? 1 : 0;
            // In generation mode, enable logits on last prompt token for sampling
        }
        batch.n_tokens = n_tokens;

        // Decode (prefill)
        auto t_decode_start = std::chrono::steady_clock::now();
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "Error: decode failed for prompt %d (ret=%d)\n", prompt_idx, ret);
            stop_producer_and_join(pfq, producer_thread, records_temp_path);
            return 1;
        }

        llama_synchronize(ctx);
        auto t_decode_end = std::chrono::steady_clock::now();

        // The shared per-prompt tail (after_prompt_done) does the bookkeeping
        // (counters, checkpoint, progress); generation has no separate
        // extract/mean phases, so those accrue as 0 under --profile.
        if (args.generate_tokens > 0) {
            GenContext g{ctx, batch, args, assignments, prompt_idx, n_tokens,
                         n_embd, n_ctx, target_layers, layer_ptrs, mean_buf,
                         accumulators};
            if (generate_assignment(g) != 0) {
                stop_producer_and_join(pfq, producer_thread, records_temp_path);
                return 1;
            }
            line_fnv = pp.line_fnv;
            // Skip the normal comprehension-mode extraction below.
            // The shared tail does the bookkeeping (counters, checkpoint,
            // progress) identically to the comprehension path.
            // Profiling: kv/decode timers are shared with comprehension mode
            // (they wrap the common prologue + generation steps); extract and
            // mean have no separate phases here, so they accrue as 0. Count
            // the prompt so --profile still prints its breakdown in
            // generation mode instead of silently printing nothing.
            if (args.profile) {
                prof_total_kv     += std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
                prof_total_decode += std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
                prof_count++;
            }
            if (!after_prompt_done()) {
                stop_producer_and_join(pfq, producer_thread, records_temp_path);
                return 1;
            }
            continue;
        }

        // Phase 2: batch-fetch all layer pointers
        auto t_extract_start = std::chrono::steady_clock::now();
        if (llama_get_hidden_states_batch(ctx, target_layers.data(), target_layers.size(), layer_ptrs.data()) != 0) {
            fprintf(stderr, "Error: llama_get_hidden_states_batch failed\n");
            stop_producer_and_join(pfq, producer_thread, records_temp_path);
            return 1;
        }

        int n_hidden = llama_get_hidden_state_n_tokens(ctx);
        if (n_hidden <= 0) {
            fprintf(stderr, "Error: decode produced no hidden state tokens for prompt %d\n", pp.prompt_idx);
            stop_producer_and_join(pfq, producer_thread, records_temp_path);
            return 1;
        }
        auto t_extract_end = std::chrono::steady_clock::now();

        // For each assignment: compute masked mean per layer and accumulate
        auto t_mean_start = std::chrono::steady_clock::now();
        for (const auto& assign : assignments) {
            if (assign.group_id < 0 || assign.group_id > 0xFFFF) {
                fprintf(stderr, "Error: group_id %d exceeds 16-bit range\n", assign.group_id);
                stop_producer_and_join(pfq, producer_thread, records_temp_path);
                return 1;
            }
            if (assign.mask_id < 0 || assign.mask_id > 0xFFFF) {
                fprintf(stderr, "Error: mask_id %d exceeds 16-bit range\n", assign.mask_id);
                stop_producer_and_join(pfq, producer_thread, records_temp_path);
                return 1;
            }

            // mask_type 0 is the common case (single full-span range): reuse
            // the hoisted buffer with clear+push_back (no allocation).
            // mask_type 1 points at the assignment's own vector -- a const
            // pointer, no copy.
            const std::vector<std::pair<int,int>>* ranges_ptr;
            if (assign.mask_type == 0) {
                ranges_buf.clear();
                ranges_buf.push_back({assign.skip, n_hidden});
                ranges_ptr = &ranges_buf;
            } else {
                ranges_ptr = &assign.ranges;
            }

            for (size_t li = 0; li < target_layers.size(); li++) {
                float* data = layer_ptrs[li];
                if (!data) {
                    fprintf(stderr, "Error: null hidden state for layer %d\n", target_layers[li]);
                    stop_producer_and_join(pfq, producer_thread, records_temp_path);
                    return 1;
                }

                std::fill(mean_buf.begin(), mean_buf.end(), 0.0f);
                int64_t tokens_in_mask = compute_masked_mean(data, n_hidden, n_embd, *ranges_ptr, mean_buf.data());
                if (tokens_in_mask < 0) {
                    fprintf(stderr, "Error: compute_masked_mean failed for prompt %d\n", pp.prompt_idx);
                    stop_producer_and_join(pfq, producer_thread, records_temp_path);
                    return 1;
                }
                if (tokens_in_mask == 0) continue;

                uint64_t flat_key = make_accum_key(assign.group_id, assign.mask_id, target_layers[li]);
                auto& av = accumulators[flat_key];
                if (av.sum.empty()) av.sum.resize(n_embd, 0.0f);
                HS_SIMD
                for (int d = 0; d < n_embd; d++) av.sum[d] += mean_buf[d];
                av.count++;

                if (args.save_per_record && records_closer) {
                    FILE* records_fp = records_closer.fp;
                    if (!records_write(&prompt_idx, sizeof(int32_t), 1, records_fp, "prompt_idx")) { stop_producer_and_join(pfq, producer_thread, records_temp_path); return 1; }
                    int32_t gid = assign.group_id;
                    int32_t mid = assign.mask_id;
                    int32_t layer_idx = target_layers[li];
                    if (!records_write(&gid, sizeof(int32_t), 1, records_fp, "group_id")) { stop_producer_and_join(pfq, producer_thread, records_temp_path); return 1; }
                    if (!records_write(&mid, sizeof(int32_t), 1, records_fp, "mask_id")) { stop_producer_and_join(pfq, producer_thread, records_temp_path); return 1; }
                    if (!records_write(&layer_idx, sizeof(int32_t), 1, records_fp, "layer_idx")) { stop_producer_and_join(pfq, producer_thread, records_temp_path); return 1; }
                    if (!records_write(mean_buf.data(), sizeof(float), n_embd, records_fp, "hidden vector")) { stop_producer_and_join(pfq, producer_thread, records_temp_path); return 1; }
                }
            }
        }
        auto t_mean_end = std::chrono::steady_clock::now();
        line_fnv = pp.line_fnv;

        // Accumulate profiling data
        if (args.profile) {
            prof_total_kv     += std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
            prof_total_decode += std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
            prof_total_extract+= std::chrono::duration<double, std::milli>(t_extract_end - t_extract_start).count();
            prof_total_mean   += std::chrono::duration<double, std::milli>(t_mean_end - t_mean_start).count();
            prof_count++;
        }

        if (!after_prompt_done()) {
            stop_producer_and_join(pfq, producer_thread, records_temp_path);
            return 1;
        }
    }

    producer_thread.join();

    // -- Finalize --
    auto end_time = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Print profiling breakdown
    if (args.profile && prof_count > 0) {
        double wall_total = total_ms;
        double avg_kv      = prof_total_kv / prof_count;
        double avg_decode  = prof_total_decode / prof_count;
        double avg_extract = prof_total_extract / prof_count;
        double avg_mean    = prof_total_mean / prof_count;
        double avg_wall    = wall_total / prof_count;
        double measured    = avg_kv + avg_decode + avg_extract + avg_mean;
        double other       = avg_wall - measured;

        fprintf(stderr, "\n=== PROFILING BREAKDOWN (%d prompts) ===\n", prof_count);
        fprintf(stderr, "Phase                   Avg ms/prompt    %% of wall\n");
        fprintf(stderr, "-----------------------------------------------------\n");
        fprintf(stderr, "KV cache clear:         %8.2f ms    %5.1f%%\n", avg_kv,      avg_kv / avg_wall * 100);
        fprintf(stderr, "Decode + sync:          %8.2f ms    %5.1f%%\n", avg_decode,  avg_decode / avg_wall * 100);
        fprintf(stderr, "HS batch fetch:         %8.2f ms    %5.1f%%\n", avg_extract, avg_extract / avg_wall * 100);
        fprintf(stderr, "Mean + accumulate:      %8.2f ms    %5.1f%%\n", avg_mean,    avg_mean / avg_wall * 100);
        fprintf(stderr, "-----------------------------------------------------\n");
        fprintf(stderr, "Measured total:         %8.2f ms    %5.1f%%\n", measured, measured / avg_wall * 100);
        fprintf(stderr, "Other (overhead):       %8.2f ms    %5.1f%%\n", other,    other / avg_wall * 100);
        fprintf(stderr, "Wall per prompt:        %8.2f ms\n", avg_wall);
        fprintf(stderr, "===============================================\n\n");
    }

    // TOCTOU guard: the producer must have consumed every prompt line. On a
    // fresh run (skip_count == 0) the counts must match exactly; on a resume
    // run the final position must equal the assignments header count. A
    // mismatch means the prompts file was modified between the count pass and
    // the producer read (or lines vanished mid-run) -- the output would be
    // silently truncated, so refuse it.
    if (prompt_idx != n_prompts_expected) {
        fprintf(stderr, "Error: producer stopped at position %d but assignments expected %d "
                        "prompts (prompts file may have been modified during extraction)\n",
                prompt_idx, n_prompts_expected);
        if (!records_temp_path.empty()) std::remove(records_temp_path.c_str());
        return 1;
    }

    if (skip_count > 0) {
        fprintf(stderr, "Done. Position %d/%d (%d new since resume) in %.2f seconds (%.2f prompts/sec)\n",
                prompt_idx, n_prompts_expected, n_processed,
                total_ms / 1000.0,
                n_processed * 1000.0 / (total_ms > 0 ? total_ms : 1));
    } else {
        fprintf(stderr, "Done. Processed %d prompts in %.2f seconds (%.2f prompts/sec)\n",
                n_processed, total_ms / 1000.0,
                n_processed * 1000.0 / (total_ms > 0 ? total_ms : 1));
    }

    // Guard: if we processed prompts but produced no accumulators (e.g. every
    // prompt hit EOS before gen_step reached token_skip, or all masked means
    // were empty), refuse to write a header-only output.bin and exit non-zero.
    // A silent empty file would be misread downstream as a valid-but-empty
    // result rather than flagged as a failed extraction.
    if (n_processed > 0 && accumulators.empty()) {
        fprintf(stderr, "Error: processed %d prompts but accumulated no hidden-state "
                        "means (check token_skip / EOS / masks); refusing to write an "
                        "empty output.bin\n", n_processed);
        // Clean up the per-record temp file on this error path.
        if (!records_temp_path.empty()) std::remove(records_temp_path.c_str());
        return 1;
    }

    // Write output
    if (!write_batch_output(accumulators, args.output_path, n_embd)) {
        // write_batch_output failed (its own temp was already cleaned inside
        // the function); remove the per-record temp file here.
        if (!records_temp_path.empty()) std::remove(records_temp_path.c_str());
        return 1;
    }

    // Per-record sidecar: close the temp file and atomically rename to the final
    // path now that the run fully succeeded.
    if (records_closer) {
        if (!records_closer.sync()) {  // flush + fsync: check for failures
            fprintf(stderr, "Error: sync failed for %s\n", records_temp_path.c_str());
            records_closer.reset();
            std::remove(records_temp_path.c_str());
            return 1;
        }
        records_closer.reset();  // close before rename
        if (rename(records_temp_path.c_str(), records_path.c_str()) != 0) {
            fprintf(stderr, "Error: cannot rename %s to %s\n",
                    records_temp_path.c_str(), records_path.c_str());
            // rename failed: remove the orphaned temp file.
            std::remove(records_temp_path.c_str());
            return 1;
        }
        if (!fsync_parent_dir(records_path.c_str())) {  // make the rename itself durable
            return 1;
        }
        fprintf(stderr, "Per-record output complete: %s\n", records_path.c_str());
    }

    // Delete checkpoint on successful completion
    std::string ckpt_path = std::string(args.output_path) + ".checkpoint";
    // Check remove() result: a stale checkpoint surviving a successful run
    // would cause a redundant --resume. Warn (not abort) since data is fine.
    if (remove(ckpt_path.c_str()) != 0 && errno != ENOENT) {
        fprintf(stderr, "Warning: could not remove checkpoint %s (run completed successfully; "
                        "delete it manually to avoid a redundant --resume)\n", ckpt_path.c_str());
    }

    return 0;
}

// -- Main ----------------------------------------------------------------

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

    // Should not reach here -- parse_args exits with usage if no mode set
    fprintf(stderr, "Error: must specify --batch, --raw, or --self-test\n");
    return 1;
}
