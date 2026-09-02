#include "common.h"
#include "llama.h"
#include "build-info.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

// RAII wrappers are in common/llama-raii.h (shared with hs-extract-batch, tests)
#include "llama-raii.h"
// Shared layer-list parser (same file as hs-extract-batch uses)
#include "layer-parse.h"

static void print_usage(const char * prog) {
    printf("usage: %s [options]\n\n", prog);
    printf("Extract per-layer hidden states from a model for a given prompt.\n\n");
    printf("options:\n");
    printf("  -h, --help                show this help and exit\n");
    printf("  -m, --model PATH          path to model file (required)\n");
    printf("  -p, --prompt TEXT         input prompt text\n");
    printf("  --raw                     interpret --prompt as comma-separated token IDs\n");
    printf("  -l, --layers LIST         comma-separated layer indices or 'all' (default: all)\n");
    printf("  -f, --file PATH           read prompt from file\n");
    printf("  --no-bos                  do not add BOS token\n");
    printf("  -c, --ctx-size N          context size (default: auto = token count, capped at n_ctx_train)\n");
    printf("  -t, --threads N           number of threads (default: 4)\n");
    printf("  -ngl, --n-gpu-layers N    number of GPU layers to offload (default: 0)\n");
    printf("  --output FILE             output JSON file (default: stdout)\n");
    printf("\nexamples:\n");
    printf("  %s -m model.gguf -p \"Hello world\"\n", prog);
    printf("  %s -m model.gguf -p \"Hello world\" --layers 0,5,10\n", prog);
    printf("  %s -m model.gguf --raw -p \"1,2,3,4\"\n", prog);
}

static std::vector<llama_token> parse_raw_tokens(const char * str, const llama_vocab * vocab) {
    std::vector<llama_token> tokens;
    std::stringstream ss(str);
    std::string item;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    while (std::getline(ss, item, ',')) {
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        item = item.substr(start, end - start + 1);
        char *endp = nullptr;
        long val = strtol(item.c_str(), &endp, 10);
        if (endp == item.c_str() || *endp != '\0') {
            fprintf(stderr, "error: invalid token '%s' (not a number)\n", item.c_str());
            return {};
        }
        if (val < 0) {
            fprintf(stderr, "error: invalid token id %ld (must be non-negative)\n", val);
            return {};
        }
        // Bounds-check against the vocabulary size. An out-of-range id passed to
        // the embedding lookup (ggml_get_rows) is an out-of-bounds GPU read, since
        // the lookup does not validate token indices. The batch tool has this check
        // (hs-extract-batch.cpp); mirror it here.
        if (val >= n_vocab) {
            fprintf(stderr, "error: token id %ld out of range [0, %d)\n", val, n_vocab);
            return {};
        }
        tokens.push_back((llama_token) val);
    }
    return tokens;
}

static std::string format_float(float v) {
    char buf[64];
    if (std::isinf(v)) { return "null"; }  // JSON has no Inf/NaN
    if (std::isnan(v)) { return "null"; }
    snprintf(buf, sizeof(buf), "%.8g", v);
    return std::string(buf);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * model_path = nullptr;
    const char * prompt_text = nullptr;
    const char * prompt_file = nullptr;
    const char * layer_str = "all";
    const char * output_file = nullptr;
    bool raw_mode = false;
    bool no_bos = false;
    int n_threads = 4;
    int n_gpu_layers = 0;
    int ctx_size = 0;  // 0 = auto: size to the token count, capped at n_ctx_train

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "error: --model requires an argument\n"); return 1; }
            model_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) { fprintf(stderr, "error: --prompt requires an argument\n"); return 1; }
            prompt_text = argv[i];
        } else if (arg == "--raw") {
            raw_mode = true;
        } else if (arg == "-l" || arg == "--layers") {
            if (++i >= argc) { fprintf(stderr, "error: --layers requires an argument\n"); return 1; }
            layer_str = argv[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) { fprintf(stderr, "error: --file requires an argument\n"); return 1; }
            prompt_file = argv[i];
        } else if (arg == "--no-bos") {
            no_bos = true;
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i >= argc) { fprintf(stderr, "error: --ctx-size requires an argument\n"); return 1; }
            char *endptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0') { fprintf(stderr, "error: --ctx-size value must be a number\n"); return 1; }
            if (errno == ERANGE) { fprintf(stderr, "error: --ctx-size value out of range\n"); return 1; }
            if (val < 1) { fprintf(stderr, "error: --ctx-size must be >= 1\n"); return 1; }
            ctx_size = (int) val;
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) { fprintf(stderr, "error: --threads requires an argument\n"); return 1; }
            char *endptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0') { fprintf(stderr, "error: --threads value must be a number\n"); return 1; }
            if (errno == ERANGE) { fprintf(stderr, "error: --threads value out of range\n"); return 1; }
            if (val < 1) { fprintf(stderr, "error: --threads must be >= 1\n"); return 1; }
            n_threads = (int) val;
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) { fprintf(stderr, "error: --n-gpu-layers requires an argument\n"); return 1; }
            char *endptr;
            errno = 0;
            long val = strtol(argv[i], &endptr, 10);
            if (*endptr != '\0') { fprintf(stderr, "error: --n-gpu-layers value must be a number\n"); return 1; }
            if (errno == ERANGE) { fprintf(stderr, "error: --n-gpu-layers value out of range\n"); return 1; }
            if (val < 0) { fprintf(stderr, "error: --n-gpu-layers must be >= 0\n"); return 1; }
            n_gpu_layers = (int) val;
        } else if (arg == "--output") {
            if (++i >= argc) { fprintf(stderr, "error: --output requires an argument\n"); return 1; }
            output_file = argv[i];
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!prompt_text && !prompt_file) {
        fprintf(stderr, "error: must specify --prompt or --file\n");
        print_usage(argv[0]);
        return 1;
    }

    if (prompt_text && prompt_file) {
        fprintf(stderr, "error: --prompt and --file are mutually exclusive\n");
        return 1;
    }

    std::string prompt_str;
    if (prompt_file) {
        std::ifstream in(prompt_file, std::ios::binary);
        if (!in) {
            fprintf(stderr, "error: could not open file '%s'\n", prompt_file);
            return 1;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        prompt_str = buf.str();
        prompt_text = prompt_str.c_str();
    }

    LlamaBackend backend;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    LlamaModel model(llama_model_load_from_file(model_path, model_params));
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path);
        return 1;
    }

    const int n_layers = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd_out(model);

    if (n_embd <= 0) {
        fprintf(stderr, "error: invalid model n_embd=%d (model corrupt or unsupported)\n", n_embd);
        return 1;
    }

    fprintf(stderr, "%s: model loaded, n_layers=%d, n_embd=%d\n", __func__, n_layers, n_embd);

    // Tokenize FIRST (needs only the vocab): the context is then sized to the
    // actual token count instead of a hardcoded 2048.
    std::vector<llama_token> tokens;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (raw_mode) {
        tokens = parse_raw_tokens(prompt_text, vocab);
        if (tokens.empty()) {
            return 1;
        }
        fprintf(stderr, "%s: parsed %zu raw tokens\n", __func__, tokens.size());
    } else {
        const bool add_bos = llama_vocab_get_add_bos(vocab) && !no_bos;
        std::string prompt(prompt_text);
        tokens = common_tokenize(vocab, prompt, add_bos, true);
        fprintf(stderr, "%s: tokenized prompt into %zu tokens\n", __func__, tokens.size());
    }

    if (tokens.empty()) {
        fprintf(stderr, "error: no tokens to process\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    // Size the context to the work, not a hardcoded 2048: auto = token count
    // (+ headroom zero: decode is single-shot), capped at n_ctx_train; or the
    // user's --ctx-size. A prompt longer than n_ctx_train is rejected upfront
    // (a too-small ctx would fail mid-decode with no pointer to the cause).
    {
        const int64_t n_ctx_train64 = llama_model_n_ctx_train(model);
        int32_t ctx_n;
        if (ctx_size > 0) {
            ctx_n = ctx_size;
        } else {
            ctx_n = (int32_t) tokens.size();
        }
        if ((int64_t) ctx_n > n_ctx_train64) {
            if (ctx_size == 0) {
                ctx_n = (int32_t) n_ctx_train64;  // auto: cap, do not reject
            } else if ((int64_t) tokens.size() <= n_ctx_train64) {
                // user asked for more ctx than needed-in-principle: allow, cap
                ctx_n = (int32_t) n_ctx_train64;
            } else {
                fprintf(stderr, "error: prompt is %zu tokens but the model's training context is only %lld tokens; use a shorter prompt\n",
                        tokens.size(), (long long) n_ctx_train64);
                return 1;
            }
        }
        if ((int64_t) tokens.size() > (int64_t) ctx_n) {
            fprintf(stderr, "error: prompt is %zu tokens but context is %d tokens; pass a larger --ctx-size or a shorter prompt\n",
                    tokens.size(), ctx_n);
            return 1;
        }
        ctx_params.n_ctx = (uint32_t) ctx_n;
        ctx_params.n_batch = (uint32_t) ctx_n;
        // Match hs-extract-batch: one ubatch per prompt keeps the full prompt
        // in a single pass (stable capture path, same code path as batch).
        ctx_params.n_ubatch = (uint32_t) ctx_n;
        fprintf(stderr, "%s: using n_ctx=%d for %zu tokens\n", __func__, ctx_n, tokens.size());
    }
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.extract_hidden_states = true;
    ctx_params.no_perf = true;

    LlamaContext ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    int32_t decode_result = llama_decode(ctx, batch);
    if (decode_result != 0) {
        fprintf(stderr, "error: llama_decode failed with code %d\n", decode_result);
        return 1;
    }

    // The getters synchronize the context before returning data
    // (see llama_get_hidden_state); no explicit sync is needed here.
    const int32_t n_tokens_out = llama_get_hidden_state_n_tokens(ctx);
    fprintf(stderr, "%s: decoded %d tokens, extracting hidden states\n", __func__, n_tokens_out);

    if (n_tokens_out <= 0) {
        fprintf(stderr, "error: invalid hidden state token count (%d)\n", n_tokens_out);
        return 1;
    }

    std::vector<int> layers = hs_parse_layer_list(layer_str, n_layers + 1);
    if (layers.empty()) {
        // Q-P1-6: silent exit-1 gave the caller nothing to act on; match the
        // batch tool's diagnostic shape (hs-extract-batch parse_layers).
        fprintf(stderr, "error: no valid layers in '%s'\n", layer_str);
        return 1;
    }

    // Stream the JSON directly (no ostringstream buffer): "all" layers with a
    // long prompt (e.g. 2048 tokens x 2560 dims) would otherwise accumulate
    // GB-scale text in RAM before the first write. When --output is given the
    // stream targets a .tmp file renamed into place only after a verified
    // flush, so a disk-full/EIO mid-write never leaves a truncated JSON at
    // the final path (same durability contract as hs-extract-batch's writers).
    std::ofstream file_out;
    std::string tmp_path;
    if (output_file) {
        tmp_path = std::string(output_file) + ".tmp";
        file_out.open(tmp_path);
        if (!file_out) {
            fprintf(stderr, "error: could not open output file '%s'\n", tmp_path.c_str());
            return 1;
        }
    }
    std::ostream& json = output_file ? static_cast<std::ostream&>(file_out) : std::cout;
    json << "{\n";
    json << "  \"n_tokens\": " << n_tokens_out << ",\n";
    json << "  \"n_embd\": " << n_embd << ",\n";
    json << "  \"n_layers\": " << n_layers << ",\n";
    json << "  \"layers\": [\n";

    for (size_t li = 0; li < layers.size(); li++) {
        int layer = layers[li];
        float * hs = llama_get_hidden_state(ctx, layer);

        if (!hs) {
            fprintf(stderr, "error: llama_get_hidden_state returned NULL for layer %d\n", layer);
            // Streaming opens the temp file before extraction; a failure
            // here must not leave an empty/truncated JSON behind.
            if (output_file) {
                file_out.close();
                std::remove(tmp_path.c_str());
            }
            return 1;
        }

        json << "    {\n";
        json << "      \"layer\": " << layer << ",\n";
        json << "      \"values\": [";
        for (size_t i = 0; i < (size_t)n_tokens_out * (size_t)n_embd; i++) {
            if (i > 0) json << ", ";
            if (i > 0 && (i % n_embd) == 0) {
                json << "\n              ";
            }
            json << format_float(hs[i]);
        }
        json << "]\n";

        json << "    }";
        if (li + 1 < layers.size()) json << ",";
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";
    json.flush();
    if (!json) {
        fprintf(stderr, "error: write/flush of output '%s' failed\n",
                output_file ? tmp_path.c_str() : "stdout");
        if (output_file) {
            file_out.close();
            std::remove(tmp_path.c_str());
        }
        return 1;
    }
    if (output_file) {
        file_out.close();
        // Atomic finalize: the final path appears only when the write is
        // complete, so a crash mid-write can never leave a truncated JSON.
        if (std::rename(tmp_path.c_str(), output_file) != 0) {
            fprintf(stderr, "error: cannot rename %s to %s\n", tmp_path.c_str(), output_file);
            std::remove(tmp_path.c_str());
            return 1;
        }
        fprintf(stderr, "%s: wrote output to '%s'\n", __func__, output_file);
    }

    return 0;
}
