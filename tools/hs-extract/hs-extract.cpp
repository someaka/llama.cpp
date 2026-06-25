#include "common.h"
#include "llama.h"
#include "build-info.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

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
    printf("  -t, --threads N           number of threads (default: 4)\n");
    printf("  -ngl, --n-gpu-layers N    number of GPU layers to offload (default: 0)\n");
    printf("  --output FILE             output JSON file (default: stdout)\n");
    printf("\nexamples:\n");
    printf("  %s -m model.gguf -p \"Hello world\"\n", prog);
    printf("  %s -m model.gguf -p \"Hello world\" --layers 0,5,10\n", prog);
    printf("  %s -m model.gguf --raw -p \"1,2,3,4\"\n", prog);
}

static std::vector<int> parse_layer_list(const char * str, int n_layers) {
    std::vector<int> layers;
    if (strcmp(str, "all") == 0) {
        for (int i = 0; i <= n_layers; i++) {
            layers.push_back(i);
        }
        return layers;
    }
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        int val = std::stoi(item);
        if (val < 0 || val > n_layers) {
            fprintf(stderr, "error: layer %d out of range [0, %d]\n", val, n_layers);
            exit(1);
        }
        layers.push_back(val);
    }
    return layers;
}

static std::vector<llama_token> parse_raw_tokens(const char * str) {
    std::vector<llama_token> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        item = item.substr(start, end - start + 1);
        tokens.push_back((llama_token) std::stoi(item));
    }
    return tokens;
}

static std::string format_float(float v) {
    char buf[64];
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
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) { fprintf(stderr, "error: --threads requires an argument\n"); return 1; }
            n_threads = std::stoi(argv[i]);
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) { fprintf(stderr, "error: --n-gpu-layers requires an argument\n"); return 1; }
            n_gpu_layers = std::stoi(argv[i]);
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

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path);
        llama_backend_free();
        return 1;
    }

    const int n_layers = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);

    fprintf(stderr, "%s: model loaded, n_layers=%d, n_embd=%d\n", __func__, n_layers, n_embd);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.extract_hidden_states = true;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens;

    if (raw_mode) {
        tokens = parse_raw_tokens(prompt_text);
        fprintf(stderr, "%s: parsed %zu raw tokens\n", __func__, tokens.size());
    } else {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab) && !no_bos;
        std::string prompt(prompt_text);
        tokens = common_tokenize(vocab, prompt, add_bos, true);
        fprintf(stderr, "%s: tokenized prompt into %zu tokens\n", __func__, tokens.size());
    }

    if (tokens.empty()) {
        fprintf(stderr, "error: no tokens to process\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    int32_t decode_result = llama_decode(ctx, batch);
    if (decode_result != 0) {
        fprintf(stderr, "error: llama_decode failed with code %d\n", decode_result);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int32_t n_tokens_out = llama_get_hidden_state_n_tokens(ctx);
    fprintf(stderr, "%s: decoded %d tokens, extracting hidden states\n", __func__, n_tokens_out);

    std::vector<int> layers = parse_layer_list(layer_str, n_layers);

    std::ostringstream json;
    json << "{\n";
    json << "  \"n_tokens\": " << n_tokens_out << ",\n";
    json << "  \"n_embd\": " << n_embd << ",\n";
    json << "  \"n_layers\": " << n_layers << ",\n";
    json << "  \"layers\": [\n";

    for (size_t li = 0; li < layers.size(); li++) {
        int layer = layers[li];
        float * hs = llama_get_hidden_state(ctx, layer);

        json << "    {\n";
        json << "      \"layer\": " << layer << ",\n";

        if (!hs) {
            fprintf(stderr, "error: llama_get_hidden_state returned NULL for layer %d\n", layer);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        } else {
            json << "      \"values\": [";
            for (int32_t i = 0; i < n_tokens_out * n_embd; i++) {
                if (i > 0) json << ", ";
                if (i > 0 && (i % n_embd) == 0) {
                    json << "\n              ";
                }
                json << format_float(hs[i]);
            }
            json << "]\n";
        }

        json << "    }";
        if (li + 1 < layers.size()) json << ",";
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";

    if (output_file) {
        std::ofstream out(output_file);
        if (!out) {
            fprintf(stderr, "error: could not open output file '%s'\n", output_file);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        out << json.str();
        fprintf(stderr, "%s: wrote output to '%s'\n", __func__, output_file);
    } else {
        printf("%s", json.str().c_str());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
