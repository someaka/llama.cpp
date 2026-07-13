# CrimsonRed llama.cpp Fork

This fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) adds
hidden-state extraction capabilities for the CrimsonRed emotion probe pipeline.

## Fork State

- **Branch:** `main` (work directly on main, no feature branches)
- **Upstream tracking:** 0 commits behind upstream `master`
- **Unique commits:** 159 (including merges)
- **Modified files:** 27 (3,679 insertions, 36 deletions)

## Features

### 1. Hidden-State Extraction API (Core Library)

Adds per-layer residual stream activation extraction to the llama.cpp compute graph:

- **`llama_context_params.extract_hidden_states`** — flag to enable extraction
- **`llama_set_extract_hidden_states()`** — runtime toggle
- **`llama_get_hidden_state(ctx, layer)`** — get all tokens for a layer
- **`llama_get_hidden_state_ith(ctx, layer, i)`** — get specific token (supports negative indexing)
- **`llama_get_hidden_state_n_tokens(ctx)`** — token count
- **`llama_get_hidden_states_batch()`** — batch-get multiple layers in one call

Model patches push the residual stream output of each decoder layer into
`t_hidden_layers` when extraction is enabled. Supported architectures:
- Gemma (gemma.cpp)
- Gemma 4 (gemma4.cpp)
- Llama (llama.cpp)
- Qwen3.5 (qwen35.cpp)

### 2. Graph Optimization: Skip lm_head

When `extract_hidden_states` is true and no samplers/embeddings are needed,
the lm_head output projection is pruned from the compute graph via
`ggml_set_output(t_logits)` skipping. This gives ~1.89× speedup for
extraction-only workloads.

### 3. `/hidden-states` HTTP Endpoint (Server)

New POST endpoint on llama-server for extracting hidden states via HTTP:

```bash
curl -X POST http://localhost:8080/hidden-states \
  -H "Content-Type: application/json" \
  -d '{"input":"text to analyze","layers":[0,5,10],"pool":"skip_mean","skip_offset":50}'
```

Pooling modes: `last` (last token), `skip_mean` (masked mean), `none` (per-token).

The `--no-hidden-states` flag disables the endpoint and enables upstream
output-allocation optimization for pure-generation workloads.

### 4. Batch Extraction CLI (`hs-extract-batch`)

Production tool for processing thousands of prompts:

- Async double-buffered pipeline (CPU masked-mean overlaps GPU decode)
- Checkpoint/resume support
- Binary I/O format (CRD2)
- Self-test mode (9 tests, no model required)
- `--profile` flag for per-step timing analysis
- Three-tier error severity (hard error / visible clamp / silent skip)

### 5. Single-Prompt CLI (`hs-extract`)

Debug/parity tool for extracting hidden states from a single prompt with JSON output.

### 6. KV-Cells Optimization

Flat position-count table replacing `std::map` for O(1) lookup.

## CI

The fork CI (`.github/workflows/fork-ci.yml`) runs on CPU-only runners:
- Builds with `GGML_NATIVE=ON` (required — CI runners have AVX)
- Runs self-test (9/9)
- Runs multi-ubatch pool=none integration test
- 11 structural integrity checks (RAII wrappers, no raw fclose, checkpoint bounds)

GPU verification (CUDA + Vulkan) is manual — see CI header comments for commands.

## Documentation

- `tools/hs-extract-batch/README.md` — batch extraction tool
- `tools/hs-extract/README.md` — single-prompt tool
- `tools/server/README.md` — `/hidden-states` endpoint (section 901+)
- `docs/history/` — historical audit reports
