# CrimsonRed llama.cpp Fork

This fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) adds
hidden-state extraction capabilities for the CrimsonRed emotion probe pipeline.

## Fork State

- **Branch:** `main` (work directly on main, no feature branches)
- **Upstream tracking:** merge-base `adb55e5148` (2026-08-14 rebase; 4 upstream commits landed since, no overlap with fork changes)
- **Fork delta:** 53 files, +7,013/-36 vs merge-base

## Features

### 1. Hidden-State Extraction API (Core Library)

Adds per-layer residual stream activation extraction to the llama.cpp compute graph:

- **`llama_context_params.extract_hidden_states`**  -  flag to enable extraction
- **`llama_set_extract_hidden_states()`**  -  runtime toggle
- **`llama_get_hidden_state(ctx, layer)`**  -  get all tokens for a layer (row i = `ptr + i * n_embd`)
- **`llama_get_hidden_state_n_tokens(ctx)`**  -  token count
- **`llama_get_hidden_states_batch()`**  -  batch-get multiple layers in one call

Model patches push the residual stream output of each decoder layer into
`t_hidden_layers` when extraction is enabled. Supported architectures:
- Gemma (gemma.cpp)
- Gemma 4 (gemma4.cpp)
- Llama (llama.cpp)
- Qwen3.5 (qwen35.cpp)

### 2. Output-Projection Interaction

Extraction does not modify the output path: logits are computed as upstream
computes them. The last-layer `ggml_get_rows` row-selection on
`inp_out_ids` is suppressed only while extraction is enabled (llama/gemma
model files), so per-layer tensors carry all tokens; the output projection
itself runs unchanged.

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
- Self-test mode (17 tests, no model required)
- `--profile` flag for per-step timing analysis
- Hard-error semantics on all range violations (no silent clamping, no graceful degradation)

### 5. Single-Prompt CLI (`hs-extract`)

Debug/parity tool for extracting hidden states from a single prompt with JSON output.

### 6. KV-Cells Optimization

Flat position-count table replacing `std::map` for O(1) lookup.

## CI

The fork CI (`.github/workflows/fork-ci.yml`) runs on CPU-only runners:
- Builds with `GGML_NATIVE=ON` (required - CI runners have AVX)
- Runs self-test (17/17)
- Runs multi-ubatch pool=none integration test
- 17 structural integrity checks (RAII wrappers, shared header, backpressure, pool=none size limit, checkpoint v2, no raw fclose, checkpoint bounds, producer-consumer pipeline)

GPU verification (CUDA + Vulkan) is manual  -  see CI header comments for commands.

## Documentation

- `tools/hs-extract-batch/README.md`  -  batch extraction tool
- `tools/hs-extract/README.md`  -  single-prompt tool
- `tools/server/README.md`  -  `/hidden-states` endpoint (section 901+)
- `docs/history/`  -  historical audit reports
