# llama.cpp Fork — Hidden-State Extraction

This fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) adds
capture-grade hidden-state extraction to core decode, the HTTP server, and
dedicated extraction tools, with a byte-identity regression gate. The
extraction layer is domain-neutral.

## Fork State

- **Branch:** `main` (work directly on main, no feature branches)
- Upstream state and the fork-vs-upstream delta are live git data, not doc
  content: `git fetch origin && git log --oneline origin/master..main`
  (commits) or `git diff --stat $(git merge-base main origin/master)..main`
  (files). This file describes what the fork contains, not repo statistics.

## Placement doctrine

Rule: if upstream could plausibly merge it, it lives at an upstream-shaped
seam; if it exists only because this is our fork, it lives in a fork-owned
path. Upstream-shaped: public API in `include/llama.h`, endpoint logic in
its own `server-*.cpp` split file, tests in `tests/`, sample data beside
its tool. Fork-owned paths: `tools/hs-extract-*/gate/`, `docs/history/`,
`.github/workflows/fork-ci.yml`. Fork code edits upstream files only at
minimal, documented seams; bulk logic goes in fork-owned TUs so future
merges touch anchors, not blocks. Never place in upstream files:
machine-lineage baselines, cross-repo paths, private artifact names. Live
docs cite dates + lessons, never paths. Examples: (1) a tool's test lives
in `tests/` (upstream shape), not `gate/` (fork-private); (2)
`digests_baseline.json` lives in `gate/` — machine-lineage bytes upstream
can't reproduce; (3) an incident report lives in `docs/history/`; the live
README keeps the date and the lesson, never the path.

## Features

### 1. Hidden-State Extraction API (Core Library)

Adds per-layer residual stream activation extraction to the llama.cpp compute graph:

- **`llama_context_params.extract_hidden_states`**  -  flag to enable extraction
- **`llama_set_extract_hidden_states()`**  -  runtime toggle
- **`llama_get_hidden_state(ctx, layer)`**  -  get all tokens for a layer (row i = `ptr + i * n_embd`)
- **`llama_get_hidden_state_n_tokens(ctx)`**  -  token count
- **`llama_get_hidden_states_batch()`**  -  batch-get multiple layers in one call
- **`llama_model_supports_hidden_states(model)`**  -  capability registry query (single source of truth in `llm_arch_supports_hidden_states()`)
- **`llama_model_arch_name(model)`**  -  architecture name string; used to name the architecture in extraction-refusal errors (context creation, runtime setter, server 400)

Model builders push the residual stream output of each decoder layer into
`t_hidden_layers` through the `capture_layer_output()` graph helper
(`src/llama-graph.cpp`); the adopted builders also call `capture_embeddings()`
to fill index 0 with the state entering block 0.

Supported architectures (capability registry,
`llm_arch_supports_hidden_states()` in `src/llama-arch.cpp`):
- Gemma (gemma.cpp)
- Gemma 4 (gemma4.cpp)
- Llama (llama.cpp)
- Qwen3.5 (qwen35.cpp)

Beyond the registry, the capture tap is already sewn into the layer loops of
the classic builders across `src/models/`. These taps are dormant: the helper
returns immediately unless extraction is enabled, and the capability registry
gates activation - context creation and the runtime setter refuse the flag
with an error naming the architecture, and the server answers unsupported
architectures with a 400-class error. Nothing is captured, and behavior is
unchanged, until an architecture joins the registry. Adoption is a two-change
recipe (builder call + registry case; for builders already carrying the tap,
only the registry case remains), classified per builder in
`docs/hidden-states-adoption-manifest.md`. The capture mechanism and the
layer index convention are documented in `docs/hidden-states-architecture.md`.
The live extent of the sweep is repo data, not doc content:
`grep -l capture_layer_output src/models/*.cpp`.

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
  -d '{"input":"text to analyze","layers":[0,5,10],"pool":"skip_mean","skip_offset":2}'

```

Pooling modes: `last` (last token), `skip_mean` (masked mean), `none` (per-token).
`input` may be a single string or an array of strings (batched; the response is
a JSON array echoing each item's `index`). `normalize: true` L2-normalizes each
pooled output vector (rejected with 400 when combined with `pool: "none"`).

Per-request limit: the input must fit in one decode call — more than `n_batch`
tokens (the server's `-b` value; upstream default 2048) is refused with 400 (`raise -b or shorten the
input`). This is a hard correctness boundary: the capture resets per decode
call, so a split prompt would silently return only its tail.

The `--no-hidden-states` flag disables the endpoint. It is a kill-switch:
nothing else in the server reads it.

### 4. Batch Extraction CLI (`hs-extract-batch`)

Production tool for processing thousands of prompts:

- Async double-buffered pipeline (CPU masked-mean overlaps GPU decode)
- Checkpoint/resume support
- Binary I/O format (magic 0x43524432, "binary accumulator format v2")
- Self-test mode (29 checks: 1-16 kernels, 16b/16c FNV vectors, 17 checkpoint fixture, 19b payload-flip reject, 18-21
  depend on 17's checkpoint write - if the fixture write fails the self-test fails rc=1
  with the attempted count dropping accordingly; 22 CRD2 output round-trip
  (mean = sum/count), 23 CRD1 reader status contract + exact EOF, 23b negative-skip
  parse-site rejection, 24 legacy v1 checkpoint restore, 25 shared layer-list
  parser acceptance/rejection table; no model required)
- `--profile` flag for per-step timing analysis
- Hard-error semantics on all range violations (no silent clamping, no graceful degradation)

### 5. Single-Prompt CLI (`hs-extract`)

Debug/parity tool for extracting hidden states from a single prompt with JSON output.

### 6. Test/Diagnostic Tools

- `tools/hs-extract-batch-test`  -  warmup/toggle equivalence check: mode 2
  (the CI mode) runs both init paths in one process — CLI-style (extract on
  from creation) and server-style (embeddings off + extract off at creation,
  BOS+EOS warmup decode, toggle on, perf reset, memory clear) — and
  compares the captured hidden states bitwise; exit 2 on any mismatch.
  Modes 0/1 print values for manual debugging.
- `tools/hs-probe`  -  logits-equivalence probe: greedy next-token argmax +
  top-8 logit value sets compared with extraction on vs off; exits 2 on
  mismatch (regression gate for the output-projection pruning fix).
  `--runtime-toggle` additionally probes the server-style path (toggle
  enabled after a warmup decode, memory cleared between passes). Runs in
  fork CI (both modes).
- `examples/hidden-states`  -  minimal public-API example (installed like
  upstream siblings).
- `tools/cvector-generator/{positive,negative}-gemma4.txt`  -  sample
  conversation pairs in Gemma chat format for the cvector generator's
  `--positive-file` / `--negative-file` options (the upstream samples are
  llama-format; these exercise the same tool against a gemma4 model).
  Data only: not referenced by any build or test target.
- `tests/test-hidden-states.cpp` / `tests/test-hidden-states.c`  -  model
  exercising tests for the public extraction API in both C++ and C (run in
  fork CI against a real GGUF).

### 7. Shared RAII Header

- `common/llama-raii.h`  -  RAII wrappers (`LlamaBackend`, `LlamaModel`,
  `LlamaContext`, `LlamaBatch`, `LlamaSampler`) shared by hs-extract,
  hs-extract-batch, hs-extract-batch-test's sibling tests
  (tests/test-hidden-states.cpp) and examples/hidden-states instead of
  per-tool copies.

## CI

The fork CI (`.github/workflows/fork-ci.yml`) runs on CPU-only runners:
- Builds with `GGML_NATIVE=OFF` (portable binaries for the CI matrix; no
  host-specific ISA assumptions)
- Runs self-test (full suite; the count is computed from the binary's own
  "N/N tests passed" line, never hardcoded here — a checkpoint-fixture write
  failure fails the run rc=1
  rather than skipping)
- Runs multi-ubatch pool=none integration test (hard row-count vs n_embd) and a decode-split refusal check (prompt > n_batch must 400)
- structural integrity checks (counted from the audit script's own check
  headers, never hardcoded here: RAII wrappers, shared header, backpressure,
  pool=none size limit, checkpoint v2+ sum records with v5 rolling content
  hash + v6 accumulator checksum, no raw fclose, checkpoint bounds,
  producer-consumer pipeline, checkpoint durability order)

GPU verification (CUDA + Vulkan) is manual  -  see CI header comments for
commands (those manual builds configure `GGML_NATIVE=ON` on the target
machine).

## Documentation

- `docs/hidden-states-architecture.md`  -  unified capture mechanism, layer
  index convention
- `docs/hidden-states-adoption-manifest.md`  -  per-builder adoption
  classification (ADOPT / REFUSE lists) and the adoption recipe
- `docs/pr-text-hidden-states.md` — archived 2026-09-18 to
  `docs/history/pr-text-hidden-states.md` (the upstream-PR path was explicitly
  dropped; the draft is kept for its API-surface prose only).
- `tools/hs-extract-batch/README.md`  -  batch extraction tool
- `tools/hs-extract/README.md`  -  single-prompt tool
- `tools/server/README.md`  -  `/hidden-states` endpoint ("POST /hidden-states" section)
- `docs/history/`  -  historical audit reports

### 8. Golden byte-identity gate

- `tools/hs-extract-batch/gate/`  -  the fork's regression gate: `gate_batch.py`
  (baseline|check; requires `HS_GATE_MODEL=<gguf>`, runs the production path,
  no fallbacks), committed anchor `digests_baseline.json` (restored
  2026-09-16 to the verified 2026-08-18 closure lineage: the 09-13
  "re-baseline" had promoted stale-cache bytes from an incremental build
  directory; the root-cause analysis is preserved in this repository's
  `docs/history/` and summarized in `gate/README.md`; fresh builds from
  clean trees reproduce this anchor exactly, AOT or PTX-JIT),
  `audit_integrity.sh` (all structural checks; the count derives from the
  script's check headers),
     `gen_inputs.py` + `golden_inputs/` (deterministic gate fixtures).
     See `gate/README.md` for the model contract and workflow.
  After each upstream sync window (a `git merge origin/master` into `main`,
  resolved and re-verified as one unit — the git recipe above shows the
  current delta), re-run the gate: fresh-build digests must match the
  committed anchor, or the anchor moves with a written cause.

### 9. Shared headers (`tools/hs-extract-common/`)

- `layer-parse.h`  -  the one layer-list parser used by both CLIs ('all',
  comma-separated hidden_states indices, negatives from the end, duplicates
  rejected).
- `tokenize.h`  -  the bounded tokenizer shared by both CLIs (per-token
  validation; replaced hand-copied per-tool variants).
