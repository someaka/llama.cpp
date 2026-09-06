# PR text — hidden-states extraction (branch `hidden-states-extraction`)

Title: `llama: hidden-state extraction API + tools + server endpoint`

## Motivation

Interpretability and analysis work on LLMs needs access to per-layer residual
stream states during normal decode. Upstream's embeddings capture
(`llama_set_embeddings`, `llama_output_seqs`) exposes embeddings and logits, but
there is **no public way to read intermediate post-block hidden states**. Today
every project that needs them carries a private fork — this PR makes the
capability available upstream.

**Relation to upstream's `llama_set_embeddings_layer_inp`:** none — different
tensor, different purpose. That internal API (declared in `src/llama-ext.h`,
used by the model files that need it) assigns the layer **input** (pre-norm `inpL`) for
speculative-decoding acceptance checks. This PR captures the post-block
**output** residual and exposes it through the public `include/llama.h` API.
Zero redundancy.

## What's included

1. **Core API** (`include/llama.h`, `src/llama-context.cpp`, graph, cparams;
   a per-block capture tap sewn into all 101 graph builders, plus an
   embeddings-capture hook in 4 of them): `llama_set_extract_hidden_states`, `llama_get_hidden_state`,
   `llama_get_hidden_state_n_tokens`,
   `llama_get_hidden_states_batch`. The getters synchronize the context
   (same convention as the logits/embeddings getters). Graph capture buffers allocated per decode;
   capture conditional on cparams flag; models append `t_hidden_layers` at the
   end of each decoder block.
2. **`hs-extract` CLI** (`tools/hs-extract/`): single-prompt JSON extraction
   with layer selection and BOS suppression.
3. **`hs-extract-batch` CLI** (`tools/hs-extract-batch/`): high-throughput
   batch extraction — thousands of prompts per model load; raw mean-pool mode
   and streaming-accumulator mode with checkpoint/resume; per-record sidecar
   output (`--save-per-record`); binary formats documented in README.
   Unknown flags and invalid flag combinations error at parse time.
4. **`/hidden-states` server endpoint** (`tools/server/`): POST prompt →
   per-layer vectors, no generation; three pooling modes (`last` default,
   `skip_mean`, per-token `none` with a 25M-float DoS cap), `normalize`,
   `skip_offset`, and `layers: "all"` request fields (full surface in
   `tools/server/README.md` ("POST /hidden-states" section));
   `--no-hidden-states` flag disables the route.
5. **Tests + example** (`tests/test-hidden-states.{cpp,c}`,
   `examples/hidden-states/`): API contract tests + minimal end-to-end example.

## Design notes

- Capture buffers sync once per decode (not per token). The runtime-toggle
  path is exercised bitwise by `llama-hs-extract-batch-test` mode 2 and
  `llama-hs-probe --runtime-toggle` (fork CI); multi-ubatch pool=none by the
  fork-CI integration test.
- Models opt in by appending to `t_hidden_layers` in the layer loop; archs
  with the last-layer `inp_out_ids` optimization also extend that guard
  (see `src/models/llama.cpp`).
- The `--no-hidden-states` flag disables the endpoint (kill-switch; nothing else in the server reads it).
- Binary output formats use stable magic constants, documented in README.

## Testing

- `test-hidden-states` (CPU): API contract — enable at creation, single +
  batch getters, boundary rejection, ladder top slot.
- `hs-extract-batch --self-test`: passes end-to-end, including the
  accumulator checkpoint/resume roundtrip.
- The stack restores upstream test coverage that intermediate branch
  surgery had silently dropped: the chunked-scan tests in
  `tests/test-backend-ops.cpp` and the
  `test-recurrent-state-rollback-nemotron-h` registration in
  `tests/CMakeLists.txt`. Both files are zero-diff (resp. add-only)
  against upstream after the restores.
- One included change is deliberately outside the extraction feature:
  `src/llama-kv-cells.h` replaces a per-position `std::map` with a flat
  count table. It is included here because sustained extraction
  workloads (200K+ prompts, millions of decode cycles) exposed heap churn
  in the rb-tree node allocator; the flat table removes that surface. Happy
  to split it out if maintainers prefer.
- Both CLIs + server endpoint exercised daily on RTX 3090 (CUDA) and AMD
  Renoir iGPU (Vulkan/RADV) — cross-backend validated, identical results
  within quantization tolerance.

## Scope note

The CRD2/STR1-named binary formats in hs-extract-batch are the wire formats of
an existing research pipeline (emotion-vector extraction over 200K+ prompts);
the magics are load-bearing for compat. Happy to add a generic-format flag or
move format docs around if maintainers prefer.
