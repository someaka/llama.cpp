# Hidden-States Architecture — unified capture mechanism

Branch `hs-arch-core` (2026-08-16). Answers: how per-layer hidden-state
capture works post-refactor, who may join, and how refusals surface.

# Layer Index Convention (canonical)

> This section is the single source of truth for layer numbering across the
> fork's hidden-state surface and the CLIs' consumers. The same wording
> appears in every doc that discusses layer indices; do not paraphrase it.

The public layer index follows the **hidden_states convention**, matching
HuggingFace `hidden_states` and upstream llama.cpp's internal layer-input
tensors (`t_layer_inp`):

| index | meaning |
|-------|---------|
| `0` | token embeddings — the state entering block 0 (after any arch-specific embedding transform, e.g. the Gemma `sqrt(n_embd)` scale) |
| `i` | the state entering block `i` — the same tensor upstream stores as `t_layer_inp[i]` and HF returns as `hidden_states[i]` |
| `N` | the output of the final block (`n_layer` = block count) — a fork extension; upstream stops at `N-1` and exposes nothing publicly |

Not captured: post-final-norm output.

Migration from the old fork numbering (post-block-i residual):

| recorded in old fork space | upstream / hidden_states space |
|---|---|
| any layer `L` | `L + 1` |
| E2B `L24` (of 35) | `25` |
| E4B `L29` (of 42) | `30` |
| Qwen `L16` (of 24) | `17` |
| Llama `L11` (of 16) | `12` |

Existing on-disk artifacts recorded before the migration stay in the old
space; readers must remap with `fork_to_upstream(i) = i + 1` and refuse
ambiguous files rather than guess.

---

## The mechanism (three parts)

### 1. Capture helper — one line per layer, shared by all architectures

`llm_graph_context::capture_layer_output(int il, ggml_tensor * cur)` in
`src/llama-graph.{h,cpp}` centralizes everything the per-arch code used to do
inline:

```cpp
// src/llama-graph.cpp
void llm_graph_context::capture_layer_output(int il, ggml_tensor * cur) {
    if (!cparams.extract_hidden_states) {
        return;
    }

    // The residual stream leaving block il is the state entering block il+1,
    // so it is stored at hidden_states index il+1: one uniform ladder mapping
    // for every builder (slot 0, the embeddings, is captured separately by
    // capture_embeddings before the loop).
    GGML_ASSERT(cur != nullptr);
    GGML_ASSERT((int) res->t_hidden_layers.size() == il + 1 &&
                "capture_layer_output called out of layer order "
                "(capture_embeddings must run first)");

    cb(cur, "hidden_state", il);
    res->t_hidden_layers.push_back(cur);
}
```

Every supported architecture calls `capture_embeddings(inpL)` once before
its layer loop (slot 0) and `capture_layer_output(il, cur)` once per layer at
the bottom of the loop (slots 1..N):

```cpp
        inpL = cur;

        capture_layer_output(il, cur);
```

The four reference adoptions: `src/models/llama.cpp`, `gemma.cpp`,
`gemma4.cpp`, `qwen35.cpp`. No model file touches
`res->t_hidden_layers` directly anymore; the helper is the only writer
(grep-verifiable: `t_hidden_layers` in `src/models/` = 0 hits).

What the helper replaced (per arch, the inline block):
`if (cparams.extract_hidden_states) { res->t_hidden_layers.push_back(cur); }`
— four copies of the same check+push, each able to drift independently.

### 2. Capability registry — the single source of truth

`llm_arch_supports_hidden_states(const llm_arch &)` in
`src/llama-arch.{h,cpp}`, beside the existing `rs_rollback` / `sm_tensor`
whitelists:

```cpp
// src/llama-arch.cpp
bool llm_arch_supports_hidden_states(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_QWEN35:
            return true;
        default:
            return false;
    }
}
```

(The trailing comment above the function in the real file states the
maintenance rule; quoted shape here.)

Three gates read it:

- **Context creation** (`src/llama-context.cpp`, after the cparams copy):
  flag set + unsupported arch → `throw std::runtime_error("hidden-state
  extraction not implemented for architecture '<name>'")`. The context is
  never created; no dead buffer allocation.
- **Runtime setter** (`llama_context::set_extract_hidden_states`): enabling
  on an unsupported arch → same throw. The server enables per request
  (server-context.cpp update_slots toggle); a throw here surfaces as a task
  error rather than a silent NULL.
- **Server `/hidden-states` route** (`tools/server/server-context.cpp`,
  post_hidden_states handler): unsupported arch → 400-class
  `ERROR_TYPE_INVALID_REQUEST` naming the arch, before any decode happens.

Backstop inside decode: if the registry says supported but the graph builder
delivered an empty `t_hidden_layers`, that is a contract violation and
aborts loudly (`GGML_ABORT`) naming the arch and the missing helper call.
The registry-not-listed path keeps its error log (context creation already
refused it; decode only sees such a context if the gates were bypassed).

### 3. Semantics — unchanged on the four reference archs

- Captured tensor: slot 0 = embeddings; slots 1..N = the block-output residual (`cur` after the final residual
  add and `build_cvec`, before the next block's norms) — the state entering
  block `il+1`. Same tensor the old inline code captured.
- Buffer layout, getters, ownership/lifetime contract, `n_embd ==
  n_embd_out` requirement: unchanged (`include/llama.h` hidden-state doc
  block).
- The getter does one `synchronize()` per call / per batch call — same as
  before the refactor.

## What a new architecture must do to join

1. Verify its builder has the standard loop tail: final residual add →
   `inpL = cur` reassignment at loop bottom. (Non-standard builders: see
   the adoption manifest — some have no single well-defined residual.)
2. Add `capture_layer_output(il, cur);` right after the reassignment.
3. Add its `LLM_ARCH_*` case to `llm_arch_supports_hidden_states`.
4. Rebuild, self-test 24/24, `test-hidden-states <gguf>` on a real GGUF of
   that arch.

Refusal needs no code: an unlisted arch is refused by name at context
creation (flag set), at the setter (enable), and at the server route.

## Costs and limits

- When the flag is off: one bool in cparams; helper is a no-op return.
- When on: `n_layers × n_tokens × n_embd` floats per decode (the capture
  buffer), plus one device sync per getter/batch-getter call.
- Not captured: the post-final-norm state. Index `i` ≡ HF
  `hidden_states[i]` (slot 0 = embeddings ≡ HF `hidden_states[0]`,
  per the canonical table at the top of this file).
- No automatic coverage of the 100+ CLASSIC archs — joining is a two-line
  adoption per the manifest; the 26 non-classic shapes (parallel-residual,
  hybrid recurrent) are refused pending per-builder semantic decisions.
  See `docs/hidden-states-adoption-manifest.md`.

## The llama/gemma extraction crash (found and fixed on this branch)

Before this branch, extracting hidden states on the llama or classic-gemma
architecture aborted inside `llama_decode`:

```
ggml-backend.cpp:194: GGML_ASSERT(buffer) failed   ← llm_graph_input_out_ids::set_input
```

Root cause (two defects, both pre-existing on main and reproducible with
main's own binaries):

1. **Orphaned input tensor.** The fork suppressed the last layer's
   in-loop `ggml_get_rows` output pruning under extraction (so all tokens'
   residuals get captured) — but nothing else in the graph consumed
   `inp_out_ids` afterwards. An input tensor no node references is never
   allocated by the scheduler; `set_input` then dereferenced its NULL
   buffer and aborted every decode.
2. **Wrong logits shape hiding behind the crash.** With the pruning
   suppressed, `t_logits` was full-token shaped (all positions), while the
   logits copy-back in `llama_context::decode` reads only
   `n_outputs × n_vocab` floats — i.e. the FIRST n_outputs positions'
   logits, not the marked output positions. Had the abort not fired, llama
   and gemma would have returned silently wrong logits under extraction.

Fix (this branch): mirror the shape gemma4/qwen35 already use — keep the
in-loop suppression (capture stays all-token), then prune with
`ggml_get_rows(cur, inp_out_ids)` **after the final output norm**, before
`lm_head`. `inp_out_ids` is consumed again (allocated, no abort), and
logits/embeddings return output-only rows exactly as without extraction.

Verification (all on real runs):

- llama-arch extraction now completes (exit 0, 2048-dim float vectors,
  no NaNs); main's own binary still aborts on the same input (bug is
  pre-existing, not introduced here).
- Logits equivalence probe (`tools/hs-probe/logits-probe.cpp`, public API
  only): identical argmax and identical top-8 logit values (6 decimal
  places) with extraction on vs off on Llama-3.2-1B f16.
- Byte-identity vs main's pre-refactor binaries on qwen35 / gemma4 E2B /
  gemma4 E4B (`cmp` on full capture dumps) — see below.


## Evidence

- Byte-equivalence with upstream access points: the same `ggml_tensor*` is
  stored by upstream's `t_layer_inp[il+1]` and our capture at loop bottom
  (`inpL = cur` — identity by construction). Full-dump `cmp` of capture
  output vs upstream's tensors was byte-equal on llama / qwen35 /
  gemma4-E2B / gemma4-E4B (verified 2026-08-16).
- Unsupported-arch behavior before this branch: decode returned 0, getters
  NULL, server answered a misleading 500 (fixed by the capability-registry
  check; the server now answers 400 naming the architecture).
