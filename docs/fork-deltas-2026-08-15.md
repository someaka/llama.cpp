# Fork Deltas vs Upstream — 2026-08-15

Fork `f810014e2` vs `origin/master` (`77918caf3`). Lead-generated (M2 delegate died on HTTP 429 after partial work; findings below are disk-verified, and the delegate's salvage — the `embeddings_layer_inp` discovery — is incorporated and verified).

## Verdict table

| # | Capability | Our files (~LOC) | Upstream equivalent? | Verdict | Rationale |
|---|---|---|---|---|---|
| 1 | Hidden-state extraction API (`extract_hidden_states`, `llama_get_hidden_state*`, `t_hidden_layers` taps, lm_head pruning, `_hs_synced` fences) | include/llama.h, src/llama-context.*, src/llama-cparams.h, 4 model files (~1,200) | **No.** Closest: `llama_set_embeddings_layer_inp` (context.cpp:3776) + `t_layer_inp` in 12 model files — but that captures layer **inputs** (pre-norm `inpL`, spec-decode acceptance plumbing, `res->t_layer_inp[il] = inpL` before norm), is exposed via no public tool, and has no batch getter. Our taps capture post-block **outputs**. Different tensor, different purpose. | **PR-CANDIDATE** (core) | Genuinely unique public API; the science surface. Must explain relation to `embeddings_layer_inp` in PR description. |
| 2 | `hs-extract-batch` (async pipeline, checkpoint/resume v2, CRD2/STR1, `--raw --mean`, `--no-bos`, `--profile`, 17-test self-test) | tools/hs-extract-batch/ (~3,250) | **No.** Upstream tools/ has 19 tools, zero activation/hidden/extract; checkpoint hits in common/ are unrelated (--predict etc.). No `--no-bos`-style tokenizer control in common args. | **PR-CANDIDATE** (flagship tool) | Largest unique artifact. Reviewer concerns: CRD2/STR1 naming, CrimsonRed-specific formats — needs a generic-format option or clear docs. |
| 3 | `hs-extract` single-prompt JSON tool | tools/hs-extract/ (~600) | **No.** | **PR-CANDIDATE** (small, pairs with #2) | Debug/parity tool; trivial to review. |
| 4 | `/hidden-states` server endpoint | tools/server/* (~350 across 6 files) | **Partial.** Upstream evolved surrounding plumbing: metrics refactor `decaf508b` (result_timings→server_slot_stats), `/metrics` during decode `77918caf3`, slot save/restore `5d9e5ac30`. No endpoint collision. | **PR-CANDIDATE** (conditional) | Adopt `server_slot_stats` conventions; verify no reliance on deleted structs (M3 probe build). |
| 5 | Vulkan debug-log enrichment | ggml-vulkan.cpp (1 line) | No | **INTERNAL-ONLY** | Too trivial for a PR; drop from PR set. |
| 6 | Fork CI + docs (fork-ci.yml, FORK_CHANGES.md, tool READMEs, docs/history) | ~1,000 | n/a | **INTERNAL-ONLY** | Fork infrastructure; PR gets upstream-style docs instead. |

## PR-candidate list (final)

1. Core extraction API + per-arch taps (`t_hidden_layers`, post-block outputs, lm_head pruning, sync fences)
2. `hs-extract-batch` production tool
3. `hs-extract` single-prompt tool
4. `/hidden-states` server endpoint — conditional on adopting upstream's newer server conventions

## What upstream has that we should adopt, not fight

- `llama_set_embeddings_layer_inp` / `t_layer_inp` (12 model files): spec-decode layer-input capture. If our PR lands, reviewers may suggest unifying tensor-capture plumbing; prepare the input-vs-output distinction argument.
- `server_slot_stats` (decaf508b): our server endpoint code should use it for any timing fields.
- Multi-output backend sampling (#25532): already merged cleanly into our decode path during rebase.

## Redundancy verdict

Nothing upstream makes any fork capability redundant. The nearest thing (`embeddings_layer_inp`) is a different mechanism serving spec-decode, not extraction.
