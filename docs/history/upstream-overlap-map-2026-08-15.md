# Upstream Overlap Map — 2026-08-15

Fork `f810014e2` vs `origin/master` (`77918caf3`); old upstream base `178ade436^2`; 213 upstream commits audited.
Lead-generated (M1 delegate died on HTTP 429 before doing work; this map is disk-derived, every claim grep-verifiable).

## Headline answer

**Upstream does NOT have native hidden-state extraction.** Evidence:
- `git grep hidden_state origin/master -- include/ src/` → only gemma3n.cpp, kimi-linear.cpp, laguna.cpp internals (1-2 hits each, unrelated to any extraction API)
- `include/llama.h` upstream: embedding support is still `llama_set_embeddings` / `llama_get_embeddings` (pooled/final-layer), nothing per-layer
- `git ls-tree origin/master tools/` → no activation/hidden/extract tool (19 tools, 0 matches)
- Only 1 of 213 commits touches the theme area (1138b851f, model-conversion refactor)

## Direct overlaps: 14 of 48 fork files touched by upstream in the window

| Overlap area | Upstream commits | Our files | Class | Note |
|---|---|---|---|---|
| Hidden-state extraction API | none — hidden_state in upstream src/ only as internals in gemma3n/kimi-linear/laguna model files | include/llama.h, src/llama-context.*, src/llama-cparams.h, src/models/*.cpp | INDEPENDENT | No upstream equivalent. Our core feature is unique. |
| hs-extract-batch tool | none — upstream tools/ has no activation/hidden/extract tool (verified: ls-tree + grep 0 hits) | tools/hs-extract-batch/* | INDEPENDENT | No upstream equivalent. |
| hs-extract tool | none | tools/hs-extract/* | INDEPENDENT | No upstream equivalent. |
| /hidden-states server endpoint | 5d9e5ac30 (slot save/restore w/ media), decaf508b (metrics refactor→server_slot_stats), 77918caf3 (/metrics during decode) | tools/server/* | PARALLEL-ADJACENT | Upstream evolved slot/state/metrics plumbing around our endpoint; no endpoint collision. result_timings deleted by decaf508b — fork DID still reference it at rebase time (M3 falsified this line); fixed in the wave-N migration commit. |
| Multi-output backend sampling | dd1ea5243 (#25532) | src/llama-context.cpp, src/llama-cparams.h, src/llama-graph.* | CONFLICT-RESOLVED-IN-REBASE | Upstream feature orthogonal to extraction; conflicts resolved by union during rebase (sampling-transaction block + our pre-allocation). Verified: get_hidden_state/synchronize md5-identical post-rebase. |
| Sampler n_vocab move | 935cad649 (#26520: n_vocab moved into penalty sampler) | tools/hs-extract-batch (call site adapted) | CONFLICT-RESOLVED-IN-REBASE | The API change that broke our build; adapted in ce80768e1. |
| Vulkan debug-log enrichment | a7cd2f0e9 (TQ2_0), 153d324bc (load-mode auto), 803b7fcae (submission batching) | ggml-vulkan.cpp (1 line) | INDEPENDENT | Our single log-line survives; upstream vulkan work is unrelated quant/system features. |
| Server tool isolation | dd2c7c447, 4ae84dea2 (docker/podman/ssh isolation) | tools/server/server.cpp | CONFLICT-RESOLVED-IN-REBASE | Upstream infra feature; our endpoint registration independent. |
| Metrics refactor | decaf508b, a035a8887, 77918caf3 | tools/server/server-context.cpp, server-task.* | CONFLICT-RESOLVED-IN-REBASE | server_slot_stats moved; our hidden-states handlers untouched by the move. |
| Load-mode auto / system config / semver | 153d324bc, 8e7f22b67, 680a9ae63, f65e568fd, a94d563ed | common/arg.cpp, common/common.h, include/llama.h | CONFLICT-RESOLVED-IN-REBASE | Common-infrastructure churn; resolved mechanically during rebase. |
| KV-cells flat table | none — upstream llama-kv-cells.h still std::map based | src/llama-kv-cells.h | INDEPENDENT | Our O(1) optimization is unique. NOTE: 67d5978bb shows upstream is migrating memory implementations (M3 MSA) — future interaction possible. |
| lm_head pruning | none found in 213 commits | src/llama-graph.cpp (fork side) | INDEPENDENT | Our ~1.89x extraction speedup remains ours. |

## Class counts

- INDEPENDENT (no upstream counterpart): 6
- CONFLICT-RESOLVED-IN-REBASE (both changed, union taken, verified): 5
- PARALLEL-ADJACENT (upstream evolved nearby plumbing, no collision): 1

## Top items needing a human decision (for Wave N)

1. **Server endpoint vs upstream's metrics refactor trajectory** — decaf508b moved timing fields into `server_slot_stats`; if we PR the endpoint, it should adopt that struct. Cosmetic-to-moderate rework.
2. **`embeddings_layer_inp.resize(n_layer()+1)` upstream change** — upstream now reserves input-embedding capture per layer; our taps use `t_hidden_layers`. Different mechanisms, but a PR conversation would ask why both exist. Prepare an explanation.
3. **Multi-output sampling union site (llama-context.cpp decode entry)** — the one true semantic merge in the rebase; worth a second read before PR (currently verified by md5 on the two critical functions only).
4. **KV-cells flat table vs upstream's M3-MSA memory migration (67d5978bb)** — if upstream keeps migrating memory code, our table may need rebasing again; PR it early or track it.
5. **Vulkan log line** — trivial; drop from any PR (not worth reviewer time).
