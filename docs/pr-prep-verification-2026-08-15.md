# PR-Prep Verification — handoff G §2 vs fork docs vs disk (2026-08-15)

Source: delegate D3 verification pass, 2026-08-14 16:13–16:22 (read-only).
Base facts: fork HEAD `30faeed63`, `fork/main` `9fff89fc1`, 333 ahead / 172 behind.
Upstream ref is `origin/master` = `77918caf3` (NO `origin/main` exists) — also the
merge-base, so `77918caf3...HEAD` is the correct delta base (51 files = wave-M's
48 + 3 docs files).

## LoC: claimed vs actual (`git diff --numstat 77918caf3...HEAD`)

| Candidate | Claimed (handoff G §2 / fork-deltas) | Actual | Verdict |
|---|---|---|---|
| Core extraction API | ~1,200 (8-file list) | 365 churn (8-file list, +361/−4); 560 incl. graph/kv-cells/raii; 1,087 widest scope (core+tests+examples+common) | overstated ~10% at widest; 8-file list ≠ claim by ~3× |
| hs-extract-batch | ~3,250 | 2,844 (3 files; cpp wc-l = 2,693); 2,953 incl. test dir | overstated ~12–17% |
| hs-extract | ~600 | 414 (332 cpp + 69 README + 13 CMake) | overstated ~45% |
| /hidden-states endpoint | ~390 / ~350 | 480 churn (6 server files, +478/−2); 434 code-only | UNDERSTATED (doc pre-migration) |

Numbers identical pre-rebase (`876a43211...9fff89fc1`: core 364, batch 2,842,
hs-extract 414, server 482) — rebase didn't distort them; estimates were loose.
**Use the actuals above in any PR text, not handoff G's figures.**

## Squash plan

Pre-rebase: 146 non-merge commits ahead of old base. Post-rebase HEAD: **120
commits** ahead of origin/master; 87 touch candidate code paths, 33 docs/CI-only.
Per-candidate: core 33, hs-extract 27, batch 20, server 26, tests/examples 25.
The 4–6 logical-commit squash is feasible; input set is 120, not "147+2".

## Input-vs-output distinction — CONFIRMED in code (PR argument sound)

- Upstream (inputs, internal): `origin/master:src/llama-context.cpp:3776`
  `llama_set_embeddings_layer_inp(...)`; declared `src/llama-ext.h:111` (NOT in
  include/llama.h). Model side assigns at TOP of layer loop, pre-norm:
  `src/models/llama.cpp:126-127` `res->t_layer_inp[il] = inpL;` (qwen35.cpp:160
  under "MTP/NextN layers" comment = spec-decode). Exactly 12 model files carry
  `t_layer_inp`.
- Ours (outputs, public): assigned at BOTTOM of loop, post-block, conditional on
  `cparams.extract_hidden_states`: `res->t_hidden_layers.push_back(cur)`.
  Public API in include/llama.h: `llama_set_extract_hidden_states`,
  `llama_get_hidden_state`, `_ith`, `_n_tokens`, `_batch` (batch getter —
  upstream has none).

## llama-server status

- `46c3776fa` confirms migration (5 refactors named, "CPU build 64 errors→0";
  server-context.cpp 643+/515−). 12/12 hidden_state markers in current
  server-context.cpp (lines 63, 2348, 2356, 2375, 2422, 2483, 2487, 3778, 3957,
  5201, 5205, 5342). Zero `result_timings`/`get_timings` references remain.
  `server_output_limits()` carve-out present at server-context.cpp:48-54.
- D3 flagged a stale Jul-26 binary in build-cuda/bin — **RESOLVED 2026-08-14
  16:25**: llama-server rebuilt in build-cuda (exit 0), `--version` reports
  `commit 30faeed63`, impl lib relinked. CUDA /hidden-states round-trip still
  pending one-CUDA-job rule (queue v3 running).

## De-brand scan — 16 hits / 6 files (all would leak into an upstream PR)

| File:line | Term(s) |
|---|---|
| `common/arg.cpp:3431` | "(CrimsonRed fork)" in --no-hidden-states help |
| `common/common.h:603` | "CrimsonRed fork: hidden-states extraction" |
| `tools/hs-extract-batch/hs-extract-batch.cpp:2` | "CrimsonRed batch hidden-state extraction CLI" |
| `:123` | "max emotion groups" |
| `:481,:1127,:1251,:1292,:1366,:2133` | CRD2 + magic 0x43524432 |
| `:1667,:1900` | STR1 + magic 0x53545231 |
| `tools/hs-extract-batch/README.md:56,70,77,81` | STR1/CRD2/emotion test prompt |
| `tools/server/server-context.cpp:54,419` | "CrimsonRed fork" comments |
| `tools/server/server.cpp:256` | "CrimsonRed fork — conditional..." |
| `tools/server/README.md:909` | "CrimsonRed / someaka" — **leaks GitHub handle** |

Clean: include/llama.h, core src/ files, tools/hs-extract/, tests/,
examples/hidden-states/ — zero hits. `EmotionWheel`: zero hits anywhere.
CRD2/STR1 are wire-format magic constants — renaming breaks CrimsonRed
consumers; fork-deltas' "generic-format option or format docs" is the right
mitigation. Also in fork delta but outside candidates:
`tools/cvector-generator/{positive,negative}-gemma4.txt` (emotion word lists) —
keep out of any PR or justify separately.
