# Upstream API Drift Audit + CPU Probe Build — Wave M3 (2026-08-15)

**Fork HEAD:** `f810014e2` (docs: FORK_CHANGES.md post-rebase state)
**Upstream window:** `178ade436^2` = `876a43211` (old) → `origin/master` = `77918caf3` (new), 213 upstream commits
**Method:** `git diff 178ade436^2 origin/master -- include/llama.h src/llama.h` (API surface), symbol cross-reference against the 48 fork-changed files (`git diff --name-only origin/master HEAD`), then a **pristine CPU probe build** via `git archive f810014e2 | tar -x -C /tmp/llm-probe` (repo checkout untouched; no git state modified).

---

## A. Public API changes, old → new upstream (`include/llama.h`; `src/llama.h` does not exist — zero changes there)

Diffstat: 1 file changed, **+41 / −21**.

| # | Symbol | Old → New | Kind |
|---|--------|-----------|------|
| 1 | `llama_sampler_init_penalties` | `(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)` → **`(n_vocab, penalty_last_n, ...)`** — new first param; `penalty_repeat` must be > 0; freq/present must be finite | **BREAKING signature change** |
| 2 | `llama_sampler_init_dry` | `(vocab, n_ctx_train, multiplier, base, allowed_length, penalty_last_n, seq_breakers, num_breakers)` → **`n_ctx_train` param removed** | **BREAKING signature change** |
| 3 | `struct llama_context_params` | **new field `uint32_t n_outputs_max_per_seq` inserted mid-struct** (after `n_outputs_max`, before `n_threads`) | layout change (ABI); source-safe unless positional init |
| 4 | `llama_sampler_iface.backend_init` | `(smpl, buft)` → **`(smpl, buft, n_outputs_max_per_seq)`** | breaking for custom sampler ifaces |
| 5 | `llama_sampler_iface` | **+ `backend_reset`**, **+ `copy_state`** vtable slots (NULL-able) | additive vtable |
| 6 | `llama_sampler_copy` | new function | additive |
| 7 | `llama_version` | new function | additive |
| 8 | `enum llama_load_mode` | **+ `LLAMA_LOAD_MODE_AUTO = -1`** | additive |
| 9 | `llama_state_seq_load_file` | comment only (`tokens_out == NULL` → count-only). **Signature unchanged** | non-change |
| 10 | `llama_get_sampled_token_ith`, `llama_sample_softmax`-family docs | comment-only semantics notes (multi-output ordering) | non-change |

Additive changes are safe for callers. The two signature breaks and the struct-field insertion are the drift that matters.

## B. Our uses of changed symbols (all 48 fork files grepped at HEAD)

| Symbol | Fork use | Verdict |
|--------|----------|---------|
| `llama_sampler_init_penalties` | `tools/hs-extract-batch/hs-extract-batch.cpp:1971` — `llama_sampler_init_penalties(n_vocab, 64, args.repeat_penalty, 0.0f, 0.0f)` with explanatory comment at :1969 | ✅ **Already fixed** by `ce80768e1`; confirmed by clean probe build + 17/17 self-test |
| `llama_context_params` | 8 sites, all `= llama_context_default_params()` + named field assignment (hs-extract.cpp:220, hs-extract-batch.cpp:1002/1543, hidden-states.cpp:67, test-hidden-states.c:26/.cpp:32, hs-extract-batch-test.cpp:36) | ✅ Safe — no positional initializers anywhere in fork code |
| `llama_sampler_init_dry` | not called in any fork file (only upstream's own `common/sampling.cpp:353`, already adapted upstream) | ✅ No action |
| `backend_init`, `backend_reset`, `copy_state`, `llama_sampler_copy`, `llama_version`, `LLAMA_LOAD_MODE_AUTO` | not referenced by fork-owned code (upstream's `common/` handles them) | ✅ No action |
| `llama_state_seq_load_file` | `tools/server/server-context.cpp:2773` — signature unchanged | ✅ Safe |
| **`result_timings`** | **`tools/server/server-context.cpp:528-529`** (`result_timings get_timings() const`) + callers at :2120, :2147 (`res->timings = slot.get_timings()`) | ❌ **BROKEN** — struct deleted upstream by `decaf508b` (server metrics refactor #26920); `git grep result_timings HEAD` finds **no definition anywhere in the tree**. ⚠️ This corrects `docs/upstream-overlap-map-2026-08-15.md:21`, which claims "fork no longer references it" — that claim is **false**. |

## C. Probe-build result (CPU-only, `/tmp/llm-probe/build-cpu`, `-DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF`, `-j4`)

Configure: clean. Build: **exit code 2**, all 64 errors in **one file** (`tools/server/server-context.cpp`); sibling server TUs (`server-task.cpp`, `server-common.cpp`, `server-queue.cpp`, `server-chat.cpp`) compiled clean. **Zero warnings** project-wide.

| Target | Verdict | Detail |
|--------|---------|--------|
| `llama-hs-extract` | ✅ **CLEAN** | Built at 100%; `--layers` flag present |
| `llama-hs-extract-batch` | ✅ **CLEAN** | Built at 100%; **`--self-test` → 17/17 passed**; `--no-bos` flag present |
| `llama-server` | ❌ **BROKEN — 64 errors, all in `server-context.cpp`** | `make Error 2` at `server-context.dir`; binary not produced |

### Exact error taxonomy (64 errors, deduplicated)

1. **`result_timings` deleted** (upstream `decaf508b`, in-window) — 5 errors:
   - `:528: error: 'result_timings' does not name a type`
   - `:2120/:2147: error: 'struct server_slot' has no member named 'get_timings'`
   - `:2120: error: 'struct server_task_result_cmpl_partial' has no member named 'timings'` / `:2147: ...cmpl_final... 'timings'`
   Upstream replacement: `res->stats = slot.stats` (`server_slot_stats`, already defined in `server-task.h:344,426` and already used by our `server-task.cpp` via `stats.to_json()` — only `server-context.cpp` lags).
2. **`common_context_seq_rm/_cp/_add` undeclared** — 12 errors at :277, :279, :700, :701, :704, :705, :3083, :3087, :3199, :3403, :3407, :3580, :3582, :4071, :4077, :4120 (some sites emit multiple). These free functions were made `static` in `common/common.cpp:1601+` behind `struct common_memory::seq_rm/seq_add/seq_cp` (upstream #26221 `ee3d1b54c`, which **predates** the 213-commit window). Upstream master's `server-context.cpp` makes **0** raw calls; our HEAD makes **12** — they were resurrected by the rebase/merge conflict resolution in `178ade436` (the pre-rebase fork file had only a comment reference; fork commits `04858c4dd`/`370530609` carried the pattern). Fix = route through `common_memory` methods, as upstream does.
3. **`server_metrics` refactor fallout** (upstream `decaf508b`, in-window) — ~44 errors:
   - `:816: error: redefinition of 'struct server_metrics'` (it moved to `server-common.h:430` — our local definition now collides)
   - `server_metrics`/`server_task_result_metrics` member removals: `t_start` ×2, `n_prompt_tokens_processed(_total)` ×5, `t_prompt_processing(_total)` ×4, `n_tokens_predicted(_total)` ×5, `t_tokens_generation(_total)` ×3, `n_decode_total` ×4, `n_busy_slots_total` ×2, `n_tokens_max` ×2, plus `on_decoded`, `on_prompt_eval`, `on_prediction` callbacks gone
   - `:4647: no matching function for call to nlohmann ... basic_json(<brace-enclosed initializer list>)` (downstream of the member errors)
4. **`common_speculative_need_embd` / `..._nextn` removed** (upstream `f785fc9ea` #26904, in-window) — 2 errors at :381, :386. Used by our fork's `need_embd()` logic around the hidden-states endpoint (the fork's output-allocation comment block in `server-task.h:200-208` depends on this path).
5. **`on_new_task` callback signature change** (upstream `77918caf3` #27041, in-window — the window's tip commit) — 1 error at :1426: `cannot convert '<lambda(server_task&&)>' to 'std::function<bool(server_task&&, bool)>'` (new `bool` param + `bool` return).
6. **`eval_llama_cmpl_schema` lost `n_ctx_slot` param** (upstream `a6aa6f545` #26524, in-window) — 1 error at :4331: `invalid initialization of reference of type 'const std::vector<llama_logit_bias>&' from expression of type 'const int'` (our call still passes `meta->slot_n_ctx` where the vector is now expected).

## D. Latent breakage list, ranked by severity

1. **CRITICAL — `llama-server` does not compile** (64 errors, all `server-context.cpp`). Consequence: the fork's `/hidden-states` server endpoint is unbuildable, and D's server/demo stack cannot be rebuilt from `f810014e2`. The CUDA-side rebuild policy (hs-extract only) masked this completely — the server had never been compiled since the rebase. Six independent upstream refactors (`decaf508b`, `ee3d1b54c` pattern, `f785fc9ea`, `77918caf3`, `a6aa6f545` + the metrics move) all land in this one file.
2. **HIGH — `server_metrics` redefinition + member purge**: our local `struct server_metrics` at `server-context.cpp:816` must be deleted and all metrics plumbing migrated to `server_slot_stats` / the new `server-common.h` definition. This is the largest single repair (~44 of the 64 errors) but mechanical once the decaf508b pattern is followed (`stats.to_json()` already used in our `server-task.cpp`).
3. **HIGH — seq-memory calls bypass `common_memory`**: 12 raw `common_context_seq_*` calls are a rebase-resolution artifact (pre-window API). Beyond compiling, this is a correctness surface: the fork's HIDDEN_STATES memory-clear fix (`04858c4dd`) must preserve its semantics when ported to `common_memory::seq_rm()` — the "Gated Delta Net / Qwen3.5" comment at :1830 documents why the fork clears memory there.
4. **MEDIUM — `common_speculative_need_embd` removal**: the fork's `need_embd()` override chain (which the hidden-states endpoint deliberately keeps `false` to match the CLI graph shape — see `server-task.h:200-208`) needs re-derivation from whatever `f785fc9ea` replaced it with.
5. **LOW — schema call sites**: `eval_llama_cmpl_schema` drop of `n_ctx_slot` and `on_new_task` signature are one-line mechanical fixes.
6. **INFO — `hs-extract`/`hs-extract-batch`: zero drift remaining.** The one real API break in the public header (`llama_sampler_init_penalties`) was already fixed at `ce80768e1`; probe build is clean, self-test 17/17, `--no-bos` and `--layers` verified present.
7. **INFO — doc correction required**: `docs/upstream-overlap-map-2026-08-15.md:21` states the fork no longer references `result_timings` — false at HEAD (references at `server-context.cpp:528-529, 2120, 2147`). This file supersedes that claim with the probe-build evidence.

---

### Reproduction

```bash
mkdir -p /tmp/llm-probe && git -C /home/a/Bureau/Work/llama.cpp archive f810014e2 | tar -x -C /tmp/llm-probe
cd /tmp/llm-probe
cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
cmake --build build-cpu --target llama-hs-extract llama-hs-extract-batch llama-server -j4   # → exit 2, 64× server-context.cpp
/tmp/llm-probe/build-cpu/bin/llama-hs-extract-batch --self-test                            # → 17/17
```

Build logs: `/tmp/probe-cmake-configure.log`, `/tmp/probe-build.log`. Probe tree: `/tmp/llm-probe` (scratch; repo checkout and git state untouched — deliverable file only, added untracked under `docs/`).
