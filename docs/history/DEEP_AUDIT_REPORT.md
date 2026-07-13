# Deep Audit Report: CrimsonRed llama.cpp Fork

**Date:** 2026-06-30
**Scope:** All 35 fork-modified files (+3416 lines)
**Auditor:** Automated deep audit

## Summary

| Severity | Count |
|----------|-------|
| Critical (data loss / crash / UB) | 4 |
| High (incorrect results / memory) | 6 |
| Medium (robustness / API) | 7 |
| Low (docs / style) | 4 |

---

## CRITICAL Findings

### [C1] src/llama-context.cpp:2046-2075 — Hidden states only captured from LAST ubatch in multi-ubatch decode — CORRECTNESS BUG

The hidden state extraction block runs AFTER the `do { ... } while (mctx->next());` loop (line 1866-2044). This loop processes the batch in multiple ubatches when `n_tokens > n_ubatch`. `gf_res_prev` (which owns `t_hidden_layers`) is overwritten on each iteration by `process_ubatch()`. Only the LAST ubatch's `t_hidden_layers` survive.

**Impact:** When a prompt exceeds `n_ubatch` (default 512, set to 512 in hs-extract-batch), hidden states for tokens in earlier ubatches are silently lost. The buffer is resized to `n_tokens * n_embd * n_layers` where `n_tokens` comes from `hres->t_hidden_layers[0]->ne[1]` — which is the ubatch token count, NOT the full batch token count. `n_hidden_tokens` is set to this smaller value. Callers reading `n_hidden_tokens * n_embd` floats per layer get data for only the last ubatch, with no error.

**Fix:** Move the hidden state extraction INSIDE the ubatch loop, accumulating into `hidden_state_buf` with proper offset tracking (like logits/embeddings do via `n_outputs_prev`/`n_tokens_prev`), or force `output_all = true` when `extract_hidden_states` is enabled.

### [C2] src/llama-context.cpp:1746 — `output_all` does not account for `extract_hidden_states` — CORRECTNESS BUG

```cpp
const bool output_all = cparams.embeddings;
```

When `extract_hidden_states = true` but `embeddings = false` (the case for ALL three tools), `output_all` is false. This means the batch allocator uses `batch.logits` to determine which tokens are outputs, splitting the batch into ubatches. Combined with C1, this means only the last ubatch's hidden states are captured.

None of the tools set `ctx_params.embeddings = true`. They only set `ctx_params.extract_hidden_states = true`.

**Fix:** `const bool output_all = cparams.embeddings || cparams.extract_hidden_states;`

### [C3] src/llama-context.cpp:2066 — `ggml_backend_tensor_get_async` without explicit synchronize before return — POTENTIAL RACE

```cpp
ggml_backend_tensor_get_async(backend_res, t, dst, 0, n_tokens * n_embd_out * sizeof(float));
```

The async copy is scheduled but `decode()` returns without calling `synchronize()`. The callers (hs-extract, hs-extract-batch, examples/hidden-states) all call `llama_synchronize()` after decode, so they are safe. BUT: the C API wrapper `llama_get_hidden_state()` (line 3820) calls `ctx->synchronize()` before returning the pointer, which provides a safety net. However, `llama_get_hidden_state_n_tokens()` (line 3832) does NOT synchronize — it returns `n_hidden_tokens` which is set synchronously, so this is fine.

**Note:** The tools are safe because they call `llama_synchronize()`. The C API is safe because `llama_get_hidden_state()` syncs. But the internal C++ method `get_hidden_state()` does NOT sync — any C++ code calling the method directly without syncing first would race. Currently no such caller exists.

**Fix:** Document that `synchronize()` must be called before `get_hidden_state()`, or add sync inside the method.

### [C4] tools/hs-extract-batch/hs-extract-batch.cpp:1159 — Signed left-shift overflow (UB) — UNDEFINED BEHAVIOR

```cpp
int key = (assign.group_id << 16) | assign.mask_id;
```

`assign.group_id` is `int32_t`. The validation at line 1143 allows values up to `0xFFFF` (65535). Shifting any value >= `0x8000` (32768) left by 16 bits overflows a signed 32-bit int, which is undefined behavior in C++. With typical 2's complement, this produces implementation-defined results but compilers may optimize assuming no overflow.

Same issue at line 927 (checkpoint restore path) and line 787/804 (output writer, though those are right-shifts which are safe).

**Fix:** Use unsigned: `int key = ((uint32_t)assign.group_id << 16) | (uint32_t)assign.mask_id;` or use `int64_t` for the key.

---

## HIGH Findings

### [H1] tools/hs-extract-batch/hs-extract-batch.cpp:613 — Unvalidated `assign.skip` from file — OOB READ

```cpp
if (fread(&a.skip, sizeof(int32_t), 1, f) != 1) return {};
```

`assign.skip` is read directly from the binary file without validation. A negative `skip` value would create a range `{negative, n_hidden}`. The `compute_masked_mean` function clamps `start = max(0, min(start, n_tokens))`, so negative values are clamped to 0 — this is actually safe. But a very large `skip` creates `{large, n_hidden}` which clamps to `{n_hidden, n_hidden}` (empty range, count=0) — also safe but silently produces zero vectors.

**Fix:** Validate `assign.skip >= 0` after reading, or document that clamping is intentional.

### [H2] tools/hs-extract-batch/hs-extract-batch.cpp:605,617 — Unvalidated `n_assignments` and `n_ranges` from file — OOM / HUGE ALLOC

```cpp
std::vector<Assignment> assignments(n_assignments);  // line 605
a.ranges.resize(n_ranges);                           // line 617
```

Both `n_assignments` and `n_ranges` are `int32_t` read directly from file. A malformed or malicious `assignments.bin` could specify `n_assignments = 0x7FFFFFFF` (2 billion), causing a multi-GB allocation that OOMs the process. Similarly for `n_ranges`.

**Fix:** Add sanity bounds: `if (n_assignments < 0 || n_assignments > MAX_ASSIGNMENTS) { error; }` with a reasonable limit (e.g., 10000).

### [H3] tools/server/server-context.cpp:42-49 — Unconditional `n_batch` output allocation wastes memory — REGRESSION

```cpp
static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch = params.n_batch;
    return n_batch;  // CrimsonRed fork: always n_batch
}
```

This allocates `n_batch * n_vocab * sizeof(float)` for logits on EVERY server context, even when hidden states are never used. For `n_batch=8192, n_vocab=128000`, that's ~4GB of wasted memory per context. The original code computed `min(n_batch, n_parallel * (1 + spec_n_max))`.

The NEXT-PASS-PLAN documents this as an intentional tradeoff, but the memory cost analysis there (`n_batch * n_embd * 4 bytes ~ 16MB`) is wrong — it's `n_batch * n_vocab * 4` for logits, which is 100-1000x larger.

**Fix:** Thread `extract_hidden_states` through `common_params` or detect at runtime, restoring the original calculation for non-HS use cases.

### [H4] tools/hs-extract-batch/hs-extract-batch.cpp:719 — Global header written before prompt processing — INCONSISTENT COUNT

```cpp
CHECKED_WRITE(&n_prompts_total, sizeof(int32_t), 1, out);  // line 719
```

`n_prompts_total` is the count of non-empty lines in prompts.txt. But `process_prompt` can fail (decode error, write error), causing the loop to `break` early. The header count will not match the actual number of prompts written. Downstream parsers reading the count will attempt to read more records than exist.

Also: empty lines are skipped in counting but also skipped in processing, so the prompt_idx sequence may have gaps if empty lines exist between prompts.

**Fix:** Write the header after processing completes (requires seeking back to position 0), or write a sentinel/actual count at the end.

### [H5] tools/hs-extract-batch/hs-extract-batch.cpp:1168-1174 — `mean_buf` zeroed per-layer but `compute_masked_mean` also relies on caller zeroing — REDUNDANT BUT CORRECT

```cpp
std::fill(mean_buf.begin(), mean_buf.end(), 0.0f);  // line 1168
compute_masked_mean(data, n_hidden, n_embd, ranges_buf, mean_buf.data());  // line 1169
```

The `compute_masked_mean` docstring says "Must be zeroed by caller" and the function only adds to `out[]`. The `std::fill` at line 1168 correctly zeros before each call. This is correct. No issue, but the contract is fragile — a future caller forgetting to zero would get garbage.

**Note:** Not a bug. Documented for completeness.

### [H6] tools/hs-extract-batch/hs-extract-batch.cpp:435-444 — `process_prompt` allocates a new `LlamaBatch` per prompt — HEAP FRAGMENTATION

```cpp
LlamaBatch batch_wrapper;
batch_wrapper.init(n_tokens, 1, false);
```

The `run_raw` mode's `process_prompt` creates a new `LlamaBatch` (heap allocation via `llama_batch_init`) on every prompt. The `run_batch` mode correctly reuses a pre-allocated batch (line 1061-1063). The raw mode does NOT reuse, contradicting the comment at line 1056-1060 which explains why reuse is necessary.

**Fix:** Hoist the batch allocation out of `process_prompt` into `run_raw`, passing it as a parameter (like the batch mode does).

---

## MEDIUM Findings

### [M1] examples/hidden-states/hidden-states.cpp:76,88,95 — Manual `llama_batch_init/free` instead of RAII — INCONSISTENT

The example uses RAII for `LlamaBackend`, `LlamaModel`, `LlamaContext` but manually calls `llama_batch_init()` and `llama_batch_free()`. The `llama_batch_free(batch)` at line 88 is in the error path, and line 95 is in the success path. This is correct but error-prone — if a new error path is added between 88 and 95 and forgets to free, it leaks.

The `LlamaBatch` RAII wrapper exists in `hs-extract-batch.cpp` (line 97-110) but is not used here.

**Fix:** Add `LlamaBatch` wrapper to `hidden-states.cpp` or share a common header.

### [M2] tests/test-hidden-states.cpp — No RAII, manual cleanup on every error path — MAINTENANCE BURDEN

The C++ test uses raw `llama_backend_init()`/`llama_free()`/`llama_model_free()`/`llama_batch_free()`/`free(tokens)` on every error path (lines 52-54, 76-80, 91-95, 107-111, 134-138). There are 5 separate cleanup blocks. Missing any one on a new error path causes a leak.

The `test-hidden-states.c` (C test) correctly uses manual cleanup (no RAII available in C).

**Fix:** Use RAII wrappers in the C++ test (the C test is fine as-is).

### [M3] tools/hs-extract/hs-extract.cpp:143,146 — `std::stoi` can throw `std::invalid_argument` — UNCAUGHT EXCEPTION

```cpp
n_threads = std::stoi(argv[i]);    // line 143
n_gpu_layers = std::stoi(argv[i]); // line 146
```

If the user passes `-t abc`, `std::stoi` throws `std::invalid_argument`, which is uncaught and calls `std::terminate()` → `abort()`. The batch tool correctly uses `strtol` with validation.

**Fix:** Replace with `strtol` + validation, or wrap in try/catch.

### [M4] tools/hs-extract/hs-extract.cpp:77,96 — `std::stoi` in `parse_layer_list`/`parse_raw_tokens` — UNCAUGHT EXCEPTION

Same issue as M3. Non-numeric input throws and aborts without a useful error message.

### [M5] src/llama-kv-cells.h:49,76 — `pos_counts` memory usage is `LLAMA_MAX_SEQ * n_cells * sizeof(int)` — LARGE ALLOCATION

```cpp
pos_counts.assign((size_t) LLAMA_MAX_SEQ * n, 0);  // line 76
```

`LLAMA_MAX_SEQ` is 64. For `n = n_ctx = 131072` (128K context), this allocates `64 * 131072 * 4 = 32MB`. This replaces the old `std::map` which allocated dynamically. The flat table is always fully allocated even if only 1 sequence is used. This is a tradeoff documented in the code.

The `reset()` at line 49 does `std::fill(pos_counts.begin(), pos_counts.end(), 0)` — a 32MB memset on every KV reset. For batch tools that reset KV per prompt, this adds overhead.

**Note:** Not a bug, but the memory and reset cost should be documented. Consider lazy allocation or per-sequence sparse tracking for large contexts.

### [M6] src/llama-kv-cells.h:545-559 — `recompute_min_max` is O(n_cells) — PERFORMANCE

```cpp
void recompute_min_max(llama_seq_id s) {
    ...
    for (uint32_t p = 0; p < n_cells; ++p) {
        if (row[p] > 0) { ... }
    }
}
```

Called when a boundary position is removed (line 526). For large contexts, this scans the entire position table. The code comments say "amortized O(n_cells) but rare" — this is correct for typical workloads but could be a hotspot for adversarial patterns.

### [M7] tools/server/server-context.cpp:5090-5093 — `GGML_ASSERT` on dynamic_cast result — ABORT ON UNEXPECTED RESULT

```cpp
GGML_ASSERT(dynamic_cast<server_task_result_hidden_states*>(res.get()) != nullptr);
```

If the server returns an error result (which is a `server_task_result_error`), this assert fires and aborts the server. But the code checks `all_results.error` before this loop (line 5085-5087), so error results should be caught. However, if a slot returns an unexpected result type, the server crashes instead of returning an HTTP error.

**Fix:** Use `dynamic_cast` with null check and return HTTP 500 instead of asserting.

---

## LOW Findings

### [L1] tools/hs-extract/README.md:50-59 — Output format documentation is WRONG

The README shows:
```json
{"prompt": "...", "tokens": [...], "hidden_states": [{"layer": 0, "data": [...]}]}
```

The actual code produces:
```json
{"n_tokens": N, "n_embd": N, "n_layers": N, "layers": [{"layer": N, "values": [...]}]}
```

No `prompt`, `tokens`, or `hidden_states` keys exist. The key is `values` not `data`. The top-level keys are `n_tokens`/`n_embd`/`n_layers`/`layers`.

**Fix:** Update README to match actual output.

### [L2] tools/hs-extract/hs-extract.cpp:21 — `llama_model_free` used but API is `llama_model_free`

The destructor calls `llama_model_free(model)`. This is correct (the API function name matches). No issue.

### [L3] .github/workflows/fork-ci.yml:139 — Self-test does not test the server endpoint — MISSING COVERAGE

The CI runs `llama-hs-extract-batch --self-test` (compute_masked_mean tests) and `--help` for tools, but does NOT:
- Start the server and test `/hidden-states` endpoint
- Run `test-hidden-states` or `test-hidden-states-c` with an actual model
- Test the `/hidden-states` endpoint with `pool=skip_mean`

The test binaries are built but never executed (they need a model file, which CI doesn't download).

**Fix:** Download a tiny test model (e.g., TinyLlama) and run the test binaries + server endpoint smoke test.

### [L4] tools/hs-extract-batch/hs-extract-batch.cpp:1009,1135 — "Pitfall #23" and "Pitfall #17" comments reference external numbering — OPAQUE

```cpp
// Create context FIRST (Qwen3.5 Gated Delta Net OOM fix -- Pitfall #23)
// CRITICAL: synchronize before reading hidden states (CUDA async race -- Pitfall #17)
```

These reference an external "Pitfalls" list that is not in the repository. Future maintainers cannot look up what Pitfall #23 or #17 are.

**Fix:** Inline the explanation or link to the document.

---

## RAII Compliance Audit

| Tool | LlamaBackend | LlamaModel | LlamaContext | FilePtr | LlamaBatch | Status |
|------|:---:|:---:|:---:|:---:|:---:|--------|
| hs-extract.cpp | ✅ | ✅ | ✅ | N/A | N/A (uses `get_one`) | PASS |
| hs-extract-batch.cpp | ✅ | ✅ | ✅ | ✅ | ✅ | PASS |
| hidden-states.cpp | ✅ | ✅ | ✅ | N/A | ❌ manual | PARTIAL (M1) |

All raw cleanup calls (`llama_backend_free`, `llama_free_model`, `llama_free`, `llama_backend_init`) are confined to RAII destructor bodies. No leaks on error paths in the two production tools.

---

## Thread Safety Audit

| Location | `llama_synchronize()` before reading HS? | Status |
|----------|:---:|--------|
| hs-extract.cpp:243 | ✅ line 243 | PASS |
| hs-extract-batch.cpp (process_prompt):458 | ✅ line 458 | PASS |
| hs-extract-batch.cpp (run_batch):1136 | ✅ line 1136 | PASS |
| hidden-states.cpp:93 | ✅ line 93 | PASS |
| server-context.cpp (send_hidden_states):2375 | ✅ line 2375 | PASS |
| llama_get_hidden_state() C API:3821 | ✅ line 3821 | PASS (safety net) |

All hidden state reads are protected by synchronize. The C API wrapper provides defense-in-depth.

---

## kv-cells Flat Table Audit

The `std::map<llama_pos, int>[LLAMA_MAX_SEQ]` → flat `std::vector<int>` rewrite is **correct**:

- **Bounds checking:** `seq_pos_inc`/`seq_pos_dec` assert `p >= 0 && (uint32_t)p < n_cells` ✅
- **Zero-init:** `reset()` does `std::fill(pos_counts.begin(), pos_counts.end(), 0)` ✅
- **Cache invalidation:** `recompute_min_max` correctly scans when boundary removed ✅
- **Thread safety:** Single-threaded (KV cache ops happen under decode lock) ✅
- **No dangling pointers:** Flat vector is owned by `llama_kv_cells`, lifetime tied to context ✅

The only concern is M5 (memory usage for large contexts) and M6 (O(n) recompute).

---

## CI Audit

The CI (`fork-ci.yml`) correctly:
- ✅ Builds all 3 tools (hs-extract, hs-extract-batch, hidden-states)
- ✅ Builds both test binaries (test-hidden-states C++ and C)
- ✅ Runs self-test (compute_masked_mean)
- ✅ Verifies RAII wrappers via grep
- ✅ Verifies CUDA sync calls via grep

The CI does NOT:
- ❌ Run test binaries with a model (L3)
- ❌ Test the `/hidden-states` server endpoint (L3)
- ❌ Test with CUDA/Vulkan backend (CPU only)

---

## API Surface Audit (llama.h)

| Function | Documented | Consistent | Notes |
|----------|:---:|:---:|-------|
| `extract_hidden_states` (ctx param) | ✅ | ✅ | Line 376, well-placed with other bools |
| `llama_set_extract_hidden_states()` | ✅ | ✅ | Line 980, mirrors `llama_set_embeddings` |
| `llama_get_hidden_state()` | ✅ | ✅ | Lines 1042-1046, documents NULL return |
| `llama_get_hidden_state_ith()` | ✅ | ✅ | Lines 1048-1052, documents negative indexing |
| `llama_get_hidden_state_n_tokens()` | ✅ | ✅ | Line 1054-1055 |

The API additions are well-documented and follow existing patterns (mirrors the embeddings API structure).

**Missing:** No `llama_get_hidden_state_n_layers()` function. Callers must use `llama_model_n_layer()` separately, which could disagree with the actual buffer if the model graph doesn't populate all layers.

---

## Dead Code / Whitespace / Unicode

- ✅ No Unicode characters (emdash, arrows, ×, …) found in any fork file
- ✅ No dead code found (previous `last_token` code was removed per commit `efdb8588c`)
- ✅ No trailing whitespace issues
- ⚠️ Opaque "Pitfall #N" comments (L4)

---

## Recommendations (Priority Order)

1. **Fix C1+C2:** Force `output_all` when `extract_hidden_states` is enabled, or move extraction inside the ubatch loop. This is a silent data corruption bug.
2. **Fix C4:** Cast to unsigned before left-shift in key computation.
3. **Fix H2:** Add bounds validation for `n_assignments` and `n_ranges` from file.
4. **Fix H3:** Restore conditional `n_outputs_max` calculation or thread the flag through.
5. **Fix H4:** Fix raw-mode header count consistency.
6. **Fix M1+M2:** Add RAII wrappers to examples and C++ test.
7. **Fix M3+M4:** Replace `std::stoi` with `strtol` in hs-extract.
8. **Fix L1:** Update README output format.
9. **Fix L3:** Add server endpoint test to CI.
