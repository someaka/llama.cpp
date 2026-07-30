# Adversarial C++ Re-Audit — CrimsonRed llama.cpp Fork (Machine D, Retry)

**Date:** 2026-07-30 · **Host:** Machine D (AMD Vulkan, no CUDA, no model)
**Scope:** fork-only code — `tools/hs-extract*`, hidden-state hunks in `src/llama-context.cpp`, `tools/server/server-context.cpp`
**Method:** read-verify of current source + execution of the prebuilt self-test binary. Report-only (no edits, no git ops).
**Auditor note:** Pass-1 timed out before writing its report; every claim below was re-derived from the working tree, not from commit messages.

---

## (A) Prior-Fix Verification Table

All fixes verified **present** by reading the live code. Evidence is file:line with quoted tokens.

| # | Prior fix | Present? | Evidence (file:line) |
|---|-----------|:--------:|----------------------|
| C1 | Buffer-overflow bounds check (hidden-state copy) | ✅ | `src/llama-context.cpp:2172-2179` — `if (dst_offset + copy_size > hidden_state_buf.size() * sizeof(float)) { ... return -1; }`; tool-side ctx bound `hs-extract-batch.cpp:1010,1637` and gen KV bound `:1799` |
| H2 | Integer-overflow guards in size math | ✅ | `src/llama-context.cpp:1916` `(size_t)n_layers * n_tokens_all * n_embd_out`; `:2169-2171` `(size_t)il * n_tokens_all * n_embd_out * sizeof(float)`; token-count guard `:2141-2147` `total_hidden > (int64_t)INT32_MAX` |
| H3 | Stream-sync before tensor readback | ✅ | `src/llama-context.cpp:2157` `ggml_backend_sched_synchronize(sched.get());` immediately before `ggml_backend_tensor_get` loop (`:2160-2182`); load-bearing pre-copy sync, plus redundant post-copy `:2193`; API-entry sync `:3972,3980,3988,4000` |
| — | `ggml_backend_sched_synchronize` at 703/1415/2157 | ✅ | confirmed: `:703` (context `synchronize()`), `:1415` (pipeline-parallel reuse), `:2157` (HS readback). Also `:2193` (defensive) |
| — | Generate-mode guards (M1/M2) | ✅ | `hs-extract-batch.cpp:1739-1744` (M1: reject `--generate`+`--save-per-story`); `:1748-1757` (M2: reject assignments with `skip>0 || mask_type==1`) |
| — | token_skip honored in generation | ✅ | `hs-extract-batch.cpp:1863` `if (gen_step >= args.token_skip)` gates accumulation; help text `:172-173` |
| — | `--no-bos` flag in `--help` | ✅ | help line: `--no-bos  Do not add BOS token (for models like Qwen3.5 ...)`; parsed `:210-211`, threaded to tokenize `:1005,1635` |
| M4 | start==end range rejection at read time | ✅ | `hs-extract-batch.cpp:873-880` `if (start == end) { ... "degenerate token span" ... return error; }`; plus `:869-872` start>end |
| — | `--batch --mean` guard (round-4) | ✅ | `hs-extract-batch.cpp:354-357` `if (args.batch_mode && args.mean_mode) { ... exit(1); }` |
| — | Atomic final-output writes | ✅ | `hs-extract-batch.cpp:1181-1194` (`write_batch_output`: temp + rename), `:1216-1234` (`write_checkpoint`), `:1557,2126-2133` (stories sidecar) |
| — | fsync-before-rename | ✅ | `FilePtr::sync()` `:89-97` (`fflush` + `fsync(fileno(fp))`); called at `:1189,1228,2127` before each `rename()` |
| — | `FilePtr::reset()` (close before rename) | ✅ | `hs-extract-batch.cpp:88` `void reset() { if (fp) { fclose(fp); fp = nullptr; } }`; called `:1190,1229,2128` |
| — | `_hs_synced` reset at decode entry | ✅ | `src/llama-context.cpp:1926` `_hs_synced.store(false, ...)` at decode start; published `true` only after multi-ubatch loop `:2223`; clear-on-disable `:1246,1249` |

**Verdict: all 13 audited prior fixes are present and correctly placed.** No regressions found.

---

## (B) Self-Test Result (exact)

```
$ ./build/bin/llama-hs-extract-batch --self-test
...
17/17 tests passed
All self-tests passed
$ echo $?  → 0
```

**Exact pass count: 17/17.** Binary: `build/bin/llama-hs-extract-batch` (built 2026-07-30 12:31, 107664 bytes).
The `Error:` lines in output are **expected negative-path tests** 5–9 (masked-mean hard errors) and 13–14 (single-range hard errors) exercising their guards — all report `PASS`. Test 17 writes/restores a real checkpoint (`42 prompts`) round-trip.

`--no-bos` confirmed in `--help` (see A).

---

## (C) New Findings

### P1 — silent all-zero / empty-result paths (the canonical fear)

**P1-1 · Null hidden-state layer is silently zero-filled, not errored** — `src/llama-context.cpp:2160-2182`
The readback copy loop does `if (t == nullptr) continue;` (`:2162`). But the buffer was pre-zeroed at `:1918` (`std::fill(...,0.0f)`), `n_hidden_tokens` is incremented for ALL tokens regardless (`:2148`), and `get_hidden_state(layer)` (`:981-995`) returns a **non-null** pointer into that zeroed region for a skipped layer. The token-count asserts (`:2129` "at least one non-null", `:2134` "non-null layers match") never assert *every* layer is non-null.
→ If any single `t_hidden_layers[il]` is null on an otherwise-supported model, the tool's `if (!data)` check (`hs-extract-batch.cpp:2006,1865`) does **not** fire (data is non-null), and `compute_masked_mean` silently averages zeros → a zero-vector contribution accumulated with `count++`. **This is exactly the silent all-zero-hidden-state bug class.** Defended only by model-level guarantees that supported archs populate all layers.
*Fix:* in the copy loop, treat an unexpected null on a supported arch as a hard error (`LLAMA_LOG_ERROR; return -1;`) rather than `continue`; or track a per-layer "populated" bitmap and have `get_hidden_state` return nullptr for unpopulated layers.

**P1-2 · `--generate N --token-skip K` with K ≥ N silently produces an empty result** — `hs-extract-batch.cpp:1863,1908-1936`
No validation that `token_skip < generate_tokens`. If `token_skip >= generate_tokens`, the gate `if (gen_step >= args.token_skip)` (`:1863`) never passes, so `gen_count[li]` stays 0 for every layer; then `:1910-1911` `if (n_gen == 0) continue;` skips every layer for every assignment. The prompt is still counted (`n_processed++`, `:1942`) and the run exits 0, but **zero accumulator entries are written** → `output.bin` with a header and no data. A 239K-prompt run with `--generate 10 --token-skip 10` completes "successfully" with nothing in it. No warning.
*Fix:* in `parse_args`, reject `token_skip >= generate_tokens` when `--generate` is set (or at least emit a loud warning); and/or after the loop, fail if `n_processed > 0` but `accumulators.empty()`.

### P1/P2 — integer overflow in server pointer arithmetic

**P1-3 · `skip_mean` and last-token pooling use `int` pointer offsets (overflow for large ctx·embd)** — `tools/server/server-context.cpp:2512, 2521-2522`
- `:2512` `const float * tok = hs + t * n_embd;` — `t` is `int32_t`, `n_embd` is `int32_t`; product computed in `int` before pointer add.
- `:2521` `hs + (n_hs_tokens - 1) * n_embd` — same `int` multiply.
For large context × embd (e.g. n_hs_tokens≈131072, n_embd≈16384 → ~2.1e9, at the `INT32_MAX` edge) the offset overflows → reads from a wrapped pointer (OOB / wrong data). Contrast the `pool=none` path (`:2492-2493`) which correctly casts `(size_t)n_hs_tokens * (size_t)n_embd`, and `hs-extract.cpp:278` which also casts to `size_t`. The two pooling paths are inconsistent with the guarded path.
*Fix:* cast to `size_t` at `:2512` and `:2521` (`hs + (size_t)t * n_embd`, `hs + (size_t)(n_hs_tokens - 1) * n_embd`).

### P2 — resource / robustness

**P2-1 · Generation sampler leaked on every generation error path** — `hs-extract-batch.cpp:1780 vs 1813/1844/1853`
`sampler` is allocated at `:1780` but only freed at `:1899-1901` after the loop completes. The error exits inside the generation loop — no-logits `:1812-1814`, decode-fail `:1841-1845`, extract-fail `:1851-1854` — all do `STOP_PRODUCER_AND_JOIN(); return 1;` **without** `llama_sampler_free(sampler)`. One sampler chain leaks per failed run (failures abort the run, so bounded to ~1, but real and easy to fix).
*Fix:* wrap the sampler in an RAII guard, or `llama_sampler_free(sampler)` before each early `return` in the generation block.

**P2-2 · `--raw` mode has no atomic write / fsync** — `hs-extract-batch.cpp:963,982`
`run_raw` opens the output directly (`fopen(output_path,"wb")`, `:963`) and writes in place with no temp+rename and no `sync()`. A crash mid-run leaves a truncated raw dump. Lower priority because `--raw` is documented debug/parity mode, but it is inconsistent with the `--batch`/checkpoint/stories atomicity guarantees and the Python parser would misread a truncated raw file.
*Fix (optional):* apply the same temp+`sync()`+`rename` pattern used by `write_batch_output`.

**P2-3 · `static thread_local` accumulation buffers are unnecessary** — `hs-extract-batch.cpp:1761-1762`
`gen_accum`/`gen_count` are `static thread_local` but `run_batch` is single-consumer; the `thread_local` storage duration adds no safety and persists buffers across calls in the same thread. Not a correctness bug (they are `std::fill`-zeroed each prompt at `:1766`), just misleading. *Fix:* make them plain locals or function-scope `static` without `thread_local`.

### Info (verified clean)
- `compute_masked_mean` / `compute_single_range_mean` bounds: every range is validated before indexing (`:552-561,592-600`); `data + t*n_embd` is only reached with `t < n_tokens`. No OOB.
- Assignment/checkpoint readers: all counts bounds-checked (`MAX_*` constants `:108-114`), n_embd cross-validated (`:1296-1300`), version-gated v1/v2 restore (`:1355-1368`).
- `process_prompt` full-mode size math cast to `size_t` (`:714`); `--batch-size>1` rejected loudly (`:1456-1461`).
- Tokenizer output bounds-checked against vocab (`:740-746`).
- Server `pool=none` DoS size cap present and correctly `size_t` (`:2447-2449`); negative `skip_offset` clamped (`:2498`); `normalize` guards `norm>0` (`:2529`).
- `hs-extract.cpp` readback: sync before read (`:244`), null check per layer (`:270-272`), `size_t` loop bound (`:278`). Clean.

---

## (D) Ranked Top-10

| Rank | Sev | Finding | Location |
|------|-----|---------|----------|
| 1 | P1 | Null layer tensor → silent zero-filled hidden states (non-null ptr defeats tool null-check) | `src/llama-context.cpp:2162` (+`:1918,:981`) |
| 2 | P1 | `--token-skip >= --generate` → silent empty output.bin, exit 0 | `hs-extract-batch.cpp:1863,1910` |
| 3 | P1/P2 | `int` pointer-offset overflow in server `skip_mean`/last-token pooling | `server-context.cpp:2512,2521` |
| 4 | P2 | Sampler chain leaked on all generation error paths | `hs-extract-batch.cpp:1813,1844,1853` |
| 5 | P2 | `--raw` mode lacks atomic write / fsync (truncation risk) | `hs-extract-batch.cpp:963,982` |
| 6 | P2 | Misleading `static thread_local` gen buffers | `hs-extract-batch.cpp:1761-1762` |
| 7 | — | Prior-fix verification: 13/13 present (C1,H2,H3,M4,guards,atomic,fsync) | see (A) |
| 8 | — | Self-test 17/17 PASS, exit 0 | `--self-test` |
| 9 | — | `--no-bos` present in `--help` and wired to tokenize | `hs-extract-batch.cpp:168,210,1005,1635` |
| 10 | — | Multi-ubatch accumulation asserts token completeness (defense vs. partial fills) | `src/llama-context.cpp:2218` |

**Bottom line:** Every previously-reported fix has landed and is correctly placed, and the self-test is green (17/17). The two findings that matter for research correctness are **P1-1** (a null layer would yield *silent zeros*, the canonical fear — currently guarded only by model-level invariants, not asserted) and **P1-2** (`--token-skip >= --generate` silently emits an empty result). P1-3 is a real but large-input-only integer overflow in the server pooling paths that should match the already-correct `pool=none` cast. No use-after-free, no uninitialized read, and no missing-error-return found on the `--raw`/`--mean`/`--generate` parse paths beyond the above.
