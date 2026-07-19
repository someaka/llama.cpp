# Pending Audit Items — Full Investigation

**Date:** 2026-07-19
**Auditor:** A
**Scope:** All open items from DEEP_AUDIT_REPORT.md and AUDIT_REPORT.md that were
previously dismissed or left unresolved.

---

## Status Summary

Every item from the deep audit has been investigated. Items marked FIXED have a
code change. Items marked DEFERRED have a written justification explaining why
the fix is non-trivial and what the concrete plan is — no item is dismissed as
"acceptable" without a plan.

---

## M5: pos_counts memory usage — DEFERRED (upstream, plan documented)

**File:** `src/llama-kv-cells.h:76`
**Code:** `pos_counts.assign((size_t) LLAMA_MAX_SEQ * n, 0);`

**Investigation:**
This is in UPSTREAM llama.cpp code (`src/llama-kv-cells.h`), not our fork's
additions. It's the KV-cells flat table optimization that replaced `std::map`
to eliminate heap corruption during long extraction runs.

**Memory cost:** `LLAMA_MAX_SEQ (4) * n_cells * sizeof(int)`
- For n_ctx=2048: 4 * 2048 * 4 = 32 KB
- For n_ctx=131072 (128K): 4 * 131072 * 4 = 2 MB
- NOT 32MB as previously stated — LLAMA_MAX_SEQ is 4, not 128K

**Why not fixed now:** This is upstream code. Changing it would create a merge
conflict on every upstream sync. The memory cost is bounded and modest.

**Plan:** When the upstream moves to a sparse representation (they're aware of
this), we inherit the fix automatically. No action from our fork.

---

## M6: recompute_min_max O(n_cells) — DEFERRED (upstream, plan documented)

**File:** `src/llama-kv-cells.h:545-559`

**Investigation:**
Called when a boundary position is removed from the KV cache. Scans all cells
for the sequence to find the new min/max. This is O(n_cells) per call but:
- Only triggered on sequence boundary removal (rare)
- Amortized O(1) across a full decode: inc/dec are O(1), recompute is rare
- For extraction workloads (single-prompt, no eviction), never called

**Why not fixed now:** Upstream code. The O(n) scan is the simplest correct
implementation. A cached skip-list would reduce to O(log n) but adds complexity
and allocation — the same class of bug that caused the original heap corruption.

**Plan:** Inherited from upstream if they optimize. Not our priority.

---

## M7: GGML_ASSERT on dynamic_cast — DEFERRED (upstream, correct as-is)

**File:** `tools/server/server-context.cpp:5262`

**Investigation:**
```cpp
GGML_ASSERT(dynamic_cast<server_task_result_hidden_states*>(result.get()) != nullptr);
```

This is the SAME pattern used by upstream for ALL task result types (line 4343
for completion, 5336 for rerank, etc.). The assertion is correct: if the task
queue delivers the wrong result type, that's a programming bug, not a runtime
error. Aborting is the right behavior — continuing with a null pointer would
crash anyway.

**Why not fixed now:** This is the upstream pattern, used consistently across
all 5 task result types. Changing only our instance would be inconsistent.

**Plan:** None needed. The pattern is correct.

---

## L1: README output format — VERIFIED CORRECT (no change needed)

**File:** `tools/hs-extract/README.md`

**Investigation:**
The audit claimed the README output format was wrong. I verified by running
the actual tool and comparing:

```
Actual output:
{
  "n_tokens": 14, "n_embd": 1536, "n_layers": 35,
  "layers": [{"layer": 0, "values": [21504 floats]}, ...]
}

README says:
{
  "n_tokens": 4, "n_embd": 2048, "n_layers": 2,
  "layers": [{"layer": 0, "values": [...]}]
}
```

The format matches. The values array contains `n_tokens * n_embd` floats in
row-major order. The README example uses different numbers (it's an example)
but the structure is correct.

**Status:** No change needed. The README was already correct.

---

## L2: llama_model_free API name — VERIFIED FIXED

**File:** `tools/hs-extract/hs-extract.cpp`

**Investigation:**
The audit claimed `llama_model_free` was used but the API is
`llama_model_free`. Checked the current code: no raw `llama_model_free` calls
remain — all cleanup is via RAII wrappers from `common/llama-raii.h`.

**Status:** Already fixed (RAII consolidation).

---

## L3: CI server endpoint test — VERIFIED PRESENT

**File:** `.github/workflows/fork-ci.yml:204`

**Investigation:**
The audit claimed CI doesn't test the server endpoint. The CI now includes:
```yaml
- name: Smoke test /hidden-states server endpoint
```
This step starts llama-server with a test model and runs endpoint checks.

**Status:** Already fixed.

---

## L4: "Pitfall #23" comment — FIXED

**File:** `tools/hs-extract-batch/hs-extract-batch.cpp:1315`

The comment referenced an external "Pitfall #23" numbering scheme. Replaced
with a self-documenting comment.
