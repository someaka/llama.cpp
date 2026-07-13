# NEXT-PASS-PLAN: llama.cpp Fork

**Date:** 2026-06-30  
**Session:** Full architectural audit + fixes

## Items Resolved This Session

| ID | Description | Verification |
|----|-------------|-------------|
| F1 | hs-extract.cpp RAII wrappers | Compiled, --help works, CI check passes |
| F2 | examples/hidden-states.cpp RAII wrappers | Compiled successfully |
| F3 | hs-extract-batch.cpp FilePtr for checkpoint I/O | Compiled, self-test 5/5 passes |
| F4 | CI checkout@v4 -> v7 | Workflow updated |
| F5 | CI ccache | hendrikmuhs/ccache-action@v1.2 added |
| F6 | Redundant #pragma GCC ivdep removed | Compiled successfully |
| F7 | cxx_std_17 in batch CMakeLists | Compiled successfully |
| F10 | Warning for unpatched models | LLAMA_LOG_WARN added |

## Items Documented, No Change Required

### F8: server_n_outputs_max unconditional n_batch
**Decision:** Keep as-is. Justified tradeoff.
**Reason:** The function receives `common_params` which has no `extract_hidden_states` field (that's a context param). The server can't know at allocation time whether a future slot will need per-token hidden states.
**Memory cost:** `n_batch * n_embd * 4` bytes per slot (~16MB for typical configs). Acceptable.
**To change:** Would require adding `extract_hidden_states` to `common_params` or `server_params`, then threading it through to `server_n_outputs_max()`. Medium complexity, low benefit.

### F9: llama_get_hidden_state() synchronizes on every call
**Decision:** Keep as-is. API correctness over 2% performance.
**Reason:** The C API contract is that `llama_get_hidden_state()` returns valid data. Callers who don't know about CUDA async semantics would get stale data without the sync. The batch tool already calls `llama_synchronize()` once before the loop, but the per-call sync adds ~2% overhead.
**To change:** Add `llama_get_hidden_state_nosync()` variant, or expose the C++ context directly (breaks ABI). Not worth the API surface change.

## Verification Block

Commands run:
- `cmake --build build --target llama-hs-extract` -> SUCCESS
- `cmake --build build --target llama-hs-extract-batch` -> SUCCESS
- `cmake --build build --target llama-hidden-states` -> SUCCESS
- `cmake --build build --target test-hidden-states test-hidden-states-c` -> SUCCESS
- `build/bin/llama-hs-extract-batch --self-test` -> 5/5 PASS
- `build/bin/llama-hs-extract --help` -> Correct output
- CI verification checks (RAII, CUDA sync, off-by-one, FilePtr) -> ALL PASS

What remains unverified:
- Full CI run on GitHub Actions (requires push)
- Server endpoint test (requires running model)
- Long-running batch extraction (requires model + dataset)
