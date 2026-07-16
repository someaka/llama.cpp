# NEXT-PASS-PLAN: llama.cpp Fork

**Date:** 2026-06-30 (last updated 2026-07-16)  
**Session:** Full architectural audit + fixes

## Items Resolved

| ID | Description | Verification |
|----|-------------|-------------|
| F1 | hs-extract.cpp RAII wrappers | Compiled, --help works, CI check passes |
| F2 | examples/hidden-states.cpp RAII wrappers | Compiled successfully |
| F3 | hs-extract-batch.cpp FilePtr for checkpoint I/O | Compiled, self-test 15/15 passes |
| F4 | CI checkout@v4 -> v7 | Workflow updated |
| F5 | CI ccache | hendrikmuhs/ccache-action@v1.2 added |
| F6 | Redundant #pragma GCC ivdep removed | Compiled successfully |
| F7 | cxx_std_17 in batch CMakeLists | Compiled successfully |
| F10 | Warning for unpatched models | LLAMA_LOG_ERROR added |
| C1 | Multi-ubatch extraction moved inside loop | n_hidden_tokens accumulates across ubatches, validated |
| C2 | output_all documented as independent of HS | Hidden states read from t_hidden_layers, not output slots |
| C3 | Race condition documented | _hs_synced atomic flag with acquire/release ordering |
| C4 | Signed left-shift fixed to uint64_t flat key | make_accum_key uses explicit uint casts |
| H1 | assign.skip validation (hard error on OOB) | compute_masked_mean returns -1 on bad ranges |
| H2 | MAX_ASSIGNMENTS/MAX_RANGES bounds validation | All file reads validated against limits |
| H4 | Raw mode: empty prompts are now hard errors | No silent skip, returns 1 |
| H6 | run_raw uses pre-allocated LlamaBatch | Heap fragmentation eliminated |
| M1 | hidden-states.cpp uses LlamaBatch RAII | Verified in CI check 3 |
| M2 | test-hidden-states.cpp uses RAII wrappers | LlamaModel/LlamaContext/LlamaBackend |
| M3/M4 | hs-extract uses strtol (not std::stoi) | All parse functions use strtol+validation |
| L1 | README updated to match actual JSON output | n_tokens/n_embd/values keys |
| L3 | CI now downloads test model + runs endpoint tests | Full server smoke test with 6 endpoint tests |
| CI-5 | fclose check rewritten to AST-style scan | Verifies all fclose inside FilePtr struct |
| CI-13 | Thread notification check updated for producer-consumer | Matches pfq.cv + producer_done pattern |

## Items Documented, No Change Required

### F8: server_n_outputs_max unconditional n_batch
**Decision:** Keep as-is. Justified tradeoff.
**Reason:** The function receives `common_params` which has no `extract_hidden_states` field (that's a context param). The server can't know at allocation time whether a future slot will need per-token hidden states. The `--no-hidden-states` flag now restores the upstream optimization for non-HS workloads.
**Memory cost:** `n_batch * n_embd * 4` bytes per slot (~16MB for typical configs). Acceptable.

### F9: llama_get_hidden_state() synchronizes on every call
**Decision:** Keep as-is. API correctness over 2% performance.
**Reason:** The C API contract is that `llama_get_hidden_state()` returns valid data. The `_hs_synced` atomic flag now prevents redundant syncs when data is already synced. `llama_get_hidden_states_batch()` does a single sync for all layers.

### M5: pos_counts memory usage (LLAMA_MAX_SEQ * n_cells * sizeof(int))
**Decision:** Acceptable tradeoff. 32MB for 128K context, eliminates heap corruption from rb-tree node alloc/free.

### M6: recompute_min_max O(n_cells)
**Decision:** Amortized O(1). Only called when boundary position removed, which is rare relative to inc/dec calls.

## Verification Block

Commands run:
- `cmake --build build-vulkan --target llama-hs-extract-batch test-hidden-states test-hidden-states-c` -> SUCCESS (0 warnings)
- `build-vulkan/bin/llama-hs-extract-batch --self-test` -> 15/15 PASS
- `build-vulkan/bin/llama-hs-extract --help` -> Correct output
- CI verification checks (13/13) -> ALL PASS

What remains unverified (genuine blockers):
- Full CI run on GitHub Actions (requires push to GitHub)
- CUDA build/test (requires A's 3090)
- Long-running batch extraction (requires model + dataset on A)
