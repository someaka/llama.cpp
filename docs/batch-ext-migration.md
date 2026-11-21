# Batch-ext migration (fork tools, tests, example)

Status: implemented 2026-11-21 on top of merge 9d0abfeb1 (upstream c622f623f).
Companion design reports: `batch-ext-design/api-semantics.md` (API contract),
`batch-ext-design/migration-inventory.md` (site map),
`batch-ext-design/audit-and-gate-risk.md` (pins and gate discipline).

## Why

Upstream a06044b93 ("llama : deprecate old batch API", #32306) marked
`llama_batch_init/free/get_one`, `llama_decode` and `llama_encode`
LLAMA_DEPRECATED. The c622f623f merge brought that deprecation into the fork,
so every fork TU still using the value-type batch API emitted -Wdeprecated
warnings. The replacement is the context-bound `llama_batch_ext` API plus
`llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)`.

## Contract (verified against src, not assumed)

- The deprecated functions are thin compat wrappers: `llama_decode` builds a
  `llama_batch_compat`, converts it to an ext batch and calls the SAME
  `decode(ext)` that `llama_process` uses (src/llama-context.cpp:5371-5379;
  conversion src/llama-batch.cpp:1542-1616). A faithful manual migration is
  therefore byte-identical by construction.
- Ext capacity is fixed at init to the context's `n_batch`
  (src/llama-batch.cpp:1225-1235); `add_token` returns -1 when full
  (:1263-1268). Prompts longer than `n_batch` must be chunked by the caller;
  ubatch splitting still happens inside decode via `memory->init_batch`.
- `llama_batch_ext_clear` resets the token list and keeps capacity; reuse
  across decodes is safe (:1256-1261).
- Positions are NOT auto-set: an unset pos silently decodes at position 0.
  Every token gets an explicit `llama_batch_ext_set_pos` (one llama_pos for
  text tokens). The old `get_one` pos=NULL auto-track (compat :1557-1586) is
  replicated with explicit 0..n-1 positions (all get_one sites decode into a
  fresh or just-cleared KV cache).
- `add_token(batch, 0, id)` reproduces n_seq_id=1 / seq_id[0]=0 exactly.
- `set_output_logits(idx, true)` == old `logits[i] != 0`; logits and embd
  output are the same single bool. The compat path defaulted logits=NULL to
  "last token only" (:1612) -- replicated at every former get_one site.
- Hidden-state capture is independent of output flags (llama-context.cpp
  :2275-2277); the ladder bytes do not depend on which rows request logits.
- Lifetime: the ext batch binds the context's memory pointer; free it before
  the context. RAII declaration order (LlamaContext before LlamaBatch) gives
  reverse-order destruction, which is correct.
- `llama_process` return codes are identical to `llama_decode` (0 success,
  1 no-KV-found, 2 aborted, -1 fatal).

## Design decisions

1. `LlamaBatch` (common/llama-raii.h) now owns `llama_batch_ext *`;
   `init(llama_context * ctx)` replaces `init(n_tokens, embd, n_seq_max)`.
   Re-init frees the previous batch (leak guard kept); still neither
   copyable nor movable.
2. Fill pattern follows upstream `common_batch::get_sub_batch`
   (common/common.cpp): clear -> add_token (check idx < 0) -> set_pos ->
   set_output_logits only when true. Error paths use each file's existing
   style (fprintf + return / exit), never a new abort class.
3. No chunking changes: hs-extract, hs-extract-batch and logits-probe size
   n_ctx = n_batch = n_ubatch (one ubatch per prompt; over-length prompts are
   rejected before decode). Ext capacity therefore covers every accepted
   prompt, and the pre-checks stay authoritative. test-hidden-states.cpp
   filler batches (200 tokens) fit the 512-token n_batch; the HS-6b
   over-capacity premise is KV-side (600 > 512) and is unchanged.
4. logits-probe and hs-extract-batch-test stay raw C API (RAII-exempt per
   audit Check 14); their batch calls were migrated in place, keeping every
   free path.
5. Audit pins preserved: no exit-family bindings in the scan_prompts_file
   window, checked_write returns still tested, no FA flag literals in the
   hs-extract TUs, no `<=` against layer limits, checkpoint durability order
   untouched.

## Sites migrated (27 call sites, 8 code files)

| File | Change |
|---|---|
| common/llama-raii.h | LlamaBatch wraps llama_batch_ext; init(ctx) |
| tools/hs-extract/hs-extract.cpp | get_one -> ext fill + llama_process |
| tools/hs-extract-batch/hs-extract-batch.cpp | 3 decode sites, 2 wrapper inits, GenContext/process_prompt signatures, 3 fill blocks |
| tools/hs-extract-batch-test/hs-extract-batch-test.cpp | warmup + main get_one -> ext (raw API) |
| tools/hs-probe/logits-probe.cpp | init/fill/decode/3 frees -> ext (raw API) |
| tests/test-hidden-states.cpp | 3 wrappers, 4 decodes, filler/recovery/multi-ubatch fills |
| tests/test-hidden-states.c | init/fill/decode/3 frees -> ext (C) |
| examples/hidden-states/hidden-states.cpp | wrapper init + fill + decode |

Docs updated: examples/hidden-states/README.md pseudocode; stale
llama_decode/llama_batch comments in the migrated TUs.

## Verification

- Build: zero warnings in fork TUs (the deprecation warnings are gone).
- gate_batch.py check vs a re-captured anchor: ALL_BYTE_IDENTICAL.
- Anchor re-capture cause (README anchor discipline): the committed anchor
  (s5/s6: 09-16-restored 2026-08-18 closure lineage; s7/s8/s9: 10-05
  re-baseline 7ff69888a for fe98cc2d1's generation-path change) predates
  upstream CUDA numerics evolution merged on 2026-11-15 (4835b6a3c,
  upstream range 2a4daf0e0..b54752df4): e96417565 "cuda: keep columns in
  registers for cooperative softmax" (#28982) and 278a89cb7 "CUDA/HIP:
  refactor the use of _sync warp primitives" (#31617). That merge was never
  gate-verified (the gate model was off disk at the time). Differential
  proof: a FRESH build of the Nov15 tree (4835b6a3c), a FRESH build of the
  merged + prune-fixed tree (82ce9de2f), and the incremental production
  build all produce IDENTICAL gate digests (all 10 entries; all three
  differ from the committed anchor in exactly the same 8 artifacts) --
  today's 201-commit merge, conflict resolution and gemma4 extraction
  guard are byte-neutral relative to Nov15, and the incremental cache is
  sound. The anchor is re-captured from the FRESH build of 82ce9de2f
  (pre-migration); model sha256 recorded in the gate README.
- Behavioral receipts: self-test 33/33; test-hidden-states (C++ and C) with
  models/hs-fixture.gguf; logits-probe plain + --runtime-toggle;
  hs-extract-batch-test mode 2 (CLI vs server bitwise); audit 20/20;
  check_doc_claims; whitespace-clean diffs.

(Receipt values are recorded in the commit message and the gate README.)