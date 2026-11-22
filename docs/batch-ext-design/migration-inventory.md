# Migration inventory — deprecated batch API in fork-owned code

Repo: `/home/a/Bureau/Work/llama.cpp` (fork `someaka/llama.cpp`), HEAD `9d0abfeb1`
("merge: upstream c622f623f (201 commits)"). Analysis date: 2026-11-21. READ-ONLY
analysis; no tracked file was modified.

Upstream change being migrated away from: `a06044b93` (#32306) marks
`llama_batch_get_one`, `llama_batch_init`, `llama_batch_free`, `llama_encode`,
`llama_decode` (and the `llama_batch` struct itself, include/llama.h:274) as
`DEPRECATED`. Replacement: opaque `llama_batch_ext` + `llama_process(ctx,
LLAMA_PROCESS_TYPE_DECODE|ENCODE, batch)`.

## 0. Method and fork footprint

`git diff --name-only b54752df4..HEAD` is polluted by the 201 upstream commits
merged at HEAD (it lists hundreds of upstream files), so the fork footprint was
taken as the task defines it: everything under `tools/hs-*` (hs-extract,
hs-extract-batch, hs-extract-batch-test, hs-extract-common, hs-probe),
`common/llama-raii.h`, `tests/test-hidden-states.cpp`,
`tools/server/server-hidden-states.*`, plus the fork-added
`tests/test-hidden-states.c` (fork commit `d67c1875a` "server+ci+tests: endpoint
pins, includes, docs (C-P2-1..7)") and `examples/hidden-states/` (both confirmed
fork-owned via `git ls-files` + fork commit history).

A regex sweep for
`\b(llama_batch_init|llama_batch_free|llama_batch_get_one|llama_decode|llama_encode)\s*\(`
over all fork-owned paths returns **29 lines = 27 real call sites + 2 comment
mentions** (tests/test-hidden-states.cpp:63 and
tools/server/server-hidden-states.h:385). Every site is listed below; the sweep
is the completeness proof. `llama_encode` has **zero** call sites in fork code.
Direct `llama_batch` struct field manipulation is inventoried per file.

## 1. Migration target API (include/llama.h at HEAD)

| Deprecated (line) | Deprecation message | Replacement |
|---|---|---|
| `llama_batch_get_one` (970–973) | "use llama_batch_ext_init and llama_batch_ext_add_token instead" | `llama_batch_ext_init(ctx)` + `llama_batch_ext_add_token` |
| `llama_batch_init` (982–986) | "use llama_batch_ext_init instead" | `llama_batch_ext_init(ctx)` |
| `llama_batch_free` (989–990) | "use llama_batch_ext_free instead" | `llama_batch_ext_free` |
| `llama_encode` (998–1001) | "use llama_process with LLAMA_PROCESS_TYPE_ENCODE instead" | `llama_process` |
| `llama_decode` (1015–1018) | "use llama_process with LLAMA_PROCESS_TYPE_DECODE instead" | `llama_process` |

New surface (include/llama.h:1024–1114, impl src/llama-batch.cpp:1455+):

- `struct llama_batch_ext;` — opaque, heap-owned.
- `llama_batch_ext_init(struct llama_context * ctx)` (1032) — **takes the
  context, not (n_tokens, embd, n_seq_max)**; sizing derives from the context
  (`new llama_batch_ext(ctx)`, src/llama-batch.cpp:1455–1457). This changes the
  RAII wrapper's `init()` signature and every call site.
- `llama_batch_ext_free` (1033), `llama_batch_ext_clear` (1034) — clear+refill
  is the natural mapping for the fork's reusable pre-allocated batches.
- `llama_batch_ext_add_token(batch, seq_id, id)` (1048) → returns batch index
  ≥0, or −1 full / −2 invalid token / −3 invalid seq id.
- `llama_batch_ext_set_pos(batch, idx, const llama_pos * pos)` (1103) —
  **positions are NOT auto-set**: "pos = not set, the caller must set it with
  llama_batch_ext_set_pos()" (1039). Old `pos[i] = i` writes map 1:1; old
  `llama_batch_get_one` sites relied on `pos = nullptr` auto-tracking and must
  now set positions explicitly (see §3 semantics notes).
- `llama_batch_ext_set_output_logits(batch, idx, bool)` (1094) — maps the old
  `batch.logits[i]` flags (default false). `set_output_embd` (1087) is currently
  equivalent.
- `llama_batch_ext_add_embd` / `add_embd_row` / `set_embd_token` /
  `set_embd_state` / `add_seq` (1049–1083) — embd/multi-seq paths; **no fork
  code uses embd or n_seq_id > 1**, so these are not needed for the migration.
- `llama_process(ctx, enum llama_process_type, llama_batch_ext *)` (1111–1114);
  "Return values are the same as llama_decode()" (1110) — the fork's
  `ret != 0` / failed-decode-invalidation contracts carry over unchanged.
- `enum llama_process_type { LLAMA_PROCESS_TYPE_ENCODE, LLAMA_PROCESS_TYPE_DECODE }`
  (308–310).

In-tree reference implementation (already migrated by upstream, useful as the
pattern to copy): `common_batch` in common/common.cpp:2207
(`batch(llama_batch_ext_init(ctx))`), `get_sub_batch` at 2290–2340
(clear → add_token → set_pos → set_output_logits), and
tools/server/server-context.cpp:4787
`llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get())`.

## 2. Complete site inventory

### 2.1 common/llama-raii.h — `LlamaBatch` wrapper (lines 63–81)

Header-only (no own target); compiled into every TU that includes it:
`llama-hs-extract` (include at hs-extract.cpp:18), `llama-hs-extract-batch`
(hs-extract-batch.cpp:64), `test-hidden-states` (tests/test-hidden-states.cpp:10),
`llama-hidden-states` (examples/hidden-states/hidden-states.cpp:10).

| Line | Current pattern | Notes |
|---|---|---|
| 64 | `llama_batch batch;` member | deprecated struct type (llama.h:274) |
| 66 | `LlamaBatch() : batch{}, initialized(false) {}` | value-init |
| 67–75 | `void init(int32_t n_tokens, int32_t embd, int32_t n_seq_max)` | signature must change: ext init takes `llama_context*` |
| 71 | `llama_batch_free(batch);` | re-init leak guard (frees previous) |
| 73 | `batch = llama_batch_init(n_tokens, embd, n_seq_max);` | the only init call |
| 76 | `~LlamaBatch() { if (initialized) llama_batch_free(batch); }` | dtor free |
| 77–80 | copy/move all `= delete` | keep non-copy/non-move semantics for the ext wrapper |
| 13–15 | header usage comment mentions `LlamaBatch batch; … .init()` | doc drift after migration |

All 6 wrapper `init()` call sites pass `embd=0, n_seq_max=1` (token-only,
single sequence): hs-extract-batch.cpp:850 and :1450 (`n_ctx`),
test-hidden-states.cpp:65 (`n_tokens`), :178 (`200`), :250 (`n_tokens`),
examples/hidden-states/hidden-states.cpp:51 (`tokens.size()`).

### 2.2 tools/hs-extract/hs-extract.cpp → target `llama-hs-extract`

(tools/hs-extract/CMakeLists.txt; links llama-common + llama)

| Line | Current pattern |
|---|---|
| 328 | `llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());` |
| 329 | `int32_t decode_result = llama_decode(ctx, batch);` |
| 331 | error string mentions `llama_decode` (prose) |

No field writes, no free (get_one owns nothing). Context: n_ctx = n_batch =
n_ubatch = ctx_n (301–305, "one ubatch per prompt … same code path as batch"),
`extract_hidden_states = true` (310). Single decode of the whole prompt;
extraction via `llama_get_hidden_state_n_tokens` (337) / `llama_get_hidden_state`
(372) — getters are NOT deprecated and stay.

### 2.3 tools/hs-extract-batch/hs-extract-batch.cpp → target `llama-hs-extract-batch`

(CMakeLists.txt builds hs-extract-batch.cpp + self-test.cpp + assignments-io.cpp
+ io-util.cpp; links llama-common + llama). **Three** `llama_decode` sites (the
task brief knew two; :1771 is the third), two `LlamaBatch` allocations, three
field-write blocks. Full decode-path description in §3.

| Line(s) | Current pattern | Role |
|---|---|---|
| 503 | `llama_batch& batch,` parameter of `process_prompt` | prefill batch passed by ref |
| 522–525 | `llama_memory_clear(mem, true)` | per-prompt KV reset (not deprecated) |
| 529–536 | `batch.token[i]=tokens[i]; batch.pos[i]=i; batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=0; … batch.n_tokens=n_tokens;` | prefill fill, run_raw path (logits ALL 0) |
| 539 | `int ret = llama_decode(ctx, batch);` | **decode #1** — process_prompt prefill (run_raw) |
| 849–850 | `LlamaBatch batch_wrapper; batch_wrapper.init(n_ctx, 0, 1);` | run_raw reusable batch |
| 879 | `process_prompt(ctx, batch_wrapper.batch, …)` | pass-through |
| 1026 | `llama_batch& batch;` member of `GenContext` | generation shares the prefill batch |
| 1041 | `llama_batch& batch = g.batch;` | alias |
| 1148 | comment mentions `llama_decode` (prose) | KV-bounds comment |
| 1193–1198 | `batch.n_tokens=1; batch.token[0]=next_token; batch.pos[0]=cur_pos; batch.n_seq_id[0]=1; batch.seq_id[0][0]=0; batch.logits[0]=(gen_step<args.generate_tokens-1)?1:0;` | single-token generation fill |
| 1200 | `ret = llama_decode(ctx, batch);` | **decode #2** — generation step |
| 1445 | comment "Allocating/freeing a llama_batch on every prompt…" (prose) | heap-fragmentation rationale |
| 1449–1451 | `LlamaBatch batch_wrapper; batch_wrapper.init(n_ctx, 0, 1); llama_batch& batch = batch_wrapper.batch;` | run_batch reusable batch |
| 1759–1767 | `batch.token[i]=tokens[i]; batch.pos[i]=i; batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=(args.generate_tokens>0 && i==n_tokens-1)?1:0; … batch.n_tokens=n_tokens;` | run_batch consumer prefill fill |
| 1771 | `int ret = llama_decode(ctx, batch);` | **decode #3** — run_batch consumer prefill |
| 1790 | `GenContext g{ctx, batch, …}` | hands same batch to generation loop |

Other TUs of this target (self-test.cpp, io-util.cpp, assignments-io.cpp,
hs-accum.h, hs-kernels.h) and tools/hs-extract-common/{layer-parse.h,
masked-mean.h, tokenize.h}: **zero** deprecated-API usage (sweep-verified).

### 2.4 tools/hs-extract-batch-test/hs-extract-batch-test.cpp → target `llama-hs-extract-batch-test`

(links llama-common + llama). Present, fork-owned, and **was missing from the
task's known-sites list**.

| Line | Current pattern |
|---|---|
| 69 | `llama_batch wb = llama_batch_get_one(warmup_tokens.data(), warmup_tokens.size());` (BOS+EOS warmup, server-style path) |
| 70 | `if (llama_decode(ctx, wb) != 0) { … exit(1); }` |
| 86 | `llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()), tokens.size());` (main decode; const_cast because get_one takes non-const) |
| 87 | `if (llama_decode(ctx, batch) != 0) { … exit(1); }` |

No field writes (get_one defaults), no frees. Raw `llama_backend_init()` at 140
— explicitly RAII-exempt per audit Check 14's comment (audit_integrity.sh:642–646).
Mode 2 compares CLI-style vs server-style captures **bitwise**; both decodes
must keep identical pos/logits semantics or the test exits 2.

### 2.5 tools/hs-probe/logits-probe.cpp → target `llama-hs-probe`

(links **llama only**, no llama-common). Raw C API, no RAII (exempt per Check 14
comment; raw `llama_backend_init()` at 159).

| Line | Current pattern |
|---|---|
| 42 | `llama_batch batch = llama_batch_init(n, 0, 1);` |
| 43–50 | fill loop: `token[i]=toks[i]; pos[i]=i; n_seq_id[i]=1; seq_id[i][0]=0; logits[i]=(i==n-1);` then `batch.n_tokens=n;` (logits on LAST token only) |
| 52 | `if (llama_decode(ctx, batch)) {` |
| 54 | `llama_batch_free(batch);` (decode-failure exit) |
| 66 | `llama_batch_free(batch);` (null-logits exit) |
| 89 | `llama_batch_free(batch);` (success exit) |

Reads `llama_get_logits_ith(ctx, n-1)` (63) — **batch-token index semantics must
survive the migration** (probe compares argmax + top-8 logits across
extraction on/off, ULP-scaled tolerance). Contexts: n_ctx=n_batch=n_ubatch=128
(179–181), prompt limit 120 tokens (173).

### 2.6 tests/test-hidden-states.cpp → target `test-hidden-states`

(tests/CMakeLists.txt:447 `llama_build(test-hidden-states.cpp)`, default link
llama-common; built only, NOT registered with CTest — see Report 2 §d).

| Line(s) | Current pattern | Role |
|---|---|---|
| 63 | comment "Decode using llama_batch_init (modern API)" | becomes stale prose |
| 64–65 | `LlamaBatch batch_wrapper; batch_wrapper.init(n_tokens, 0, 1);` | main prompt batch |
| 66 | `llama_batch & batch = batch_wrapper.batch;` | alias |
| 67–74 | fill: token/pos=i/n_seq_id=1/seq_id[0]=0/`logits[i]=(i==n_tokens-1)?1:0`; n_tokens set | logits last-only |
| 76 | `int ret = llama_decode(ctx, batch);` | **decode #1** (ctx: n_ctx=n_batch=n_ubatch=512, extract=true, lines 30–33) |
| 177–178 | `LlamaBatch fbatch_wrapper; fbatch_wrapper.init(200, 0, 1);` | HS-6b filler batch |
| 179 | `llama_batch & fbatch = fbatch_wrapper.batch;` | alias |
| 181–188 | fill ×3 rounds: `pos[i]=b*200+i`, `logits[i]=(i==199)?1:0` ("upstream requires >=1 output per decode"), n_tokens=200 | over-capacity premise: 3rd decode must FAIL |
| 189 | `int fret = llama_decode(ctx, fbatch);` | **decode #2** (in 3-iteration loop; b<2 must succeed, b==2 must fail and invalidate capture) |
| 215–218 | recovery reuse of first batch: `batch.n_tokens=n_tokens;` + logits rewrite loop | batch REUSE after failed decode |
| 219 | `if (llama_decode(ctx, batch) != 0)` | **decode #3** (recovery; capture must repopulate) |
| 249–250 | `LlamaBatch batch2_wrapper; batch2_wrapper.init(n_tokens, 0, 1);` | HS-6a multi-ubatch batch |
| 251 | `llama_batch & batch2 = batch2_wrapper.batch;` | alias |
| 252–259 | fill identical to main (logits last-only) | |
| 260 | `if (llama_decode(ctx2, batch2) != 0)` | **decode #4** (ctx2: n_ubatch=8, line 242 — forces multi-ubatch accumulation; captures compared byte-identical via memcmp at 276/282) |

Migration-sensitive contracts: failed-decode invalidation (getters empty after
the over-capacity decode, 202–210), recovery (224–227), and **byte-identical
multi-ubatch capture** (276–285). `llama_process` return codes are documented
identical to `llama_decode`, so the b==2 failure premise should carry over —
verify empirically.

### 2.7 tests/test-hidden-states.c → target `test-hidden-states-c`

**Fork-owned** (commit d67c1875a; tests/CMakeLists.txt:448–449 via
`LLAMA_TEST_NAME`). C TU — the ext API is plain C, so migration stays C-clean.
Missing from the task's known-sites list.

| Line | Current pattern |
|---|---|
| 56 | `struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);` |
| 57–63 | fill: token/pos=i/n_seq_id=1/seq_id[0]=0/`logits[i]=0` |
| 64 | `batch.logits[n_tokens - 1] = 1;` (last-token logits, separate statement) |
| 65 | `batch.n_tokens = n_tokens;` |
| 67 | `if (llama_decode(ctx, batch) != 0) {` |
| 69 | `llama_batch_free(batch);` (decode-failure cleanup) |
| 82 | `llama_batch_free(batch);` (no-hidden-tokens cleanup) |
| 170 | `llama_batch_free(batch);` (normal cleanup) |

Context n_ctx=64 (27), extract=true (28); prompt "Hello world", 16-slot token
buffer. Checks ladder top slot n_layer, out-of-range rejection, NaN scans.

### 2.8 examples/hidden-states/hidden-states.cpp → target `llama-hidden-states`

(examples/hidden-states/CMakeLists.txt; links llama + llama-common). Fork-added
example; one of the 13 files the audit reads.

| Line | Current pattern |
|---|---|
| 50–51 | `LlamaBatch batch_wrapper; batch_wrapper.init(tokens.size(), 0, 1);` |
| 52 | `llama_batch & batch = batch_wrapper.batch;` |
| 53–60 | fill: token/`pos[i]=(llama_pos)i`/n_seq_id=1/seq_id[0]=0/`logits[i]=(i==tokens.size()-1)?1:0`; `batch.n_tokens=tokens.size();` |
| 62 | `if (llama_decode(ctx, batch) != 0) {` |

Context: n_ctx=512, extract=true (28–29); n_batch/n_ubatch left at defaults.

### 2.9 tools/server/server-hidden-states.* and server-context.* — NO batch API

Sweep-verified: **zero** deprecated calls and zero `llama_batch` struct
manipulation in all four files. Only prose mentions:
server-hidden-states.h:385, server-context.cpp:530, 1174, 4117, 4675 (comments
about capture resetting per decode). The server already runs the new API via
upstream's migration: `common_batch` (common/common.cpp:2207, wraps
`llama_batch_ext_init`) and `llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE,
batch.view.get())` at server-context.cpp:4787. The fork's hidden-states seam
(HIDDEN_STATES slot isolation server-context.cpp:570–579, toggles
3661–3681/4818–4853, `send_hidden_states` in server-hidden-states.h, serializers
in server-hidden-states.cpp) never constructs or fills batches. **No migration
work needed here**; optionally refresh the stale `llama_decode()` comment
wording (not gated; audit strips comments).

### 2.10 Documentation mentions (not compiled, will go stale)

- examples/hidden-states/README.md:44 — "`llama_batch_get_one` is not used there."
- examples/hidden-states/README.md:57 — pseudocode
  `llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));`
- tests/test-hidden-states.cpp:63, common/llama-raii.h:13–15,
  hs-extract-batch.cpp:1148/1445 — comments naming the old API.

None of these are pinned by check_doc_claims.py (it gates only
tools/server/README.md's parity table + 3 code facts), but the fork's
doc-honesty practice (and audit Check 20's doc-binding style) argues for
updating them in the migration commit.

## 3. hs-extract-batch.cpp decode paths — migration detail

### 3.1 Context sizing and chunking: NO tool-side chunking

`load_model_session` (737–794), shared by run_raw and run_batch:
- `n_ctx` = `--ctx-size` or auto-derived from the prompts pre-scan
  (`ctx_from_prompts`, 682), capped at `n_ctx_train` (773).
- `cparams.n_ctx = cparams.n_batch = cparams.n_ubatch = s.n_ctx` (783–785).
  Comment at 780–781: "n_ubatch = n_ctx keeps the full prompt in one ubatch
  pass (stable CUDA graph capture)."
- `cparams.extract_hidden_states = true` (786).

**The tool never splits prompts into n_batch chunks itself.** Every prompt is
decoded in exactly one `llama_decode` call; over-length prompts are refused up
front (run_raw 872–877; producer marks `pp.skip`, consumer aborts 1734–1743).
The library-internal multi-ubatch split is exercised only by
tests/test-hidden-states.cpp HS-6a (ctx2 n_ubatch=8). Migration must preserve
n_batch=n_ubatch=n_ctx sizing — it is load-bearing for CUDA graph stability and
byte-identity.

### 3.2 Batch allocation and reuse

Exactly two `LlamaBatch` objects, both `init(n_ctx, 0, 1)` — **embd=0 (token
path only; there is NO embd path anywhere in the fork tools), n_seq_max=1**:
- run_raw: 849–850, passed by reference into `process_prompt` (879).
- run_batch: 1449–1451, used by the consumer prefill and handed to
  `generate_assignment` via `GenContext` (1790).

Allocated ONCE per run and reused across all prompts (comment 1444–1448:
per-prompt init/free caused glibc heap-corruption aborts over 90K+ iterations).
Natural ext mapping: one `llama_batch_ext_init(ctx)` per run +
`llama_batch_ext_clear()` per prompt/step, then re-add tokens. The wrapper's
re-init-frees-previous guard (llama-raii.h:70–72) has no ext analogue needed if
init happens once; keep the RAII free-on-scope-exit contract.

### 3.3 Path A — prefill (two variants, one fill pattern)

**A1: `process_prompt` (501–588), used by run_raw only.**
1. `llama_memory_clear(mem, true)` per prompt (522–525) — KV reset, seq 0 reused.
2. Fill loop 529–536: `token[i]=tokens[i]`, `pos[i]=i` (positions restart at 0
   every prompt because KV is cleared), `n_seq_id[i]=1`, `seq_id[i][0]=0`,
   **`logits[i]=0` for ALL i** ("No logits needed — we only extract hidden
   states"), `batch.n_tokens=n_tokens`.
3. `llama_decode` (539); `ret != 0` → fatal per-prompt error.
4. Getters self-synchronize (545–547): `llama_get_hidden_state_n_tokens`, then
   per-layer `llama_get_hidden_state` (566); full or mean-mode write.

**A2: run_batch consumer prefill (1751–1784).**
1. KV clear (1752–1753), profiling timer.
2. Fill loop 1759–1767: identical to A1 EXCEPT
   **`logits[i] = (args.generate_tokens > 0 && i == n_tokens - 1) ? 1 : 0`**
   (1764) — logits on the LAST prompt token only in generation mode, so the
   first generated token can be sampled from prefill. Comprehension mode: all 0.
3. `llama_decode` (1771); failure aborts via `stop_producer_and_join`.
4. Explicit `llama_synchronize(ctx)` (1783) to bracket the decode profile window.
5. Branch: `--generate` → `generate_assignment` (1789–1816, then `continue`);
   else comprehension extraction `llama_get_hidden_states_batch` (1820) +
   masked-mean accumulation (1836+, second `make_accum_key` use site at 1870).

### 3.4 Path B — generation loop `generate_assignment` (1039–1313)

State: `cur_pos = n_tokens` (1102, first position AFTER the prompt);
`next_token = LLAMA_TOKEN_NULL`; per-prompt sampler chain built at 1116–1145
(`LlamaSampler` RAII; penalties/top-k/top-p/temp/dist(seed=0) when any sampling
param non-default; otherwise manual greedy argmax). Chain is deliberately
per-prompt for RNG reproducibility (1109–1112) — do not hoist.

Loop `for gen_step in [0, args.generate_tokens)` (1147):
1. **KV bounds check** `cur_pos >= n_ctx` → warning + break (1152–1156).
2. gen_step==0: sample from PREFILL logits at batch index `n_tokens-1`
   (1158–1185): `llama_sampler_sample(sampler, ctx, n_tokens-1)` or
   `llama_get_logits_ith(ctx, n_tokens-1)` + argmax with NaN guard.
3. EOG check → break (1188–1190).
4. **Single-token fill** (1193–1198): `n_tokens=1`, `token[0]=next_token`,
   `pos[0]=cur_pos`, `n_seq_id[0]=1`, `seq_id[0][0]=0`,
   **`logits[0] = (gen_step < args.generate_tokens - 1) ? 1 : 0`** — logits on
   every generated step EXCEPT the last (no next token to sample).
5. `llama_decode` (1200); failure → return 1. `cur_pos++` (1206).
6. `llama_get_hidden_states_batch` (1209); capture must hold **exactly 1 token**
   (1217–1224) — each single-token decode refills the capture from scratch.
7. Accumulate row 0 per layer into `gen_accum`, skipping the first
   `args.token_skip` steps (1232–1244).
8. Sample next token from THIS step's logits at index 0 (1247–1272), unless
   last step.

After the loop: per-layer mean = gen_accum/gen_count → per-assignment
accumulators via `make_accum_key(group, mask, target_layers[li])` (1280–1309).

### 3.5 Rules summary (must survive migration byte-for-byte)

| Aspect | Rule |
|---|---|
| Chunking | none tool-side; n_batch=n_ubatch=n_ctx; one decode per prompt; over-length refused |
| Positions | prefill `pos[i]=i` from 0 (KV cleared per prompt); generation `pos=cur_pos` continuing from n_tokens, +1 per step; bound `cur_pos < n_ctx` |
| Sequences | always seq_id 0, n_seq_id 1 (single sequence; KV cleared between prompts) |
| Logits | process_prompt: all 0. run_batch prefill: last=1 iff `--generate>0`, else all 0. Generation step: `logits[0]=1` except final step. (Other tools: last-token=1, except get_one sites which request NO logits.) |
| embd | none anywhere (all `init(...,0,1)`; get_one embd=nullptr) |
| Batch lifetime | one allocation per run, reused; generation reuses the prefill batch object |
| Decode returns | `ret != 0` fatal everywhere except the test's deliberate over-capacity failure |

### 3.6 Semantics notes for the ext migration

- `llama_batch_get_one` defaults (src/llama-batch.cpp:1145–1157): pos=nullptr
  (auto-tracked by decode), n_seq_id/seq_id=nullptr (seq 0), logits=nullptr
  (no logits rows). The ext API requires explicit `set_pos` (llama.h:1039).
  For hs-extract (fresh ctx, single decode) and batch-test (memory_clear before
  the real decode, comment at hs-extract-batch-test.cpp:81–82 "so the decode
  starts at pos 0"), explicit positions 0..n−1 reproduce the auto-tracked
  values exactly — but this must be verified per site, incl. the warmup decode
  (positions 0..1 before the clear).
- All-false logits batches decode successfully on extraction contexts today
  (process_prompt comprehension mode), so "no logits" get_one sites map to
  simply never calling `set_output_logits(...,true)`.
- `llama_get_logits_ith` indices used by the fork (n_tokens−1 at prefill, 0 at
  generation steps, n−1 in the probe) are BATCH-token indices; ext batch order
  must equal the old fill order (it will, if tokens are added in loop order).
- `llama_process` return values are documented identical to `llama_decode`
  (llama.h:1110), preserving the failed-decode-invalidation test premise.

## 4. Target compilation map (summary)

| File | CMake target | Links |
|---|---|---|
| common/llama-raii.h | header-only → 4 TUs below | — |
| tools/hs-extract/hs-extract.cpp | llama-hs-extract | llama-common, llama |
| tools/hs-extract-batch/hs-extract-batch.cpp (+self-test, assignments-io, io-util) | llama-hs-extract-batch | llama-common, llama |
| tools/hs-extract-batch-test/hs-extract-batch-test.cpp | llama-hs-extract-batch-test | llama-common, llama |
| tools/hs-probe/logits-probe.cpp | llama-hs-probe | llama only |
| tests/test-hidden-states.cpp | test-hidden-states (llama_build) | llama-common (default) |
| tests/test-hidden-states.c | test-hidden-states-c (llama_build + LLAMA_TEST_NAME) | llama-common (default) |
| examples/hidden-states/hidden-states.cpp | llama-hidden-states | llama, llama-common |
| tools/server/server-hidden-states.cpp, server-context.cpp | server-context lib → llama-server | (no batch migration needed) |

**Totals: 27 deprecated call sites across 8 code files** (3 in llama-raii.h,
2 hs-extract, 3 hs-extract-batch, 4 batch-test, 5 logits-probe, 4
test-hidden-states.cpp, 5 test-hidden-states.c, 1 example) **+ 6 wrapper
init() call sites + ~10 direct field-write blocks**; zero llama_encode; zero
batch API in server files and hs-extract-common.
