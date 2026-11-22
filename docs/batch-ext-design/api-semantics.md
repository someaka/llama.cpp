# Semantic contract: deprecated `llama_batch` → `llama_batch_ext` migration

Repo: `/home/a/Bureau/Work/llama.cpp` @ `9d0abfeb1` (merge of upstream `c622f623f`, 201 commits).
All citations are `file:line` at this HEAD. Read-only analysis; no tracked file modified, no build run.

**Headline finding:** the deprecated `llama_decode`/`llama_encode` no longer have their own
compute path. They convert the value-type batch into a `llama_batch_ext` via
`llama_batch_compat` and call the *same* `llama_context::decode(const llama_batch_ext &)` /
`encode(const llama_batch_ext &)` that `llama_process` dispatches to. A manual ext migration
that reproduces the compat conversion's token/pos/seq/output values is therefore
**numerically identical by construction** — both paths execute the identical function.

---

## 1. How the deprecated functions are implemented now

### 1.1 Batch constructors/free are still plain value-type helpers (NOT ext wrappers)

- `llama_batch_get_one` — `src/llama-batch.cpp:1145-1157`. Returns a non-owning view:
  ```c
  return {
      /*n_tokens =*/ n_tokens,
      /*tokens   =*/ tokens,
      /*embd     =*/ nullptr,
      /*pos      =*/ nullptr,
      /*n_seq_id =*/ nullptr,
      /*seq_id   =*/ nullptr,
      /*logits   =*/ nullptr,
  };
  ```
  The NULL `pos`/`n_seq_id`/`seq_id`/`logits` fields are *defaults* interpreted later by the
  compat layer (§4, §5, §6), not by this function.
- `llama_batch_init` — `src/llama-batch.cpp:1159-1187`. Still raw `malloc` of
  `token`/`embd`, `pos`, `n_seq_id`, `seq_id` (row pointers + sentinel NULL at
  `[n_tokens_alloc]`), `logits`. All payload left uninitialized. No ext involved.
- `llama_batch_free` — `src/llama-batch.cpp:1189-1201`. Frees those arrays (walks `seq_id`
  rows until the NULL sentinel). No ext involved.

### 1.2 decode/encode ARE thin wrappers over ext — the conversion code

`src/llama-context.cpp:5369-5379` (comment in source: `// compat: llama_batch -> llama_batch_ext -> encode/decode`):

```cpp
int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}
```

Public C entry points just add error logging:
- `llama_encode` — `src/llama-context.cpp:5383-5392` → `ctx->encode(batch)`.
- `llama_decode` — `src/llama-context.cpp:5394-5404` → `ctx->decode(batch)`; logs unless
  `ret == 0 || ret == 1`.

`llama_batch_compat` is RAII (`src/llama-batch.h:277-288`): ctor `new llama_batch_ext(ctx)`
then `init(...)`, dtor `delete batch_ext` (`src/llama-batch.cpp:1618-1625`). A fresh ext
batch is built **per decode call**.

The full conversion, `llama_batch_compat::init` — `src/llama-batch.cpp:1542-1616` (key parts quoted):

```cpp
static const llama_seq_id default_seq_id    = 0;      // :1553
static const int32_t      default_n_seq_id  = 1;      // :1554

// auto-generates positions locally when batch_inp.pos is null, continuing from memory
std::vector<llama_pos> pos_next(batch_ext->n_seq_max);                    // :1557
for (llama_seq_id s = 0; s < (llama_seq_id) batch_ext->n_seq_max; ++s) {
    pos_next[s] = llama_memory_seq_pos_max(batch_ext->mem, s) + 1;        // :1559
}

for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
    const int32_t      n_sid = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : default_n_seq_id;  // :1563
    const llama_seq_id * sids = batch_inp.seq_id  ? batch_inp.seq_id[i]     : &default_seq_id; // :1564

    llama_batch_ext::token t;
    for (int32_t s = 0; s < n_sid; ++s) {
        t.seq_ids.insert(sids[s]);                                        // :1570
    }

    if (batch_inp.pos) {
        if (has_token) {
            t.pos[0] = batch_inp.pos[i];                                  // :1577  (token: 1 pos)
        } else {
            // embedding batch (M-RoPE): section-major pos[j*n_tokens + i]
            for (uint32_t j = 0; j < batch_ext->n_pos_per_embd; ++j) {
                t.pos[j] = batch_inp.pos[(int32_t) j * batch_inp.n_tokens + i];  // :1581
            }
        }
    } else {
        t.pos[0] = pos_next[sids[0]]++;                                   // :1586  (auto-track)
    }

    if (has_token) {
        t.id = batch_inp.token[i];                                        // :1591
    }
    // ... embd / MTP-state copy at :1594-1606 (not used by plain-text tools) ...

    // if no logits array is given, default to only the last token being an output
    t.output = batch_inp.logits
        ? (batch_inp.logits[i] != 0)                                      // :1610-1611
        : (i == batch_inp.n_tokens - 1);                                  // :1612

    batch_ext->tokens.push_back(t);                                       // :1614
}
```

**Consequence for the migration:** the mapping old→ext is exactly:

| old field | ext equivalent |
|---|---|
| `b.token[i] = tok` | `llama_batch_ext_add_token(batch, seq, tok)` (id part) |
| `b.pos[i] = p` | `llama_batch_ext_set_pos(batch, idx, &p)` (1 value for tokens) |
| `b.n_seq_id[i]=1; b.seq_id[i][0]=s` | `add_token(batch, s, tok)` (seq part) |
| `b.logits[i] = 0/1` | `llama_batch_ext_set_output_logits(batch, idx, false/true)` |
| `b.pos == NULL` (get_one) | **no ext equivalent** — caller must compute positions (§4) |
| `b.logits == NULL` (get_one) | **no ext equivalent** — default is *all false*, not last-true (§6) |

Both paths then run `llama_context::decode(const llama_batch_ext &)`
(`src/llama-context.cpp:2230`), which sanitizes via `llama_batch_allocr::init`
(`src/llama-batch.cpp:28`), splits into ubatches, builds graphs, extracts outputs.
There is no second, divergent compute path.

---

## 2. Capacity: fixed at init from ctx `n_batch`; no growth; no >n_batch prompts

- Ctor `llama_batch_ext::llama_batch_ext(llama_context * ctx)` — `src/llama-batch.cpp:1225-1235`:
  ```cpp
  n_tokens_max(llama_n_batch(ctx)),
  ...
  n_seq_max(llama_n_seq_max(ctx)),
  mem(llama_get_memory(ctx)),
  n_vocab(llama_vocab_n_tokens(...)),
  n_pos_per_embd(llama_get_model(ctx)->hparams.n_pos_per_embd())
  ```
  `n_tokens_max` is `const size_t` (`src/llama-batch.h:103`). **Fixed at init = ctx `n_batch`.
  No dynamic growth anywhere** (storage is `std::vector<token> tokens`, `src/llama-batch.h:130`,
  but the cap is enforced on every add).
- The **-1 "batch is full"** trigger — `llama_batch_ext::add_token`, `src/llama-batch.cpp:1263-1268`:
  ```cpp
  if (tokens.size() >= n_tokens_max) {
      return -1; // size limit reached
  }
  if (seq_id < 0 || seq_id >= n_seq_max) {
      return -3; // invalid sequence id
  }
  ```
  i.e. -1 fires exactly when the batch already holds `n_batch` entries. Note the check order:
  capacity is checked *before* seq-id validity. The C wrapper `llama_batch_ext_add_token`
  (`src/llama-batch.cpp:1471-1480`) propagates -1/-3 from `add_token`, and returns **-2** when
  `set_token_id` fails (`id < 0 || id >= n_vocab`, `src/llama-batch.cpp:1295-1304`).
- **One ext batch cannot hold a prompt longer than `n_batch`.** Even if it could, decode
  hard-asserts: `GGML_ASSERT(n_tokens_all <= cparams.n_batch);` — `src/llama-context.cpp:2329`.
  Callers must chunk long prompts into ≤`n_batch` ext batches / `llama_process` calls, exactly
  as old-API callers had to (the assert is on the shared path, so old `llama_decode` had the
  same limit).
- **ubatch splitting still happens inside `llama_process`/decode**, unchanged:
  `mctx = memory->init_batch(*balloc, cparams.n_ubatch, embd_all);` —
  `src/llama-context.cpp:2358`, then the `do { ... } while (mctx->next());` ubatch loop at
  `src/llama-context.cpp:2483-2772`. Splitting machinery (`llama_batch_allocr::split_simple /
  split_equal / split_seq`, `src/llama-batch.h:192-201`) is shared by old and new paths because
  the old path *is* the new path (§1). Non-causal attention requires `n_ubatch >= n_tokens`
  (assert at `src/llama-context.cpp:2331`).

---

## 3. `llama_batch_ext_clear`: resets content, keeps capacity, safe to reuse

`src/llama-batch.cpp:1256-1261` (C wrapper `llama_batch_ext_clear` at `:1463-1465`):

```cpp
void llama_batch_ext::clear() {
    tokens.clear();
    embd  .clear();
    state .clear();
    n_embd = 0;
}
```

- Resets the token count to 0 (`tokens.size()`), drops embd/state payload, resets the embd
  row-width latch. `n_tokens_max` is const and untouched; `std::vector::clear` does not release
  capacity, so reuse is allocation-cheap.
- **Safe to reuse across decode calls.** After `clear`, `add_token` returns indices starting
  at 0 again. Each new entry is a fresh value-initialized `token` struct
  (`src/llama-batch.h:116-129`): `id = LLAMA_TOKEN_NULL`, `output = false`,
  `pos = {0,0,0,0}`, empty `seq_ids` — no stale pos/output/seq leaks from the previous fill.
- The ctor calls `clear()` too (`src/llama-batch.cpp:1234, 1253`).
- Caveat: a *failed* fill (e.g. -1 mid-prompt) leaves a partial batch; `clear` before refilling.
  `llama_process` on an empty batch returns -1 (`src/llama-context.cpp:2246-2249`).

---

## 4. Position handling — the main migration hazard

### 4.1 `set_pos` for plain text tokens: ONE `llama_pos` value

`llama_batch_ext_set_pos` (`src/llama-batch.cpp:1508-1510`) → `set_token_pos`
(`src/llama-batch.cpp:1411-1427`):

```cpp
size_t n_pos = t.id != LLAMA_TOKEN_NULL ? 1 : n_pos_per_embd;
for (size_t i = 0; i < n_pos; ++i) {
    t.pos[i] = pos_in[i];
}
```

For an entry added with `add_token` (has a token id): **copies exactly 1 value** — pass
`&p` where `p` is a single `llama_pos`. Only `pos[0]` is written; the other M-RoPE sections
stay 0 and are irrelevant for text (see expansion below). For embd-only entries
(`add_embd`/`add_embd_row`), it copies `n_pos_per_embd` values in `[t, y, x, z]` order
(header contract: `include/llama.h:1099-1106`). `pos_in == NULL` or bad idx → returns false.

### 4.2 If pos is never set: NO error — silent position 0

- `add_token` comment: `// position is left undefined; call set_token_pos() before decoding`
  (`src/llama-batch.cpp:1271`), and header: `pos = not set, the caller must set it with
  llama_batch_ext_set_pos()` (`include/llama.h:1039`).
- But the storage is value-initialized: `std::array<llama_pos, GGML_MROPE_SECTIONS> pos = {0,0,0,0};`
  (`src/llama-batch.h:128`). Neither `llama_batch_allocr::init` nor `decode(ext)` validates
  that pos was explicitly set — allocr copies `tok.pos[j]` verbatim
  (`src/llama-batch.cpp:184-196`). **An unset pos decodes at position 0**: wrong KV-cache
  placement, wrong attention, corrupted outputs — with no error code. This is the single
  biggest byte-safety hazard of the migration; the old pattern always wrote `b.pos[i]` explicitly.

### 4.3 Old semantics, including `get_one` auto-track

- Explicit old `b.pos[i]`: compat copies it 1:1 into `t.pos[0]` (`src/llama-batch.cpp:1574-1577`).
  Manual equivalent: `llama_batch_ext_set_pos(batch, idx, &b.pos[i])`.
- Old `pos == NULL` (`llama_batch_get_one`): auto-track lives **only in the compat layer**,
  not in ext: `pos_next[s] = llama_memory_seq_pos_max(mem, s) + 1` per sequence
  (`src/llama-batch.cpp:1557-1560`), then `t.pos[0] = pos_next[sids[0]]++` per token, keyed on
  the **first** seq id (`src/llama-batch.cpp:1586`). Empty sequence → `seq_pos_max` returns -1
  (`src/llama-kv-cells.h:509-518`; C wrapper `llama_memory_seq_pos_max` returns -1 for NULL mem,
  `src/llama-context.cpp:5259-5266`) → first auto pos = 0. Positions are sampled from memory
  **at decode-call time** (compat ctor), then incremented locally across the batch.
  Manual ext equivalent for a single seq 0: before filling, query
  `llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1` as the base, then
  `set_pos(idx, &(base + i))` — or track `n_past` in the tool as before.
- M-RoPE expansion is shared and unchanged: allocr expands a token's `[p]` to `[p, p, p, 0]`
  across the `n_pos_per_embd` sections (`src/llama-batch.cpp:184-196`, comment at :191).
  Both old and new paths get the identical expansion because both run allocr.

---

## 5. seq_id: `add_token(batch, 0, id)` reproduces `n_seq_id=1, seq_id[0]=0` exactly

- `llama_batch_ext_add_token(batch, seq_id, id)` — `src/llama-batch.cpp:1471-1480` — calls
  `add_token(seq_id)` which validates `0 <= seq_id < n_seq_max` and does
  `t.seq_ids.insert(seq_id)` (`src/llama-batch.cpp:1267-1273`), then `set_token_id(idx, id)`
  validating `0 <= id < n_vocab` (`src/llama-batch.cpp:1295-1304`).
- For `seq_id = 0`: entry ends with `seq_ids = {0}`. allocr rebuilds the flat arrays from the
  set: `n_seq_id[i] = seq_ids.size()` (=1), `seq_id[i][0] = 0`
  (`src/llama-batch.cpp:199-232`). **Bit-for-bit the old `n_seq_id[i]=1; seq_id[i][0]=0`.**
- `n_seq_max` comes from `llama_n_seq_max(ctx)` (ctor, `src/llama-batch.cpp:1230`); with the
  usual `n_seq_max=1` context, seq 0 is the only valid id; anything else → -3.
- Multi-seq caveats (not relevant to single-seq-0 tools, but part of the contract):
  - `seq_ids` is `std::unordered_set<llama_seq_id>` (`src/llama-batch.h:127`): **duplicate seq
    ids are deduplicated** (old arrays could repeat an id) and **iteration order is
    unspecified** for >1 ids. allocr's coupling logic anchors on `seq_id[i][0]`
    (`src/llama-batch.cpp:296-310`), so multi-seq ordering can differ from an old explicit
    array order. Single-id entries are unaffected.
  - `llama_batch_ext_add_seq(batch, idx, s)` (`src/llama-batch.cpp:1503-1505` → `add_seq`,
    `:1280-1293`) adds another id to an existing entry; header says call it before other
    `_set()` functions (`include/llama.h:1051-1057`).

---

## 6. logits flag: `set_output_logits(idx, true)` == old `logits[i] = 1`; nothing extra is set

- Both C setters are the **same function**: `llama_batch_ext_set_output_embd`
  (`src/llama-batch.cpp:1524-1526`) and `llama_batch_ext_set_output_logits`
  (`src/llama-batch.cpp:1528-1530`) each call `batch->set_output(idx, value)`
  (`src/llama-batch.cpp:1429-1435`), which sets the single per-token bool
  `token.output` (`src/llama-batch.h:124`, annotated `// TODO: have dedicated output flags`).
  The header documents the equivalence both ways (`include/llama.h:1085-1097`).
  There is **no separate embd-output flag** in this implementation — one bool drives both.
- Old path mapping: compat sets `t.output = logits[i] != 0` when a logits array exists
  (`src/llama-batch.cpp:1608-1612`). So `set_output_logits(idx, true/false)` is exactly
  `logits[i] = 1/0`.
- **Default difference:** old `logits == NULL` (i.e. `llama_batch_get_one`) meant
  *last token only* output (`src/llama-batch.cpp:1612`). Ext default is `output = false` for
  **every** token (`src/llama-batch.h:124`). Migrating `get_one` users who consume
  logits/embeddings must explicitly mark the last token (or whichever tokens they read).
- Downstream, allocr copies `output` into the internal `batch.logits`
  (`src/llama-batch.cpp:235-254, 278`) and counts `n_outputs`; when `cparams.embeddings`
  (`output_all`) it force-marks all tokens with a warning
  (`src/llama-batch.cpp:244-252`; `decode(ext)` passes `output_all = cparams.embeddings`,
  `src/llama-context.cpp:2278`). This is identical for old and new paths.
- **Hidden-state bytes are independent of the output flag.** The fork's capture reads
  per-layer `t_hidden_layers[]` during the forward pass — source comment:
  "extract_hidden_states reads from per-layer t_hidden_layers[] during the model forward
  pass, which is independent of the output slot mapping" (`src/llama-context.cpp:2275-2277`).
  Capture covers **every token of the batch** regardless of `output` marking
  (buffer sized `n_slots * n_tokens_all * n_embd_out`, `src/llama-context.cpp:2410-2445`;
  per-ubatch copy loop `src/llama-context.cpp:2680-2770`). The output flag only changes
  logits/embd slot counts (`n_outputs`), not the hidden-state tensors. Marking outputs
  differently therefore cannot change extracted hidden-state bytes — but *not marking the
  last token* will break any tool that also reads `llama_get_logits_ith`.

---

## 7. Ownership / lifetime / thread safety

- `llama_batch_ext_init(ctx)` = `new llama_batch_ext(ctx)` (`src/llama-batch.cpp:1455-1457`);
  `llama_batch_ext_free(batch)` = `delete batch` (`src/llama-batch.cpp:1459-1461`). Plain
  heap object; the destructor is implicit (frees the vectors only).
- **The batch does not own or retain the context.** The ctor copies scalars out of ctx and
  stores one raw pointer: `mem = llama_get_memory(ctx)` (`src/llama-batch.cpp:1231`,
  member `src/llama-batch.h:108`, "memory for position inference"). No back-pointer, no
  registration, no refcount.
- Lifetime rule: the context (its memory module) must outlive every **use** of the batch
  (`llama_process`, and compat-style pos inference reads `mem`). `ext_free` itself never
  dereferences `mem`, so freeing the batch after ctx death is technically harmless — but the
  clean discipline is **free the ext batch before `llama_free(ctx)`**, mirroring the old
  `llama_batch_free` before `llama_free` pattern. A batch created from one ctx must not be
  processed on another (its `n_tokens_max`/`n_vocab`/`mem` are bound to the creating ctx).
- Thread safety: no locks in any ext function; fill and process are single-threaded-per-ctx
  operations, same contract as the old API. `decode(ext)`/getters synchronize internally
  where needed (`synchronize()` at `src/llama-context.cpp:2336`; hidden-state getters
  `llama_get_hidden_state` / `llama_get_hidden_state_n_tokens` / `llama_get_hidden_states_batch`
  all call `ctx->synchronize()`, `src/llama-context.cpp:5036-5088`; the capture loop drains the
  backend sched before `ggml_backend_tensor_get`, `src/llama-context.cpp:2720-2723`).
  Concurrent `llama_process` calls on one ctx remain unsupported.

---

## 8. `llama_process` return codes vs `llama_decode`

`llama_process` — `src/llama-context.cpp:5471-5478`:

```cpp
int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}
```

Enum at `include/llama.h:308-311`; header states "Return values are the same as
llama_decode()" (`include/llama.h:1110-1114`). Since it dispatches to the very functions the
deprecated wrappers call, codes are **identical by construction**:

| code | meaning (decode) | source |
|---|---|---|
| 0 | success | `src/llama-context.cpp:2857` |
| 1 | no KV slot for batch (non-fatal; retry smaller batch / defrag happens internally once) | `:2387` |
| 2 | aborted (processed ubatches remain in memory) | `:2536` |
| -1 | invalid input: empty batch `:2246-2249`, embd-width mismatch `:2251-2255`, allocr init failure `:2312-2315`, hidden-state contract failures `:2417-2445, 2690-2769`, unknown process type `:5477` | |
| -2 | memory init/compute failure `:2358-2381`, output-reserve failure `:2401-2404`, GGML alloc failure `:2537` | |
| -3 | graph compute GGML_STATUS_FAILED | `:2538` |

Old `llama_decode` header semantics (`include/llama.h:1003-1021`): positive = warning;
on fatal error/abort processed ubatches remain in memory; otherwise memory restored.
Unchanged. `encode(ext)` returns 0 / -1 (empty `:1979-1982`, width mismatch `:1986-1990`,
allocr failure `:2002-2005`, reserve failure `:2038-2041`, graph statuses `:2064-2066`).
Note `decode(ext)` on a memory-less ctx silently redirects to `encode(ext)`
(`src/llama-context.cpp:2241-2244`) — same as before.

---

## 9. Other behavioral differences that could change extracted bytes

1. **No pos auto-track in ext** (§4.3). Tools that used `llama_batch_get_one` must now set
   positions explicitly; forgetting this silently decodes at pos 0 (§4.2) — the only
   *silent* corruption mode found.
2. **No last-token output default in ext** (§6). Affects logits/embd reads, not hidden-state
   capture bytes.
3. **Capacity enforced at add time** (§2): old `llama_batch_init(n_tokens_alloc, ...)` could
   allocate any size (failure surfaced later at decode's assert); ext rejects the
   `n_batch+1`-th token with -1 mid-fill, leaving a partial batch that must be cleared.
   Chunk boundaries must match the old tool's chunking to keep identical per-call batches.
4. **seq_id dedup + unspecified multi-id ordering** (§5) — only for entries with >1 seq id;
   irrelevant for the fork's single-seq-0 pattern.
5. **Token ordering / ubatch splitting / padding: no divergence.** ext preserves insertion
   order (`tokens.push_back`, `src/llama-batch.cpp:1275, 1614`); sanitization, M-RoPE pos
   expansion, seq-set grouping, ubatch splitting and any padding all live in the shared
   `llama_batch_allocr` + `memory->init_batch` machinery (`src/llama-batch.cpp:28-330`,
   `src/llama-context.cpp:2358, 2483-2772`). Old and new paths run the same code on the same
   values → same ubatches → same graphs → same bytes.
6. **Hidden-state buffer layout is unchanged and path-independent**: layer-major,
   `(n_layer+1) * n_tokens_all * n_embd_out` floats, tokens accumulated in ubatch processing
   order at offset `il*n_tokens_all*n_embd_out + n_tokens_prev*n_embd_out`
   (`src/llama-context.cpp:2425-2445, 2734-2747`). For single-sequence causal batches ubatch
   order == insertion order. `n_hidden_tokens` is invalidated up front on every decode and on
   every failure exit (`src/llama-context.cpp:2231-2240, 2519-2523`), and getters expose
   nothing after a failed call — migration does not alter this contract.
7. **Validation moved earlier**: invalid token id → -2 at `add_token` time (old API deferred
   to decode's allocr, `src/llama-batch.cpp:138-142`); invalid seq → -3 at add time. Behavior
   on *valid* inputs is unchanged.
8. **MTP/LiliCorr state width**: ext carries `n_embd_state` (nonzero only for MTP contexts or
   `lilicorr_candidate_topk > 0`, `src/llama-batch.cpp:1213-1223`); plain decoder text batches
   have `n_embd_state = 0` and no state payload — no effect on the fork's tools.

---

## MIGRATION EQUIVALENCE VERDICT

**Yes — old → ext is mechanically byte-safe for the fork's hidden-states tools**, because the
deprecated `llama_decode`/`llama_encode` are now thin compat wrappers that build a
`llama_batch_ext` and call the *identical* `decode(ext)`/`encode(ext)` implementation that
`llama_process` uses (`src/llama-context.cpp:5371-5379, 5471-5478`). There is no parallel
compute path that could diverge. For the canonical pattern
`llama_batch_init(n,0,1)` + `token[i]/pos[i]=i/n_seq_id=1/seq_id[0]=0/logits[i]` + `llama_decode`,
the exact replacement is:

```c
llama_batch_ext * b = llama_batch_ext_init(ctx);          // capacity = n_batch, fixed
for (i = 0; i < n; ++i) {                                  // n <= n_batch, chunk as before
    int32_t idx = llama_batch_ext_add_token(b, 0, tok[i]); // == seq {0}, id validated
    // check idx >= 0 (-1 full, -2 bad token, -3 bad seq)
    llama_pos p = /* exactly the old b.pos[i] value */;
    llama_batch_ext_set_pos(b, idx, &p);                   // MANDATORY: 1 value for tokens
    llama_batch_ext_set_output_logits(b, idx, old_logits_i != 0);
}
int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b); // same return codes as llama_decode
llama_batch_ext_clear(b);                                  // reuse; or ext_free before llama_free(ctx)
```

Equivalence holds **iff** the migration reproduces, per token: the same id, the same pos
value, seq set {0}, and the same output flag — and keeps the same per-call chunking
(≤ `n_batch` tokens per batch, contiguous positions across calls).

**Residual risks (ranked):**
1. **Unset pos is silent, not an error** — defaults to 0 (`src/llama-batch.h:128`); no
   validation at process time. Every `add_token` must be paired with `set_pos`. This is the
   only found mode that corrupts bytes without any error code. (High impact, easy to guard:
   assert `set_pos` returned true for every idx.)
2. **`get_one` auto-track has no ext counterpart** — tools relying on `pos=NULL` must compute
   base = `llama_memory_seq_pos_max(mem, 0) + 1` at fill time and increment per token
   (mirrors `src/llama-batch.cpp:1557-1586`); an off-by-one here shifts KV positions and
   changes outputs.
3. **`get_one` logits default lost** — ext defaults all outputs false; mark the last token if
   logits/embd are consumed (hidden-state bytes themselves are output-flag-independent,
   `src/llama-context.cpp:2275-2277`).
4. **Partial batch on -1** mid-fill (prompt > `n_batch`): must `clear` and chunk; feeding the
   partial batch would decode a truncated prompt.
5. **Lifetime**: batch binds `mem` of the creating ctx; free ext before ctx, never process a
   batch after its ctx is freed, never cross contexts.
6. **Multi-seq only**: unordered_set dedup/ordering (§5) — not applicable to single-seq-0
   extraction, but do not generalize the pattern to coupled sequences without care.
7. No thread-safety change: still one in-flight process per ctx.

Golden-digest gate expectation: with rules 1–4 enforced, digests must match byte-for-byte;
any mismatch indicates a pos/output/chunking discrepancy in the migration, not an API
semantic difference.
