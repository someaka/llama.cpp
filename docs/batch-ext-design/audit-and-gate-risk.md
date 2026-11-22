# Audit & gate risk report — batch-ext migration

Repo: `/home/a/Bureau/Work/llama.cpp`, HEAD `9d0abfeb1`. Companion to
`migration-inventory.md`. All audit/gate/self-test runs below were executed
READ-ONLY against the pristine tree on 2026-11-21 (no tracked file modified,
nothing built, nothing committed).

## HEADLINE FINDINGS (read first)

1. **The audit has 20 checks at HEAD, not 19.** `audit_integrity.sh` contains
   `Check 1` … `Check 20`; fork-ci.yml:679 says "The 20 structural checks live
   in a committed script". Check 20 (architecture census) was added after the
   "19-check" description was written (script last touched by fork commits
   `ea7247ba6`/`43a7781ea`; file mtime Oct 29). The verifier skill's "19/19"
   and "self-test 24/24" receipts are stale: **self-test is now 33/33**.
2. **The audit is NOT all-clean at HEAD — pre-existing, migration-unrelated
   failure.** Verified run: Checks 1–19 PASS, **Check 20 FAIL**:
   ```
   FAIL: architecture census drift: doc states '101 of [the] 187' 0 times, expected 3;
   mechanism section does not state the full census sentence ('...101 of the 187 ... 97 of those 101...');
   doc does not state 'the 43 non-classic'; doc does not state 'none of the 43 carries a hook'
   ```
   Cause: the 201-commit upstream merge grew `src/models/*.cpp` from 180 to
   **187 builders** (101 hook-bearing — still matches the pin; **97 hook-only**;
   **43 non-classic**), but `docs/hidden-states-architecture.md` still says
   "101 of 180" (lines 73–74, 162, 184–185) and "the 39 non-classic" (185).
   The doc was last updated at the previous merge (`4835b6a3c`). **The
   migration's acceptance criterion "audit must pass" cannot be met without
   also fixing this doc census** (doc-only edit; the check recomputes from the
   tree, no pin change needed). This is independent of the batch-API migration
   but blocks the same gate.
3. `check_doc_claims.py` PASSES at HEAD ("6 parity rows honour the cell
   contract; 3 code facts hold"). `--self-test` PASSES 33/33 on the existing
   `build-cuda/bin/llama-hs-extract-batch`.
4. **Concurrent-edit notice:** at 17:08 (after this report's audit run) a
   tracked-file modification appeared that was NOT made by this analysis:
   `src/models/gemma4.cpp` (+2/−1: `prune_shared` gains
   `&& !cparams.extract_hidden_states`, a hidden-states pruning fix by a
   parallel worker). It touches neither `capture_layer_output` nor `inpL = cur`,
   so the Check 20 census counts above are unaffected — but the parent's
   "git status 0 tracked-dirty" receipt must account for it.
5. **The gate model is not on disk and no HuggingFace repo is named in the gate
   README** (quote in §b). `HS_GATE_MODEL` is unset in this environment; the
   gate REFUSES without it. A pre-migration binary backup exists at
   `/home/a/Bureau/Work/fork-premigration-bin-20261121/` (contains
   llama-hs-extract-batch, llama-hs-extract, llama-hs-probe,
   llama-hs-extract-batch-test, llama-hidden-states, test-hidden-states,
   test-hidden-states-c + libs) — useful as the trusted pre-refactor binary,
   but the anchor must never be re-captured from the binary under test.

## (a) audit_integrity.sh — all checks, purposes, migration exposure

Mechanics: `set -euo pipefail`; `strip_comments()` (lines 3–34) removes block
comments (space-filled), line comments, then string/char literals **per line**,
so all identifier pins verify code, not prose. Run from the repo root:
`bash tools/hs-extract-batch/gate/audit_integrity.sh`. CI step: fork-ci.yml
"Quick audit fix integrity check" (line ~677–681).

Flag legend: **EDITED** = check scans a file the migration will edit
(llama-raii.h, hs-extract.cpp, hs-extract-batch.cpp, test-hidden-states.cpp,
example, probe, batch-test, test-c); **UNTOUCHED** = check scans files the
migration should not edit (io-util.cpp, src/, server files, docs) — it must
simply remain true.

| # | Purpose (one line) | Files scanned | Exposure | What must remain true after migration |
|---|---|---|---|---|
| 1 | Flat accumulator key is uint64_t (no group/mask/layer truncation) | hs-accum.h; all tools/hs-extract-batch/*.cpp | **EDITED** (use-site count spans hs-extract-batch.cpp) | `using AccumulatorMap = std::unordered_map<uint64_t,` and `inline uint64_t make_accum_key` in hs-accum.h; **≥2** `uint64_t <id> = make_accum_key(` use sites across the TUs (live: hs-extract-batch.cpp:1296 generation, :1870 comprehension) — keep both |
| 2 | `output_all = cparams.embeddings` polarity in the context | src/llama-context.cpp | UNTOUCHED | both greps (`const bool output_all = cparams.embeddings;`, `if (output_all) {`) still present; do not edit src/llama-context.cpp in this migration |
| 3 | RAII LlamaBackend used; no raw backend init | examples/hidden-states/hidden-states.cpp, tools/hs-extract/hs-extract.cpp, tools/hs-extract-batch/hs-extract-batch.cpp | **EDITED** (all 3) | each TU keeps a `LlamaBackend <name>;` declaration (use-site pin) AND contains **no** `llama_backend_init(` call (comment-stripped). If probe/batch-test are RAII-converted as part of migration, they must be ADDED to this check's list (gate edit, see Check 14 note) |
| 4 | Hidden-state getter synchronizes as FIRST statement | src/llama-context.cpp | UNTOUCHED | `llama_get_hidden_state` impl keeps `ctx->synchronize()` before first return and as first body statement (position-pinned) |
| 5 | Every fclose lives inside FilePtr RAII | hs-extract-batch.cpp, io-util.cpp, io-util.h, assignments-io.cpp, self-test.cpp, hs-extract.cpp | **EDITED** (2 of 6 scanned) | no `fclose` outside `struct FilePtr {…}` in any scanned file; migration adds none |
| 6 | Every checked_write() return is tested; literal-stripper fail-closed | same 5 TUs (no io-util.h) | **EDITED** (2 of 5) | all `checked_write`/`checked_write_h` calls stay inside if/return/&&/\|\|/assignment; raw vs comment-stripped call-site counts must stay EQUAL (stripper mis-sync = FAIL). Verified live count: **21 call statements (26 call sites)** — keep the I/O code untouched so this stays stable (count is printed, not numerically pinned) |
| 7 | Assignment reader returns explicit error status; call site tests it | assignments-io.cpp; hs-extract-batch.cpp | **EDITED** (call-site pin) | ≥5 `AssignmentReadStatus::error` returns in assignments-io.cpp (untouched); hs-extract-batch.cpp keeps `status != AssignmentReadStatus::ok` (live at :1422 and :1570) |
| 8 | Prompt pre-scan is pure; callers fail by return, never exit | hs-extract-batch.cpp | **EDITED** (whole-TU scans) | `static bool scan_prompts_file(` stays in this TU; its body has no exit-family call; `struct PromptsScan` present; **exactly 2** `if (!scan_prompts_file(` guarded callers and **exactly 3** total occurrences (def + 2 calls); each guarded action returns a literal/identifier nonzero failure (no exit, no nested if, no ternary/arithmetic return, no success-constant identifier); **TU-wide bans**: no `= <rhs>;` initializer mentioning any exit-family name (alias ban), no new exit-helper function called from a guarded window (transitive closure). New migration code in this TU must not introduce exit-family bindings anywhere |
| 9 | No off-by-one `<=` against layer-count limits or aliases | hs-extract.cpp, hs-extract-batch.cpp | **EDITED** (both) | new loops use `<` against `layers.size()/target_layers.size()/n_layers_total/n_layer/n_layers` and aliases; alias collector must still fire (anti-vacuity: zero aliases = FAIL) — keep e.g. `const size_t n_layers_total = target_layers.size();` (:1081); LIMITS grammar canary (`target_` coverage) intact |
| 10 | Checkpoint n_masks/n_layers_data bounds enforced | io-util.cpp | UNTOUCHED | both `if (n_masks < 0 \|\| n_masks > MAX_MASKS)` / `if (n_layers_data < 0 \|\| … > MAX_LAYERS)` guards keep `return false;` within 3 lines |
| 11 | Output writer uses pre-built gm_pairs index (no O(K²) rescan) | io-util.cpp | UNTOUCHED | `gm_pairs[<id>]` subscript use; `gm_pairs[gm_idx].group_id == group_id` advance walk; **exactly 1** range-scan over gm_pairs |
| 12 | Checkpoint count/layer_idx validation enforced | io-util.cpp | UNTOUCHED | `if (count < 0)` and `layer_idx < 0 \|\| layer_idx > 0xFFFF` guards keep `return false;` |
| 13 | Producer-consumer error notification (no deadlock) | hs-extract-batch.cpp | **EDITED** (consumer loop is a migration edit zone) | a `cv.notify(_one/_all)?(` call shape; `pfq.producer_done` present; `if (prompts_fin.bad())` handler (live :1618) keeps `producer_done = true;` BEFORE `pfq.cv.notify_all();` in that order |
| 14 | Shared RAII header exists and is included (not commented out) | common/llama-raii.h; hs-extract.cpp, hs-extract-batch.cpp, tests/test-hidden-states.cpp | **EDITED** (header rewritten + all 3 TUs) | `common/llama-raii.h` must still exist at that exact path; each listed TU keeps a raw `#include "llama-raii.h"` AND a surviving stripped include (live includes: hs-extract.cpp:18, hs-extract-batch.cpp:64, test-hidden-states.cpp:10). If `LlamaBatch` is replaced by an ext wrapper, keep it in this same header. **Explicit exemption note (script lines 642–646)**: logits-probe.cpp and hs-extract-batch-test.cpp still call raw llama_backend_init/free; if the migration converts them to RAII, they must be added to Check 14's AND Check 3's lists (a gate-file change requiring its own mutation verification) |
| 15 | Async pipeline backpressure (bounded queue) | hs-extract-batch.cpp | **EDITED** | `cv_space` present; the EXACT predicate string `pfq.cv_space.wait(lk, [&]{ return pfq.queue.size() < MAX_PREFETCH \|\| pfq.producer_done.load(); })` (live :1608) verbatim; `stop_producer_and_join` present; **exactly one** `MAX_PREFETCH` declaration, plain decimal literal in [1,1024] (live `static constexpr size_t MAX_PREFETCH = 64;` :976), no `#define MAX_PREFETCH` |
| 16 | /hidden-states pool=none response size cap enforced | server-hidden-states.h | UNTOUCHED | `if (total_all_layers > HS_MAX_POOL_NONE_FLOATS)` guard keeps error-return within 6 lines |
| 17 | Checkpoint v6 sum-based records (lossless) | io-util.cpp | UNTOUCHED | `CHECKPOINT_VERSION = 6`, `bool write_sum` param, positive `if (write_sum)` branch |
| 18 | Checkpoint/output durability: fsync → reset → rename | io-util.cpp | UNTOUCHED | ordered sequence `if (!f.sync()) {…} f.reset(); rename(temp_path.c_str(), ckpt_path…)` and same for `out`/output_path |
| 19 | Capture toggle cleared structurally on all return paths | server-hidden-states.h | UNTOUCHED | `struct hs_toggle_reset {… llama_set_extract_hidden_states(c, false);` + `hs_reset{slot.ctx_tgt}` instantiated as FIRST statement of `send_hidden_states`, before any return |
| 20 | Architecture census doc matches the tree | src/models/*.cpp, src/llama-arch.cpp, docs/hidden-states-architecture.md | UNTOUCHED by migration — **but FAILING at HEAD** (see Headline 2) | hooked builders == 101 (pin); doc states "101 of [the] 187" ×3, the full mechanism-section census sentence with 97 hook-only, "the 43 non-classic", "none of the 43 carries a hook"; registry case count/names match; no builder touches `t_hidden_layers`. **Requires a doc-census fix commit (180→187, 39→43, hook-only sentence) before the audit can pass** — migration must not touch src/models or the census otherwise |

### check_doc_claims.py (separate CI step, "Parity table cell contract")

Gates `tools/server/README.md` parity table SHAPE (6 axis rows; measured rows
must name metric ∈ {abs, rel-L2, rel-max, bit-identical}, `pool=` within 35
chars of each figure, concrete layer set, n=/trials/"n not recorded"; no
bound-style wording; preamble says "not bounds") plus **3 code facts**:

| Code fact | File | Migration exposure |
|---|---|---|
| HS CLIs have NO Flash-Attention flag: no `"-fa"`/`"--flash-attn"` string literal | **hs-extract.cpp, hs-extract-batch.cpp** | **EDITED** — migration must not add any FA flag literal to these TUs |
| Server prints `"%.9g"` (9 significant digits) | server-hidden-states.cpp | UNTOUCHED — keep |
| `n_parallel is set to auto` paired with `params.kv_unified = true` | server.cpp | UNTOUCHED — keep |

Verified PASS at HEAD. The parity table's NUMBERS are deliberately not pinned;
if migration changes measured deviations the rows may be updated, but the cell
contract must hold.

### Other CI pins touching migration-edited files (fork-ci.yml)

- **PR-surface whitespace check** (~line 658–674): `git diff <merge-base>..HEAD
  --check` over tools/server, tools/hs-extract, tools/hs-extract-batch,
  tools/hs-extract-common, tools/hs-extract-batch-test, **common/llama-raii.h**,
  include/llama.h — migration diffs must be whitespace-clean (no trailing
  whitespace / space-before-tab; comment also demands no blank line at EOF).
- **Behavioral receipts** that must stay green after migration:
  `llama-hs-extract-batch --self-test` (verified **33/33** at HEAD);
  `test-hidden-states` + `test-hidden-states-c` with models/hs-fixture.gguf;
  `llama-hs-probe` plain + `--runtime-toggle`; `llama-hs-extract-batch-test … 2`
  (bitwise CLI-vs-server equivalence); /hidden-states server endpoint smokes
  (last_token pool, skip_offset rejection, envelope parity grep on
  server-hidden-states.cpp). The probe/batch-test/test binaries are the
  behavioural proof that the ext migration preserved decode semantics.
- Envelope-parity check greps server-hidden-states.cpp (field set of
  to_json/to_json_text) — untouched by migration.

## (b) Running the golden gate in check mode

### Exact command lines (gate/README.md §"Running the gate after a refactor", lines 96–110)

```bash
cmake --build build-cuda --target llama-hs-extract-batch -j$(nproc)
./build-cuda/bin/llama-hs-extract-batch --self-test  # must pass

python3 tools/hs-extract-batch/gate/gate_batch.py check \
    build-cuda/bin/llama-hs-extract-batch
```

"Exit 0 + `ALL_BYTE_IDENTICAL` = the refactor is byte-neutral." (README:110)

### Required env and refusals (gate_batch.py)

- `HS_GATE_MODEL=/path/to/model.gguf` — **REQUIRED, no default**
  (gate_batch.py:53 `DEFAULT_E2B = os.environ.get("HS_GATE_MODEL")`; README:80).
  Unset → `REFUSED: set HS_GATE_MODEL=… -- the gate runs the production path;
  no fallbacks.` (235–238); missing file → REFUSED (239–241). Currently UNSET
  in this environment.
- **CUDA-only binary**: `libggml-cuda.so*` must sit next to the binary or the
  driver exits `REFUSED: … CPU-only build` (227–232). `build-cuda/bin/` already
  contains `libggml-cuda.so*` and `llama-hs-extract-batch` (verified present).
- **Model pre-flight**: parses GGUF KV `*.block_count`; refuses if
  `max_needed (35) >= n_layer + 1` (246–252) — i.e. the model must have ≥35
  layers (36 slots). The committed `models/hs-fixture.gguf` (llama arch,
  5 layers, sha256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`
  pinned in fork-ci.yml:248) **CANNOT run the gate** (README:87–92).
- Fixtures present check: prompts.txt, assignments.bin, assignments_gen.bin
  under `golden_inputs/` else `REFUSED: inputs missing` (253–255).

### Where the model comes from — README quote (gate/README.md:80–92)

> - Set `export HS_GATE_MODEL=/path/to/model.gguf` (REQUIRED — no default).
>   The committed baseline digests were captured with `gemma-4-E2B.Q4_K_M.gguf`
>   (gemma4 architecture, 35 layers) on the production CUDA path; the file
>   formerly lived under CrimsonRed/data/models/ and has since moved off disk.
>   The model's sha256 is NOT recorded anywhere: `digests_baseline.json` stores
>   only artifact digests, so the anchor identifies the model by name only --
>   always run baseline and check against the same GGUF file. Note that
>   `models/hs-fixture.gguf` (committed in-repo; sha256 pinned in
>   `.github/workflows/fork-ci.yml`) is a different, much smaller CI fixture
>   (llama arch, 5 layers): it CANNOT run the gate scenarios (s5/s6 request
>   layer 17 and s7-s9 request layer 35) and must not be used as HS_GATE_MODEL.
>   Using any model other than the E2B Q4_K_M invalidates byte-identity
>   comparison.

**The gate README names NO HuggingFace repo.** It identifies the model by file
name only (gemma-4-E2B.Q4_K_M.gguf, gemma4 arch, 35 layers) and says it moved
off disk from CrimsonRed/data/models/. Verified: no `*E2B*.gguf` /
`*gemma*.gguf` model under ~/Bureau/Work (only the vocab-only fixture
`models/ggml-vocab-gemma-4.gguf`, which is not a runnable model). For
re-acquisition, the tree's UPSTREAM docs mention candidate repos — NOT the gate
README: docs/multimodal.md:99/139 `-hf ggml-org/gemma-4-E2B-it-GGUF` and
docs/backend/OPENVINO.md:170 `bartowski/google_gemma-4-E2B-it-Q4_K_M`
(huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF). Since the baseline GGUF's
sha256 was never recorded, any re-downloaded Q4_K_M must be treated as
name-matched only; byte-identity against the 2026-08-18 anchor additionally
assumes the same quantization lineage — if digests mismatch with a re-acquired
file, model provenance (not the migration) is the first suspect.

### Runtime expectations

- 5 scenarios, all on the production CUDA path, `-ngl 99`, original
  tokenization (no `--no-bos`), one model (README:10–33; gate_batch.py:59–72):

| scenario | layers | extra args | assignments | resume? |
|---|---|---|---|---|
| s5_e2b_comp_records | 0,17,35 | `--save-per-record` | assignments.bin | no |
| s6_e2b_resume | 0,17,35 | — | assignments.bin | kill+resume |
| s7_e2b_gen_greedy | 0,35 | `--generate 8 --token-skip 2` | assignments_gen.bin | no |
| s8_e2b_gen_sampled | 0,35 | `--generate 8 --token-skip 2 --temperature 0.8 --top-k 40 --repeat-penalty 1.1` | assignments_gen.bin | no |
| s9_e2b_gen_resume | 0,35 | same as s8 | assignments_gen.bin | kill+resume |

- Base command shape (gate_batch.py:83–86):
  `<bin> <model> golden_inputs/prompts.txt <layers> <out> --batch
  --assignments golden_inputs/<assignments file> -ngl 99 [extra]`.
- Resume mechanics (147–193): full run first; then a kill run with
  `--checkpoint-every 5`, SIGKILL at the **2nd "Checkpoint saved" stderr line**;
  then `--resume` run; resumed output must byte-equal the full run.
- Timeouts: 1800 s wall clock per full run (137–142) and per kill phase
  (162–183); a wedged binary fails the gate instead of hanging.
- Scratch: `/tmp/hs-batch-golden/proc-<pid>/` per-process (34), stale proc-*
  trees >1 day pruned (36–47). Inputs are the COMMITTED fixtures, not /tmp.
- Determinism: verified empirically on NVIDIA RTX 3090 (24 GB, CUDA) 2026-08-18
  incl. resume parity and sampled generation; **re-verify on any other device**
  before trusting the anchor (README:35–38). Gate is a same-machine refactor
  gate, not cross-backend (README:128–136).
- Anchor discipline (README:138–147, the 2026-09-13 incident): an anchor only
  moves with a written cause and only from a **fresh-build** digest set — never
  incremental builds, never the binary under test.

### How digests_baseline.json is consumed (gate_batch.py:280–320)

1. Check mode first looks for `/tmp/hs-batch-golden/proc-<pid>/digests_baseline.json`
   (volatile); if absent, **falls back to the committed anchor**
   `tools/hs-extract-batch/gate/digests_baseline.json` (286–289); if neither
   exists → REFUSED with "never re-baseline from the binary under test".
2. Refuses baseline scenario keys not in SCENARIOS (stale-key guard, 297–300).
3. Per scenario, compares the UNION of artifact keys; missing artifact on
   either side = FAIL; sha256 string equality = ok; boolean
   `fullrun_equals_resumed` compared too. Prints one line per artifact.
4. Exit 0 + `ALL_BYTE_IDENTICAL -- refactor changed structure, not numerics`;
   exit 1 + `N MISMATCHES -- refactor changed bytes`. Setup problems print
   `REFUSED`/`inputs missing` instead (README:31–33).
5. Baseline mode (`gate_batch.py baseline <bin>`) writes
   `GOLD/digests_baseline.json` and prints the path; operator copies it to the
   committed anchor (README:121–126).

Current committed anchor content (verified): s5 output.bin
`107fa44977a5…` + records sidecar `2eb6509acb15…`; s6 fullrun == resumed ==
`107fa44977a5…` (same bytes as s5's output.bin) + `fullrun_equals_resumed: true`;
s7 `84811fda8cb0…`; s8 `ca9cc7171704…`; s9 fullrun == resumed == `ca9cc7171704…`
+ true. Migration must reproduce ALL of these exactly — note s7/s8 exercise the
generation decode path (single-token batches, logits flags, sampler RNG) and
s6/s9 exercise checkpoint/resume parity, so any pos/logits/ordering drift in the
ext migration surfaces here.

## (c) golden_inputs fixtures — exist and are committed

`tools/hs-extract-batch/gate/golden_inputs/` (all tracked in git, verified via
`git ls-files`):

| file | size | content |
|---|---|---|
| prompts.txt | 1863 B | 24 deterministic ASCII prompts, varied lengths, all ≥8 tokens (BOS added by binary) |
| assignments.bin | 1165 B | main comprehension assignments (explicit ranges within [0,6)) |
| assignments_gen.bin | 525 B | generation-mode assignments (skip=0, mask_type=0) |

`gen_inputs.py` regenerates them deterministically (atomic temp+os.replace
writes; optional argv[1] output dir; flag-looking args REFUSED); regeneration
must be byte-identical to the committed bytes (README:53–60). The gate driver
reads these committed fixtures directly (`INPUTS`, gate_batch.py:51).

## (d) tests/test-hidden-states.cpp registration and run requirements

- **Registered in tests/CMakeLists.txt:446–450**:
  ```cmake
  # hidden states extraction test (needs a model file to run)
  llama_build(test-hidden-states.cpp)
  set(LLAMA_TEST_NAME test-hidden-states-c)
  llama_build(test-hidden-states.c)
  unset(LLAMA_TEST_NAME)
  ```
  `llama_build` (tests/CMakeLists.txt:7–31) only creates the executable
  (default link `llama-common`, cxx_std_17). **No `llama_test`/`add_test` call
  → the targets are built but NOT registered with CTest**; they run only when
  invoked manually or by fork-ci.
- Targets: `test-hidden-states` (C++) and `test-hidden-states-c` (C).
- **Needs a model**: both take `<model.gguf>` as argv[1] (usage strings at
  test-hidden-states.cpp:14, test-hidden-states.c:9). fork-ci runs them as
  `LD_LIBRARY_PATH=build/bin build/bin/test-hidden-states models/hs-fixture.gguf`
  (and likewise `-c`) — the committed 5-layer llama-arch fixture suffices
  (C++ test uses n_ctx=n_batch=n_ubatch=512 and prompt "Hello world"; C test
  n_ctx=64; neither requests deep layers). The gate model is NOT needed for
  these tests.
- Migration relevance: the C++ test is the behavioural guard for
  failed-decode invalidation, recovery, and **byte-identical multi-ubatch
  capture** (n_ubatch=8 ctx2, memcmp at lines 276/282); the C test guards the
  C-API surface (ext API is C-compatible). Both must pass post-migration;
  both are in the audit's Check 14 include-pin list (C++ only) and the
  fork-ci whitespace surface.

## Risk checklist for the migration commit(s)

1. Fix the Check 20 doc census FIRST or in the same series (180→187 builders,
   39→43 non-classic, 97 hook-only sentence in the mechanism section, "101 of
   [the] 187" ×3) — otherwise the audit cannot pass regardless of migration
   quality. Doc-only; recompute from tree, don't hand-pin.
2. Keep llama-raii.h at its path with the include pins intact (Check 14); keep
   `LlamaBackend <name>;` declarations and zero raw backend init in the 3
   pinned TUs (Check 3). If probe/batch-test go RAII, update Check 3+14 lists
   in the same commit and mutation-verify both directions.
3. In hs-extract-batch.cpp preserve verbatim: MAX_PREFETCH=64 single declaration,
   the exact cv_space.wait predicate (:1608), stop_producer_and_join,
   prompts_fin.bad() handler order (:1618), scan_prompts_file + its 2 guarded
   callers, both make_accum_key use sites, both status checks, the
   n_layers_total alias; add no exit-family bindings, no `<=` layer bounds,
   no FA flag literals, no raw fclose, no untested checked_write.
4. Preserve decode semantics exactly (positions, seq 0, logits flag patterns,
   single-ubatch sizing n_batch=n_ubatch=n_ctx, per-prompt KV clear, batch
   reuse order) — the gate's 9 artifact digests + 2 resume-parity booleans are
   the acceptance proof; run `--self-test` (33/33), probe (both modes),
   batch-test mode 2 (bitwise), both test-hidden-states binaries with
   hs-fixture, then the gate with a name-matched gemma-4-E2B.Q4_K_M.gguf on the
   SAME CUDA device class the anchor was captured on (RTX 3090 lineage).
5. Gate model must be re-acquired (off disk; no HF repo named in gate README;
   sha256 never recorded) — treat any digest mismatch with a re-downloaded
   file as provenance-suspect before blaming the migration; never re-baseline
   from the migrated binary.
6. Whitespace-clean diffs on all PR-surface files; update stale doc/comment
   mentions of the old API (examples/hidden-states/README.md:44,57;
   test-hidden-states.cpp:63; llama-raii.h:13–15) — not gate-pinned, but the
   fork's doc-honesty standard applies.
