set -euo pipefail
# Strip // and /* */ comments so identifier checks verify code, not prose
strip_comments() {
  python3 - "$1" <<'PY'
import sys, re
src = open(sys.argv[1]).read()
src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
src = re.sub(r'^\s*//.*$', '', src, flags=re.M)
src = re.sub(r'//[^\n"]*$', '', src, flags=re.M)
# A trailing // comment containing a double quote defeats the sub above (the
# char class stops at the quote), so the line survives stripping with its
# comment tail intact. Over-deletion is safe here: it can only cause a
# false FAIL, never a false PASS.
src = re.sub(r'//[^\n]*"[^"\n]*$', '', src, flags=re.M)
# String/char literals are stripped per LINE: a file-level scan can mis-sync on
# an escaped quote or continuation and swallow real code (this ate the pool=none
# guard when run whole-file). Per-line keeps any mis-sync local to one line.
_out = []
for _line in src.splitlines(keepends=True):
    if _line.count('"') >= 2:
        _line = re.sub(r'"(?:\\.|[^"\\])*"', '""', _line)
    if _line.count("'") >= 2:
        _line = re.sub(r"'(?:\\.|[^'\\])*'", "''", _line)
    _out.append(_line)
src = "".join(_out)
try:
    sys.stdout.write(src)
    sys.stdout.flush()
except BrokenPipeError:
    import os
    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    os._exit(0)  # downstream grep -q closed early on a match; that is success
PY
}

echo "=== Check 1: uint64_t key (flat accumulator) ==="
strip_comments tools/hs-extract-batch/hs-accum.h | grep -q "inline uint64_t make_accum_key" && echo "PASS" || { echo "FAIL: uint64_t make_accum_key missing in code"; exit 1; }
echo "=== Check 2: output_all defined ==="
strip_comments src/llama-context.cpp | grep -q "const bool output_all *= *cparams\.embeddings;" && echo "PASS" || { echo "FAIL: output_all missing or polarity wrong"; exit 1; }
strip_comments src/llama-context.cpp | grep -q "if (output_all) {" && echo "PASS" || { echo "FAIL: output_all primary use missing or inverted"; exit 1; }
echo "=== Check 3: RAII LlamaBackend ==="
for f in examples/hidden-states/hidden-states.cpp tools/hs-extract/hs-extract.cpp; do
  strip_comments "$f" | grep -qE "LlamaBackend +[A-Za-z_][A-Za-z_0-9]*;" && echo "PASS: $f" || { echo "FAIL: missing RAII LlamaBackend declaration in $f"; exit 1; }
done
echo "=== Check 4: hidden-state getters synchronize (upstream getter idiom) ==="
for f in src/llama-context.cpp; do
  grep -q "llama_context::get_hidden_state\b" "$f" || { echo "FAIL: get_hidden_state impl missing"; exit 1; }
  python3 - "$f" <<'CHK'
import sys, re
src = open(sys.argv[1]).read()
src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
src = re.sub(r'^\s*//.*$', '', src, flags=re.M)
body = src[src.find("float * llama_get_hidden_state("):]
sync = body.find("ctx->synchronize()")
first_ret = body.find("return")
if sync == -1:
    print("FAIL: llama_get_hidden_state does not synchronize"); raise SystemExit(1)
# first-statement pin: synchronize must precede the first return AND be the
# first statement of the function body (catches throw/reorder hoists)
_first_ret = body.find("return")
if _first_ret != -1 and sync > _first_ret:
    print("FAIL: synchronize must precede the first return"); raise SystemExit(1)
_brace = body.find("{")
_stmt = re.sub(r"//[^\n]*", " ", body[_brace + 1 : sync]).strip()
if _stmt:
    print("FAIL: synchronize must be the first statement of llama_get_hidden_state"); raise SystemExit(1)
print("PASS: getters sync (position-pinned)")
CHK
done
echo "=== Check 5: All fclose calls are inside FilePtr RAII wrapper ==="
python3 - <<'PY'
import re
from pathlib import Path
# post-decomposition the tool is 5 TUs; FilePtr lives in io-util.h.
# Scan every TU so a target file with zero fclose() cannot vacuous-pass.
tool_tus = [
    'tools/hs-extract-batch/hs-extract-batch.cpp',
    'tools/hs-extract-batch/io-util.cpp',
    'tools/hs-extract-batch/assignments-io.cpp',
    'tools/hs-extract-batch/self-test.cpp',
    'tools/hs-extract/hs-extract.cpp',
]
outside = []
for tu in tool_tus:
  raw = Path(tu).read_text()
  # Strip comments and string/char literals first: a literal containing
  # 'fclose(' must not fail the gate (F-7), and a comment mentioning it
  # must not either.
  raw = re.sub(r'/\*.*?\*/', ' ', raw, flags=re.S)
  raw = re.sub(r'^\s*//.*$', '', raw, flags=re.M)
  raw = re.sub(r'"(?:\\.|[^"\\])*"', '""', raw)
  raw = re.sub(r"'(?:\\.|[^'\\])*'", "''", raw)
  lines = raw.splitlines()
  in_fileptr = False
  brace_depth = 0
  for lineno, line in enumerate(lines, 1):
      stripped = line.strip()
      if stripped.startswith('struct FilePtr'):
          in_fileptr = True
          brace_depth = 0
      if in_fileptr:
          brace_depth += line.count('{') - line.count('}')
          if brace_depth <= 0:
              in_fileptr = False
      elif re.search(r'\bfclose\s*\(', line.split('//')[0]) and not stripped.startswith(('//', '*', '/*')):
          outside.append(f'{tu}:{lineno}: {line.strip()}')
if outside:
    print(f'FAIL: {len(outside)} fclose calls outside FilePtr:')
    for o in outside: print(f'  {o}')
    raise SystemExit(1)
print('PASS: all fclose calls inside FilePtr RAII wrapper')
PY
echo "=== Check 6: every checked_write() call site tests its bool return ==="
python3 - <<'PY'
from pathlib import Path
# CHECKED_WRITE macro became checked_write() (bool-returning inline fn,
# io-util.h) in the decomposition; the invariant it guarded is that no
# write result is ever ignored. Scan every tool TU for ignored returns.
tool_tus = [
    'tools/hs-extract-batch/hs-extract-batch.cpp',
    'tools/hs-extract-batch/io-util.cpp',
    'tools/hs-extract-batch/assignments-io.cpp',
    'tools/hs-extract-batch/self-test.cpp',
    'tools/hs-extract/hs-extract.cpp',
]
import re
bad = []
total_calls = 0
sum_raw = 0
for tu in tool_tus:
    raw = Path(tu).read_text()
    raw_calls = len(re.findall(r"\bchecked_write\s*\(", raw))
    # Strip line comments FIRST: apostrophes inside // comments can pair into a
    # fake multi-line char literal during the literal pass and swallow real code
    # (this ate 3 of 8 checked_write sites in hs-extract-batch.cpp once).
    raw = re.sub(r'//[^\n]*', '', raw)
    # Strip block comments and string/char literals BEFORE any paren/depth
    # analysis: parens inside them must not poison the continuation joiner
    # (D8: '"( v6"' used to disarm the whole rest of the TU).
    raw = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), raw, flags=re.S)
    raw = re.sub(r'"(?:\\.|[^"\\])*"', '""', raw)
    raw = re.sub(r"'(?:\\.|[^'\\])*'", "''", raw)
    stripped_calls = len(re.findall(r"\bchecked_write\s*\(", raw))
    if stripped_calls < raw_calls:
        # fail-closed: the literal stripper mis-synced and ate real call sites
        print(f"FAIL: literal stripper ate {raw_calls - stripped_calls} checked_write call sites in {tu}; fix the strip order")
        raise SystemExit(1)
    sum_raw += raw_calls
    lines = raw.splitlines()
    # Join continuation lines so multi-line if(...) conditions are seen as
    # one logical statement: a call on a continuation line whose statement
    # opened with if(/return/etc counts as tested. (Track open paren depth
    # of the last control keyword; a bare-depth call is an ignored return.)
    joined = []  # (first_lineno, logical_line)
    buf, start, depth = '', 0, 0
    for lineno, line in enumerate(lines, 1):
        code = line.split('//')[0]
        if not buf:
            start = lineno
        buf += ('' if buf.endswith(' ') or not buf else ' ') + code.strip() if buf else code.strip()
        depth += code.count('(') - code.count(')')
        if depth <= 0 and buf.strip():
            joined.append((start, buf))
            buf, depth = '', 0
    if buf.strip():
        joined.append((start, buf))
    for lineno, code in joined:
        if 'checked_write(' in code:
            total_calls += 1
            # a tested call is inside if(...)/return/&&/|| or assigns to a var
            if not re.search(r'(if\s*\(|return|&&|\|\||=\s*\w+$|bool\s+ok)', code):
                bad.append(f'{tu}:{lineno}: {code.strip()}')
if bad:
    print('\n'.join(bad))
    raise SystemExit(1)
print(f'PASS ({total_calls} checked_write call statements ({sum_raw} call sites), all tested)')
PY
echo "=== Check 7: assignment reader uses explicit status ==="
strip_comments tools/hs-extract-batch/assignments-io.h | grep -q "AssignmentReadStatus::error" && echo "PASS" || { echo "FAIL: explicit assignment read status missing"; exit 1; }
echo "=== Check 8: prompt pre-scan is pure and does not call exit ==="
# auto_size_ctx was removed in the single-pass pre-scan
# refactor; its successor must uphold the same contract: pure
# stream walk, returns bool, never exits.
# The function must exist where this check scans it: if it moves TU the
# sed range goes empty and the exit() scan would pass vacuously.
if ! grep -qE "static bool scan_prompts_file\(" tools/hs-extract-batch/hs-extract-batch.cpp; then
echo "FAIL: scan_prompts_file not found in hs-extract-batch.cpp (moved TU? update this check's target)"; exit 1
fi
if sed -n '/static bool scan_prompts_file/,/^}/p' tools/hs-extract-batch/hs-extract-batch.cpp | grep -qE '\b(exit|abort|quick_exit|terminate|_Exit)[[:space:]]*\('; then
echo "FAIL: scan_prompts_file calls an exit-family function"; exit 1
fi
if ! grep -q "struct PromptsScan" tools/hs-extract-batch/hs-extract-batch.cpp; then
echo "FAIL: PromptsScan struct missing (pre-scan contract changed; update this check)"; exit 1
fi
echo "PASS"
echo "=== Check 9: No off-by-one ==="
# Alias-tolerant: collect every identifier bound to a layer-count expression
# (e.g. 'const size_t nl = layers.size();'), then reject '<=' against the
# canonical limits AND every collected alias. Python pass so aliases are
# resolved, not pattern-matched. (Verifier bypass history: 'li <= layers.size()'
# and 'li <= nl' both slipped earlier literal greps.)
python3 - <<'PY'
import re
LIMITS = r"(?:(?:target_|hs_)?layers?\.size\(\)|n_layers_total|n_layer\b|n_layers\b)"
failed = False
for tu in ["tools/hs-extract/hs-extract.cpp", "tools/hs-extract-batch/hs-extract-batch.cpp"]:
    src = open(tu).read()
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    src = re.sub(r'^\s*//.*$', '', src, flags=re.M)
    src = re.sub(r'//[^\n"]*$', '', src, flags=re.M)
    src = re.sub(r'//[^\n]*"[^"\n]*$', '', src, flags=re.M)
    stripped = src
    aliases = set(re.findall(r"(\w+)\s*=\s*[^;\n]*" + LIMITS, stripped))
    alias_alt = "|".join(re.escape(a) for a in sorted(aliases))
    pat = r"(^|[^A-Za-z0-9_])\w+\s*<=\s*(?:" + LIMITS
    if alias_alt:
        pat += "|" + alias_alt
    pat += r")"
    for lineno, line in enumerate(stripped.splitlines(), 1):
        if re.search(pat, line):
            print(f"FAIL: off-by-one risk in {tu}:{lineno}: {line.strip()}")
            failed = True
if not aliases:
    # fail-closed: a collector that never fires means the grammar has rotted
    print(f"FAIL: Check 9 alias collector matched nothing in {tu} (grammar rot?)")
    failed = True
# canary: the grammar itself must cover the live consumer limit
# (target_layers.size()); catches silent grammar reversion.
if "target_" not in LIMITS:
    print("FAIL: Check 9 canary lost: LIMITS no longer covers target_layers.size()")
    failed = True
if failed:
    raise SystemExit(1)
print("PASS: no <= against layer-count limits or their aliases")
PY
echo "=== Check 10: checkpoint n_masks/n_layers_data bounds validation ==="
# Pin ENFORCEMENT, not text: the guarded block must contain 'return false;'
# (a guard whose body was emptied must fail). -A3 reaches the body on
# stripped source; the condition line and the return must co-occur.
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -A3 -F "if (n_masks < 0 || n_masks > MAX_MASKS)" | grep -q "return false;"; then
  echo "FAIL: checkpoint n_masks bounds check not enforced"; exit 1
fi
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -A3 -F "if (n_layers_data < 0 || n_layers_data > MAX_LAYERS)" | grep -q "return false;"; then
  echo "FAIL: checkpoint n_layers_data bounds check not enforced"; exit 1
fi
echo "PASS"
echo "=== Check 11: output writer uses pre-built index (not O(K^2) rescan) ==="
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -qE 'gm_pairs\[[A-Za-z_][A-Za-z_0-9]*\]'; then
  echo "FAIL: output writer pre-built index (gm_pairs) missing at use site"; exit 1
fi
echo "PASS"
echo "=== Check 12: checkpoint count/layer_idx validation ==="
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -A3 -E "if \(count < 0\)" | grep -q "return false;"; then
  echo "FAIL: checkpoint count validation not enforced"; exit 1
fi
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -A3 -E "layer_idx < 0 \|\| layer_idx > 0xFFFF" | grep -q "return false;"; then
  echo "FAIL: checkpoint layer_idx validation not enforced"; exit 1
fi
echo "PASS"
echo "=== Check 13: producer-consumer pipeline error notification ==="
# The batch tool uses a producer-consumer pipeline where a prefetch
# thread (producer) reads prompts + assignments while the main thread
# (consumer) does GPU decode. Error paths must notify the consumer via
# the condition variable to avoid deadlock.
# Anchored to call shapes, not bare identifiers: a comment mentioning
# pfq.cv.notify must not satisfy this check.
if ! strip_comments tools/hs-extract-batch/hs-extract-batch.cpp | grep -qE "^\s*(pfq\.)?cv\.notify(_one|_all)?\("; then
  echo "FAIL: producer-consumer cv notification call missing"; exit 1
fi
if ! strip_comments tools/hs-extract-batch/hs-extract-batch.cpp | grep -q "pfq.producer_done"; then
  echo "FAIL: producer_done flag missing"; exit 1
fi
python3 - <<'PY'
import re
src = open("tools/hs-extract-batch/hs-extract-batch.cpp").read()
src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
src = re.sub(r'^\s*//.*$', '', src, flags=re.M)
i = src.find("if (prompts_fin.bad())")
if i == -1:
    print("FAIL: bad-stream handler missing (prompts_fin.bad())"); raise SystemExit(1)
j = src.find("return;", i)
block = src[i:j] if j != -1 else src[i:]
has_done = "producer_done = true;" in block
has_notify = re.search(r"pfq\.cv\.notify_all\(\);", block) is not None
ordered = has_done and has_notify and block.find("producer_done = true;") < block.find("pfq.cv.notify_all();")
if not (has_done and has_notify and ordered):
    print("FAIL: bad-stream handler lost its consumer wakeup (must set producer_done then notify_all)"); raise SystemExit(1)
print("PASS: bad-stream handler notifies consumer")
PY
echo "PASS"
echo "=== Check 14: shared RAII header used by all fork tools ==="
# RAII wrappers (LlamaBackend, LlamaModel, LlamaContext, LlamaBatch)
# are extracted to common/llama-raii.h. All fork tools must include it
# instead of defining their own copies.
if [ ! -f common/llama-raii.h ]; then
  echo "FAIL: common/llama-raii.h missing"; exit 1
fi
for f in tools/hs-extract/hs-extract.cpp tools/hs-extract-batch/hs-extract-batch.cpp tests/test-hidden-states.cpp; do
  # code, not prose: the raw file must carry the include, AND the stripped
  # file must still carry an include statement at that spot (so a commented-out
  # include cannot satisfy this; the literal pass empties quoted names).
  raw_has=$(grep -cE '^#[[:space:]]*include[[:space:]]*"llama-raii\.h"' "$f" || true)
  stripped_has=$(strip_comments "$f" | grep -cE '^#[[:space:]]*include[[:space:]]*""' || true)
  if [ "$raw_has" -lt 1 ] || [ "$stripped_has" -lt 1 ]; then
    echo "FAIL: $f does not include llama-raii.h (raw=$raw_has stripped_includes=$stripped_has)"; exit 1
  fi
done
echo "PASS"
echo "=== Check 15: async pipeline backpressure (bounded queue) ==="
# The producer-consumer queue must have backpressure to prevent
# unbounded memory growth on 200K-prompt runs.
# Anchored to use-sites: the bound must appear in a comparison (the
# backpressure predicate), not merely be declared or mentioned.
if ! strip_comments tools/hs-extract-batch/hs-extract-batch.cpp | grep -q "cv_space"; then
  echo "FAIL: backpressure cv_space missing"; exit 1
fi
if ! strip_comments tools/hs-extract-batch/hs-extract-batch.cpp | grep -qF 'pfq.cv_space.wait(lk, [&]{ return pfq.queue.size() < MAX_PREFETCH || pfq.producer_done.load(); })'; then
  echo "FAIL: live backpressure wait predicate missing"; exit 1
fi
if ! strip_comments tools/hs-extract-batch/hs-extract-batch.cpp | grep -q "stop_producer_and_join"; then
  echo "FAIL: producer-stop-and-join path missing"; exit 1
fi
echo "PASS"
echo "=== Check 16: server pool=none response size limit ==="
# The /hidden-states endpoint must cap pool=none response size to
# prevent DoS via enormous JSON responses. Pin the guard predicate at
# its use site (the error message mentioning the constant is not proof).
if ! strip_comments tools/server/server-context.cpp | grep -A6 -F "if (total_all_layers > MAX_POOL_NONE_FLOATS)" | grep -qE "server_task_result_error|send_error"; then
  echo "FAIL: pool=none response size limit not enforced"; exit 1
fi
echo "PASS"
echo "=== Check 17: checkpoint v2+ sum-based records (no precision loss) ==="
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -q "CHECKPOINT_VERSION = 6"; then
  echo "FAIL: checkpoint version is not 6 (v6: accumulator-region checksum)"; exit 1
fi
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -q "bool write_sum"; then
  echo "FAIL: write_sum parameter missing from _write_accumulator_to_file signature"; exit 1
fi
# Pin the use polarity: the parameter must gate lossless sum writes positively
# (inverting it to if (!write_sum) would silently swap payloads to lossy means).
if ! strip_comments tools/hs-extract-batch/io-util.cpp | grep -qE "if \(write_sum\)"; then
  echo "FAIL: write_sum positive-use branch missing (polarity inverted?)"; exit 1
fi
echo "PASS"
echo "=== Check 18: checkpoint durability order (fsync before rename) ==="
# The durability guarantee: the tmp file is flushed+fsynced and closed before
# the rename publishes it. Pin the ordered sequence inside write_checkpoint's
# tmp-handling: if (!f.sync()) -> f.reset() -> rename(...). A reorder that
# publishes unsynced bytes must fail here.
python3 - <<'PY'
import re
src = open("tools/hs-extract-batch/io-util.cpp").read()
src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
src = re.sub(r'^\s*//.*$', '', src, flags=re.M)
src = re.sub(r'//[^\n"]*$', '', src, flags=re.M)
src = re.sub(r'//[^\n]*"[^\n]*$', '', src, flags=re.M)

def ordered(sync_handle, rename_arg):
    """True iff the exact sequence sync-handle-check -> handle.reset(); ->
    rename(rename_arg) occurs contiguously (only whitespace between the
    reset and the rename)."""
    i = src.find(f"rename({rename_arg}")
    if i == -1:
        return False
    pat = (r"if \(!" + re.escape(sync_handle) + r"\.sync\(\)\) \{"
           r"[^}]*}"                       # sync failure block, closed
           r"\s*" + re.escape(sync_handle) + r"\.reset\(\);\s*"
           r"(?:if \(\s*)?rename\(" + re.escape(rename_arg))
    return re.search(pat, src) is not None

ok_ckpt = ordered("f", "temp_path.c_str(), ckpt_path")
ok_out = ordered("out", "temp_path.c_str(), output_path")
if not ok_ckpt:
    print("FAIL: checkpoint durability order broken (fsync -> reset -> rename(ckpt))")
if not ok_out:
    print("FAIL: output durability order broken (fsync -> reset -> rename(output))")
if ok_ckpt and ok_out:
    print("PASS: fsync -> reset -> rename order enforced on both writers")
    raise SystemExit(0)
raise SystemExit(1)
PY
echo "=== Check 19: /hidden-states capture toggle cleared on all return paths ==="
# The capture disable must be structural (a scope guard declared at function
# entry), not a manual call at the end of the success path: manual disables
# leave the toggle enabled on early-error returns.
python3 - <<'PY'
import re
src = open("tools/server/server-context.cpp").read()
src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
src = re.sub(r'^\s*//.*$', '', src, flags=re.M)
ok_struct = re.search(r"struct hs_toggle_reset \{[^}]*llama_set_extract_hidden_states\(c, false\);", src)
ok_use = re.search(r"hs_reset\{slot\.ctx_tgt\}", src)
if not ok_struct:
    print("FAIL: hs_toggle_reset scope guard missing in send_hidden_states"); raise SystemExit(1)
if not ok_use:
    print("FAIL: hs_toggle_reset guard not instantiated with slot.ctx_tgt"); raise SystemExit(1)
# position pin: the guard must be instantiated BEFORE the first return in the
# function body - a guard moved below the early returns re-leaks the toggle.
body_start = src.find("void send_hidden_states")
body = src[body_start:]
guard_pos = body.find("hs_reset{slot.ctx_tgt}")
if guard_pos == -1:
    print("FAIL: hs_toggle_reset guard not instantiated"); raise SystemExit(1)
first_return = body.find("return")
if first_return != -1 and first_return < guard_pos:
    print("FAIL: hs_toggle_reset guard instantiated after an early return (leak path)"); raise SystemExit(1)
# first-statement pin: nothing but whitespace between the function's opening
# brace and the guard (catches return- AND throw-class early exits)
_first_brace = body.find("{")
_guard_decl = body.find("struct hs_toggle_reset")
between = re.sub(r"//[^\n]*", " ", body[_first_brace + 1 : _guard_decl]).strip()
if between:
    print("FAIL: hs_toggle_reset guard must be the first statement of send_hidden_states"); raise SystemExit(1)
print("PASS: capture disable is structural (scope guard)")
PY
echo "OK: All audit fix patterns verified"
