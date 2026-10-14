#!/usr/bin/env python3
"""Golden byte-identity gate for run_batch (the hs-extract-batch refactor).

Two modes:
  baseline <cuda-bin>  run all scenarios with the PRE-refactor CUDA binary,
                       save every produced artifact under baseline/
  check <cuda-bin>     run all scenarios with the POST-refactor CUDA binary,
                       byte-compare every artifact against baseline/

Methodology: the PRODUCTION path -- CUDA binary (libggml-cuda.so* present),
-ngl 99, gemma-E2B only, hard requirements, no fallbacks. Byte-identity of
every artifact is the pass condition. Exit 0 = all identical.

Resume scenarios kill the binary at the 2nd "Checkpoint saved" stderr line
(deterministic checkpoint boundary), then re-run with --resume; the resumed
output must be byte-identical to the full run's output (checkpoints v2+
store raw sums losslessly, so float accumulation order is preserved).
"""
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time

# Per-process scratch root: two concurrent gate instances must never share
# scenario dirs (one's rmtree would delete the other's in-flight writes).
# Digest comparison is against the committed anchor, so distinct scratch roots
# change nothing about what is compared.
GOLD = pathlib.Path("/tmp/hs-batch-golden") / f"proc-{os.getpid()}"

def _prune_stale_scratch():
    """Remove proc-* scratch roots older than a day (crashed gate runs)."""
    root = pathlib.Path("/tmp/hs-batch-golden")
    if not root.is_dir():
        return
    now = time.time()
    for d in root.glob("proc-*"):
        try:
            if now - d.stat().st_mtime > 86400:
                shutil.rmtree(d)
        except OSError:
            pass  # a live instance owns it; skip

# Golden INPUT fixtures are committed to the repo (deterministic, ~4KB);
# volatile scenario outputs stay under /tmp by design.
INPUTS = pathlib.Path(__file__).resolve().parent / "golden_inputs"

DEFAULT_E2B = os.environ.get("HS_GATE_MODEL")
MODELS = {
    "e2b": DEFAULT_E2B,
}

# name, model-key, layers, extra args, assignments-file key, is_resume_scenario
SCENARIOS = [
    ("s5_e2b_comp_records",   "e2b",   "0,17,35", ["--save-per-record"], "main", False),
    ("s6_e2b_resume",         "e2b",   "0,17,35", [], "main", True),
    ("s7_e2b_gen_greedy",     "e2b",   "0,35",    ["--generate", "8", "--token-skip", "2"], "gen", False),
    ("s8_e2b_gen_sampled",    "e2b",   "0,35",    ["--generate", "8", "--token-skip", "2",
                                                  "--temperature", "0.8", "--top-k", "40",
                                                  "--repeat-penalty", "1.1"], "gen", False),
    # Resume parity with generation state: gen mode carries sampling params
    # in the checkpoint fingerprint and has its own records path; only a
    # kill-and-resume exercises that resume path end-to-end.
    ("s9_e2b_gen_resume",     "e2b",   "0,35",    ["--generate", "8", "--token-skip", "2",
                                                  "--temperature", "0.8", "--top-k", "40",
                                                  "--repeat-penalty", "1.1"], "gen", True),
]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

ASSIGN_FILES = {"main": "assignments.bin", "gen": "assignments_gen.bin"}

def base_cmd(bin_path, model, layers, out, extra, assign_key="main"):
    return [bin_path, model, str(INPUTS / "prompts.txt"), layers, out,
            "--batch", "--assignments", str(INPUTS / ASSIGN_FILES[assign_key]),
            "-ngl", "99"] + list(extra)

def model_n_layer(model_path):
    """Pre-flight: read *.block_count from the GGUF KV table."""
    import struct

    def read_fmt(f, fmt):
        b = f.read(struct.calcsize(fmt))
        if len(b) != struct.calcsize(fmt):
            sys.exit(f"REFUSED: {model_path}: truncated GGUF metadata")
        return struct.unpack(fmt, b)

    with open(model_path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"REFUSED: {model_path} is not a GGUF file")
        (version,) = read_fmt(f, "<I")
        cnt = "<Q" if version >= 3 else "<I"
        read_fmt(f, cnt)          # n_tensors (wire order: n_tensors, then n_kv)
        (n_kv,) = read_fmt(f, cnt)

        def rv(f, vtype):
            if vtype in (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12):
                return read_fmt(f, {0: "<b", 1: "<B", 2: "<h", 3: "<H", 4: "<i",
                                    5: "<I", 6: "<f", 7: "<?", 10: "<q",
                                    11: "<Q", 12: "<d"}[vtype])[0]
            if vtype == 8:
                (ln,) = read_fmt(f, cnt)
                return f.read(ln)
            if vtype == 9:
                (et,) = read_fmt(f, "<I")
                (n,) = read_fmt(f, cnt)
                return [rv(f, et) for _ in range(n)]
            sys.exit(f"REFUSED: {model_path}: unknown GGUF value type {vtype}")

        for _ in range(n_kv):
            (klen,) = read_fmt(f, cnt)
            key = f.read(klen).decode("utf-8", "replace")
            (vtype,) = read_fmt(f, "<I")
            val = rv(f, vtype)
            if key.endswith("block_count"):
                if isinstance(val, list):
                    sys.exit(f"REFUSED: {model_path}: block_count is not a scalar")
                return int(val)
    sys.exit(f"REFUSED: {model_path}: no *.block_count key in GGUF metadata")

def run_full(bin_path, model, layers, out, extra, assign_key="main", timeout=1800):
    cmd = base_cmd(bin_path, model, layers, out, extra, assign_key)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        sys.exit(f"FAIL: run exceeded {timeout}s wall clock: {' '.join(cmd)}")
    if r.returncode != 0:
        sys.exit(f"FAIL: run exited {r.returncode}: {' '.join(cmd)}\nstderr tail:\n{r.stderr[-1500:]}")
    return r

def run_kill_and_resume(bin_path, model, layers, out, extra, assign_key="main"):
    """Full run -> full_out; kill at 2nd checkpoint; resume -> out. Return paths."""
    full_out = str(out) + ".fullrun.bin"
    run_full(bin_path, model, layers, full_out, extra, assign_key)
    # kill run
    cmd = base_cmd(bin_path, model, layers, out, extra + ["--checkpoint-every", "5"], assign_key)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_src = proc.stderr
    if stderr_src is None:  # unreachable: stderr=PIPE
        proc.kill()
        sys.exit("FAIL: Popen did not provide a stderr pipe for the kill run")
    # Wall-clock bound on the kill phase: stderr is drained on a helper thread
    # and the deadline is enforced by proc.wait(timeout) on the main thread, so
    # a wedged binary (GPU hang after checkpoint 1) fails the gate instead of
    # blocking forever inside readline().
    deadline = time.monotonic() + 1800
    checkpoints_seen = 0
    killed = False

    def _drain():
        nonlocal checkpoints_seen, killed
        for line in stderr_src:
            if "Checkpoint saved" in line:
                checkpoints_seen += 1
                if checkpoints_seen == 2:
                    proc.send_signal(signal.SIGKILL)
                    killed = True
                    return

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    try:
        proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        sys.exit(f"FAIL: kill run exceeded 1800s wall clock (saw {checkpoints_seen} checkpoints)")
    drainer.join(timeout=60)
    if not killed:
        proc.kill()
        sys.exit(f"FAIL: never saw 2nd checkpoint in kill run (saw {checkpoints_seen})")
    ckpt = str(out) + ".checkpoint"
    if not pathlib.Path(ckpt).exists():
        sys.exit("FAIL: no checkpoint file after kill run")
    # resume run
    run_full(bin_path, model, layers, out, extra + ["--resume"], assign_key)
    return full_out, out

def artifacts(dir_path):
    return sorted(p.name for p in pathlib.Path(dir_path).iterdir() if p.is_file())

def do_scenario(bin_path, name, model_key, layers, extra, assign_key, is_resume, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    model = MODELS[model_key]
    out = str(workdir / "output.bin")
    if is_resume:
        full_out, resume_out = run_kill_and_resume(bin_path, model, layers, out, extra, assign_key)
        return {"output.bin.fullrun": sha256(full_out), "output.bin.resumed": sha256(resume_out),
                "fullrun_equals_resumed": sha256(full_out) == sha256(resume_out)}
    run_full(bin_path, model, layers, out, extra, assign_key)
    result = {}
    for f in artifacts(workdir):
        result[f] = sha256(pathlib.Path(workdir) / f)
    # sanity: expected artifacts must exist
    if "output.bin" not in result:
        sys.exit(f"FAIL: {name} produced no output.bin")
    if "--save-per-record" in extra and "output.bin.records.bin" not in result:
        sys.exit(f"FAIL: {name} produced no records sidecar")
    return result

def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("baseline", "check"):
        sys.exit(f"usage: {sys.argv[0]} baseline|check <cuda-binary>")
    mode, bin_path = sys.argv[1], os.path.abspath(sys.argv[2])
    if not os.path.exists(bin_path):
        sys.exit(f"binary not found: {bin_path}")
    # Refuse CPU-only binaries: the gate runs the PRODUCTION path (CUDA
    # binary, -ngl 99). A CUDA-off build produces different gemma numerics
    # than the production build even at -ngl 0 (verified 2026-08-18), so it
    # can never serve as a valid gate binary.
    bin_dir = pathlib.Path(bin_path).parent
    if not any(bin_dir.glob("libggml-cuda.so*")):
        sys.exit(
            f"REFUSED: {bin_path} is a CPU-only build (no libggml-cuda.so* in "
            f"{bin_dir}). The gate runs the production path: CUDA binary, "
            f"-ngl 99. Rebuild with -DGGML_CUDA=ON and rerun.")
    # Hard requirements -- no fallbacks. The e2b model is the gate's only
    # model; a missing model aborts the gate, it does not fall back.
    for model_key, model_path in MODELS.items():
        if not model_path:
            sys.exit("REFUSED: set HS_GATE_MODEL=/path/to/model.gguf -- "
                     "the gate runs the production path; no fallbacks.")
        if not pathlib.Path(model_path).exists():
            sys.exit(f"REFUSED: model {model_key} not found at {model_path} -- "
                     "the gate runs the production path; no fallbacks.")
        # Pre-flight: the scenarios request layer indices up to 35. Refuse a
        # model that cannot satisfy them BEFORE the first scenario burns GPU
        # time (hs_parse_layer_list would reject it mid-run with a confusing
        # argument error).
        max_needed = max(int(t) for _, _, layers, *_ in SCENARIOS
                         for t in layers.split(","))
        n_layer = model_n_layer(model_path)
        if max_needed >= n_layer + 1:
            sys.exit(f"REFUSED: model {model_key} has {n_layer} layers "
                     f"(n_slots={n_layer + 1}) but scenario layers request index "
                     f"{max_needed} - wrong model for the gate")
    for req in ("prompts.txt", "assignments.bin", "assignments_gen.bin"):
        if not (INPUTS / req).exists():
            sys.exit(f"REFUSED: inputs missing at {INPUTS / req} - run gen_inputs.py")

    _prune_stale_scratch()

    target = GOLD / mode
    if target.exists():
        for p in target.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
    target.mkdir(parents=True, exist_ok=True)

    digests = {}
    for name, model_key, layers, extra, assign_key, is_resume in SCENARIOS:
        print(f"[{mode}] {name} ...", flush=True)
        digests[name] = do_scenario(bin_path, name, model_key, layers, extra, assign_key, is_resume, target / name)

    dig_path = GOLD / f"digests_{mode}.json"
    dig_path.write_text(json.dumps(digests, indent=2))

    if mode == "baseline":
        n = sum(len(v) for v in digests.values())
        print(f"baseline captured: {len(digests)} scenarios, {n} artifacts")
        print(f"baseline digests written to {dig_path}")
        return 0

    # check mode: compare
    dig_path = GOLD / "digests_baseline.json"
    if not dig_path.exists():
        # Fall back to the committed anchor next to this script. Never point
        # the operator at 'record' mode: re-baselining from the binary under
        # test would overwrite the anchor with self-certifying bytes.
        committed = pathlib.Path(__file__).resolve().parent / "digests_baseline.json"
        if committed.exists():
            print(f"note: {dig_path} missing - using the committed anchor at {committed}")
            dig_path = committed
        else:
            sys.exit(f"REFUSED: baseline digests missing at {dig_path} and no committed anchor at "
                     f"{committed} - restore gate/digests_baseline.json from a trusted commit; "
                     f"never re-baseline from the binary under test")
    base = json.loads(dig_path.read_text())
    # A baseline containing scenarios that were renamed/removed (or a typo'd
    # key) would silently shrink the comparison to the intersection. Refuse.
    stale = sorted(set(base) - {s[0] for s in SCENARIOS})
    if stale:
        sys.exit(f"REFUSED: baseline scenario keys drifted: {', '.join(stale)} - "
                 f"re-capture the baseline from a trusted commit")
    fails = 0
    for name, _, _, _, _, _ in SCENARIOS:
        b, c = base.get(name, {}), digests[name]
        keys = sorted(set(b) | set(c))
        for k in keys:
            bv, cv = b.get(k), c.get(k)
            if bv is None or cv is None:
                status, detail = "FAIL", "missing artifact"
            elif b[k] == c[k]:
                status, detail = "ok", (cv[:16] if isinstance(cv, str) else str(cv))
            else:
                status, detail = "FAIL", f"{bv[:16] if isinstance(bv, str) else bv} != {cv[:16] if isinstance(cv, str) else cv}"
            print(f"  {name:26s} {k:28s} {status} {detail}")
            if status == "FAIL":
                fails += 1
    if fails:
        print(f"\n{fails} MISMATCHES -- refactor changed bytes")
        return 1
    print("\nALL_BYTE_IDENTICAL -- refactor changed structure, not numerics")
    return 0

if __name__ == "__main__":
    sys.exit(main())
