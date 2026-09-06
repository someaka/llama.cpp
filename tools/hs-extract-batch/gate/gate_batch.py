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
import os
import pathlib
import signal
import subprocess
import sys

# Per-process scratch root: two concurrent gate instances must never share
# scenario dirs (one's rmtree would delete the other's in-flight writes).
# Digest comparison is against the committed anchor, so distinct scratch roots
# change nothing about what is compared.
GOLD = pathlib.Path("/tmp/hs-batch-golden") / f"proc-{os.getpid()}"
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

def run_full(bin_path, model, layers, out, extra, assign_key="main", timeout=1800):
    cmd = base_cmd(bin_path, model, layers, out, extra, assign_key)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
    checkpoints_seen = 0
    killed = False
    for line in proc.stderr:
        if "Checkpoint saved" in line:
            checkpoints_seen += 1
            if checkpoints_seen == 2:
                proc.send_signal(signal.SIGKILL)
                killed = True
                break
    proc.wait()
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
                     f"the gate runs the production path; no fallbacks.")
    for req in ("prompts.txt", "assignments.bin", "assignments_gen.bin"):
        if not (INPUTS / req).exists():
            sys.exit(f"REFUSED: inputs missing at {INPUTS / req} - run gen_inputs.py")

    target = GOLD / mode
    if target.exists():
        for p in target.iterdir():
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
    target.mkdir(parents=True, exist_ok=True)

    digests = {}
    for name, model_key, layers, extra, assign_key, is_resume in SCENARIOS:
        print(f"[{mode}] {name} ...", flush=True)
        digests[name] = do_scenario(bin_path, name, model_key, layers, extra, assign_key, is_resume, target / name)

    (GOLD / f"digests_{mode}.json").write_text(__import__("json").dumps(digests, indent=2))

    if mode == "baseline":
        n = sum(len(v) for v in digests.values())
        print(f"baseline captured: {len(digests)} scenarios, {n} artifacts")
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
    base = __import__("json").loads(dig_path.read_text())
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
