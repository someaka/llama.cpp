#!/usr/bin/env python3
"""Golden byte-identity gate for run_batch (the hs-extract-batch refactor).

Two modes:
  baseline <cpu-bin>   run all scenarios with the PRE-refactor CPU binary,
                       save every produced artifact under baseline/
  check <cpu-bin>      run all scenarios with the POST-refactor CPU binary,
                       byte-compare every artifact against baseline/

Methodology (from HANDOFF-2026-08-16): CPU-only binary (CUDA-off build),
-ngl 0, original tokenization flags (no --no-bos). Byte-identity of every
artifact is the pass condition. Exit 0 = all identical.

Resume scenarios kill the binary at the 2nd "Checkpoint saved" stderr line
(deterministic checkpoint boundary), then re-run with --resume; the resumed
output must be byte-identical to the full run's output (v2 checkpoint stores
raw sums losslessly, so float accumulation order is preserved).
"""
import hashlib
import os
import pathlib
import signal
import subprocess
import sys

GOLD = pathlib.Path("/tmp/hs-batch-golden")
INPUTS = GOLD / "inputs"

MODELS = {
    "llama": "/tmp/llama-1b-q4_k_m.gguf",
    "e2b": "/home/a/Bureau/Work/CrimsonRed/data/models/gemma-4-E2B.Q4_K_M.gguf",
}

# name, model-key, layers, extra args, assignments-file key, is_resume_scenario
SCENARIOS = [
    ("s1_llama_comp_records", "llama", "0,8,16", ["--save-per-record"], "main", False),
    ("s2_llama_resume",       "llama", "0,8,16", [], "main", True),
    ("s3_llama_gen_greedy",   "llama", "0,16",   ["--generate", "8", "--token-skip", "2"], "gen", False),
    ("s4_llama_gen_sampled",  "llama", "0,16",   ["--generate", "8", "--token-skip", "2",
                                                  "--temperature", "0.8", "--top-k", "40",
                                                  "--repeat-penalty", "1.1"], "gen", False),
    ("s5_e2b_comp_records",   "e2b",   "0,17,35", ["--save-per-record"], "main", False),
    ("s6_e2b_resume",         "e2b",   "0,17,35", [], "main", True),
    ("s7_e2b_gen_greedy",     "e2b",   "0,35",    ["--generate", "8", "--token-skip", "2"], "gen", False),
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
            "-ngl", "0"] + list(extra)

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
        sys.exit(f"usage: {sys.argv[0]} baseline|check <cpu-binary>")
    mode, bin_path = sys.argv[1], os.path.abspath(sys.argv[2])
    if not os.path.exists(bin_path):
        sys.exit(f"binary not found: {bin_path}")
    if not (INPUTS / "assignments.bin").exists():
        sys.exit(f"inputs missing -- run gen_inputs.py first")

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
    base = __import__("json").loads((GOLD / "digests_baseline.json").read_text())
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
