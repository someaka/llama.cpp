# hs-extract-batch golden gate

A byte-identity regression gate for the `llama-hs-extract-batch` batch
extraction path. Any refactor of the batch pipeline (streaming, checkpointing,
generation, records sidecar) must reproduce, byte for byte, the artifact
digests recorded in `digests_baseline.json`. If the bytes change, the refactor
changed numerics, not just structure.

## What the gate is

Seven scenarios run on CPU only, across two models:

- llama 1B Q4_K_M
- gemma E2B Q4_K_M

Per model: comprehension + records sidecar, kill-and-resume, and greedy
generation. The llama model additionally runs sampled generation. In detail:

| scenario                 | model | layers    | exercise                                 |
|--------------------------|-------|-----------|------------------------------------------|
| s1_llama_comp_records    | llama | 0,8,16    | comprehension + `--save-per-record`       |
| s2_llama_resume          | llama | 0,8,16    | kill at 2nd checkpoint, `--resume`        |
| s3_llama_gen_greedy      | llama | 0,16      | `--generate 8`, greedy                    |
| s4_llama_gen_sampled     | llama | 0,16      | `--generate 8`, temp 0.8, top-k 40        |
| s5_e2b_comp_records      | e2b   | 0,17,35   | comprehension + `--save-per-record`       |
| s6_e2b_resume            | e2b   | 0,17,35   | kill at 2nd checkpoint, `--resume`        |
| s7_e2b_gen_greedy        | e2b   | 0,35      | `--generate 8`, greedy                    |

Methodology: a CUDA-off CPU binary, `-ngl 0`, original tokenization flags
(no `--no-bos`). Every produced artifact (`output.bin`, the `.records.bin`
sidecar, resume full-run vs resumed output) is sha256-compared against the
baseline. Exit 0 means all bytes identical; exit 1 means the refactor changed
bytes. Resume scenarios additionally assert the resumed output is
byte-identical to the uninterrupted full run.

The driver refuses CUDA-ON binaries up front (a `libggml-cuda.so*` next to
the binary aborts with `REFUSED`): with a visible CUDA device, gemma-graph
op placement differs from a CUDA-off build even at `-ngl 0`, so a CUDA-ON
binary can never validly compare against this baseline (verified 2026-08-18
on the pass-5 gate #3 incident).

## Files

- `gate_batch.py` - the gate driver. Two modes: `baseline <cpu-bin>` (run all
  scenarios with a pre-refactor binary and record digests) and
  `check <cpu-bin>` (run all scenarios with the binary under test and compare
  against `digests_baseline.json`).
- `gen_inputs.py` - regenerates the deterministic gate inputs: `prompts.txt`
  (24 fixed prompts), `assignments.bin` (mixed skip/range assignments), and
  `assignments_gen.bin` (generation-compatible assignments).
- `digests_baseline.json` - the recorded pre-refactor sha256 digests. This is
  the anchor: any future refactor must reproduce these exact digests.

## Required models

The gate hardcodes two model paths (in the `MODELS` dict of `gate_batch.py`):

- `/home/a/Bureau/Work/CrimsonRed/data/models/llama-1b-q4_k_m.gguf` - llama 1B
  Q4_K_M (durable copy, sha256 `ddb21816ad55ccb1...`; falls back to the
  historical `/tmp/llama-1b-q4_k_m.gguf` if the durable copy is absent)
- `/home/a/Bureau/Work/CrimsonRed/data/models/gemma-4-E2B.Q4_K_M.gguf` - gemma E2B Q4_K_M

Both files must exist at those paths, or the constants must be adjusted to
point at byte-identical copies.

## The /tmp path assumption

`gate_batch.py` hardcodes `GOLD = /tmp/hs-batch-golden`. Inputs are read from
`GOLD/inputs`, artifacts are written under `GOLD/baseline` or `GOLD/check`,
and digests are read/written as `GOLD/digests_*.json`. To run the gate you
must either recreate that directory layout under /tmp (copy this directory's
contents to `/tmp/hs-batch-golden` and generate inputs into
`/tmp/hs-batch-golden/inputs`), or adjust the `GOLD` constant at the top of
`gate_batch.py`. Note that `check` mode compares against
`GOLD/digests_baseline.json`, so that file must be the committed baseline
digests, not a freshly regenerated one.

## Running the gate

1. Recreate the /tmp layout and generate inputs:

   ```bash
   mkdir -p /tmp/hs-batch-golden/inputs
   cp gate_batch.py gen_inputs.py digests_baseline.json /tmp/hs-batch-golden/
   python3 gen_inputs.py /tmp/hs-batch-golden/inputs
   ```

2. Build a CPU-only binary of llama-hs-extract-batch from the tree under
   test (CUDA off):

   ```bash
   cmake -B build-chk -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
   cmake --build build-chk --target llama-hs-extract-batch -j$(nproc)
   ```

3. Run the gate against that binary:

   ```bash
   python3 gate_batch.py check <path-to-build-chk>/bin/llama-hs-extract-batch
   ```

   All scenarios print per-artifact `ok`/`FAIL` lines; the run ends with
   `ALL_BYTE_IDENTICAL` (pass) or a mismatch count (fail).

To regenerate the anchor from scratch instead (only meaningful from a
pre-refactor checkout of the extraction code), run
`python3 gate_batch.py baseline <cpu-bin>` in the same /tmp layout, then copy
the resulting `digests_baseline.json` back into this directory and commit it.

## Regenerable vs irreplaceable

The baseline binaries and baseline output artifacts under /tmp are
regenerable: they can be rebuilt from git at the recorded base by following
the procedure above. The digests in `digests_baseline.json` are the
irreplaceable anchor - they pin the exact bytes the pre-refactor code
produced on CPU. Without them there is nothing for a refactor to prove
itself against, which is why they are committed here rather than left in
volatile /tmp storage.
