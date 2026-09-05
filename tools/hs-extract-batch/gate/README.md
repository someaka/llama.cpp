# Golden byte-identity gate for `llama-hs-extract-batch`

A byte-identity regression gate for the `llama-hs-extract-batch` batch
extraction tool. After any refactor of the tool, run the gate to prove the
refactor changed structure, not bytes: every artifact the tool produces must
be byte-identical to the recorded baseline.

## Methodology: the production path

Five scenarios run on the PRODUCTION CUDA binary (`build-cuda`,
`libggml-cuda.so*` present), `-ngl 99`, one model:

- gemma E2B Q4_K_M

Each scenario exercises a distinct output path: plain comprehension
extraction with `--save-per-record`, kill-and-`--resume` checkpoint parity,
greedy generation, and sampled generation (seed=0 dist sampler). In detail:

| scenario            | model | layers    | what                                     |
|---------------------|-------|-----------|------------------------------------------|
| s5_e2b_comp_records | e2b   | 0,17,35   | comprehension + `--save-per-record`      |
| s6_e2b_resume       | e2b   | 0,17,35   | kill at 2nd checkpoint, `--resume`       |
| s7_e2b_gen_greedy     | e2b   | 0,35      | `--generate 8 --token-skip 2`, greedy    |
| s8_e2b_gen_sampled  | e2b   | 0,35      | `--generate 8 --token-skip 2 --temperature 0.8 --top-k 40 --repeat-penalty 1.1` |
| s9_e2b_gen_resume   | e2b   | 0,35      | same flags as s8; kill at 2nd checkpoint, then `--resume` |

Methodology: a CUDA-ON binary (`libggml-cuda.so*` next to the binary is
REQUIRED; a CPU-only build is refused), `-ngl 99`, original tokenization
flags (no `--no-bos`). Every produced artifact (`output.bin`, the
`.records.bin` sidecar, resume full-run vs resumed output) is sha256-compared
against the baseline. Exit 0 means all bytes identical; exit 1 means
mismatch OR setup/run failure (messages distinguish them: `REFUSED` /
`inputs missing` are setup problems, `MISMATCHES` is a byte regression).

Determinism at `-ngl 99` on this GPU was verified empirically (2026-08-18:
every scenario type byte-identical across repeat runs, including resume
parity and sampled generation).

The driver refuses CPU-only binaries up front (no `libggml-cuda.so*` next to
the binary aborts with `REFUSED`): the gate runs the production path, and a
CUDA-off build produces different gemma numerics than the production build
(verified 2026-08-18), so it can never serve as a valid gate binary. A
missing model file likewise aborts (`REFUSED`) -- the gate never falls back.

## Files

- `gate_batch.py` - the gate driver. Two modes: `baseline <cuda-bin>` (run all
  scenarios with the PRE-refactor binary, save every artifact digest under
  `/tmp/hs-batch-golden/baseline/` and write `digests_baseline.json`), and
  `check <cuda-bin>` (run all scenarios with the binary under test and compare
  every artifact digest against the baseline).
- `gen_inputs.py` - deterministic generator for the gate inputs (24 prompts,
  two assignment files). The generated fixtures are COMMITTED at
  `golden_inputs/`; running the script regenerates them to verify
  determinism (output must be byte-identical to the committed bytes).
- `golden_inputs/` - the committed gate input fixtures (`prompts.txt`,
  `assignments.bin`, `assignments_gen.bin`). These are the bytes every gate
  run consumes; regenerate with `gen_inputs.py` to verify they are still
  reproducible.
- `digests_baseline.json` - the anchor. Keys are scenario names; values map
  artifact names to sha256 digests.

## The /tmp path assumption

Input fixtures are COMMITTED at `golden_inputs/` and are what the driver
reads (`INPUTS`) -- they are not /tmp-dependent. The driver still uses
`/tmp/hs-batch-golden` as its working tree for VOLATILE scenario outputs,
baseline artifacts, and digests (`GOLD`). `/tmp` is volatile; if the layout
is gone, the check mode never needs the baseline artifacts (it compares
digests, not files), but it does need `/tmp/hs-batch-golden/
digests_baseline.json` (see "Re-capturing the baseline"). To relocate the
volatile tree, adjust the `GOLD` constant.

## Required model

- Set `export HS_GATE_MODEL=/path/to/model.gguf` (REQUIRED — no default). Gate
  model: gemma E2B Q4_K_M (sha256
  `389c868898bffed97fd178646f88562cafecc6f60983a636bac53b131fd068a2`).

## Running the gate after a refactor

1. Rebuild the tool from the tree under test (the production build):

   ```bash
   cmake --build build-cuda --target llama-hs-extract-batch -j$(nproc)
   ./build-cuda/bin/llama-hs-extract-batch --self-test  # must pass
   ```

2. Run the check:

   ```bash
   python3 tools/hs-extract-batch/gate/gate_batch.py check \
       build-cuda/bin/llama-hs-extract-batch
   ```

   Exit 0 + `ALL_BYTE_IDENTICAL` = the refactor is byte-neutral.

## Re-capturing the baseline

Only meaningful from a commit that is TRUSTED (e.g. the tree before a
refactor lands). The current baseline was captured 2026-08-18 on the
production CUDA path (RTX 3090, from-scratch `build-cuda` at fork commit
`e6a499a7e`, `-ngl 99`, inputs from `gen_inputs.py`). The CPU-era digests
(before 2026-08-18) were captured on a CUDA-off binary at `-ngl 0` and are
NOT comparable -- never mix anchors across backends; re-capture instead:

```bash
python3 tools/hs-extract-batch/gate/gate_batch.py baseline \
    build-cuda/bin/llama-hs-extract-batch
cp /tmp/hs-batch-golden/digests_baseline.json \
   tools/hs-extract-batch/gate/digests_baseline.json
```

## Why byte-identity on GPU is sound here

The gate is a REFACTOR gate, not a cross-backend comparison: baseline and
check run on the same machine, same GPU, same driver, same binary
configuration, same inputs. It proves a source change did not alter the
produced bytes -- nothing more. Cross-backend (CUDA vs Vulkan vs CPU)
comparisons are out of scope by design (numerics legitimately differ).
Without the gate there is nothing for a refactor to prove byte-neutrality
with, and numeric drift ships silently.
