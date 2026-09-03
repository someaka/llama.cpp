# hs-extract-batch

High-performance batch hidden state extraction tool for llama.cpp models.

Processes thousands of prompts efficiently using streaming architecture and optional GPU acceleration.

## Features

- **Streaming processing** - Handles datasets larger than RAM
- **GPU acceleration** - Up to 100x faster with CUDA/Vulkan
- **Checkpoint/resume** - Save progress and resume interrupted runs

> Hidden-state values are reproducible per backend/`-ngl` configuration; CPU and
> CUDA results for the same prompt are not value-comparable at deep layers. Pin
> the backend (and `-ngl`) for research data.

- **Self-test mode** - Validate setup without requiring a model
- **Binary I/O** - Compact output format for downstream processing

## Usage

### Batch mode (production)

```bash
llama-hs-extract-batch model.gguf prompts.txt all output.bin \
  --batch --assignments assignments.bin -ngl 99
```

### Raw mode (debug/parity testing)

```bash
llama-hs-extract-batch model.gguf prompts.txt 0,5,10 output.bin --raw
```

### Self-test (no model required)

```bash
llama-hs-extract-batch --self-test
```

## Options

**Batch mode (production):**
```
<model> <prompts.txt> [layers] <output.bin> --batch --assignments <file> [flags]
  ([layers]: `all`, decimal indices, or negative indices resolved Python-style; CLI-only sugar - the server API takes non-negative integers)
```

**Raw mode (debug):**
```
<model> <prompts.txt> [layers] <output.bin> --raw [flags]
```

**Common flags:**
- `--mean` - Output token means instead of full per-token data (raw mode only)
- `--token-skip N` - Skip first N tokens for mean computation (default: 0). Applies only to `--raw --mean` and `--batch --generate`; rejected elsewhere.
- `--assignments FILE` - Path to assignments.bin (required for batch mode)
- `--ctx-size N` - Override auto context sizing (default: auto from prompts)
- `--checkpoint-every N` - Save checkpoint every N prompts (N >= 1, default: 10000)
- `--resume` - Resume from last checkpoint
- `-ngl, --n-gpu-layers N` - Number of layers to offload to GPU (default: 99 = all)
- `--save-per-record` - Also write per-record vectors to `<output>.records.bin` (per-record sidecar format: prompt_idx + group_id + mask_id + layer_idx + float32[n_embd] per record). Batch mode only; rejected with `--raw`, `--generate`, and `--resume`.
- `--batch-size N` - Only 1 is supported (multi-prompt batching is not implemented: shared KV-cache semantics would corrupt extraction). The flag is parsed so old driver scripts keep working; values > 1 are rejected with an error at runtime.
- `--profile` - Print per-phase timing breakdown (KV clear, decode, sync, extract, mean, accumulate)
- `--no-bos` - Force BOS off. Default: BOS follows the tokenizer's
  `add_bos_token` (added for gemma-class vocabs, not for qwen-class), with
  `--no-bos` as a force-off override — the same semantics as `hs-extract`
  (unified 2026-09-01; before that this tool added BOS unconditionally).
- `--generate N` - Generation-based extraction: generate N tokens after each prompt and extract hidden states from the generated tokens only (comprehension-based extraction otherwise). Incompatible with `--save-per-record`
- `--temperature F` - Sampling temperature for generation mode (0 = greedy argmax, default)
- `--top-k K` - Top-k sampling, generation mode only (default: 0 = disabled)
- `--top-p F` - Nucleus sampling threshold, generation mode only (default: 1.0 = disabled)
- `--repeat-penalty F` - Repeat penalty, generation mode only (default: 1.0 = disabled)

**Layer specification:**
- `all` - Extract from all layers (default)
- `0,5,10` - Comma-separated layer indices
- Layer indices follow the hidden_states convention: 0 = embeddings, i = state
  entering block i (= HF hidden_states[i]), N = final block output (range
  [0, n_layer] inclusive); negative indices resolve Python-style from the end
  (-1 = last slot)

## Input Format

**prompts.txt** - One prompt per line:
```
The quick brown fox jumps over the lazy dog.
Hello, world!
This is a test prompt.
```

**assignments.bin** - Binary format (little-endian, all integers are int32, all floats float32) defining which tokens to extract from each prompt and which label group each extraction belongs to. There is no separate groups file: the group name table is part of the assignments file header.

```
Header:
  magic:           int32 (0x43524431)
  n_prompts:       int32 (must match the number of lines in prompts.txt)
  n_embd_expected: int32 (must match the model's hidden dimension; 0 skips the check)
  n_groups:        int32
  Per group:
    name_len:      int32
    name:          name_len bytes (UTF-8 label for this group_id; parsed and validated, not currently used in output)

Per prompt (n_prompts times, in the same order as prompts.txt lines):
  n_assignments:   int32 (0..100000)
  Per assignment:
    group_id:      int32 (index into the group name table above)
    mask_id:       int32 (distinguishes multiple masks on the same group)
    mask_type:     int32
    If mask_type == 0 (skip-prefix):
      skip:        int32 (use tokens [skip, n_tokens))
    If mask_type == 1 (explicit ranges):
      n_ranges:    int32 (0..100000)
      Per range:
        start:     int32
        end:       int32 (half-open [start, end) token range; start < end enforced)
```

All counts and bounds (`n_prompts`, `n_embd_expected`, `n_groups`, `name_len`, `n_assignments`, `n_ranges`) are validated at read time; a file that violates them is rejected as corrupt rather than trusted.

## Output Format

### Batch mode (production) - binary accumulator format

```
Header:
  magic:     int32 (0x43524432 = "binary format v2")
  n_groups:  int32
  n_layers:  int32
  n_embd:    int32

Per group (sorted by group_id):
  group_id:    int32
  n_masks:     int32
  Per mask (sorted by mask_id):
    mask_id:        int32
    n_layers_data:  int32
    Per layer (sorted by layer_idx):
      layer_idx: int32
      count:     int32
      mean:      float32[n_embd]  (sum / count)
```

### Raw mode (debug) - per-prompt dump

```
Header:
  n_prompts_total: int32

Per prompt:
  prompt_idx: int32
  n_tokens:   int32
  n_layers:   int32
  layer_indices: int32[n_layers]
  Per layer:
    data: float32[n_tokens][hidden_dim]       (full mode)
       OR float32[hidden_dim]                  (with --mean: mean over [token_skip, n_tokens))
```

### Per-record sidecar (with `--save-per-record`)

Written alongside the batch accumulator output as `<output>.records.bin`:

```
Header:
  magic:          int32 (0x53545231)
  n_embd:         int32
  n_layers:       int32

Per record (one per prompt x assignment x layer, in processing order):
  prompt_idx:     int32
  group_id:       int32
  mask_id:        int32
  layer_idx:      int32
  mean:           float32[n_embd]
```

## Checkpoint and Resume

For long-running jobs, use checkpointing:

```bash
llama-hs-extract-batch model.gguf prompts.txt all output.bin \
  --batch --assignments assignments.bin \
  -ngl 99 --checkpoint-every 5000 --resume
```

If interrupted, re-run with `--resume` to continue from the last checkpoint.

### Checkpoint file format (v6)

A checkpoint is `<output>.checkpoint`; the format is versioned. Physical
field order (all little-endian):

| # | field | type | notes |
|---|---|---|---|
| 1 | `version` | i32 | 6 for current checkpoints; 1-5 load with degraded guarantees + warning |
| 2 | `n_iterated` | i32 | number of prompts fully processed (resume skip count) |
| 3 | `n_fp_layers` | i32 | run fingerprint: layer count |
| 4 | `generate_mode` | u8(bool) | fingerprint: `--generate` was active |
| 5 | `generate_tokens` | i32 | fingerprint: `--generate N` |
| 6 | `token_skip` | i32 | fingerprint: `--token-skip` |
| 7 | `layers` | i32[n_fp_layers] | fingerprint: sorted target layers |
| 8 | `content_fnv64` | u64 | rolling FNV-1a-64 over every consumed non-empty prompts line, each line followed by a `\n` byte (since v5); v4 stored a single-line hash on a retired basis (not revalidated) |
| 9 | `n_prompts` | i32 | expected total prompts; must match the current run |
| 10 | accumulator region | binary | `OUTPUT_MAGIC` i32 (`0x43524432`, "CRD2"), then `n_groups`/`n_layers`/`n_embd` i32, then per-group records: `group_id` i32, `n_masks` i32, then per mask: `mask_id` i32, `n_layers_data` i32, then per layer: `layer_idx` i32, `count` i32, `f32[n_embd]` raw sums |
| 11 | `acc_fnv64` | u64 | v6 trailer: FNV-1a-64 over bytes of field 10 only, so any in-range payload bit flip is refused |

FNV-1a-64 definition: offset basis `14695981039346656037`
(`0xcbf29ce484222325`), prime `1099511628211`; for each line, hash the line
bytes then the `'\n'` byte; the state carries across lines in file order.
A stored `content_fnv64` of 0 with `n_iterated > 0` is refused as corrupt.
On resume the same rolling hash is re-derived from the current prompts file
and a mismatch refuses the resume (both hashes printed). The v6 accumulator
checksum is recomputed on resume and a mismatch refuses with both digests
(any in-range payload bit flip — group ids, counts, float bytes — is caught).
Legacy versions: v5 resumes without the accumulator checksum (warning);
v4 with the count check only (warning); v1/v2/v3 predate content checking
(v3 still enforces the run fingerprint).

## Performance

Tested on RTX 3090 (CUDA):

| Dataset Size | GPU Layers | Time | Throughput |
|--------------|------------|------|------------|
| 10K prompts  | 0 (CPU)    | 52min | 3.2/s |
| 10K prompts  | 99 (GPU)   | 28s   | 357/s |
| 200K prompts | 99 (GPU)   | 9min  | 370/s |

## Notes

- The hidden-state getters synchronize the context before returning data; no explicit `llama_synchronize()` is needed
- Checkpoints include full state for exact resumption
- Memory usage scales with batch size, not dataset size
- For single-prompt extraction, use [hs-extract](../hs-extract/) instead
