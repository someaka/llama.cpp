# hs-extract-batch

High-performance batch hidden state extraction tool for llama.cpp models.

Processes thousands of prompts efficiently using streaming architecture and optional GPU acceleration.

## Features

- **Streaming processing** - Handles datasets larger than RAM
- **GPU acceleration** - Up to 100x faster with CUDA/Vulkan
- **Checkpoint/resume** - Save progress and resume interrupted runs
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
```

**Raw mode (debug):**
```
<model> <prompts.txt> [layers] <output.bin> --raw [flags]
```

**Common flags:**
- `--mean` - Output token means instead of full per-token data (raw mode only)
- `--token-skip N` - Skip first N tokens for mean computation (default: 0)
- `--assignments FILE` - Path to assignments.bin (required for batch mode)
- `--ctx-size N` - Override auto context sizing (default: auto from prompts)
- `--checkpoint-every N` - Save checkpoint every N prompts (default: 10000)
- `--resume` - Resume from last checkpoint
- `-ngl, --n-gpu-layers N` - Number of layers to offload to GPU (default: 99 = all)
- `--save-per-record` - Also write per-record vectors to `<output>.records.bin` (per-record sidecar format: prompt_idx + group_id + mask_id + layer_idx + float32[n_embd] per record). Batch mode only; rejected with `--raw` and `--generate`.
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
