# hs-extract

Single-prompt hidden state extraction tool for llama.cpp models.

Extracts residual stream activations from a single prompt and outputs them as JSON.

Layer indices follow the hidden_states convention: 0 = embeddings, i = state
entering block i (= HF hidden_states[i]), N = final block output (range
[0, n_layer] inclusive); negative indices resolve Python-style from the end
(-1 = last slot).
Supported architectures: llama, gemma, gemma4, qwen35. Extraction requires
n_embd == n_embd_out (models with a separate output projection are rejected
with an error).

## Usage

```bash
llama-hs-extract -m model.gguf -p "Hello, world!"
```

## Options

- `-m, --model MODEL` - Path to GGUF model file (required)
- `-p, --prompt TEXT` - Input prompt text (`argv` cannot carry a NUL byte)
- `-f, --file FILE` - Read prompt from file as raw bytes
- `-l, --layers LIST` - Comma-separated layer indices or `all` (default: all) (negative indices resolve Python-style; CLI-only sugar - the server API takes non-negative integers)
- `--raw` - Interpret prompt as comma-separated token IDs (a NUL byte in that
  input is refused: raw mode takes ids, not text)
- `--no-bos` - Don't add BOS token. Default: BOS is added only when the
  model's tokenizer declares `add_bos_token` (e.g. gemma yes, qwen no).
- `-t, --threads N` - Number of CPU threads (default: 4)
- `-c, --ctx-size N` - Context size (default: auto = token count, capped at n_ctx_train)
- `-ngl, --n-gpu-layers N` - Number of layers to offload to GPU (default: 0)
- `--output FILE` - Output JSON file (default: stdout)

### Prompt bytes

`-f` reads the file as bytes (binary, no decoding) and strips exactly one
trailing `\n` and then one trailing `\r`. A **single-line** file therefore
tokenizes like the batch tool's `getline`-per-line reading of that line; a
multi-line file is one prompt here and N prompts there (the batch tool defines
the line semantics), and an interior `\n` is kept as a byte. An empty file, or
one holding only a newline, is refused (`prompt is empty`), matching the batch
tool's refusal of a prompts file with no non-empty prompts. Interior NUL bytes are **preserved
and tokenized**: a byte-level vocabulary renders them as `<0x00>`, so a file
`abc\0def` tokenizes to 8 tokens (7 byte tokens + BOS) and matches
`hs-extract-batch` at float32 bit equality on every captured layer. `-p` cannot carry a NUL at all (`argv` is
NUL-terminated), and `--raw` takes comma-separated token IDs rather than text,
so a NUL in that input is refused with an error instead of being truncated.
The server's `/hidden-states` `input` field is a JSON string and does carry
NULs: it tokenizes the full string and counts the NUL as its own token (see
`tools/server/README.md`).

> Hidden-state values are reproducible per backend/`-ngl` configuration; CPU and
> CUDA results for the same prompt are not value-comparable at deep layers. Pin
> the backend (and `-ngl`) for research data.
>
> JSON serialization uses 9 significant digits, which round-trips float32
> losslessly (JSON text → float32 reproduces the exact binary bits), so any
> output can be compared bit-for-bit against binary captures on the same
> pinned backend. Front-END outputs (hs-extract vs hs-extract-batch vs the
> server) are bit-identical to each other only when the four parity conditions
> are pinned: the same backend build, the same effective Flash-Attention
> resolution, the same effective compute placement, and a KV layout that does
> not share one pool across decoding requests. Only the server can move the FA
> axis (`-fa on|off`); this CLI has no `-fa` flag and always uses the library
> default (auto), which is also the server default. See
> `tools/server/README.md` ("Parity protocol") for the measurement definitions
> and the per-axis deviations. hs-extract-batch is the parity-of-record for
> probe data; the server is parity-of-record only under those pins.


## Examples

Extract all layers:
```bash
llama-hs-extract -m model.gguf -p "The quick brown fox"
```

Extract specific layers:
```bash
llama-hs-extract -m model.gguf -p "Hello" -l 0,5,10
```

Read prompt from file:
```bash
llama-hs-extract -m model.gguf -f prompt.txt --output result.json
```

GPU acceleration:
```bash
llama-hs-extract -m model.gguf -p "Test" -ngl 99 --output output.json
```

## Output Format

JSON output structure:
```json
{
  "n_tokens": 4,
  "n_embd": 2048,
  "n_layers": 35,
  "layers": [
    {"layer": 0, "values": [0.123, -0.456, ...]},
    {"layer": 5, "values": [0.234, -0.567, ...]}
  ]
}
```
(`n_layers` is the model's block count, `llama_model_n_layer` — the number
of transformer blocks, 35 on the gate model — not the hidden-state slot
count (36) and not the number of layers requested.)

Each layer entry contains a flat array of `n_tokens * n_embd` float values
(row-major: token 0 dims 0..n_embd-1, then token 1, etc.).

## Notes

- The hidden-state getters synchronize the context before returning data; no explicit `llama_synchronize()` is needed
- Supports both CPU and GPU inference via `-ngl` flag
- For batch processing, use [hs-extract-batch](../hs-extract-batch/) instead
