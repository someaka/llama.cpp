#!/usr/bin/env python3
"""Generate models/hs-refusal-fixture.gguf: a minimal loadable gemma2 model.

gemma2 is supported by upstream but NOT in the fork's hidden-state capability
registry (llm_arch_supports_hidden_states), so it is the refusal-contract
fixture: creation with extract_hidden_states=true must fail, and the runtime
toggle must return -1 with the context otherwise functional.

Run from the fork root with a python that has numpy:
  PYTHONPATH=gguf-py python3 tests/gen-refusal-fixture.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf  # noqa: E402

N_EMBD = 8
N_FF = 16
N_HEAD = 1
N_HEAD_KV = 1
N_LAYER = 1
N_WORDS = 32
N_VOCAB = N_WORDS + 256  # word tokens + byte fallback (any text tokenizes)
N_CTX = 64

out = Path(__file__).resolve().parents[1] / "models" / "hs-refusal-fixture.gguf"
w = gguf.GGUFWriter(str(out), "gemma2")

w.add_name("hs-refusal-fixture")
w.add_context_length(N_CTX)
w.add_embedding_length(N_EMBD)
w.add_block_count(N_LAYER)
w.add_feed_forward_length(N_FF)
w.add_head_count(N_HEAD)
w.add_head_count_kv(N_HEAD_KV)
w.add_rope_dimension_count(N_EMBD)
w.add_vocab_size(N_VOCAB)
w.add_layer_norm_rms_eps(1e-6)
w.add_bos_token_id(0)
w.add_eos_token_id(1)

w.add_tokenizer_model("llama")
tokens = ["<bos>", "<eos>"] + [f"tok{i}" for i in range(N_WORDS - 2)]
tokens += [f"<0x{i:02X}>" for i in range(256)]
scores = [0.0, 0.0] + [-float(i + 1) for i in range(N_WORDS - 2)]
scores += [-float(1000 + i) for i in range(256)]
types = [gguf.TokenType.CONTROL] * 2 + [gguf.TokenType.NORMAL] * (N_WORDS - 2)
types += [gguf.TokenType.BYTE] * 256
w.add_token_list(tokens)
w.add_token_scores(scores)
w.add_token_types(types)

rng = np.random.default_rng(42)


def t(name, shape):
    w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.02)


t("token_embd.weight", (N_VOCAB, N_EMBD))
t("output_norm.weight", (N_EMBD,))
# gemma2's output is TENSOR_DUPLICATED of token_embd -- no separate tensor.
for il in range(N_LAYER):
    p = f"blk.{il}."
    t(p + "attn_norm.weight", (N_EMBD,))
    t(p + "attn_q.weight", (N_EMBD, N_EMBD))
    t(p + "attn_k.weight", (N_EMBD, N_EMBD))
    t(p + "attn_v.weight", (N_EMBD, N_EMBD))
    t(p + "attn_output.weight", (N_EMBD, N_EMBD))
    t(p + "post_attention_norm.weight", (N_EMBD,))
    t(p + "ffn_norm.weight", (N_EMBD,))
    t(p + "ffn_gate.weight", (N_FF, N_EMBD))
    t(p + "ffn_up.weight", (N_FF, N_EMBD))
    t(p + "ffn_down.weight", (N_EMBD, N_FF))
    t(p + "post_ffw_norm.weight", (N_EMBD,))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out} ({out.stat().st_size} bytes)")