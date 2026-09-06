# Hidden-States Adoption Manifest - all architecture graph builders

Generated 2026-08-16 from the exhaustive classification pass over src/models/
(151 arch files at the 2026-09-04 HEAD; the 2026-08-17 syncs brought bailingmoe3
(earlier same-day sync, 373336672, before the ec5368b2a merge);
127 builders with graph loops classified; every row carries
file:line evidence from source reads, not sampling).


# Layer Index Convention (canonical)

> This section is the single source of truth for layer numbering across the
> fork's hidden-state surface and the CLIs' consumers. The same wording
> appears in every doc that discusses layer indices; do not paraphrase it.

The public layer index follows the **hidden_states convention**, matching
HuggingFace `hidden_states` and upstream llama.cpp's internal layer-input
tensors (`t_layer_inp`):

| index | meaning |
|-------|---------|
| `0` | token embeddings — the state entering block 0 (after any arch-specific embedding transform, e.g. the Gemma `sqrt(n_embd)` scale) |
| `i` | the state entering block `i` — the same tensor upstream stores as `t_layer_inp[i]` and HF returns as `hidden_states[i]` |
| `N` | the output of the final block (`n_layer` = block count) — a fork extension; upstream stops at `N-1` and exposes nothing publicly |

Not captured: post-final-norm output.

Migration from the old fork numbering (post-block-i residual):

| recorded in old fork space | upstream / hidden_states space |
|---|---|
| any layer `L` | `L + 1` |
| E2B `L24` (of 35) | `25` |
| E4B `L29` (of 42) | `30` |
| Qwen `L16` (of 24) | `17` |
| Llama `L11` (of 16) | `12` |

Existing on-disk artifacts recorded before the migration stay in the old
space; readers must remap with `fork_to_upstream(i) = i + 1` and refuse
ambiguous files rather than guess.

---

## Mechanism (what a new arch joins)

The unified mechanism has three parts (branch hs-arch-core):

1. **Capture helper** - `llm_graph_context::capture_layer_output(il, cur)` in
   src/llama-graph.{h,cpp}: centralizes the flag check, the push into
   `res->t_hidden_layers`, output marking, and tensor naming. Every supported
   architecture calls it at the bottom of its layer loop, right after the final
   residual reassignment (`inpL = cur;` or equivalent):

   ```cpp
        inpL = cur;

        capture_layer_output(il, cur);
   ```

2. **Capability registry** - `llm_arch_supports_hidden_states(const llm_arch &)`
   in src/llama-arch.{h,cpp} beside `llm_arch_supports_rs_rollback`: the single
   source of truth for supported arch values. Context creation and the runtime
   setter throw naming the architecture when the flag is set for an unsupported
   arch. The llama.h doc block carries a comment pointing at the registry as the
   maintenance point.

3. **Server request validation** - unsupported arch gets a 400-class error naming
   the arch, not the misleading 500-class "decode may have failed".

## Registry

| arch value | builder file | status |
|---|---|---|
| LLM_ARCH_LLAMA | src/models/llama.cpp | ADOPTED (reference) |
| LLM_ARCH_GEMMA | src/models/gemma.cpp | ADOPTED (reference) |
| LLM_ARCH_GEMMA4 | src/models/gemma4.cpp | ADOPTED (reference) |
| LLM_ARCH_QWEN35 | src/models/qwen35.cpp | ADOPTED (reference) |

Adoption adds a supported arch in exactly two places: the call in the builder and
a registry case. Refusals need no code change.

## ADOPT list - CLASSIC builders

Adoption point: bottom of the layer loop, immediately after the final residual
reassignment that follows the final residual add listed below.

| # | builder | file | final residual add (line) |
|---|---|---|---|
| 1 | llama_model_afmoe | afmoe.cpp | 262 |
| 2 | llama_model_apertus | apertus.cpp | 146 |
| 3 | llama_model_arcee | arcee.cpp | 131 |
| 4 | llama_model_arctic | arctic.cpp | 154 |
| 5 | llama_model_baichuan | baichuan.cpp | 130 |
| 6 | llama_model_bailingmoe | bailingmoe.cpp | 155 |
| 7 | llama_model_bailingmoe2 | bailingmoe2.cpp | 188 |
| 8 | bailingmoe3 | bailingmoe3.cpp | 389 |
| 9 | bert | bert.cpp | 218 |
| 10 | bitnet | bitnet.cpp | 143 |
| 11 | bloom | bloom.cpp | 128 |
| 12 | chameleon | chameleon.cpp | 164 |
| 13 | chatglm | chatglm.cpp | 138 |
| 14 | codeshell | codeshell.cpp | 130 |
| 15 | cogvlm | cogvlm.cpp | 138 |
| 16 | dbrx | dbrx.cpp | 128 |
| 17 | deci | deci.cpp | 168 |
| 18 | deepseek | deepseek.cpp | 172 |
| 19 | deepseek2 | deepseek2.cpp | 693 |
| 20 | deepseek32 | deepseek32.cpp | 456 |
| 21 | dots1 | dots1.cpp | 171 |
| 22 | dream | dream.cpp | 116 |
| 23 | ernie4_5 | ernie4-5.cpp | 142 |
| 24 | ernie4_5_moe | ernie4-5-moe.cpp | 110 |
| 25 | eurobert | eurobert.cpp | 109 |
| 26 | exaone | exaone.cpp | 113 |
| 27 | exaone4 | exaone4.cpp | 166 |
| 28 | exaone_moe | exaone-moe.cpp | 218 |
| 29 | gemma | gemma.cpp | 116 |
| 30 | gemma2 | gemma2.cpp | 148 |
| 31 | gemma3 | gemma3.cpp | 192 |
| 32 | gemma4 | gemma4.cpp | 362 |
| 33 | gemma4_assistant | gemma4-assistant.cpp | 178 |
| 34 | gemma_embedding | gemma-embedding.cpp | 161 |
| 35 | glm4 | glm4.cpp | 165 |
| 36 | glm4_moe | glm4-moe.cpp | 416 |
| 37 | glm_dsa | glm-dsa.cpp | 500 |
| 38 | gpt2 | gpt2.cpp | 125 |
| 39 | granite | granite.cpp | 149 |
| 40 | grok | grok.cpp | 190 |
| 41 | grovemoe | grovemoe.cpp | 170 |
| 42 | hunyuan_moe | hunyuan-moe.cpp | 164 |
| 43 | hunyuan_vl | hunyuan-vl.cpp | 167 |
| 44 | hy_v3 | hy-v3.cpp | 204 |
| 45 | internlm2 | internlm2.cpp | 115 |
| 46 | jais | jais.cpp | 110 |
| 47 | jais2 | jais2.cpp | 140 |
| 48 | jamba | jamba.cpp | 177 |
| 49 | laguna | laguna.cpp | 316 |
| 50 | llada | llada.cpp | 136 |
| 51 | llada_moe | llada-moe.cpp | 139 |
| 52 | llama | llama.cpp | 221 |
| 53 | llama4 | llama4.cpp | 245 |
| 54 | maincoder | maincoder.cpp | 127 |
| 55 | mellum | mellum.cpp | 197 |
| 56 | mimo2 | mimo2.cpp | 222 |
| 57 | minicpm3 | minicpm3.cpp | 223 |
| 58 | minimax_01 | minimax-01.cpp | 457 |
| 59 | minimax_m2 | minimax-m2.cpp | 144 |
| 60 | minimax_m3 | minimax-m3.cpp | 582 |
| 61 | mistral3 | mistral3.cpp | 210 |
| 62 | modern_bert | modern-bert.cpp | 159 |
| 63 | mpt | mpt.cpp | 148 |
| 64 | muse_glimmer | muse-glimmer.cpp | 175 |
| 65 | nanbeige | nanbeige.cpp | 156 |
| 66 | nemotron | nemotron.cpp | 125 |
| 67 | neo_bert | neo-bert.cpp | 119 |
| 68 | olmo | olmo.cpp | 117 |
| 69 | olmo2 | olmo2.cpp | 182 |
| 70 | olmoe | olmoe.cpp | 150 |
| 71 | openai_moe | openai-moe.cpp | 147 |
| 72 | openelm | openelm.cpp | 148 |
| 73 | orion | orion.cpp | 117 |
| 74 | paddleocr | paddleocr.cpp | 85 |
| 75 | pangu_embed | pangu-embed.cpp | 132 |
| 76 | phi2 | phi2.cpp | 117 |
| 77 | phi3 | phi3.cpp | 166 |
| 78 | plamo | plamo.cpp | 112 |
| 79 | plamo3 | plamo3.cpp | 174 |
| 80 | plm | plm.cpp | 183 |
| 81 | pockettts | pockettts.cpp | 123 |
| 82 | qwen | qwen.cpp | 116 |
| 83 | qwen2 | qwen2.cpp | 127 |
| 84 | qwen2moe | qwen2moe.cpp | 170 |
| 85 | qwen2vl | qwen2vl.cpp | 119 |
| 86 | qwen3 | qwen3.cpp | 135 |
| 87 | qwen35 | qwen35.cpp | 196 |
| 88 | qwen3moe | qwen3moe.cpp | 155 |
| 89 | qwen3vl | qwen3vl.cpp | 154 |
| 90 | qwen3vlmoe | qwen3vlmoe.cpp | 166 |
| 91 | refact | refact.cpp | 136 |
| 92 | rnd1 | rnd1.cpp | 153 |
| 93 | seed_oss | seed-oss.cpp | 126 |
| 94 | smallthinker | smallthinker.cpp | 166 |
| 95 | smollm3 | smollm3.cpp | 127 |
| 96 | stablelm | stablelm.cpp | 147 |
| 97 | starcoder | starcoder.cpp | 123 |
| 98 | starcoder2 | starcoder2.cpp | 134 |
| 99 | step35 | step35.cpp | 336 |
| 100 | talkie | talkie.cpp | 126 |
| 101 | xverse | xverse.cpp | 114 |

Of the 127 builders reviewed, 101 are ADOPT and 26 are REFUSE. Of the 101
ADOPT rows, 4 are the active registry references (llama, gemma, gemma4,
qwen35); 97 carry the dormant `capture_layer_output()` call (the 2026-08-16
hs-arch-core sweep, plus granite and jais2 restored to the adopt point after
the manifest audit, plus nanbeige and bailingmoe3 added 2026-08-17 when the
classic tail was verified to hold there too, bringing files-with-a-tap to
101 = 4 registry + 97 dormant). No REFUSE-listed builder
carries a call; refusals are recorded above with named
reasons and stay out until a per-builder semantic decision exists. Dormant
calls are no-ops until the arch is added to the registry; per-adoption steps:

1. read the builder's loop tail against this table (adopt point = loop
   bottom, after the final residual reassign);
2. check the last-layer get_rows pruning site — under extraction it must
   be suppressed (or moved post-final-norm, as llama/gemma now do) or the
   graph aborts (orphaned `inp_out_ids`);
3. add the arch value to `llm_arch_supports_hidden_states()`;
4. run the real-GGUF extraction test + logits-probe equivalence.

## REFUSE list - with named reasons

| builder | file | shape | reason (evidence) |
|---|---|---|---|
| llama_model_arwkv7 | src/models/arwkv7.cpp | HYBRID-OTHER | arwkv7.cpp:182 — RWKV7 attention + llama-style FFN; residual at 182 |
| llama_model_cohere2 | src/models/cohere2.cpp | PARALLEL-RESIDUAL | cohere2.cpp:124-134 — FFN consumes ffn_inp=norm(inpL) (80,120); block out = inpL + ffn_out + attn_out (133-134) |
| llama_model_cohere2moe | src/models/cohere2moe.cpp | PARALLEL-RESIDUAL | cohere2moe.cpp:265-266 — same three-way sum: cur=cur+inpL then cur=cur+attn_out |
| llama_model_command_r | src/models/command-r.cpp | PARALLEL-RESIDUAL | command-r.cpp:118-119 — same three-way sum: cur=cur+inpL then cur=cur+attn_out |
| llama_model_deepseek4 | src/models/deepseek4.cpp | HYBRID-OTHER | deepseek4.cpp:1368 — inpL = build_hc_post(cur, residual, post, comb, il): hyper-connection mixing; next-layer input is an hc-transformed com |
| llama_model_dflash | src/models/dflash.cpp | HYBRID-OTHER | dflash.cpp — MTP draft arch: graph<false> main loop 427-481 classic (add 477), graph_dsv4 loop 597-655 hc-post (653); draft/target coupling  |
| llama_model_eagle3 | src/models/eagle3.cpp | HYBRID-OTHER | eagle3.cpp:137-158,164+ — MTP encoder/decoder pair, n_layer==1 asserted (168); not a plain stack |
| llama_model_falcon | src/models/falcon.cpp | PARALLEL-RESIDUAL | falcon.cpp:121-135 — FFN consumes attn_norm (block input norm), not the post-attention residual; block out = inpL + attn_out + ffn_out via t |
| llama_model_falcon_h1 | src/models/falcon-h1.cpp | HYBRID-OTHER | falcon-h1.cpp:165 — attn_out + ssm_out summed mid-block; single residual thread at bottom (187) but mamba2 hybrid |
| llama_model_gemma3n | src/models/gemma3n.cpp | HYBRID-OTHER | gemma3n.cpp:253-256 — alt-up: block output is a 3D concat [n_embd, n_tokens, n_altup] of corrected slices, not [n_embd, n_tokens] |
| llama_model_gptneox | src/models/gptneox.cpp | PARALLEL-RESIDUAL | gptneox.cpp:143-172 — hparams.use_par_res runtime gate; parallel branch x = x + attn(ln1(x)) + ffn(ln2(x)) (163-166); sequential else-branch |
| llama_model_granite_hybrid | src/models/granite-hybrid.cpp | HYBRID-OTHER | granite-hybrid.cpp:175 — mamba2/attn/ffn per-layer dispatch via llm_build_mamba_base; residual add inside build_layer_ffn helper (granite-hy |
| llama_model_granite_switch | src/models/granite-switch.cpp | HYBRID-OTHER | granite-switch.cpp:310 — adapter-routed ffn; residual inside helper (granite-switch.cpp:420) |
| llama_model_kimi_k3 | src/models/kimi-k3.cpp | HYBRID-OTHER | kimi-k3.cpp:323 — prefix_sum = ggml_add(prefix_sum, cur) accumulated ACROSS layers (running sum), not a per-block residual |
| llama_model_kimi_linear | src/models/kimi-linear.cpp | HYBRID-OTHER | kimi-linear.cpp:528 — single-thread residual at loop bottom, but blocks dispatch on is_recr(il) between linear-attention and full-attention  |
| llama_model_lfm2 | src/models/lfm2.cpp | HYBRID-OTHER | lfm2.cpp:259,268 — hybrid shortconv/attn (is_recr), single residual thread |
| llama_model_mamba | src/models/mamba.cpp | HYBRID-OTHER | mamba.cpp:115 — pure SSM; residual at 115 but all blocks are mamba/mamba2 recurrent |
| llama_model_nemotron_h | src/models/nemotron-h.cpp | HYBRID-OTHER | nemotron-h.cpp:191-207 — per-layer is_recr/n_ff dispatch (mamba2/attn/ffn); residual at 207; res->t_layer_inp[il] = inpL at 183 records bloc |
| llama_model_plamo2 | src/models/plamo2.cpp | HYBRID-OTHER | plamo2.cpp:176 — classic residual thread; build_plamo2_mamba_layer / build_plamo2_attn_layer per-layer dispatch (loop 125-181) |
| llama_model_qwen35moe | src/models/qwen35moe.cpp | HYBRID-OTHER | qwen35moe.cpp:222 — classic residual thread; is_recr delta/attn dispatch (161); sibling of adopted qwen35 |
| llama_model_qwen3next | src/models/qwen3next.cpp | HYBRID-OTHER | qwen3next.cpp:196 — classic residual thread (ffn_residual); per-layer is_recr dispatch between gated-delta-net and full attn; same shape as  |
| llama_model_rwkv6 | src/models/rwkv6.cpp | HYBRID-OTHER | rwkv6.cpp:162 — RWKV time/channel mix with token_shift state (116-148); per-layer output well-defined but recurrence semantics require decis |
| llama_model_rwkv6qwen2 | src/models/rwkv6qwen2.cpp | HYBRID-OTHER | rwkv6qwen2.cpp — RWKV6 backbone + qwen2 head; hybrid |
| llama_model_rwkv7 | src/models/rwkv7.cpp | HYBRID-OTHER | rwkv7.cpp:191 — same as rwkv6 |
| llama_model_t5 | src/models/t5.cpp | HYBRID-OTHER | t5.cpp:145-255 — encoder-decoder: separate enc/dec loops; classic per-layer thread BUT cross-attn consumes external embd_enc; dec loop uses  |
| llama_model_wavtokenizer_dec | src/models/wavtokenizer-dec.cpp | HYBRID-OTHER | wavtokenizer-dec.cpp — posnet/convnext loops (130,218), n_embd_out != n_embd (llama-model.cpp:1124) |

## Adoption order

Wave 1 (branch hs-arch-core): the 4 reference archs - llama, gemma, gemma4, qwen35.
Wave 2 (same-family extensions with identical loop tails): llama4,
smollm3, gemma2/gemma3 (verify each against the ADOPT table above first).
(qwen35moe was listed here initially but is REFUSED above — HYBRID-OTHER,
is_recr dispatch — and refusals win over adoption-order suggestions.)
Later waves: remaining CLASSIC rows, family cluster by family cluster.
Refused archs stay out until a per-builder semantic decision is made and recorded.

## The 24 files not yet classified (151 on disk - 127 classified; upstream drift since the 2026-08-17 syncs added clip, deepseek2ocr, delta-net-base, dots3note, granite-moe/swa, hunyuan-dense, jina-bert-v2/v3, lfm2moe, llama-embed, mamba-base/2, minicpm, mistral4, nemotron-h-moe, nomic-bert/-moe, phimoe, qwen3tts, qwen4exp, rwkv6/7-base, t5encoder)

21 of the 24 are covered by proxy through a classified builder; 3 are
standalone builders with no dormant tap and no classification yet — they are
real omissions of this manifest, listed at the bottom.

- **11 registered alias archs** inherit a builder's graph (and its dormant
  tap) via `using graph =` aliases in models.h: llama-embed -> llama,
  phimoe -> phi3, granite-moe & minicpm -> granite, mistral4 &
  deepseek2ocr -> deepseek2, jina-bert-v2/v3 & nomic-bert &
  nomic-bert-moe -> bert, hunyuan-dense -> hunyuan_vl. Their taps are
  deliberate dormant no-ops until the registry lists them.
- **4 inherit refusals** the same way: mamba2 -> mamba, lfm2moe -> lfm2,
  nemotron-h-moe -> nemotron_h, t5encoder -> t5.
- **4 are base helpers** other builders compose (delta-net-base, mamba-base,
  rwkv6-base, rwkv7-base) -- no standalone graph to classify.
- **clip** is a vision encoder (not a text decoder ladder); **qwen3tts** is a
  registered arch whose graph comes from another builder.

**Not covered by proxy — open work, not silently safe:**

- `dots3note.cpp` (476 lines): registered standalone arch with a CLASSIC
  residual tail (`cur = ggml_add(ctx0, cur, ffn_inp)` at :455) — adoptable,
  no dormant tap sewn yet.
- `granite-swa.cpp` (319 lines): same — classic tail at :312, adoptable,
  no dormant tap sewn yet.
- `qwen4exp.cpp` (1279 lines): registered standalone arch, hc-stream hybrid
  residual — needs a refusal-or-adopt decision before a tap can be honest.
