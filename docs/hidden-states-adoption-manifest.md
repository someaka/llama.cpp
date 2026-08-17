# Hidden-States Adoption Manifest - all architecture graph builders

Generated 2026-08-16 from the exhaustive classification pass over src/models/
(148 arch files after the 2026-08-17 upstream merge brought bailingmoe3;
127 builders with graph loops classified; every row carries
file:line evidence from source reads, not sampling).


# Layer Index Convention (canonical)

> This section is the single source of truth for layer numbering across the
> fork's hidden-state surface and CrimsonRed's consumers. The same wording
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
| 1 | llama_model_afmoe | src/models/afmoe.cpp | 262 |
| 2 | llama_model_apertus | src/models/apertus.cpp | 146 |
| 3 | llama_model_arcee | src/models/arcee.cpp | 131 |
| 4 | llama_model_arctic | src/models/arctic.cpp | 154 |
| 5 | llama_model_baichuan | src/models/baichuan.cpp | 130 |
| 6 | llama_model_bailingmoe | src/models/bailingmoe.cpp | 155 |
| 7 | llama_model_bailingmoe2 | src/models/bailingmoe2.cpp | 191 |
| 8 | bailingmoe3 | src/models/bailingmoe3.cpp | 381 |
| 9 | bert | src/models/bert.cpp | 218 |
| 10 | bitnet | src/models/bitnet.cpp | 143 |
| 11 | bloom | src/models/bloom.cpp | 128 |
| 12 | chameleon | src/models/chameleon.cpp | 164 |
| 13 | chatglm | src/models/chatglm.cpp | 138 |
| 14 | codeshell | src/models/codeshell.cpp | 130 |
| 15 | cogvlm | src/models/cogvlm.cpp | 138 |
| 16 | dbrx | src/models/dbrx.cpp | 128 |
| 17 | deci | src/models/deci.cpp | 168 |
| 18 | deepseek | src/models/deepseek.cpp | 172 |
| 19 | deepseek2 | src/models/deepseek2.cpp | 692 |
| 20 | deepseek32 | src/models/deepseek32.cpp | 462 |
| 21 | dots1 | src/models/dots1.cpp | 171 |
| 22 | dream | src/models/dream.cpp | 116 |
| 23 | ernie4_5 | src/models/ernie4-5.cpp | 142 |
| 24 | ernie4_5_moe | src/models/ernie4-5-moe.cpp | 110 |
| 25 | eurobert | src/models/eurobert.cpp | 109 |
| 26 | exaone | src/models/exaone.cpp | 113 |
| 27 | exaone4 | src/models/exaone4.cpp | 169 |
| 28 | exaone_moe | src/models/exaone-moe.cpp | 221 |
| 29 | gemma | src/models/gemma.cpp | 116 |
| 30 | gemma2 | src/models/gemma2.cpp | 148 |
| 31 | gemma3 | src/models/gemma3.cpp | 192 |
| 32 | gemma4 | src/models/gemma4.cpp | 377 |
| 33 | gemma4_assistant | src/models/gemma4-assistant.cpp | 181 |
| 34 | gemma_embedding | src/models/gemma-embedding.cpp | 161 |
| 35 | glm4 | src/models/glm4.cpp | 169 |
| 36 | glm4_moe | src/models/glm4-moe.cpp | 259 |
| 37 | glm_dsa | src/models/glm-dsa.cpp | 508 |
| 38 | gpt2 | src/models/gpt2.cpp | 125 |
| 39 | granite | src/models/granite.cpp | 175 (in build_layer_ffn helper) |
| 40 | grok | src/models/grok.cpp | 190 |
| 41 | grovemoe | src/models/grovemoe.cpp | 170 |
| 42 | hunyuan_moe | src/models/hunyuan-moe.cpp | 164 |
| 43 | hunyuan_vl | src/models/hunyuan-vl.cpp | 167 |
| 44 | hy_v3 | src/models/hy-v3.cpp | 208 |
| 45 | internlm2 | src/models/internlm2.cpp | 115 |
| 46 | jais | src/models/jais.cpp | 110 |
| 47 | jais2 | src/models/jais2.cpp | 140 |
| 48 | jamba | src/models/jamba.cpp | 177 |
| 49 | laguna | src/models/laguna.cpp | 316 |
| 50 | llada | src/models/llada.cpp | 136 |
| 51 | llada_moe | src/models/llada-moe.cpp | 139 |
| 52 | llama | src/models/llama.cpp | 221 |
| 53 | llama4 | src/models/llama4.cpp | 245 |
| 54 | maincoder | src/models/maincoder.cpp | 127 |
| 55 | mellum | src/models/mellum.cpp | 197 |
| 56 | mimo2 | src/models/mimo2.cpp | 225 |
| 57 | minicpm3 | src/models/minicpm3.cpp | 231 |
| 58 | minimax_01 | src/models/minimax-01.cpp | 494 |
| 59 | minimax_m2 | src/models/minimax-m2.cpp | 144 |
| 60 | minimax_m3 | src/models/minimax-m3.cpp | 582 |
| 61 | mistral3 | src/models/mistral3.cpp | 210 |
| 62 | modern_bert | src/models/modern-bert.cpp | 159 |
| 63 | mpt | src/models/mpt.cpp | 148 |
| 64 | muse_glimmer | src/models/muse-glimmer.cpp | 175 |
| 65 | nanbeige | src/models/nanbeige.cpp | 155 (dormant tap added 2026-08-17) |
| 66 | nemotron | src/models/nemotron.cpp | 125 |
| 67 | neo_bert | src/models/neo-bert.cpp | 119 |
| 68 | olmo | src/models/olmo.cpp | 117 |
| 69 | olmo2 | src/models/olmo2.cpp | 182 |
| 70 | olmoe | src/models/olmoe.cpp | 150 |
| 71 | openai_moe | src/models/openai-moe.cpp | 147 |
| 72 | openelm | src/models/openelm.cpp | 148 |
| 73 | orion | src/models/orion.cpp | 117 |
| 74 | paddleocr | src/models/paddleocr.cpp | 85 |
| 75 | pangu_embed | src/models/pangu-embed.cpp | 132 |
| 76 | phi2 | src/models/phi2.cpp | 117 |
| 77 | phi3 | src/models/phi3.cpp | 166 |
| 78 | plamo | src/models/plamo.cpp | 112 |
| 79 | plamo3 | src/models/plamo3.cpp | 174 |
| 80 | plm | src/models/plm.cpp | 191 |
| 81 | pockettts | src/models/pockettts.cpp | 123 |
| 82 | qwen | src/models/qwen.cpp | 116 |
| 83 | qwen2 | src/models/qwen2.cpp | 127 |
| 84 | qwen2moe | src/models/qwen2moe.cpp | 170 |
| 85 | qwen2vl | src/models/qwen2vl.cpp | 119 |
| 86 | qwen3 | src/models/qwen3.cpp | 135 |
| 87 | qwen35 | src/models/qwen35.cpp | 200 |
| 88 | qwen3moe | src/models/qwen3moe.cpp | 155 |
| 89 | qwen3vl | src/models/qwen3vl.cpp | 154 |
| 90 | qwen3vlmoe | src/models/qwen3vlmoe.cpp | 166 |
| 91 | refact | src/models/refact.cpp | 136 |
| 92 | rnd1 | src/models/rnd1.cpp | 153 |
| 93 | seed_oss | src/models/seed-oss.cpp | 126 |
| 94 | smallthinker | src/models/smallthinker.cpp | 166 |
| 95 | smollm3 | src/models/smollm3.cpp | 127 |
| 96 | stablelm | src/models/stablelm.cpp | 147 |
| 97 | starcoder | src/models/starcoder.cpp | 123 |
| 98 | starcoder2 | src/models/starcoder2.cpp | 134 |
| 99 | step35 | src/models/step35.cpp | 340 |
| 100 | talkie | src/models/talkie.cpp | 126 |
| 101 | xverse | src/models/xverse.cpp | 114 |

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

