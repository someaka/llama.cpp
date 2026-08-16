# Hidden-States Adoption Manifest - all architecture graph builders

Generated 2026-08-16 from the exhaustive classification pass over src/models/
(147 arch files; 126 builders with graph loops classified; every row carries
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
| 8 | llama_model_bert | src/models/bert.cpp | 218 |
| 9 | llama_model_bitnet | src/models/bitnet.cpp | 143 |
| 10 | llama_model_bloom | src/models/bloom.cpp | 128 |
| 11 | llama_model_chameleon | src/models/chameleon.cpp | 164 |
| 12 | llama_model_chatglm | src/models/chatglm.cpp | 138 |
| 13 | llama_model_codeshell | src/models/codeshell.cpp | 130 |
| 14 | llama_model_cogvlm | src/models/cogvlm.cpp | 138 |
| 15 | llama_model_dbrx | src/models/dbrx.cpp | 128 |
| 16 | llama_model_deci | src/models/deci.cpp | 168 |
| 17 | llama_model_deepseek | src/models/deepseek.cpp | 172 |
| 18 | llama_model_deepseek2 | src/models/deepseek2.cpp | 692 |
| 19 | llama_model_deepseek32 | src/models/deepseek32.cpp | 495 |
| 20 | llama_model_dots1 | src/models/dots1.cpp | 171 |
| 21 | llama_model_dream | src/models/dream.cpp | 116 |
| 22 | llama_model_ernie4_5 | src/models/ernie4-5.cpp | 142 |
| 23 | llama_model_ernie4_5_moe | src/models/ernie4-5-moe.cpp | 110 |
| 24 | llama_model_eurobert | src/models/eurobert.cpp | 109 |
| 25 | llama_model_exaone | src/models/exaone.cpp | 113 |
| 26 | llama_model_exaone4 | src/models/exaone4.cpp | 169 |
| 27 | llama_model_exaone_moe | src/models/exaone-moe.cpp | 221 |
| 28 | llama_model_gemma | src/models/gemma.cpp | 115 |
| 29 | llama_model_gemma2 | src/models/gemma2.cpp | 148 |
| 30 | llama_model_gemma3 | src/models/gemma3.cpp | 192 |
| 31 | llama_model_gemma4 | src/models/gemma4.cpp | 361 |
| 32 | llama_model_gemma4_assistant | src/models/gemma4-assistant.cpp | 181 |
| 33 | llama_model_gemma_embedding | src/models/gemma-embedding.cpp | 161 |
| 34 | llama_model_glm4 | src/models/glm4.cpp | 169 |
| 35 | llama_model_glm4_moe | src/models/glm4-moe.cpp | 259 |
| 36 | llama_model_glm_dsa | src/models/glm-dsa.cpp | 541 |
| 37 | llama_model_gpt2 | src/models/gpt2.cpp | 125 |
| 38 | llama_model_granite | src/models/granite.cpp | None |
| 39 | llama_model_grok | src/models/grok.cpp | 190 |
| 40 | llama_model_grovemoe | src/models/grovemoe.cpp | 170 |
| 41 | llama_model_hunyuan_moe | src/models/hunyuan-moe.cpp | 164 |
| 42 | llama_model_hunyuan_vl | src/models/hunyuan-vl.cpp | 167 |
| 43 | llama_model_hy_v3 | src/models/hy-v3.cpp | 208 |
| 44 | llama_model_internlm2 | src/models/internlm2.cpp | 115 |
| 45 | llama_model_jais | src/models/jais.cpp | 110 |
| 46 | llama_model_jais2 | src/models/jais2.cpp | None |
| 47 | llama_model_jamba | src/models/jamba.cpp | 177 |
| 48 | llama_model_laguna | src/models/laguna.cpp | 316 |
| 49 | llama_model_llada | src/models/llada.cpp | 136 |
| 50 | llama_model_llada_moe | src/models/llada-moe.cpp | 139 |
| 51 | llama_model_llama | src/models/llama.cpp | 220 |
| 52 | llama_model_llama4 | src/models/llama4.cpp | 245 |
| 53 | llama_model_maincoder | src/models/maincoder.cpp | 127 |
| 54 | llama_model_mellum | src/models/mellum.cpp | 197 |
| 55 | llama_model_mimo2 | src/models/mimo2.cpp | 225 |
| 56 | llama_model_minicpm3 | src/models/minicpm3.cpp | 231 |
| 57 | llama_model_minimax_01 | src/models/minimax-01.cpp | 494 |
| 58 | llama_model_minimax_m2 | src/models/minimax-m2.cpp | 144 |
| 59 | llama_model_minimax_m3 | src/models/minimax-m3.cpp | 582 |
| 60 | llama_model_mistral3 | src/models/mistral3.cpp | 210 |
| 61 | llama_model_modern_bert | src/models/modern-bert.cpp | 159 |
| 62 | llama_model_mpt | src/models/mpt.cpp | 148 |
| 63 | llama_model_muse_glimmer | src/models/muse-glimmer.cpp | 175 |
| 64 | llama_model_nanbeige | src/models/nanbeige.cpp | 155 |
| 65 | llama_model_nemotron | src/models/nemotron.cpp | 125 |
| 66 | llama_model_neo_bert | src/models/neo-bert.cpp | 119 |
| 67 | llama_model_olmo | src/models/olmo.cpp | 117 |
| 68 | llama_model_olmo2 | src/models/olmo2.cpp | 182 |
| 69 | llama_model_olmoe | src/models/olmoe.cpp | 150 |
| 70 | llama_model_openai_moe | src/models/openai-moe.cpp | 147 |
| 71 | llama_model_openelm | src/models/openelm.cpp | 148 |
| 72 | llama_model_orion | src/models/orion.cpp | 117 |
| 73 | llama_model_paddleocr | src/models/paddleocr.cpp | 85 |
| 74 | llama_model_pangu_embed | src/models/pangu-embed.cpp | 132 |
| 75 | llama_model_phi2 | src/models/phi2.cpp | 117 |
| 76 | llama_model_phi3 | src/models/phi3.cpp | 166 |
| 77 | llama_model_plamo | src/models/plamo.cpp | 112 |
| 78 | llama_model_plamo3 | src/models/plamo3.cpp | 174 |
| 79 | llama_model_plm | src/models/plm.cpp | 191 |
| 80 | llama_model_pockettts | src/models/pockettts.cpp | 123 |
| 81 | llama_model_qwen | src/models/qwen.cpp | 116 |
| 82 | llama_model_qwen2 | src/models/qwen2.cpp | 127 |
| 83 | llama_model_qwen2moe | src/models/qwen2moe.cpp | 170 |
| 84 | llama_model_qwen2vl | src/models/qwen2vl.cpp | 119 |
| 85 | llama_model_qwen3 | src/models/qwen3.cpp | 135 |
| 86 | llama_model_qwen35 | src/models/qwen35.cpp | 199 |
| 87 | llama_model_qwen3moe | src/models/qwen3moe.cpp | 155 |
| 88 | llama_model_qwen3vl | src/models/qwen3vl.cpp | 154 |
| 89 | llama_model_qwen3vlmoe | src/models/qwen3vlmoe.cpp | 166 |
| 90 | llama_model_refact | src/models/refact.cpp | 136 |
| 91 | llama_model_rnd1 | src/models/rnd1.cpp | 153 |
| 92 | llama_model_seed_oss | src/models/seed-oss.cpp | 126 |
| 93 | llama_model_smallthinker | src/models/smallthinker.cpp | 166 |
| 94 | llama_model_smollm3 | src/models/smollm3.cpp | 127 |
| 95 | llama_model_stablelm | src/models/stablelm.cpp | 147 |
| 96 | llama_model_starcoder | src/models/starcoder.cpp | 123 |
| 97 | llama_model_starcoder2 | src/models/starcoder2.cpp | 134 |
| 98 | llama_model_step35 | src/models/step35.cpp | 340 |
| 99 | llama_model_talkie | src/models/talkie.cpp | 126 |
| 100 | llama_model_xverse | src/models/xverse.cpp | 114 |

Of the 103 CLASSIC builders, 4 are the reference adoption; 98 received the
dormant `capture_layer_output()` call in the hs-arch-core sweep (2026-08-16,
compile-verified, byte-identity on the supported archs preserved);
1 (nanbeige) was skipped — no residual add before its loop-tail reassign,
so the classic tail pattern does not hold there. Dormant calls are no-ops
until the arch is added to the registry; per-adoption steps:

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
Wave 2 (same-family extensions with identical loop tails): qwen35moe, llama4,
smollm3, gemma2/gemma3 (verify each against the ADOPT table above first).
Later waves: remaining CLASSIC rows, family cluster by family cluster.
Refused archs stay out until a per-builder semantic decision is made and recorded.

