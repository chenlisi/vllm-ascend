# Kimi-K3

## 1 Introduction

Kimi-K3 is a hybrid-attention Mixture-of-Experts model. Its defining feature is **per-layer heterogeneous attention**: the 93-layer stack interleaves two attention families rather than using one throughout.

| Layer group | Count | Attention | Cache |
| --- | --- | --- | --- |
| KDA (linear attention) | 69 | `KimiDeltaAttention` (gated delta net with short conv) | fixed-size conv + recurrent state |
| MLA (full attention) | 24 | `KimiMLAAttention` (compressed latent KV, NoPE, output gate) | paged KV, grows with sequence |

Layer selection is by membership in the 1-based `linear_attn_config.kda_layers`, not by an arithmetic rule on the layer index — the final MLA layer breaks the otherwise regular every-4th cadence.

This document records the validation steps for Kimi-K3 in the vLLM-Ascend environment. Kimi-K3 is first supported in **vLLM-Ascend v0.23.1rc1**; this document was validated against that version.

> **Scope of the current validation.** The text path is validated. On this checkpoint the published weights are git-LFS pointer files, so an end-to-end run with real weights (and therefore a numerical accuracy baseline) has **not** been executed — see §6. Everything in §4–§5 was measured with `--load-format dummy`.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

Stage 1 status for this model:

| Feature | Status |
| --- | --- |
| BF16 inference | Supported |
| Tensor parallel | Supported (TP=8 validated) |
| Expert parallel | Not enabled in the validated configuration |
| Eager mode | Supported (validated) |
| ACLGraph / piecewise | Not yet validated |
| Prefix caching | Not yet validated for mixed linear attention |
| Quantization | Not applicable (checkpoint ships BF16 `quantization_config: null`) |
| Speculative decoding | Not applicable (`num_nextn_predict_layers = 0`) |
| Multi-modal (vision) | Not in scope — text path only |

## 3 Prerequisites

### 3.1 Model Weight

| Model | Hardware Requirement | Download |
| --- | --- | --- |
| Kimi-K3 (BF16) | Atlas 800I A3 (64G, 8 cards for TP=8) | vendor checkpoint |

### 3.2 Environment

- vLLM-Ascend **v0.23.1rc1** (or later)
- vLLM **v0.23.1rc1** and its companion transformers version
- `torch_npu` with a CANN build that includes the `CausalConv1d` custom op for the target SoC

## 4 Deployment

### 4.1 Single-node, TP=8

```bash
vllm serve <MODEL_PATH> \
  --served-model-name Kimi-K3 \
  --trust-remote-code \
  --tokenizer-mode kimi_k3 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --no-enable-chunked-prefill \
  --max-model-len 32768
```

Three model-specific details:

- **`--tokenizer-mode kimi_k3` is required.** The checkpoint ships no Jinja chat template; its chat encoding is programmatic (Python, via `encoding_k3.py`). vLLM-Ascend provides the `kimi_k3` tokenizer mode and renderer so this encoder is used. Without it, chat requests fail with HTTP 400 (`default chat template is no longer allowed`).
- **`--no-enable-chunked-prefill` is required.** Chunked prefill is enabled by default upstream, but this model keeps a cross-layer attention-residual chain (`attn_res_block_size`) as per-request forward-pass state. Chunked prefill restarts that chain on every chunk, so later chunks are mixed against an incomplete accumulator — the model runs and produces plausible but wrong output, with no error. Starting with chunked prefill enabled now fails fast in `KimiK3ForConditionalGeneration.__init__` rather than degrading silently; set `VLLM_ASCEND_KIMI_K3_ALLOW_CHUNKED_PREFILL=1` only if you deliberately want the old behaviour.
- **`--trust-remote-code` is required.** The checkpoint tokenizer overrides `encode` with a different signature than the base class, so the generic tokenizer cannot substitute for it.

For a first-run smoke test with random weights, add `--load-format dummy`. Note that a dummy run with the full 93 layers and 896 experts needs roughly `93 × 59 GB / TP` of expert memory, which exceeds a single die — reduce layers and experts via a derived config (see §6).

### 4.2 Notes on memory

The routed experts dominate the memory footprint: each MoE layer holds `num_experts (896) × (2 × routed_expert_hidden_size (3584) × moe_intermediate_size (3072) + 3072 × 3584)` ≈ 30 G parameters ≈ 59 GB in BF16, divided by TP. KV cache is comparatively small: MLA stores `kv_lora_rank (512) + qk_rope_head_dim (64)` per token per layer.

The aligned attention block size for this model is driven by the KDA recurrent state, which is stored in **float32** (not bfloat16). At TP=8 that gives a 768-token block; the platform derives this automatically and logs it at startup.

## 5 Validation

The validated configuration and its expected startup signals:

| Signal | Expected |
| --- | --- |
| Attention block size | `Setting attention block size to 768 tokens` |
| Mamba page padding | `Padding mamba page size by 3.03%` |
| Readiness | `GET /v1/models` returns 200 |
| Completion | `POST /v1/completions` returns 200 with non-empty output |

## 6 Known Limitations

- **Accuracy is not yet validated.** The checkpoint's `model.safetensors.index.json` and all 96 `model-*.safetensors` shards are git-LFS pointer files (~135 bytes each), so real weights were not available. All results above were produced with `--load-format dummy`, which validates that the model runs and produces output — **not** that the output is correct.
- **Multi-modal is not supported.** The model is a VL architecture (`KimiK3ForConditionalGeneration`) but only the text path is implemented: no vision tower, projector, or vision preprocessing. Image inputs are not accepted.
- **Graph mode is not validated.** Only eager execution is validated.
- **Prefix caching and P/D disaggregation are not validated** for this mixed linear-attention architecture.
