# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
#
"""Kimi-K3 configuration for vllm-ascend.

Why this file exists
--------------------
The Kimi-K3 checkpoint is a *nested* config: the top level declares
``architectures=["KimiK3ForConditionalGeneration"]`` / ``model_type="kimi_k3"``
and every model geometry field lives inside ``text_config``, whose own
``architectures`` is ``["KimiLinearForCausalLM"]`` / ``model_type="kimi_linear"``.

Upstream vLLM registers only the *language* side (``kimi_linear``). The
top-level ``kimi_k3`` key is unregistered, so ``ModelRegistry`` cannot resolve
``KimiK3ForConditionalGeneration`` and engine startup fails with
"Model architectures [...] are not supported for now".

The checkpoint ships an ``auto_map`` pointing at ``configuration_kimi_k3.py``,
but that path is unusable in this environment: the same directory's
``modeling_kimi_k3.py`` imports ``transformers.utils.generic.OutputRecorder``,
which does not exist in the installed transformers (5.14.1), so the dynamic
module fails to import.

So we register a real config class here instead of relying on remote code.

Design notes
------------
* ``KimiK3Config`` exposes ``text_config`` as a **plain dict-like namespace**
  rather than a nested object on purpose (see below) -- the language-side
  implementation reads geometry through ``hf_text_config`` and expects the
  usual attribute access, which works identically for either form. Keeping the
  raw dict is what upstream vLLM does for the language side already, and it
  avoids re-instantiating ``KimiLinearConfig`` a second time.
* ``KimiLinearConfig`` (upstream, ``vllm.transformers_utils.configs.kimi_linear``)
  already preserves the five Kimi-K3-only fields
  (``mla_use_output_gate``, ``attn_res_block_size``, ``latent_moe_use_norm``,
  ``routed_expert_hidden_size``, ``activation_situ_*``) plus ``topk_method``
  and ``use_grouped_topk`` through its ``**kwargs`` passthrough. Verified by
  constructing it from the real ``text_config`` and reading each field back.
  This contradicts the earlier design claim of "5 config field gaps" -- on the
  language side there is no gap; the gaps are in the *implementation*, not the
  config class.
"""

from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


class KimiK3Config(PretrainedConfig):
    """Top-level config for Kimi-K3 (VL shell).

    Stage 1 uses the **text path only**, so this class deliberately carries no
    vision fields beyond stashing the raw ``vision_config`` dict for
    round-tripping. Nothing in the text path reads it.
    """

    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | KimiLinearConfig | None = None,
        vision_config: dict | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        # Mirror the vendor behaviour: surface a text-side quantization_config
        # at the top level so the platform's quantization detection sees it.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)


def register_kimi_k3_config() -> None:
    """Register the ``kimi_k3`` model_type and tokenizer mode.

    Both registries must be populated: ``AutoConfig`` (used by
    ``transformers_utils.config.get_config``) and vLLM's own
    ``_CONFIG_REGISTRY``. Idempotent.
    """
    from transformers import AutoConfig

    AutoConfig.register("kimi_k3", KimiK3Config, exist_ok=True)

    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        if "kimi_k3" not in _CONFIG_REGISTRY:
            _CONFIG_REGISTRY["kimi_k3"] = KimiK3Config
    except Exception:  # pragma: no cover - registry layout is upstream-internal
        # Not fatal: AutoConfig registration above is the primary path, and
        # _CONFIG_REGISTRY is consulted only as a fallback.
        pass

    register_kimi_k3_tokenizer()


def register_kimi_k3_tokenizer() -> None:
    """Register the self-hosted ``kimi_k3`` tokenizer mode + renderer.

    Needed because the checkpoint's chat encoding is programmatic (Python, via
    `encoding_k3.py`) rather than a Jinja template, so vLLM's chat entrypoint
    would otherwise refuse every chat request with HTTP 400. See
    ``tokenizer.py`` for the full rationale.

    The renderer is registered too: vLLM selects the renderer by the *same*
    tokenizer-mode string, so registering only the tokenizer fails later with
    ``No renderer registered for renderer_mode='kimi_k3'``. The `hf` renderer is
    the correct target -- it is the generic one that calls the tokenizer's
    `apply_chat_template`, which is exactly what our wrapper exposes.
    """
    try:
        from vllm.tokenizers.registry import TokenizerRegistry

        if "kimi_k3" not in TokenizerRegistry.tokenizers:
            TokenizerRegistry.register(
                "kimi_k3", "vllm_ascend.models.kimi_k3.tokenizer", "KimiK3Tokenizer"
            )
    except Exception:  # pragma: no cover - registry layout is upstream-internal
        pass

    try:
        from vllm.renderers.registry import RENDERER_REGISTRY

        if "kimi_k3" not in RENDERER_REGISTRY.renderers:
            RENDERER_REGISTRY.register("kimi_k3", "vllm.renderers.hf", "HfRenderer")
    except Exception:  # pragma: no cover - registry layout is upstream-internal
        pass
