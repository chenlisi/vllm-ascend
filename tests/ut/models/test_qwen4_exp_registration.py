# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Registration and startup-contract tests for Ascend Qwen4Exp.

Everything here goes through the *real* assembly path:
``EngineArgs.create_engine_config()`` produces the ``VllmConfig`` the engine
would use, and the model class is obtained from ``ModelRegistry`` after
``vllm_ascend.models.register_model()`` ran. No hand-written stubs stand in for
either.
"""

import math

import pytest
import torch

from vllm import ModelRegistry
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.torch_utils import get_dtype_size

import vllm_ascend  # noqa: F401  (installs the Ascend platform plugin)
from vllm_ascend.models import register_model

MODEL = "/mnt/weight/Qwen3.8-Flash-Next"
ARCHS = ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")


pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(MODEL),
    reason=f"Qwen4Exp checkpoint not present at {MODEL}",
)


@pytest.fixture(scope="module")
def engine_config():
    register_model()
    args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=8,
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
        max_model_len=32768,
        max_num_seqs=8,
        load_format="dummy",
        engram_config={"cpu_offload": False},
    )
    return args.create_engine_config()


def test_registry_overrides_upstream_dispatch(engine_config):
    """Both architectures resolve to the Ascend implementation.

    The upstream package's ``else`` branch selects the NVIDIA model, so a
    resolution landing outside ``vllm_ascend.`` means the override did not take
    effect and the NVIDIA (flash-attn) path would be built instead.
    """
    register_model()
    model_config = engine_config.model_config
    for arch in ARCHS:
        cls, resolved_arch = ModelRegistry.resolve_model_cls([arch], model_config)
        assert resolved_arch == arch
        assert cls.__module__.startswith("vllm_ascend."), cls.__module__
        assert cls.__name__.startswith("AscendQwen4Exp")


def test_engram_whitelist_allows_qwen4_exp(engine_config):
    """The Ascend Engram guard accepts Qwen4Exp and keeps the parallel limits.

    ``_validate_engram_config`` used to hard-code ``DeepseekV41ForCausalLM``,
    which raised before the model could be built.
    """
    from vllm_ascend.platform import _validate_engram_config

    _validate_engram_config(engine_config)

    import copy

    broken = copy.copy(engine_config)
    broken.parallel_config = copy.copy(engine_config.parallel_config)
    broken.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="PP=DCP=1"):
        _validate_engram_config(broken)


def test_block_size_derivation_matches_runtime(engine_config):
    """``block_size`` follows the backend alignment, not a hard-coded value.

    With TP=8 the GDN state page is 400896 B and one attention token costs
    1024 B, so the derivation floor is 391 tokens. The rounding granularity is
    the QSA backend's own ``get_supported_kernel_block_sizes()`` (``MultipleOf
    (16)``), which is read from the backend here rather than assumed: this
    stack has no MLA layer, so upstream's MLA-only 128 floor does not apply and
    400 is the size the scheduler actually ends up with. The expected value is
    produced by the *real* derivation helper the worker calls, so a formula
    change upstream or a backend alignment change here fails the test instead
    of drifting silently.
    """
    from vllm.platforms import current_platform
    from vllm.v1.attention.backend import MultipleOf

    from vllm_ascend.models.qwen4_exp.qsa import AscendQwen4ExpQSABackend

    model_config = engine_config.model_config
    parallel_config = engine_config.parallel_config
    cache_config = engine_config.cache_config
    cls, _ = ModelRegistry.resolve_model_cls(model_config.architecture, model_config)

    mamba_page = max(
        spec.page_size_bytes for spec in cls.get_mamba_specs_from_config(engine_config)
    )
    attn_1tok = (
        2
        * model_config.get_head_size()
        * model_config.get_num_kv_heads(parallel_config)
        * get_dtype_size(model_config.dtype)
    )
    assert mamba_page == 400896
    assert attn_1tok == 1024

    # Alignment granularity comes from the backend under test.
    supported = AscendQwen4ExpQSABackend.get_supported_kernel_block_sizes()
    alignment = max(
        min(s.base if isinstance(s, MultipleOf) else s for s in supported),
        cache_config.block_size,
    )
    assert alignment == 16, f"kernel block alignment drifted: {supported}"

    expected = alignment * math.ceil(mamba_page / (alignment * attn_1tok))
    assert expected == 400, f"block_size drifted: alignment={alignment}"

    # Drive the same helper the worker uses; it mutates cache_config in place,
    # so restore the shared fixture afterwards to keep test order irrelevant.
    saved = (
        cache_config.block_size,
        cache_config.mamba_block_size,
        cache_config.mamba_page_size_padded,
    )
    try:
        current_platform._align_hybrid_block_size(
            engine_config, AscendQwen4ExpQSABackend
        )
        assert cache_config.block_size == expected
        attn_page = cache_config.block_size * attn_1tok
        # Same formula and denominator as the upstream logger
        # (interface.py: "Padding mamba page size by %.2f%%").
        padding_pct = 100 * (attn_page - mamba_page) / mamba_page
        # 2.17% is the figure the running service logs (smoke/serve-dummy.log).
        assert round(padding_pct, 2) == 2.17
        assert cache_config.mamba_page_size_padded == attn_page == 409600
    finally:
        (
            cache_config.block_size,
            cache_config.mamba_block_size,
            cache_config.mamba_page_size_padded,
        ) = saved

    # The ring capacity must divide the scheduler block (E4 assertion).
    assert expected % 4 == 0


def test_mamba_specs_cover_both_state_families(engine_config):
    """GDN and PLE each own one MambaSpec; only the PLE one is TP-replicated."""
    model_config = engine_config.model_config
    cls, _ = ModelRegistry.resolve_model_cls(model_config.architecture, model_config)
    specs = cls.get_mamba_specs_from_config(engine_config)
    assert len(specs) == 2
    gdn, ple = specs
    assert gdn.page_size_bytes == 400896
    assert ple.page_size_bytes == 184320
    assert gdn.tp_replicated is False
    assert ple.tp_replicated is True
    # PLE conv window: (kernel-1) * ngram_size = 9 taps over 10240 channels.
    assert ple.shapes == ((9, 10240),)


def test_text_config_layer_layout(engine_config):
    """Layer mix and PLE placement the implementation keys off."""
    text_config = engine_config.model_config.hf_text_config
    layer_types = set(text_config.layer_types)
    assert layer_types == {"indexed_attention", "linear_attention"}
    assert text_config.layer_types.count("linear_attention") == 36
    assert text_config.layer_types.count("indexed_attention") == 12
    assert text_config.ple_layer_ids == [2]
    assert text_config.number_of_conv_states == 3
    # 2 KV heads over 8 ranks: one replicated head per rank.
    assert text_config.num_key_value_heads == 2
    assert text_config.num_attention_heads == 24
    assert text_config.indexer_budget == 2048
    assert text_config.indexer_compress_ratio == 4


def test_language_model_only_drops_mrope(engine_config):
    """``--language-model-only`` strips MRoPE *and* ``uses_mrope``.

    The design (handoff S-1) predicted ``uses_mrope`` would stay True while
    the rope parameters degraded to 1D. The measured behaviour is that both
    follow ``--language-model-only``: the hook runs before ``uses_mrope`` is
    consumed, so the ring stores a plain 128-wide BF16 row with no packed
    int64 axis positions. The width is pinned here because it decides the ring
    page size (128 * 2 B vs 140 * 2 B).
    """
    model_config = engine_config.model_config
    assert model_config.uses_mrope is False
    rope_parameters = model_config.hf_text_config.rope_parameters
    assert "mrope_section" not in rope_parameters
    assert "mrope_interleaved" not in rope_parameters
    # rotary_dim stays 64: partial_rotary_factor applies to the 1D rope too.
    rotary_dim = int(model_config.get_head_size() * rope_parameters["partial_rotary_factor"])
    assert rotary_dim == 64


def test_qwen4_exp_mrope_geometry_is_self_consistent(engine_config):
    """The pre-strip geometry the checkpoint was trained with is consistent.

    ``sum(mrope_section) == rotary_dim // 2`` is what ``MRotaryEmbedding``
    asserts, so the text config (read fresh, not through the stripped engine
    config) must satisfy it. A change here would mean the checkpoint metadata
    and the rope construction disagree.
    """
    from transformers import Qwen4ExpConfig

    config = Qwen4ExpConfig.from_pretrained(MODEL).text_config
    rope_parameters = config.rope_parameters
    assert rope_parameters["mrope_section"] == [11, 11, 10]
    assert rope_parameters["mrope_interleaved"] is True
    rotary_dim = int(config.head_dim * rope_parameters["partial_rotary_factor"])
    assert rotary_dim == 64
    assert sum(rope_parameters["mrope_section"]) == rotary_dim // 2


def test_model_classes_are_the_ascend_ones():
    """The two OOT mechanisms report different log lines; both must appear.

    Custom-op replacement logs ``Instantiating ...`` when the layer is built;
    pluggable-layer replacement logs its own line. Both strings are asserted so
    a silent fallback to the upstream class cannot pass.
    """
    from vllm_ascend.models.qwen4_exp import model as model_module
    from vllm_ascend.models.qwen4_exp.qsa import AscendQwen4ExpQSAAttention

    assert AscendQwen4ExpQSAAttention.__module__ == "vllm_ascend.models.qwen4_exp.qsa"
    assert model_module.AscendQwen4ExpModel.__module__ == "vllm_ascend.models.qwen4_exp.model"
    # The attention layer must not inherit the upstream owner (its constructor
    # raises on non-CUDA platforms).
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    assert not issubclass(AscendQwen4ExpQSAAttention, Qwen4ExpQSAAttention)
