# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Ascend Qwen4Exp MTP drafter: registration, construction chain and guards.

Everything here drives the *real* assembly path:

* ``EngineArgs.create_engine_config()`` produces the ``VllmConfig`` (including
  the rewritten draft config) the engine would use -- the drafter contract is
  never hand-assembled from a dict;
* the class under test comes from ``ModelRegistry`` after
  ``vllm_ascend.models.register_model()``, so a registry regression fails;
* the drafter is instantiated for real (``torch.device("meta")``), which is
  what surfaces the NVIDIA ``requires FlashAttention`` constructor that the
  Ascend override exists to avoid.

Discriminating power: on the pre-fix tree ``_try_load_model_cls("Qwen4ExpMTP")``
returns ``vllm.models.qwen4_exp.nvidia.mtp.Qwen4ExpMTP`` and constructing it
raises ``NotImplementedError("Qwen4Exp QSA requires FlashAttention")`` from
``nvidia/qsa.py``. Tests 1 and 2 fail there.
"""

import copy
import logging

import pytest
import torch

from vllm import ModelRegistry
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

import vllm_ascend  # noqa: F401  (installs the Ascend platform plugin)
from vllm_ascend.models import register_model

MODEL = "/mnt/weight/Qwen3.8-Flash-Next"

pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(MODEL),
    reason=f"Qwen4Exp checkpoint not present at {MODEL}",
)


# ---------------------------------------------------------------------------
# Distributed-group stubs
# ---------------------------------------------------------------------------
def _group(size: int = 1):
    """Minimal stand-in for a vLLM ``GroupCoordinator``.

    The linear / MoE constructors read the TP, EP, DP, PP, PCP and DCP groups
    at construction time even for a single rank, and ``get_*_group()`` asserts
    when none exists in the process. Only the attributes those constructors
    touch are provided.
    """
    from unittest.mock import MagicMock

    group = MagicMock()
    group.world_size = size
    group.rank_in_group = 0
    group.is_first_rank = True
    group.is_last_rank = True
    group.size.return_value = size
    device_group = MagicMock()
    device_group.size.return_value = size
    group.device_group = device_group
    return group


@pytest.fixture
def parallel_groups(monkeypatch):
    """Install single-rank TP/EP/DP/PP/PCP/DCP/ETP groups for construction."""
    import vllm.distributed.parallel_state as ps

    for name in ("_TP", "_EP", "_DP", "_PP", "_PCP", "_DCP", "_ETP"):
        monkeypatch.setattr(ps, name, _group(1), raising=False)
    yield


# ---------------------------------------------------------------------------
# Engine configs
# ---------------------------------------------------------------------------
def _engine_config(num_speculative_tokens: int, tensor_parallel_size: int = 1):
    register_model()
    args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
        max_model_len=32768,
        max_num_seqs=4,
        load_format="dummy",
        engram_config={"cpu_offload": False},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
    )
    return args.create_engine_config()


@pytest.fixture(scope="module")
def mtp_engine_config():
    """The F3 cumulative config with ``K=1``.

    Built once: ``create_engine_config`` loads the checkpoint config from disk
    and the drafter construction below mutates ``static_forward_context``, so
    every test that constructs takes a deep copy.
    """
    return _engine_config(num_speculative_tokens=1)


def _build_draft(config):
    """Instantiate the registered drafter for real, on the meta device."""
    from vllm_ascend.utils import register_ascend_customop

    register_ascend_customop(config)
    model_config = config.speculative_config.draft_model_config
    cls, arch = ModelRegistry.resolve_model_cls(
        model_config.architecture, model_config
    )
    assert arch == "Qwen4ExpMTP"
    with set_current_vllm_config(config), torch.device("meta"):
        return cls(vllm_config=config, prefix="")


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------
def test_mtp_registration_overrides_upstream_nvidia_dispatch(mtp_engine_config):
    """``Qwen4ExpMTP`` resolves to the Ascend class, not the NVIDIA one.

    The upstream package dispatches on platform with an ``else`` branch that
    selects the NVIDIA implementation; on NPU that draft layer raises in its
    constructor. Resolution landing outside ``vllm_ascend.`` therefore means
    F3 cannot build a drafter at all.
    """
    register_model()
    model_config = mtp_engine_config.speculative_config.draft_model_config
    cls, arch = ModelRegistry.resolve_model_cls(model_config.architecture, model_config)
    assert arch == "Qwen4ExpMTP"
    assert cls.__module__ == "vllm_ascend.models.qwen4_exp.mtp", cls.__module__
    assert cls.__name__ == "AscendQwen4ExpMTP"


def test_draft_config_carries_the_mtp_contract(mtp_engine_config):
    """The rewritten draft config is what the drafter keys off."""
    spec = mtp_engine_config.speculative_config
    assert spec.method == "mtp"
    assert spec.num_speculative_tokens == 1
    draft = spec.draft_model_config
    assert draft.architectures == ["Qwen4ExpMTP"]
    text_config = draft.hf_text_config
    # ``mtp_start_layer_idx`` and the PLE-off predicate both read these.
    assert text_config.num_hidden_layers == 48
    assert text_config.mtp_num_hidden_layers == 1
    assert text_config.ple_layer_ids == [2]
    assert text_config.hc_count == 4
    # ``hc_mult`` is injected by the speculative-config normalisation and
    # widens the drafter's hidden-state buffers upstream.
    assert draft.hf_config.hc_mult == 4
    assert spec.draft_model_config.hf_config.index_share_for_mtp_iteration is False


def test_draft_rope_is_aligned_with_the_target(mtp_engine_config):
    """The drafter runs the target's 1D RoPE, so both ring pages agree.

    ``Qwen4ExpMTPConfig`` keeps MRoPE whenever the draft config still carries a
    ``vision_config``, and the draft config is derived from the multimodal
    checkpoint -- so without the strip the drafter stores three int64 axis
    positions per key (head width 140) while the target, under
    ``--language-model-only``, runs 1D (width 128).

    That difference is load-bearing, not cosmetic: the QSA ring page is
    ``capacity * head_size * 2`` bytes, so 2240 vs 2048. The KV pool unifies
    every page to the largest one (409600), and 2240 does not divide it -- the
    ring view cannot be built and KV-cache initialisation fails with
    ``shape '[..., 8, 140]' is invalid``. 2048 divides exactly, giving the same
    ``block_size`` / page the target ring gets.

    The config is driven in the shape the drafter is really loaded with:
    ``load_eagle_model`` calls ``get_model(vllm_config=<target>, model_config=
    <draft>)``, so ``vllm_config.model_config`` is the **target** when the
    draft's ``__init__`` runs. A strip that reads ``vllm_config.model_config``
    therefore aligns the target (already 1D) and leaves the draft at 140 --
    the assertion on ``vllm_config.model_config`` below is what fails on that
    tree, not just the draft's rope parameters.
    """
    from vllm.models.qwen4_exp.nvidia.mtp import _make_draft_vllm_config
    from vllm_ascend.models.qwen4_exp.mtp import _align_draft_rope_with_target

    config = copy.deepcopy(mtp_engine_config)
    draft = config.speculative_config.draft_model_config

    # This is the pre-state the fix exists for, and it is a property of the
    # checkpoint + spec normalisation, not an assumption about them.
    assert draft.uses_mrope is True
    # The real loading shape: model_config is still the target.
    assert config.model_config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert config.model_config is not draft

    # The module logs through ``vllm.logger.logger`` -- the convention in this
    # repo; the ``vllm_ascend`` namespace has a file-only handler, so a
    # dedicated logger there would not reach stdout. ``vllm`` sets
    # ``propagate=False``, so caplog cannot see the record: attach a local
    # handler to that logger for the duration of the call instead.
    from vllm.logger import logger as vllm_logger

    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=logging.INFO)
    vllm_logger.addHandler(handler)
    try:
        returned = _align_draft_rope_with_target(config)
    finally:
        vllm_logger.removeHandler(handler)

    assert returned is draft.hf_text_config
    assert draft.uses_mrope is False
    rope_parameters = draft.hf_text_config.rope_parameters
    assert "mrope_section" not in rope_parameters
    assert "mrope_interleaved" not in rope_parameters
    # 1D RoPE keeps the target's own parameters; the strip removes the axes
    # only, it does not rewrite theta or the partial-rotary factor.
    assert rope_parameters["rope_theta"] == 10000000
    assert rope_parameters["partial_rotary_factor"] == 0.25
    # The gate the change order asks for: the alignment must be visible in the
    # log, so an operator can tell an aligned drafter from a silently
    # misaligned one.
    messages = [r.getMessage() for r in records]
    assert any("aligned to the target's 1D RoPE" in m for m in messages), messages

    # The config the draft layers actually read (``_make_draft_vllm_config``)
    # is built from the same object, so it inherits the alignment.
    draft_vllm_config = _make_draft_vllm_config(config, 48)
    assert draft_vllm_config.model_config is draft
    assert draft_vllm_config.model_config.uses_mrope is False

    # Page geometry that follows. The ring stores one *indexer* key per slot,
    # so its width is ``index_head_dim`` (128) plus the MRoPE axes if they were
    # kept -- not the attention ``head_dim`` (256).
    from vllm_ascend.models.qwen4_exp.qsa import AscendQSAKeyStateCache

    capacity = 8  # compress_ratio * ceil((compress_ratio + K) / compress_ratio)
    index_head_dim = 128  # three MRoPE axes were 3 * 4 B on top of this
    unified_page = 1600 * index_head_dim * 2  # target ring, measured at runtime
    assert unified_page == 409600
    for head_size, divides in ((128, True), (140, False)):
        page = capacity * head_size * 2
        assert (unified_page % page == 0) is divides, head_size
    assert capacity * (unified_page // (capacity * 128 * 2)) == 1600
    # The ring keeps its own page geometry rather than being padded up to the
    # shared hybrid page -- that flag is what lets the 2048-byte page survive.
    assert AscendQSAKeyStateCache.align_kv_cache_with_mamba is False


def test_draft_rope_strip_spares_a_genuine_mrope_target(mtp_engine_config):
    """A target that does run MRoPE is left alone (no blanket downgrade)."""
    from unittest.mock import patch

    from vllm_ascend.models.qwen4_exp.mtp import _align_draft_rope_with_target

    config = copy.deepcopy(mtp_engine_config)
    # ``uses_mrope`` is derived from the rope parameters, so drive the reported
    # target mode directly: the guard reads ``target_model_config.uses_mrope``.
    with patch.object(
        type(config.speculative_config.target_model_config),
        "uses_mrope",
        new_callable=lambda: property(lambda self: True),
    ):
        before = copy.deepcopy(
            config.speculative_config.draft_model_config.hf_text_config.rope_parameters
        )
        _align_draft_rope_with_target(config)
        after = config.speculative_config.draft_model_config.hf_text_config.rope_parameters
        assert after == before


def test_draft_rope_strip_refuses_silent_no_op(mtp_engine_config):
    """Alignment is idempotent, but a draft that stays MRoPE must raise.

    The pre-fix implementation returned quietly whenever it found nothing to
    strip. That is exactly how the width-140 bug survived: it aligned the
    *target*, found the target already 1D, logged nothing and returned, and the
    caller read a clean return as "the draft is aligned". Two properties are
    pinned here: a second call on an aligned config is a no-op, and a draft
    that still reports MRoPE while carrying no section (i.e. one this function
    failed to align) raises instead of returning.

    The failure case is driven with a stub because the real
    ``draft_model_config`` and ``target_model_config`` are both ``ModelConfig``
    instances: patching ``uses_mrope`` on the class would report MRoPE for the
    target too and take the "genuine MRoPE target" early return instead.
    """
    from types import SimpleNamespace

    from vllm_ascend.models.qwen4_exp.mtp import _align_draft_rope_with_target

    config = copy.deepcopy(mtp_engine_config)
    _align_draft_rope_with_target(config)
    draft = config.speculative_config.draft_model_config
    assert draft.uses_mrope is False

    # Idempotent: the drafter may be constructed against an aligned config.
    _align_draft_rope_with_target(config)
    assert draft.uses_mrope is False

    # The failure mode: reports MRoPE, carries nothing left to pop.
    unaligned_draft = SimpleNamespace(
        hf_config=None,
        hf_text_config=SimpleNamespace(
            rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25}
        ),
        uses_mrope=True,
    )
    stub_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=unaligned_draft, target_model_config=None
        )
    )
    with pytest.raises(RuntimeError, match="no mrope_section was found"):
        _align_draft_rope_with_target(stub_config)


# ---------------------------------------------------------------------------
# 2. Construction chain
# ---------------------------------------------------------------------------
def test_ascend_mtp_constructs_with_ascend_owners(mtp_engine_config, parallel_groups):
    """The drafter builds, and every swappable owner is the Ascend one.

    This is the end-to-end replacement for the NVIDIA constructor that raises
    ``Qwen4Exp QSA requires FlashAttention`` on NPU.
    """
    from vllm_ascend.models.qwen4_exp.hyperconnection import AscendGatedResidual
    from vllm_ascend.models.qwen4_exp.mtp import (
        AscendQwen4ExpMTP,
        AscendQwen4ExpMultiTokenPredictor,
    )
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpDecoderLayer
    from vllm_ascend.models.qwen4_exp.qsa import AscendQwen4ExpQSAAttention

    config = copy.deepcopy(mtp_engine_config)
    draft = _build_draft(config)

    assert isinstance(draft, AscendQwen4ExpMTP)
    assert isinstance(draft.model, AscendQwen4ExpMultiTokenPredictor)
    assert len(draft.model.layers) == 1

    layer = draft.model.layers[0]
    assert isinstance(layer, AscendQwen4ExpDecoderLayer)
    # ``mtp_start_layer_idx = num_hidden_layers`` (48), so PLE -- which lives
    # at layer index 2 -- must not attach even though the checkpoint's text
    # config still lists it.
    assert layer.layer_idx == 48
    assert layer.ple is None
    assert isinstance(layer.self_attn, AscendQwen4ExpQSAAttention)

    mixer = draft.model.hyper_connection_mixer
    assert isinstance(mixer, AscendGatedResidual)
    # The checkpoint ships no ``hyper_connection_mixer.block_inject_weight``.
    assert mixer.use_combine is False

    # Draft layers sit past the target stack, so their cache keys do not
    # collide with the target's.
    assert layer.self_attn.layer_name == "mtp.layers.48.self_attn.attn"


def test_draft_layer_contract_guard_rejects_wrong_owner(mtp_engine_config, parallel_groups):
    """A draft layer with a substituted attention owner is refused.

    Guard-side evidence for ``_assert_draft_layer_contract``: it must catch a
    silently-substituted owner rather than trusting the constructor chain, and
    it must not refuse the real chain (asserted immediately before).
    """
    config = copy.deepcopy(mtp_engine_config)
    draft = _build_draft(config)
    # Real chain is accepted ...
    draft.model._assert_draft_layer_contract()
    # ... and a swapped owner is not.
    layer = draft.model.layers[0]
    layer.self_attn = torch.nn.Identity()
    with pytest.raises(TypeError, match="expected the Ascend QSA"):
        draft.model._assert_draft_layer_contract()


def test_moe_parameters_expose_the_draft_layer(mtp_engine_config, parallel_groups):
    """EPLB bookkeeping sees the draft MoE layer.

    ``AscendQwen4ExpMTP`` inherits the upstream ``Qwen4ExpMixtureOfExperts``
    through ``Qwen4ExpMTP``, and C3 linearisation puts it before the Ascend
    mixin; without the explicit override its ``isinstance`` test matches no
    layer and the drafter would silently report zero experts.
    """
    config = copy.deepcopy(mtp_engine_config)
    draft = _build_draft(config)
    assert draft.num_moe_layers == 1
    assert draft.num_routed_experts == 512
    assert len(draft.moe_mlp_layers) == 1


# ---------------------------------------------------------------------------
# 3. Ring capacity / K constraint
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_speculative_tokens,capacity,constructs",
    [
        (1, 8, True),
        (2, 8, True),
        (3, 8, True),
        (4, 8, True),
        # K >= 5 raises the ring capacity to 12, which does not divide the
        # 400-token attention block, so construction must fail rather than
        # produce a cache geometry the worker cannot lay out.
        (5, 12, False),
    ],
)
def test_ring_capacity_follows_num_speculative_tokens(
    num_speculative_tokens, capacity, constructs
):
    """``capacity = compress_ratio * ceil((compress_ratio + K) / compress_ratio)``.

    Driven through the real spec builder (``AscendQSAKeyStateCache``) so the
    formula and the divisibility assertion are the production ones.
    """
    config = _engine_config(num_speculative_tokens=num_speculative_tokens)
    text_config = config.model_config.hf_text_config
    compress_ratio = text_config.indexer_compress_ratio
    assert compress_ratio == 4

    from vllm_ascend.models.qwen4_exp.qsa import AscendQSAKeyStateCache

    cache = AscendQSAKeyStateCache.__new__(AscendQSAKeyStateCache)
    cache.compress_ratio = compress_ratio
    cache.head_size = 128
    cache.dtype = torch.bfloat16
    cache.cache_config = config.cache_config

    if constructs:
        spec = cache.get_kv_cache_spec(config)
        assert spec.block_size == capacity
        assert config.cache_config.block_size % capacity == 0
    else:
        with pytest.raises(AssertionError, match="QSA ring capacity"):
            cache.get_kv_cache_spec(config)


# ---------------------------------------------------------------------------
# 4. Target hidden-state hand-off
# ---------------------------------------------------------------------------
def test_target_exposes_the_mtp_hidden_buffer(mtp_engine_config, parallel_groups):
    """``get_mtp_target_hidden_states`` returns the buffer only for ``mtp``.

    The speculator keys the drafter's input width off this hook
    (``_target_feeds_hc_residual``) and the model runner swaps the collapsed
    hidden states for whatever it returns, so both directions matter: a
    non-None buffer under ``method != "mtp"`` would feed the drafter a stale
    tensor, and a None buffer under ``method == "mtp"`` would silently fall
    back to the collapsed stream (the scheme-A contract would be broken).
    """
    from vllm.model_executor.model_loader.utils import get_model_cls

    cls = get_model_cls(mtp_engine_config.model_config)
    assert cls.__module__.startswith("vllm_ascend."), cls.__module__
    assert hasattr(cls, "get_mtp_target_hidden_states")

    config = copy.deepcopy(mtp_engine_config)
    with set_current_vllm_config(config), torch.device("meta"):
        model = cls(vllm_config=config, prefix="")

    text_config = config.model_config.hf_text_config
    buffer = model.get_mtp_target_hidden_states()
    assert buffer is not None
    assert buffer.shape == (
        config.scheduler_config.max_num_batched_tokens,
        text_config.hc_count * text_config.hidden_size,
    )

    # Same class, no speculative config: the buffer must not be allocated.
    plain = copy.deepcopy(mtp_engine_config)
    plain.speculative_config = None
    with set_current_vllm_config(plain), torch.device("meta"):
        target = cls(vllm_config=plain, prefix="")
    assert target.get_mtp_target_hidden_states() is None
