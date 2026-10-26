# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Parallel-topology contracts of the Qwen4Exp QSA stack (Stage 2, TP=8).

Stage 2 freezes TP=8 / PP=1 / DP=1 / DCP=1 / PCP=1 / EP=off, and every one of
those decisions is carried by a fail-fast predicate rather than by the launch
command alone. Until now those predicates were only observable by launching a
service; the tests below drive the *real* assembly path instead:

* the ``VllmConfig`` comes from ``EngineArgs.create_engine_config()``, so a
  change in the upstream derivation (block size, KV-head handling, engram
  validation) fails here rather than at serve time, and
* the layer under test is the production ``AscendQwen4ExpQSAAttention``, so the
  predicates run in their real position (before any projection is built).

The tensor-parallel world size is read from the process group, which a single
process UT cannot initialize. Only that one accessor is patched; everything
else -- head counts, KV-head replication, block size, ring capacity -- is read
from the checkpoint config and the real cache config.

Measurement notes that differ from the design document:

* The ring page is ``4 * 128 * 2 = 1024`` B, not the 1120 B the design derived.
  1120 assumes ``cache_rope_positions=True``; ``--language-model-only`` strips
  MRoPE *and* ``uses_mrope`` (pinned by
  ``test_language_model_only_drops_mrope``), so the ring stores a plain 128-wide
  BF16 row. 1024 is the real geometry and the one asserted here.
"""

import copy
import os

import pytest
import torch

from vllm import ModelRegistry
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

import vllm_ascend  # noqa: F401  (installs the Ascend platform plugin)
from vllm_ascend.models import register_model
from vllm_ascend.models.qwen4_exp import qsa

MODEL = "/mnt/weight/Qwen3.8-Flash-Next"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL),
    reason=f"Qwen4Exp checkpoint not present at {MODEL}",
)


class _FakeTPGroup:
    """Stand-in for the tensor-parallel ``GroupCoordinator``.

    Only the attributes the linear layers read are provided. Any other access
    raises ``AttributeError`` on purpose: if a future change starts reading
    another group attribute, the test should fail instead of silently getting
    a default.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.rank_in_group = 0

    @property
    def device_group(self):
        return None


@pytest.fixture(scope="module")
def engine_config():
    """The Stage 2 frozen launch configuration.

    Mirrors the serve command in ``parallel/impl/launch-params.md``: TP=8,
    eager, language-model-only, max-len 32768, 16 sequences, no prefix caching
    (APC is a Stage 3 feature). ``load_format="dummy"`` keeps it weight-free.
    """
    register_model()
    args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=8,
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
        max_model_len=32768,
        max_num_seqs=16,
        load_format="dummy",
        enable_prefix_caching=False,
        engram_config={"cpu_offload": False},
    )
    return args.create_engine_config()


def _fake_tp(tp_size: int):
    """Patch every accessor that reads the tensor-parallel group.

    Three separate modules resolve the group at call time (the QSA layer, the
    Ascend linear op dispatcher, and upstream's linear base), and patching only
    one leaves the others hitting ``get_tp_group()``, which asserts when no
    distributed group exists in this process. ``exit_stack``-style nesting is
    used so the whole set is restored together.
    """
    from contextlib import ExitStack
    from unittest.mock import patch

    stack = ExitStack()
    stack.enter_context(patch.object(qsa, "get_tensor_model_parallel_world_size", lambda: tp_size))
    stack.enter_context(
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            lambda: tp_size,
        )
    )
    for target in (
        "vllm_ascend.ops.linear_op.get_tp_group",
        "vllm.distributed.parallel_state.get_tp_group",
    ):
        stack.enter_context(patch(target, lambda: _FakeTPGroup(tp_size)))
    return stack


def _build_attention(
    engine_config,
    *,
    tp_size: int,
    block_size: int | None = None,
    dcp_size: int = 1,
    pcp_size: int = 1,
    prefix: str = "model.layers.0",
):
    """Construct the real QSA owner under a synthetic parallel topology.

    ``vllm_config`` is a deep copy so the module-scoped fixture stays pristine
    across parameterized cases (the layer writes into
    ``compilation_config.static_forward_context`` during construction).
    """
    from unittest.mock import patch

    config = copy.deepcopy(engine_config)
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.decode_context_parallel_size = dcp_size
    config.parallel_config.prefill_context_parallel_size = pcp_size
    if block_size is not None:
        config.cache_config.block_size = block_size

    with _fake_tp(tp_size), set_current_vllm_config(config):
        layer = qsa.AscendQwen4ExpQSAAttention(
            vllm_config=config,
            config=config.model_config.hf_text_config,
            layer_id=0,
            quant_config=None,
            prefix=prefix,
        )
    return layer, config


# ---------------------------------------------------------------------------
# TP divisibility and KV-head replication
# ---------------------------------------------------------------------------
def test_tp8_divides_heads_and_replicates_kv_heads(engine_config):
    """TP=8: 24 heads shard to 3/rank, 2 KV heads replicate to 1/rank.

    This is the whole reason ``num_key_value_heads < TP`` is legal here: the
    KV plane is *copied* per rank, not split, which is what the cache budget
    in the design assumes (1024 B/token/rank for the main K/V plane).
    """
    layer, config = _build_attention(engine_config, tp_size=8)
    hf = config.model_config.hf_text_config

    assert hf.num_attention_heads == 24
    assert hf.num_key_value_heads == 2
    assert layer.total_num_heads % 8 == 0
    assert layer.num_heads == 3
    # 2 < 8 -> replicate: every rank keeps one whole KV head.
    assert layer.num_kv_heads == 1
    assert layer.head_dim == 256
    assert layer.kv_size == 256


@pytest.mark.parametrize("tp_size", [5, 7, 16])
def test_heads_not_divisible_by_tp_is_refused(engine_config, tp_size):
    """A TP that does not divide 24 heads must fail, not silently floor.

    TP=16 is the candidate the S2.0 matrix rejected on this exact arithmetic;
    the guard has to catch it before a projection is sharded unevenly.
    """
    with pytest.raises(ValueError, match="QSA attention heads must be divisible by TP"):
        _build_attention(engine_config, tp_size=tp_size)


def test_tp_that_cannot_replicate_kv_heads_is_refused(engine_config):
    """With fewer KV heads than ranks, TP must divide the KV-head count.

    ``num_key_value_heads=2`` means only even TP sizes can hand every rank a
    whole head. TP=3 is the single odd size that still divides the 24 attention
    heads, so it is the one case where this second predicate -- and not the
    head check -- has to fire.
    """
    assert 24 % 3 == 0
    with pytest.raises(ValueError, match="replicated QSA KV heads"):
        _build_attention(engine_config, tp_size=3)


def test_dcp_gt_1_is_refused(engine_config):
    """DCP is structurally incompatible with QSA, so it must fail closed.

    ``supports_dcp = False`` on the owner and on the side caches is a
    declaration; this predicate is what actually stops the engine, because
    QSA never enters the attention-backend validation path that reads the
    declaration.
    """
    with pytest.raises(NotImplementedError, match="context parallel"):
        _build_attention(engine_config, tp_size=8, dcp_size=2)


def test_pcp_gt_1_is_refused(engine_config):
    """Same predicate covers prefill context parallelism."""
    with pytest.raises(NotImplementedError, match="context parallel"):
        _build_attention(engine_config, tp_size=8, pcp_size=2)


def test_supports_dcp_declaration_is_false():
    """The layer and both side caches declare no DCP support."""
    from vllm.models.qwen4_exp.common.qsa_cache import _QSAStateCache

    assert qsa.AscendQwen4ExpQSAAttention.supports_dcp is False
    assert _QSAStateCache.supports_dcp is False
    assert issubclass(qsa.AscendQSAKeyStateCache, _QSAStateCache)
    assert issubclass(qsa.AscendQSACompressedKeyStateCache, _QSAStateCache)


# ---------------------------------------------------------------------------
# KV-cache geometry (the numbers the design freezes)
# ---------------------------------------------------------------------------
def test_main_kv_spec_matches_the_frozen_geometry(engine_config):
    """FullAttentionSpec at block_size=400 -> page 409600, 1 KV head/rank."""
    layer, config = _build_attention(engine_config, tp_size=8, block_size=400)
    spec = layer.get_kv_cache_spec(config)

    assert type(spec).__name__ == "FullAttentionSpec"
    assert spec.block_size == 400
    assert spec.num_kv_heads == 1
    assert spec.head_size == 256
    assert spec.page_size_bytes == 409600
    assert spec.dcp_sharded is True


def test_compressed_key_spec_uses_page_level_compression(engine_config):
    """AscendMLAAttentionSpec: 400 tokens / ratio 4 = 100 rows, 25600 B page.

    The spec class is load-bearing: only ``AscendMLAAttentionSpec`` divides by
    ``tokens_per_state`` in ``get_storage_block_size``, and the worker's
    reshape guard rejects the view otherwise.
    """
    layer, config = _build_attention(engine_config, tp_size=8, block_size=400)
    spec = layer.indexer.compressed_key_cache.get_kv_cache_spec(config)

    assert type(spec).__name__ == "AscendMLAAttentionSpec"
    assert spec.block_size == 400
    assert spec.tokens_per_state == 4
    assert spec.page_size_bytes == 25600
    # storage rows per page, not tokens: what the kernel layout is built from.
    assert spec.block_size // spec.tokens_per_state == 100


def test_ring_spec_semantics(engine_config):
    """The raw ring is one block per request, key-only, never prefix-cached.

    Page is ``capacity(4) * 128 * 2 = 1024`` B here (no MRoPE positions under
    ``--language-model-only``), and the spec must opt out of every feature that
    assumes token-indexed pages.
    """
    layer, config = _build_attention(engine_config, tp_size=8, block_size=400)
    spec = layer.indexer.raw_key_cache.get_kv_cache_spec(config)

    assert type(spec).__name__ == "AscendQSARingSpec"
    assert spec.block_size == 4  # ring *capacity*, not a scheduler block
    assert spec.head_size == 128
    assert spec.page_size_bytes == 1024
    assert spec.max_num_blocks_per_req(config, 32768) == 1
    assert spec.max_memory_usage_bytes(config) == spec.page_size_bytes
    # skip / skip / no-op semantics the design assigns to this role.
    assert spec.prefix_cacheable is False
    assert spec.uses_slot_mapping is False
    # The ring page must not be padded up to the shared 409600 hybrid page.
    assert qsa.AscendQSAKeyStateCache.align_kv_cache_with_mamba is False
    assert qsa.AscendQSACompressedKeyStateCache.align_kv_cache_with_mamba is False


def test_ring_capacity_must_divide_the_scheduler_block(engine_config):
    """The ring joins the LCM that sets the scheduler block, so this must hold.

    ``block_size=400`` is divisible by the capacity 4 (E4 assertion). A block
    size that is not must raise instead of producing a partial ring.
    """
    layer, config = _build_attention(engine_config, tp_size=8, block_size=400)
    assert config.cache_config.block_size % 4 == 0
    ring_spec = layer.indexer.raw_key_cache.get_kv_cache_spec(config)
    assert ring_spec.block_size == 4

    # Same owner, a block size the capacity cannot divide. The config is
    # mutated in place and restored: after construction it holds the built
    # layer in ``static_forward_context``, which cannot be deep-copied.
    saved = config.cache_config.block_size
    try:
        config.cache_config.block_size = 6
        with pytest.raises(AssertionError, match="must divide the attention block"):
            layer.indexer.raw_key_cache.get_kv_cache_spec(config)
    finally:
        config.cache_config.block_size = saved


# ---------------------------------------------------------------------------
# Quantization is N/A: no silent bf16 fallback may hide behind a quant flag
# ---------------------------------------------------------------------------
def test_no_quantization_signature_in_the_checkpoint(engine_config):
    """The checkpoint carries no quantization config, which is why S2 says N/A."""
    assert engine_config.quant_config is None
    assert getattr(engine_config.model_config.hf_text_config, "quantization_config", None) is None
    assert not os.path.exists(os.path.join(MODEL, "quant_model_description.json"))


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "int8"])
def test_non_bf16_kv_cache_is_refused(engine_config, cache_dtype):
    """Non-BF16 KV cache must raise, not degrade to bf16 behind the flag."""
    from unittest.mock import patch

    config = copy.deepcopy(engine_config)
    config.cache_config.cache_dtype = cache_dtype
    with (
        patch.object(qsa, "get_tensor_model_parallel_world_size", lambda: 8),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            lambda: 8,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tp_group",
            lambda: _FakeTPGroup(8),
        ),
        set_current_vllm_config(config),
        pytest.raises(NotImplementedError, match="BF16 main KV cache"),
    ):
        qsa.AscendQwen4ExpQSAAttention(
            vllm_config=config,
            config=config.model_config.hf_text_config,
            layer_id=0,
            quant_config=None,
            prefix="model.layers.0",
        )


def test_kv_quant_scheme_is_refused(engine_config):
    """A ``kv_cache_scheme`` on quant_config must not silently apply."""
    from types import SimpleNamespace
    from unittest.mock import patch

    config = copy.deepcopy(engine_config)
    with (
        patch.object(qsa, "get_tensor_model_parallel_world_size", lambda: 8),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            lambda: 8,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tp_group",
            lambda: _FakeTPGroup(8),
        ),
        set_current_vllm_config(config),
        pytest.raises(NotImplementedError, match="KV quantization"),
    ):
        qsa.AscendQwen4ExpQSAAttention(
            vllm_config=config,
            config=config.model_config.hf_text_config,
            layer_id=0,
            quant_config=SimpleNamespace(kv_cache_scheme=object()),
            prefix="model.layers.0",
        )


# ---------------------------------------------------------------------------
# EP / PP / DP: the frozen "off" side of the parallel matrix
# ---------------------------------------------------------------------------
def test_ep_off_selects_allgather(engine_config):
    """With EP off the MoE comm path is ALLGATHER regardless of expert count.

    The design's argument for EP=off is that ``experts/device`` is 64 either
    way, so this asserts the mechanism: the selector short-circuits on
    ``enable_expert_parallel`` before it ever looks at expert density.
    """
    from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method

    config = copy.deepcopy(engine_config)
    assert config.parallel_config.enable_expert_parallel is False
    assert config.parallel_config.data_parallel_size == 1
    assert config.parallel_config.pipeline_parallel_size == 1

    method = select_moe_comm_method(num_tokens=16, vllm_config=config)
    assert method is MoECommType.ALLGATHER


def test_eplb_requires_expert_parallel(engine_config):
    """EPLB without EP is rejected upstream; the frozen config has both off."""
    from vllm.config import ParallelConfig

    assert engine_config.parallel_config.enable_eplb is False
    with pytest.raises(ValueError, match="enable_expert_parallel must be True"):
        ParallelConfig(enable_eplb=True, enable_expert_parallel=False)


def test_qwen4_model_states_carry_no_vision_tower(engine_config):
    """``--language-model-only`` is what makes DP=1 meaningful (text only).

    The model refuses the multimodal path outright, so the DP sizing question
    from S2.0 (encoder TP mode) cannot arise in this deployment.
    """
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForConditionalGeneration

    cls, _ = ModelRegistry.resolve_model_cls(
        "Qwen4ExpForConditionalGeneration", engine_config.model_config
    )
    assert cls is AscendQwen4ExpForConditionalGeneration
    assert engine_config.model_config.multimodal_config.language_model_only is True
    assert engine_config.model_config.uses_mrope is False
