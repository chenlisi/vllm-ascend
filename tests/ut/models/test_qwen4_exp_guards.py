# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Guard tests for the semantic-sensitive QSA branches.

The mechanism x branch matrix in ``design/worker-design-spec.md`` §8 marks the
gate (output sigmoid), norm (Gemma RMSNorm on q/k/indexer), and quantization
families as branch-bearing. Two kinds of evidence are required per branch:

* the guard raises for an input outside the supported predicate, and
* the default path still works for the supported input (so the guard is not a
  blanket refusal).

Every test here is a pure-tensor check: no NPU and no engine are needed.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp.qsa import (
    AscendQwen4ExpQSAAttnImpl,
    _sparse_gqa_torch,
    gemma_rmsnorm_torch,
)

PAGE, BLOCKS, KV_HEADS, HEAD_DIM, NUM_HEADS = 2, 2, 1, 4, 2


def _impl(num_heads=NUM_HEADS, num_kv_heads=KV_HEADS):
    return AscendQwen4ExpQSAAttnImpl(
        num_heads=num_heads, head_size=HEAD_DIM, scale=0.5, num_kv_heads=num_kv_heads
    )


def _caches(seed=0):
    torch.manual_seed(seed)
    key = torch.randn(BLOCKS, PAGE, KV_HEADS, HEAD_DIM)
    return key, torch.randn_like(key)


def _metadata(num_tokens=1):
    return SimpleNamespace(
        num_actual_tokens=num_tokens,
        block_table=torch.arange(BLOCKS, dtype=torch.int32).reshape(1, -1),
        token_to_req=torch.zeros(num_tokens, dtype=torch.int32),
    )


def _layer(**overrides):
    fields = dict(alibi_slopes=None, sinks=None, sliding_window=None, kv_cache_dtype="auto")
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _selection(rows=1, topk=2):
    # Leading columns are token indices, the trailing column is the valid count.
    sel = torch.full((rows, topk + 1), -1, dtype=torch.int32)
    for row in range(rows):
        sel[row, : min(topk, 2)] = torch.arange(min(topk, 2))
        sel[row, -1] = min(topk, 2)
    return sel


# ---------------------------------------------------------------------------
# gate family: the QSA output sigmoid must see exactly one scalar per lane
# ---------------------------------------------------------------------------
def test_output_gate_applies_sigmoid_per_lane():
    """The default path multiplies the projection by sigmoid(gate)."""
    impl = _impl()
    key, value = _caches()
    torch.manual_seed(1)
    query = torch.randn(1, NUM_HEADS, HEAD_DIM)
    gate = torch.randn(1, NUM_HEADS * HEAD_DIM) * 4.0
    out = torch.zeros(1, NUM_HEADS, HEAD_DIM)

    impl.forward_qsa(_layer(), query, (key, value), _metadata(), out, _selection(), gate)

    # Independent recomputation of the two halves.
    attn = _sparse_gqa_torch(
        query,
        key,
        value,
        _selection()[:, :-1],
        torch.arange(BLOCKS, dtype=torch.int32).reshape(1, -1),
        torch.zeros(1, dtype=torch.int32),
        0.5,
    )
    expected = attn * torch.sigmoid(gate.reshape(1, NUM_HEADS, HEAD_DIM))
    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("gate_width", [NUM_HEADS * HEAD_DIM - 1, NUM_HEADS * HEAD_DIM + 1])
def test_output_gate_width_guard(gate_width):
    """A gate that is not one scalar per (head, lane) is rejected, not broadcast."""
    impl = _impl()
    key, value = _caches()
    query = torch.randn(1, NUM_HEADS, HEAD_DIM)
    out = torch.zeros(1, NUM_HEADS, HEAD_DIM)
    with pytest.raises(ValueError, match="one scalar per"):
        impl.forward_qsa(
            _layer(), query, (key, value), _metadata(), out, _selection(),
            torch.randn(1, gate_width),
        )


def test_output_gate_guard_does_not_fire_on_valid_input():
    """Control: the same call with the correct width returns normally."""
    impl = _impl()
    key, value = _caches()
    query = torch.randn(1, NUM_HEADS, HEAD_DIM)
    out = torch.zeros(1, NUM_HEADS, HEAD_DIM)
    result = impl.forward_qsa(
        _layer(), query, (key, value), _metadata(), out, _selection(),
        torch.zeros(1, NUM_HEADS * HEAD_DIM),
    )
    assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# unsupported-attention-feature branch: named predicates, explicit raise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"alibi_slopes": torch.ones(NUM_HEADS)}, "ALiBi or attention sinks"),
        ({"sinks": torch.zeros(NUM_HEADS)}, "ALiBi or attention sinks"),
        ({"sliding_window": 8}, "sliding-window"),
        ({"kv_cache_dtype": "fp8"}, "BF16 main KV cache"),
    ],
)
def test_unsupported_feature_branches_raise(overrides, message):
    impl = _impl()
    key, value = _caches()
    query = torch.randn(1, NUM_HEADS, HEAD_DIM)
    out = torch.zeros(1, NUM_HEADS, HEAD_DIM)
    with pytest.raises(NotImplementedError, match=message):
        impl.forward_qsa(
            _layer(**overrides), query, (key, value), _metadata(), out, _selection(),
            torch.zeros(1, NUM_HEADS * HEAD_DIM),
        )


def test_unbound_cache_guard():
    """An empty cache means the allocator never bound this layer."""
    impl = _impl()
    query = torch.randn(1, NUM_HEADS, HEAD_DIM)
    out = torch.zeros(1, NUM_HEADS, HEAD_DIM)
    empty = torch.empty(0)
    with pytest.raises(RuntimeError, match="not bound"):
        impl.forward_qsa(
            _layer(), query, (empty, empty), _metadata(), out, _selection(),
            torch.zeros(1, NUM_HEADS * HEAD_DIM),
        )


# ---------------------------------------------------------------------------
# GQA head mapping: the operator cannot express kv_heads > q_heads
# ---------------------------------------------------------------------------
def test_gqa_head_guard():
    """More KV heads than query heads has no defined grouping."""
    with pytest.raises(ValueError, match="query heads divisible by KV heads"):
        _sparse_gqa_torch(
            torch.zeros(1, 2, HEAD_DIM),
            torch.zeros(BLOCKS, PAGE, 4, HEAD_DIM),
            torch.zeros(BLOCKS, PAGE, 4, HEAD_DIM),
            torch.tensor([[0]], dtype=torch.int32),
            torch.arange(BLOCKS, dtype=torch.int32).reshape(1, -1),
            torch.zeros(1, dtype=torch.int32),
            0.5,
        )


# ---------------------------------------------------------------------------
# norm family: the Gemma variant has no mean subtraction
# ---------------------------------------------------------------------------
def test_gemma_norm_matches_reference_and_differs_from_layernorm():
    """Gemma RMSNorm scales by (1 + w) and skips mean-centering.

    A plain LayerNorm would subtract the mean; the difference is checked so an
    implementation that silently falls back to LayerNorm fails here.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    w = torch.randn(8)
    got = gemma_rmsnorm_torch(x, w, 1e-6)

    xf = x.float()
    reference = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * (1.0 + w.float())
    assert torch.allclose(got, reference, atol=1e-6)

    layernorm = torch.nn.functional.layer_norm(x, (8,), eps=1e-6) * (1.0 + w)
    assert not torch.allclose(got, layernorm, atol=1e-3)


# ---------------------------------------------------------------------------
# metadata assembly: a populated dict must carry this layer's entry
# ---------------------------------------------------------------------------
#
# Both fallbacks below used to emit plausible output when the per-layer entry
# was missing (a zeroed QSA output, a PLE result without its conv term), which
# turns a naming/registration mismatch into silent numerical garbage. These
# guards are what makes such a mismatch show up at the first forward.

MODEL_PREFIX = "model.layers.0"


def _forward_context(metadata):
    return SimpleNamespace(attn_metadata=metadata)


def _patch_forward_context(monkeypatch, metadata):
    """Point every consumer at a fake forward context.

    ``qsa`` binds ``get_forward_context`` at import time while ``ple`` imports
    it lazily inside the call, so both the shared module and its importer are
    patched here.
    """
    import vllm.forward_context as ctx_mod
    import vllm_ascend.models.qwen4_exp.qsa as qsa_mod

    fake = lambda: _forward_context(metadata)  # noqa: E731
    monkeypatch.setattr(ctx_mod, "get_forward_context", fake)
    monkeypatch.setattr(qsa_mod, "get_forward_context", fake)


def _ple_metadata():
    """A real ``PleShortConvAttentionMetadata`` with the required fields set."""
    import dataclasses

    from vllm.v1.attention.backends.short_conv_attn import (
        PleShortConvAttentionMetadata,
    )

    kwargs = {}
    for field in dataclasses.fields(PleShortConvAttentionMetadata):
        if (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        ):
            continue
        kwargs[field.name] = torch.zeros(1, dtype=torch.int32)
    return PleShortConvAttentionMetadata(**kwargs)


def _ple_layer_stub(conv_state):
    return SimpleNamespace(prefix=f"{MODEL_PREFIX}.ple", kv_cache=(conv_state,))


def test_qsa_missing_layer_metadata_raises(monkeypatch):
    """A non-empty metadata dict without this layer's key is a defect."""
    from vllm_ascend.models.qwen4_exp.qsa import AscendQwen4ExpQSAAttention

    layer_name = f"{MODEL_PREFIX}.self_attn.attn"
    metadata = {f"{MODEL_PREFIX}.linear_attn": object()}
    _patch_forward_context(monkeypatch, metadata)

    layer = SimpleNamespace(layer_name=layer_name)
    out = torch.zeros(1, NUM_HEADS, HEAD_DIM)
    with pytest.raises(RuntimeError, match="missing from attention metadata") as excinfo:
        AscendQwen4ExpQSAAttention._run_qsa(
            layer,
            torch.zeros(1, HEAD_DIM),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, NUM_HEADS, HEAD_DIM),
            torch.zeros(1, KV_HEADS, HEAD_DIM),
            torch.zeros(1, KV_HEADS, HEAD_DIM),
            out,
            torch.zeros(1, NUM_HEADS * HEAD_DIM),
        )
    # The message must name the layer and expose the actual keys, otherwise it
    # cannot distinguish a prefix mismatch from a builder omission.
    message = str(excinfo.value)
    assert layer_name in message
    assert "linear_attn" in message
    # The raise must not half-write the buffer.
    assert torch.equal(out, torch.zeros(1, NUM_HEADS, HEAD_DIM))


def test_qsa_profiling_without_metadata_still_zeroes(monkeypatch):
    """Control: profiling passes no metadata at all and must keep working."""
    from vllm_ascend.models.qwen4_exp.qsa import AscendQwen4ExpQSAAttention

    _patch_forward_context(monkeypatch, None)
    layer = SimpleNamespace(layer_name=f"{MODEL_PREFIX}.self_attn.attn")
    out = torch.ones(1, NUM_HEADS, HEAD_DIM)
    AscendQwen4ExpQSAAttention._run_qsa(
        layer,
        torch.zeros(1, HEAD_DIM),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, NUM_HEADS, HEAD_DIM),
        torch.zeros(1, KV_HEADS, HEAD_DIM),
        torch.zeros(1, KV_HEADS, HEAD_DIM),
        out,
        torch.zeros(1, NUM_HEADS * HEAD_DIM),
    )
    assert torch.equal(out, torch.zeros(1, NUM_HEADS, HEAD_DIM))


def test_ple_missing_layer_metadata_raises(monkeypatch):
    """A non-empty metadata dict without this layer's key is a defect."""
    from vllm_ascend.models.qwen4_exp.ple import AscendQwen4ExpPLELayer

    metadata = {f"{MODEL_PREFIX}.self_attn.attn": _ple_metadata()}
    _patch_forward_context(monkeypatch, metadata)

    layer = _ple_layer_stub(torch.zeros(2, 3, 4))
    residual = torch.full((1, 4), 2.0)
    with pytest.raises(RuntimeError, match="missing from attention metadata") as excinfo:
        AscendQwen4ExpPLELayer._short_conv(layer, torch.zeros(1, 4), residual, torch.ones(1, 4))
    assert f"{MODEL_PREFIX}.ple" in str(excinfo.value)
    assert "self_attn" in str(excinfo.value)
    # Dropping the conv term must not happen silently: residual stays untouched.
    assert torch.equal(residual, torch.full((1, 4), 2.0))


def test_ple_unbound_conv_state_raises(monkeypatch):
    """Metadata is present but the state slot was never bound."""
    from vllm_ascend.models.qwen4_exp.ple import AscendQwen4ExpPLELayer

    prefix = f"{MODEL_PREFIX}.ple"
    _patch_forward_context(monkeypatch, {prefix: _ple_metadata()})

    layer = _ple_layer_stub(torch.empty(0))
    residual = torch.full((1, 4), 2.0)
    with pytest.raises(RuntimeError, match=r"unbound \(empty\) conv state") as excinfo:
        AscendQwen4ExpPLELayer._short_conv(
            layer, torch.zeros(1, 4), residual, torch.ones(1, 4)
        )
    assert prefix in str(excinfo.value)
    assert torch.equal(residual, torch.full((1, 4), 2.0))


def test_ple_profiling_without_metadata_adds_outer_residual(monkeypatch):
    """Control: with no metadata the layer contributes only the outer residual."""
    from vllm_ascend.models.qwen4_exp.ple import AscendQwen4ExpPLELayer

    _patch_forward_context(monkeypatch, None)
    layer = _ple_layer_stub(torch.empty(0))
    residual = torch.zeros(1, 4)
    outer = torch.ones(1, 4)
    AscendQwen4ExpPLELayer._short_conv(layer, torch.zeros(1, 4), residual, outer)
    assert torch.equal(residual, outer)
