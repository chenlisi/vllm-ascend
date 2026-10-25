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
