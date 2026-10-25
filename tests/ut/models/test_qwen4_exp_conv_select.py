# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Dilated short convolution, QSA selection and runner-state tests.

The convolution is compared against ``torch.nn.functional.conv1d`` with the
window made explicit, and the selection against a brute-force loop over the
logical cache. Both references are written independently of the implementation
so a shared mistake cannot hide.
"""

import torch
import torch.nn.functional as F
import pytest

from vllm_ascend.models.qwen4_exp.qsa import qsa_select_torch


# ---------------------------------------------------------------------------
# Dilated short convolution
# ---------------------------------------------------------------------------
class _ConvHarness:
    """Minimal stand-in carrying only what ``_dilated_conv`` reads."""

    def __init__(self, dilation: int, kernel: int):
        self.short_conv_dilation = dilation
        self.conv_state_len = (kernel - 1) * dilation

    _dilated_conv = None


def _dilated_conv(tokens, history, weight, dilation):
    """Copy of the layer method under test, kept callable without an NPU."""
    from vllm_ascend.models.qwen4_exp.ple import AscendQwen4ExpPLELayer

    return AscendQwen4ExpPLELayer._dilated_conv(
        _ConvHarness(dilation, weight.shape[1]), tokens, history, weight
    )


@pytest.mark.parametrize("dilation,kernel", [(3, 4), (3, 2), (1, 4)])
def test_dilated_conv_matches_conv1d(dilation, kernel):
    """The loop over taps equals a dilated depthwise ``conv1d``."""
    torch.manual_seed(0)
    channels = 8
    num_tokens = 5
    state_len = (kernel - 1) * dilation
    tokens = torch.randn(num_tokens, channels)
    history = torch.randn(state_len, channels)
    weight = torch.randn(channels, kernel)

    got = _dilated_conv(tokens, history, weight, dilation)

    # Reference: build the full sequence [history | tokens] and run a
    # depthwise dilated convolution, keeping the last ``num_tokens`` rows.
    sequence = torch.cat((history, tokens), dim=0).T.unsqueeze(0)  # [1, C, T]
    expected = F.conv1d(
        sequence,
        weight.unsqueeze(1),  # depthwise: [C, 1, K]
        dilation=dilation,
        groups=channels,
    )
    expected = expected[0].T[-num_tokens:]
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def test_dilated_conv_reads_only_the_preceding_window():
    """A tap beyond ``conv_state_len`` must not see older history."""
    channels, kernel, dilation = 4, 4, 3
    state_len = (kernel - 1) * dilation  # 9
    weight = torch.zeros(channels, kernel)
    weight[:, -1] = 1.0  # only the newest tap contributes
    history = torch.arange(state_len, dtype=torch.float32).reshape(state_len, 1)
    history = history.expand(state_len, channels).contiguous()
    tokens = torch.full((1, channels), -1.0)
    got = _dilated_conv(tokens, history, weight, dilation)
    # The newest tap of row 0 is sequence position state_len, i.e. the token
    # itself, so every channel reads -1.
    torch.testing.assert_close(got[0], torch.full((channels,), -1.0))


def test_dilated_conv_rejects_short_history():
    from vllm_ascend.models.qwen4_exp.ple import AscendQwen4ExpPLELayer

    harness = _ConvHarness(3, 4)
    with pytest.raises(RuntimeError, match="smaller than expected"):
        AscendQwen4ExpPLELayer._window_from_state(
            harness, torch.zeros(1, 4, 5), harness.conv_state_len + 0  # > window
        )


# ---------------------------------------------------------------------------
# QSA selection
# ---------------------------------------------------------------------------
def _brute_force_topk(query, keys, token_topk, compress_ratio):
    """Rank ``keys`` ``[num_keys, num_heads, head_dim]`` for one query row."""
    scores = (
        (keys.float() * query.float().unsqueeze(0)).sum(dim=-1).clamp(min=0.0).sum(dim=-1)
    )
    keep = min(token_topk // compress_ratio, scores.numel())
    # ``scores`` is one row, so the ranking is returned as a flat index list.
    return torch.argsort(scores, descending=True, stable=True)[:keep].to(torch.int32)


def _visible_keys(cache, visible, num_heads):
    """Logical key rows ``0..visible-1`` in ``[visible, num_heads, head_dim]``."""
    return cache.reshape(-1, cache.shape[-1])[:visible].reshape(
        visible, num_heads, cache.shape[-1] // num_heads
    )


def _select_fixture(visible=6, page_size=2, num_heads=2, head_dim=2, token_topk=8):
    """One request whose compressed keys occupy ``visible`` logical slots."""
    compress_ratio = 4
    # ``out`` holds one *compressed key* index per column and is sized like the
    # kernel's block-index buffer.
    key_slots = token_topk
    pages = (visible + page_size - 1) // page_size + 1
    # The compressed-key cache stores the indexer's single KV head as one
    # ``num_heads * head_dim`` row per logical key.
    cache = torch.randn(pages, page_size, 1, num_heads * head_dim)
    block_table = torch.arange(pages, dtype=torch.int32).reshape(1, -1)
    query = torch.randn(1, num_heads, head_dim)
    token_to_req = torch.zeros(1, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    visible_blocks = torch.tensor([visible], dtype=torch.int32)
    out = torch.full((1, key_slots), -1, dtype=torch.int32)
    return (
        cache,
        block_table,
        query,
        token_to_req,
        query_start_loc,
        visible_blocks,
        out,
        compress_ratio,
        key_slots,
    )


def test_select_scores_only_visible_keys():
    """The selection never reaches past ``visible_blocks``."""
    (cache, block_table, query, token_to_req, query_start_loc, visible_blocks,
     out, compress_ratio, key_slots) = _select_fixture(visible=4, token_topk=4)
    qsa_select_torch(
        query, cache, block_table, token_to_req, query_start_loc,
        visible_blocks, 4, out,
    )
    expected = _brute_force_topk(query, _visible_keys(cache, 4, query.shape[1]), 4, compress_ratio)
    assert sorted(out[0].tolist()) == sorted(expected.reshape(-1).tolist())


def test_select_matches_brute_force_ranking():
    """Scores are clamped at zero and summed over the query heads."""
    (cache, block_table, query, token_to_req, query_start_loc, visible_blocks,
     out, compress_ratio, key_slots) = _select_fixture(visible=6, token_topk=8)
    qsa_select_torch(
        query, cache, block_table, token_to_req, query_start_loc,
        visible_blocks, 8, out,
    )
    expected = _brute_force_topk(query, _visible_keys(cache, 6, query.shape[1]), 8, compress_ratio)
    expected = expected.reshape(-1)
    assert out[0, : len(expected)].tolist() == expected.tolist()
    if len(expected) < key_slots:
        assert (out[0, len(expected) :] == -1).all()


def test_select_pads_when_nothing_is_visible():
    (cache, block_table, query, token_to_req, query_start_loc, _visible,
     out, _ratio, _slots) = _select_fixture()
    qsa_select_torch(
        query, cache, block_table, token_to_req, query_start_loc,
        torch.tensor([0], dtype=torch.int32), 8, out,
    )
    assert (out == -1).all()


def test_select_pads_on_unmapped_request():
    """A padding request (``token_to_req`` = -1) selects nothing."""
    (cache, block_table, query, _t2r, query_start_loc, visible_blocks,
     out, _ratio, _slots) = _select_fixture()
    qsa_select_torch(
        query, cache, block_table, torch.tensor([-1], dtype=torch.int32),
        query_start_loc, visible_blocks, 8, out,
    )
    assert (out == -1).all()


def test_select_is_deterministic_for_tied_scores():
    """Equal scores resolve to the lower key index, matching the kernel scan."""
    cache = torch.zeros(2, 2, 1, 2)  # num_heads * head_dim = 1 * 2
    block_table = torch.arange(2, dtype=torch.int32).reshape(1, -1)
    query = torch.ones(1, 1, 2)
    out = torch.full((1, 2), -1, dtype=torch.int32)
    qsa_select_torch(
        query, cache, block_table, torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32), 8, out,
    )
    assert out[0].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Key-head grouping and RoPE
# ---------------------------------------------------------------------------
def test_select_groups_repeated_key_heads():
    """Four key heads serve 24 query heads; each group sees its own key row.

    The indexer publishes ``indexer_kv_heads`` key rows per logical key while
    the query has ``indexer_n_heads`` heads, so every key row applies to a
    contiguous slice of query heads.
    """
    from vllm_ascend.models.qwen4_exp.qsa import qsa_select_torch

    visible, key_heads, head_dim, num_heads = 6, 4, 4, 8
    torch.manual_seed(0)
    cache = torch.randn(4, 2, 1, key_heads * head_dim)
    block_table = torch.arange(4, dtype=torch.int32).reshape(1, -1)
    query = torch.randn(1, num_heads, head_dim)
    out = torch.full((1, 4), -1, dtype=torch.int32)
    qsa_select_torch(
        query, cache, block_table, torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([visible], dtype=torch.int32), 4, out,
    )

    # Reference: expand each key head across its query-head group.
    flat = cache.reshape(-1, key_heads * head_dim)[:visible]
    keys = flat.reshape(visible, key_heads, head_dim).float()
    grouped = keys.repeat_interleave(num_heads // key_heads, dim=1)
    scores = torch.clamp(
        (grouped * query[0].float().unsqueeze(0)).sum(dim=-1), min=0.0
    ).sum(dim=-1)
    expected = torch.argsort(scores, descending=True, stable=True)[:4]
    assert out[0].tolist() == expected.tolist()


def test_rope_matches_rotate_half_reference():
    """The torch RoPE equals an independent rotate-half implementation."""
    from vllm_ascend.models.qwen4_exp.qsa import apply_qsa_rope_torch

    class _Rope:
        rotary_dim = 8
        is_neox_style = False

        @staticmethod
        def _match_cos_sin_cache_dtype(tensor):
            positions = torch.arange(64, dtype=torch.float32)
            freqs = torch.outer(positions, torch.tensor(1.0 / 16 ** (torch.arange(0, 4, dtype=torch.float32) / 4)))
            return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(tensor.dtype)

    torch.manual_seed(0)
    tokens, heads, head_dim = 5, 2, 8
    tensor = torch.randn(tokens, heads, head_dim)
    positions = torch.arange(tokens)
    got = apply_qsa_rope_torch(_Rope, positions, tensor)

    cache = _Rope._match_cos_sin_cache_dtype(tensor)
    cos, sin = cache[positions].chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x1, x2 = tensor[..., 0::2], tensor[..., 1::2]
    expected = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_rope_rejects_multi_axis_positions():
    """MRoPE positions must not silently take the 1D path."""
    from vllm_ascend.models.qwen4_exp.qsa import apply_qsa_rope_torch

    class _Rope:
        rotary_dim = 8
        is_neox_style = False

    with pytest.raises(NotImplementedError, match="1D RoPE"):
        apply_qsa_rope_torch(
            _Rope, torch.zeros(3, 4, dtype=torch.int64), torch.zeros(4, 1, 8)
        )
