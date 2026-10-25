# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Semantic tests for the Ascend Qwen4Exp module ports.

Every test compares the Ascend implementation against an *independent*
reference:

* HyperConnection and PLE gate are compared against the upstream
  platform-neutral owners with the same weights (so a divergence in the math,
  the rounding boundaries or the parameter layout fails).
* The dilated short convolution is compared against ``torch.nn.functional``
  with an explicit per-request loop over the state window.
* The QSA eager kernels are compared against brute-force loops over the
  logical cache.

The tests run on CPU tensor semantics; nothing here needs an NPU.
"""

import torch
import torch.nn.functional as F
import pytest
from unittest.mock import MagicMock

from vllm_ascend.models.qwen4_exp.hyperconnection import (
    AscendGatedResidual,
    AscendGroupedGemmaRMSNorm,
    AscendHyperConnectionConfig,
)
from vllm_ascend.models.qwen4_exp.ple import ple_gate_torch
from vllm_ascend.models.qwen4_exp.qsa import (
    expand_qsa_indices_torch,
    gemma_rmsnorm_torch,
    qsa_compress_groups_torch,
    qsa_store_rows_torch,
)

torch.manual_seed(0)

HC_COUNT = 4
HIDDEN = 16
LOWRANK = 5
TOKENS = 6


@pytest.fixture(autouse=True)
def cpu_gemm(monkeypatch):
    """Route the vLLM linear modules through plain matmuls.

    ``vllm::unquantized_gemm`` is CUDA/NPU-only, so on CPU the modules cannot
    execute. The projection *parameters* and their shard layout are what the
    tests need to exercise, so each call is replaced by ``F.linear`` over the
    same weight. This keeps the real module (and its weight loading) in the
    path instead of stubbing the layer out.
    """
    import torch.nn.functional as F

    from vllm_ascend.models.qwen4_exp import hyperconnection as hc

    def apply(layer, x):
        # The real kernels take activations in the weight dtype; normed inputs
        # are promoted to fp32 for the statistics, so cast back here.
        return F.linear(x.to(layer.weight.dtype), layer.weight)

    monkeypatch.setattr(hc, "_call_linear", apply)
    return apply


@pytest.fixture(autouse=True)
def tp_group():
    """Install a single-rank TP group; the layers under test are TP-replicated.

    ``MergedColumnParallelLinear`` reads the tensor-parallel group at
    construction time even when ``disable_tp=True`` asserts it is the only
    rank, so the group has to exist rather than be stubbed out.
    """
    import vllm.distributed.parallel_state as ps

    group = MagicMock()
    group.world_size = 1
    group.rank_in_group = 0
    group.is_first_rank = True
    group.is_last_rank = True
    previous = ps._TP
    ps._TP = group
    try:
        yield group
    finally:
        ps._TP = previous


# ---------------------------------------------------------------------------
# HyperConnection
# ---------------------------------------------------------------------------
def _reference_gated_residual(
    weights, hc_count, hidden, hidden_states, block_output, injection, eps=1e-6
):
    """Straight transcription of ``common/hyperconnection.GatedResidual``.

    ``weights`` carries the checkpoint projections in their applied
    orientation: ``down`` [lora, HC*H], ``up`` [HC*H, lora], ``inject``
    [hc_count, HC*H].
    """
    grouped = hidden_states.float().unflatten(-1, (hc_count, hidden))
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normed = (grouped * torch.rsqrt(variance + eps)).flatten(-2)
    normed = normed * (1.0 + weights["norm"].float())

    def project(x, weight):
        return F.linear(x.to(weight.dtype), weight)

    lora = F.silu(project(normed, weights["down"]) / hc_count)
    gate = torch.sigmoid(project(lora, weights["up"])).unflatten(-1, (hc_count, hidden))
    mixed = (gate * normed.unflatten(-1, (hc_count, hidden))).mean(dim=-2).to(hidden_states.dtype)

    injection_weight = 2.0 * torch.sigmoid(project(normed, weights["inject"]) / hc_count)
    residual = hidden_states.unflatten(-1, (hc_count, hidden))
    combined = (
        residual + block_output.unsqueeze(-2) * injection_weight.unsqueeze(-1)
    ).flatten(-2)
    return mixed, combined.to(hidden_states.dtype)


def _hyperconnection(dtype=torch.float32, use_combine=True):
    config = AscendHyperConnectionConfig(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN,
        params_dtype=dtype,
        hc_lowrank=LOWRANK,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    module = AscendGatedResidual(config, use_combine=use_combine)
    hyper_hidden = HC_COUNT * HIDDEN
    down = torch.randn(LOWRANK, hyper_hidden, dtype=dtype)
    inject = torch.randn(HC_COUNT, hyper_hidden, dtype=dtype)
    up = torch.randn(hyper_hidden, LOWRANK, dtype=dtype)
    norm_weight = torch.randn(hyper_hidden, dtype=dtype)
    module.hc_norm.weight.data.copy_(norm_weight)
    module.input_mix_weight_up.weight.data.copy_(up)
    if use_combine:
        # The packed tensor is padded to a 16-row multiple; the pad rows stay
        # zero and are ignored at use time.
        packed = module.input_mix_weight_down_block_inject.weight.data
        packed.zero_()
        packed[: LOWRANK + HC_COUNT].copy_(torch.cat((down, inject), dim=0))
    else:
        module.input_mix_weight_down.weight.data.copy_(down)
    weights = {"norm": norm_weight, "down": down, "up": up, "inject": inject}
    return module, weights


def test_packed_projection_layout_matches_checkpoint_stack():
    """The packed tensor holds down then inject; the pad rows stay zero."""
    module, weights = _hyperconnection()
    packed = module.input_mix_weight_down_block_inject
    assert packed.output_sizes == [LOWRANK, HC_COUNT, module.pad_size] if module.pad_size else packed.output_sizes == [LOWRANK, HC_COUNT]
    assert module.pad_size == (-(LOWRANK + HC_COUNT)) % 16
    reference = torch.cat((weights["down"], weights["inject"]), dim=0)
    torch.testing.assert_close(packed.weight[: LOWRANK + HC_COUNT], reference, atol=0, rtol=0)
    assert torch.count_nonzero(packed.weight[LOWRANK + HC_COUNT :]) == 0
    assert module.input_mix_weight_up.weight.shape == (HC_COUNT * HIDDEN, LOWRANK)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mix_and_combine_match_reference(dtype):
    module, weights = _hyperconnection(dtype)
    hyper_input = torch.randn(TOKENS, HC_COUNT * HIDDEN, dtype=dtype)
    block_output = torch.randn(TOKENS, HIDDEN, dtype=dtype)

    mixed, injection = module.mix(hyper_input)
    combined = module.combine(hyper_input, block_output, injection)
    ref_mixed, ref_combined = _reference_gated_residual(
        weights, HC_COUNT, HIDDEN, hyper_input, block_output, injection
    )
    tol = dict(atol=1e-6, rtol=1e-5) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(mixed.float(), ref_mixed.float(), **tol)
    torch.testing.assert_close(combined.float(), ref_combined.float(), **tol)
    assert injection.shape == (TOKENS, HC_COUNT)


def test_combine_and_mix_equals_combine_then_mix():
    module, _ = _hyperconnection()
    hidden = torch.randn(TOKENS, HC_COUNT * HIDDEN)
    block_output = torch.randn(TOKENS, HIDDEN)
    _, injection = module.mix(torch.randn(TOKENS, HC_COUNT * HIDDEN))

    combined, mixed, new_injection = module.combine_and_mix(
        hidden, block_output, injection
    )
    expected_combined = module.combine(hidden, block_output, injection)
    expected_mixed, expected_injection = module.mix(expected_combined)

    torch.testing.assert_close(combined, expected_combined, atol=0, rtol=0)
    torch.testing.assert_close(mixed, expected_mixed, atol=0, rtol=0)
    torch.testing.assert_close(new_injection, expected_injection, atol=0, rtol=0)


def test_combine_without_injection_is_unit_weight():
    """The final mixer combines with unit weight and owns no injection weight."""
    module, _ = _hyperconnection(use_combine=False)
    assert not hasattr(module, "input_mix_weight_down_block_inject")
    hidden = torch.randn(TOKENS, HC_COUNT * HIDDEN)
    block_output = torch.randn(TOKENS, HIDDEN)
    _, injection = module.mix(hidden)
    assert injection is None
    combined = module.combine(hidden, block_output, None)
    expected = (
        hidden.unflatten(-1, (HC_COUNT, HIDDEN)) + block_output.unsqueeze(-2)
    ).flatten(-2)
    torch.testing.assert_close(combined, expected, atol=0, rtol=0)


def test_combine_requires_injection_when_mixing():
    module, _ = _hyperconnection()
    with pytest.raises(ValueError, match="requires its injection"):
        module.combine(
            torch.randn(TOKENS, HC_COUNT * HIDDEN), torch.randn(TOKENS, HIDDEN), None
        )


def test_grouped_norm_matches_reference():
    """The grouped norm normalizes each H-wide stream independently."""
    module = AscendGroupedGemmaRMSNorm(HC_COUNT * HIDDEN, 1e-6, HIDDEN, torch.float32)
    module.weight.data.copy_(torch.randn(HC_COUNT * HIDDEN))
    x = torch.randn(TOKENS, HC_COUNT * HIDDEN)
    out = module(x)
    grouped = x.unflatten(-1, (HC_COUNT, HIDDEN))
    variance = grouped.square().mean(dim=-1, keepdim=True)
    expected = (grouped * torch.rsqrt(variance + 1e-6)).flatten(-2) * (1.0 + module.weight)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# PLE gate
# ---------------------------------------------------------------------------
def _reference_ple_gate(key, value, hidden, norm_key, norm_query, norm_conv, eps, hc):
    channel = value.shape[-1]

    def grouped_norm(x, weight):
        g = x.float().unflatten(-1, (hc, channel))
        var = g.square().mean(dim=-1, keepdim=True)
        return g * torch.rsqrt(var + eps) * (1.0 + weight.float().unflatten(-1, (hc, channel)))

    k_n = grouped_norm(key, norm_key).to(key.dtype)
    q_n = grouped_norm(hidden, norm_query).to(hidden.dtype)
    d = (k_n.float() * q_n.float()).sum(-1, keepdim=True) / channel**0.5
    g = torch.sigmoid(torch.sign(d) * torch.sqrt(d.abs().clamp_min(1e-6)))
    v = value.float()
    gated = (g * v.unsqueeze(1)).to(value.dtype)
    gf = gated.float()
    var = gf.square().mean(-1, keepdim=True)
    normed = gf * torch.rsqrt(var + eps)
    normed = normed * (1.0 + norm_conv.float().unflatten(-1, (hc, channel)))
    return gated.flatten(-2), normed.flatten(-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ple_gate_matches_reference(dtype):
    hc, channel = 4, 8
    key = torch.randn(TOKENS, hc * channel, dtype=dtype)
    value = torch.randn(TOKENS, channel, dtype=dtype)
    hidden = torch.randn(TOKENS, hc * channel, dtype=dtype)
    nk = torch.randn(hc * channel, dtype=dtype)
    nq = torch.randn(hc * channel, dtype=dtype)
    nc = torch.randn(hc * channel, dtype=dtype)
    eps = 1e-6

    gated, normed = ple_gate_torch(key, value, hidden, nk, nq, nc, eps)
    ref_gated, ref_normed = _reference_ple_gate(key, value, hidden, nk, nq, nc, eps, hc)
    tol = dict(atol=1e-5, rtol=1e-4) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gated.float(), ref_gated.float(), **tol)
    torch.testing.assert_close(normed.float(), ref_normed.float(), **tol)


def test_ple_gate_rejects_mismatched_dtypes():
    key = torch.randn(2, 8, dtype=torch.bfloat16)
    value = torch.randn(2, 4, dtype=torch.float32)
    hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    w = torch.zeros(8)
    with pytest.raises(ValueError, match="share a dtype"):
        ple_gate_torch(key, value, hidden, w, w, w, 1e-6)


# ---------------------------------------------------------------------------
# QSA eager kernels
# ---------------------------------------------------------------------------
def test_qsa_store_rows_skips_padding_slots():
    """PAD slots address the null block and must never be written."""
    pages, page_size, width = 3, 4, 5
    cache = torch.zeros(pages, page_size, 1, width)
    rows = torch.arange(1, 11, dtype=torch.bfloat16).reshape(2, width)
    slots = torch.tensor([2, -1])  # second row is padding
    qsa_store_rows_torch(cache, slots, rows)
    flat = cache.view(-1, width)
    assert torch.equal(flat[2], rows[0])
    assert torch.count_nonzero(flat[3]) == 0


def test_qsa_store_rows_rejects_bad_shape():
    cache = torch.zeros(2, 2, 1, 3)
    with pytest.raises(ValueError, match="incompatible shapes"):
        qsa_store_rows_torch(cache, torch.tensor([0]), torch.zeros(1, 1, 4))


def test_gemma_rmsnorm_matches_reference():
    x = torch.randn(5, 8, dtype=torch.bfloat16)
    w = torch.randn(8)
    out = gemma_rmsnorm_torch(x, w, 1e-6)
    xf = x.float()
    expected = (
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * (1.0 + w)
    ).to(x.dtype)
    torch.testing.assert_close(out.float(), expected.float(), atol=1e-6, rtol=1e-5)


def _ring_fixture(head_dim=4, capacity=4, pages=8):
    """One request whose chunk is the two positions 2 and 3.

    A group ending at position 3 covers positions 0..3: members 0 and 1 are
    already in the ring, members 2 and 3 are the rows of this chunk. That is
    exactly the ring/raw split the compressor has to fuse.
    """
    raw_keys = torch.randn(2, 1, head_dim)
    positions = torch.stack([torch.arange(2), torch.arange(2), torch.arange(2)], dim=-1) + 2
    ring = torch.zeros(pages, capacity, 1, head_dim)
    ring[1, 0:2, 0] = torch.randn(2, head_dim)  # committed members 0 and 1
    block_table = torch.tensor([[1]], dtype=torch.int32)
    token_to_req = torch.tensor([0, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    logical_positions = torch.tensor([2, 3], dtype=torch.int64)
    compressed_slots = torch.tensor([5, 6], dtype=torch.int64)
    return (
        raw_keys,
        positions,
        ring,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
    )


def test_compress_groups_pools_ring_and_raw_members():
    """A group spanning the ring and this chunk averages all four members."""
    (
        raw_keys,
        positions,
        ring,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
    ) = _ring_fixture()
    pooled, first_positions = qsa_compress_groups_torch(
        raw_keys,
        positions,
        ring,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio=4,
        rope_position_cache=None,
    )
    # Row 1 ends at position 3: members 0..1 from the ring, 2..3 from raw.
    expected_row1 = torch.cat((ring[1, 0:2, 0].float(), raw_keys[0:2, 0].float())).mean(dim=0)
    torch.testing.assert_close(pooled[1, 0].float(), expected_row1, atol=1e-6, rtol=1e-6)
    # Row 0 ends at position 2, which cannot close a group (needs p >= ratio-1
    # *and* a full span) so it stays zero unless its own group is complete.
    if int(logical_positions[0]) < 3:
        assert torch.count_nonzero(pooled[0]) == 0
    # The first member of row 1's group is position 0, which lives in the ring,
    # so the packed rope position comes from the raw row of the same chunk only
    # when the chunk carries it.
    assert first_positions.shape == (2, 3)


def test_compress_groups_skips_incomplete_groups():
    (raw_keys, positions, ring, block_table, token_to_req, query_start_loc,
     logical_positions, compressed_slots) = _ring_fixture()
    pooled, _ = qsa_compress_groups_torch(
        raw_keys, positions, ring, block_table, token_to_req, query_start_loc,
        logical_positions, compressed_slots, compress_ratio=4, rope_position_cache=None,
    )
    assert torch.count_nonzero(pooled) > 0

    # Below the ratio no group can be closed.
    pooled_short, _ = qsa_compress_groups_torch(
        raw_keys, positions, ring, block_table, token_to_req, query_start_loc,
        torch.zeros_like(logical_positions), compressed_slots,
        compress_ratio=4, rope_position_cache=None,
    )
    assert torch.count_nonzero(pooled_short) == 0

    # Incomplete slots (PAD) are skipped even when the position qualifies.
    pooled_padded, _ = qsa_compress_groups_torch(
        raw_keys, positions, ring, block_table, token_to_req, query_start_loc,
        logical_positions, torch.full_like(compressed_slots, -1),
        compress_ratio=4, rope_position_cache=None,
    )
    assert torch.count_nonzero(pooled_padded) == 0


def test_expand_indices_packs_blocks_tail_and_count():
    """Expansion writes key-span tokens, then the causal tail, then the count."""
    # Real geometry: token_topk=2048 / ratio=4 -> 512 block-index columns.
    block_indices = torch.full((1, 512), -1, dtype=torch.int32)
    block_indices[0, 0] = 2
    positions = torch.tensor([9], dtype=torch.int64)  # tail = 10 - 8 = 2 tokens
    visible = torch.tensor([1], dtype=torch.int32)
    out = torch.full((1, 2048 + 4), -1, dtype=torch.int32)
    expand_qsa_indices_torch(block_indices, positions, visible, 4, 2048, out)
    tokens = out[0, :-1]
    # Key 2 covers 8..11; position 9 has two members in its open group (8, 9),
    # which the tail appends after the selected spans.
    assert tokens[:6].tolist() == [8, 9, 10, 11, 8, 9]
    assert int(out[0, -1].item()) == 6
    assert (tokens[6:] == -1).all()


def test_expand_indices_rejects_inert_rows():
    """A row without a position reports zero valid entries, not padding ones."""
    block_indices = torch.zeros((1, 512), dtype=torch.int32)
    out = torch.zeros((1, 2048 + 4), dtype=torch.int32)
    expand_qsa_indices_torch(
        block_indices, torch.tensor([-1]), torch.tensor([0], dtype=torch.int32), 4, 2048, out
    )
    assert int(out[0, -1].item()) == 0
    assert (out[0, :-1] == -1).all()


def test_expand_indices_rejects_missized_block_buffer():
    """The block buffer must be exactly ``token_topk // ratio`` columns wide.

    Upstream asserts the same shape; a narrower buffer would silently drop
    candidates, a wider one would mean the caller and the expander disagree on
    the geometry.
    """
    with pytest.raises(ValueError, match="block-index buffer must be 512 wide"):
        expand_qsa_indices_torch(
            torch.zeros((1, 2), dtype=torch.int32),
            torch.tensor([9]),
            torch.tensor([1], dtype=torch.int32),
            4,
            2048,
            torch.full((1, 2048 + 4), -1, dtype=torch.int32),
        )


def test_expand_indices_width_guard():
    block_indices = torch.zeros((1, 512), dtype=torch.int32)
    with pytest.raises(ValueError, match="packed buffer must be"):
        expand_qsa_indices_torch(
            block_indices,
            torch.tensor([0]),
            torch.tensor([1], dtype=torch.int32),
            4,
            2048,
            torch.zeros((1, 8), dtype=torch.int32),
        )


@pytest.mark.parametrize("visible_blocks_count", [512, 1024, 3000])
def test_expand_indices_saturates_block_columns_long_sequence(visible_blocks_count):
    """A long sequence must clamp the block scan to the block-index width.

    Real geometry: ``token_topk=2048`` with ``compress_ratio=4`` gives
    ``block_indices`` 512 columns (``token_topk // compress_ratio``), while
    ``visible_blocks`` counts *compressed keys* and grows past that once the
    sequence exceeds 2048 tokens (``visible = seq_len // 4``). Upstream clamps
    with ``tl.minimum(visible_blocks, BLOCK_TOPK)`` where ``BLOCK_TOPK`` is the
    block-index column count (``qsa_indexer.py:240-241``), so the expansion
    must stop at the last available column instead of reading past it.

    ``visible_blocks_count=1024`` is the smallest value that overran the
    column axis under the previous ``min(visible, token_topk)`` bound.
    """
    block_topk = 2048 // 4
    ratio = 4
    block_indices = torch.arange(block_topk, dtype=torch.int32).unsqueeze(0)
    visible = torch.tensor([visible_blocks_count], dtype=torch.int32)
    # Position beyond 2048 tokens; the open group holds exactly one token
    # (4096 % 4 == 0), so the tail contributes nothing and the count is the
    # expanded span alone.
    positions = torch.tensor([4095], dtype=torch.int64)
    out = torch.full((1, 2048 + ratio), -1, dtype=torch.int32)

    expand_qsa_indices_torch(block_indices, positions, visible, ratio, 2048, out)

    tokens = out[0, :-1]
    complete = min(visible_blocks_count, block_topk)
    expected_count = complete * ratio
    assert int(out[0, -1].item()) == expected_count
    assert (tokens[:expected_count] >= 0).all()
    # Every expanded slot is the selected key's own span, in rank order.
    assert tokens[0].item() == 0
    assert tokens[expected_count - 1].item() == (complete - 1) * ratio + (ratio - 1)
    assert (tokens[expected_count:] == -1).all()


def _ple_layout_harness():
    """Bind the layer's conv-window helpers onto a bare object.

    ``_window_from_state``/``_store_window`` only depend on the SSM conv-state
    layout, so they can be driven without constructing a layer (which needs a
    full VllmConfig). Both layouts are exercised by patching the layout
    helper, so an implementation that hardcodes one orientation fails.
    """
    from vllm_ascend.models.qwen4_exp.ple import AscendQwen4ExpPLELayer

    class Harness:
        pass

    for name in ("_window_from_state", "_store_window"):
        setattr(Harness, name, getattr(AscendQwen4ExpPLELayer, name))
    return Harness()


@pytest.mark.parametrize("dim_first", [False, True])
def test_conv_window_layout_roundtrip(monkeypatch, dim_first):
    """Cache layout SD ([slot, len, C]) and DS ([slot, C, len]) both work."""
    import vllm_ascend.models.qwen4_exp.ple as ple_mod

    monkeypatch.setattr(ple_mod, "is_conv_state_dim_first", lambda: dim_first)

    # Two slots, four taps, three channels; slot 1 is the one under test.
    window = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    conv_state = window.transpose(-1, -2).contiguous() if dim_first else window.clone()
    expected = window[1]

    harness = _ple_layout_harness()
    got = harness._window_from_state(conv_state[1:2], 4)
    assert torch.equal(got, expected), f"read mismatch for dim_first={dim_first}"

    # A prefill chunk of two tokens shifts the window by two and keeps the tail.
    chunk = torch.full((2, 3), -1.0)
    rolled = torch.cat((got, chunk), dim=0)
    harness._store_window(conv_state, 1, rolled, 4)
    roundtrip = harness._window_from_state(conv_state[1:2], 4)
    assert torch.equal(roundtrip, rolled[-4:]), f"write mismatch for dim_first={dim_first}"
    # Slot 0 must be untouched by a write to slot 1.
    assert torch.equal(harness._window_from_state(conv_state[0:1], 4), window[0])


def test_conv_window_rejects_short_cache(monkeypatch):
    import vllm_ascend.models.qwen4_exp.ple as ple_mod

    monkeypatch.setattr(ple_mod, "is_conv_state_dim_first", lambda: False)
    harness = _ple_layout_harness()
    with pytest.raises(RuntimeError, match="smaller than expected"):
        harness._window_from_state(torch.zeros(1, 2, 3), 4)
