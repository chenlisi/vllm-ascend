# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Speculative-decode (MTP) path of the Ascend PLE dilated short convolution.

The reference is the upstream kernel contract written out row by row
(``_simulate_spec_step``): output row ``j`` reads taps ``j + k * dilation``
from the state while they fall inside ``conv_state_len`` and from the chunk
afterwards, the write-back shifts the live window by one and appends the whole
chunk, and the next step reads that window from ``num_accepted - 1``. The
cases below pin the read window, the acceptance rollback, the unused-tail
preservation and the mixed-batch row mapping.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

MODEL_PREFIX = "model.layers.2"

# Small but structurally faithful geometry: a 4-tap kernel dilated by 3 gives
# a 9-token window, which is wider than any chunk used below.
CHANNELS = 4
KERNEL = 4
DILATION = 3
STATE_LEN = (KERNEL - 1) * DILATION


def _metadata(**kwargs):
    """A real ``PleShortConvAttentionMetadata`` (dataclass, not a stub)."""
    from vllm.v1.attention.backends.short_conv_attn import (
        PleShortConvAttentionMetadata,
    )

    values = {}
    for field in dataclasses.fields(PleShortConvAttentionMetadata):
        if (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        ):
            continue
        values[field.name] = torch.zeros(1, dtype=torch.int32)
    values.setdefault("num_actual_tokens", 0)
    values.update(kwargs)
    return PleShortConvAttentionMetadata(**values)


class Harness:
    """Layer methods bound onto a bare object with the geometry attributes."""

    def __init__(self, conv_state, num_spec_tokens, dtype=torch.float32):
        from types import MethodType

        from vllm_ascend.models.qwen4_exp.ple import AscendQwen4ExpPLELayer

        for name in (
            "_window_from_state",
            "_store_window",
            "_dilated_conv",
            "_slice_history",
            "_short_conv_spec",
            "_row_map",
            "_short_conv_eager",
            "_short_conv",
        ):
            setattr(self, name, MethodType(getattr(AscendQwen4ExpPLELayer, name), self))
        self.prefix = f"{MODEL_PREFIX}.ple"
        self.conv_state_len = STATE_LEN
        self.short_conv_dilation = DILATION
        self.num_spec_tokens = num_spec_tokens
        self.kv_cache = (conv_state,)
        self._conv_weight_2d = torch.arange(
            CHANNELS * KERNEL, dtype=torch.float32
        ).reshape(CHANNELS, KERNEL) / (CHANNELS * KERNEL)
        self.dtype = dtype


def _conv_weight(dtype):
    weight = torch.arange(CHANNELS * KERNEL, dtype=torch.float32).reshape(
        CHANNELS, KERNEL
    )
    return (weight / (CHANNELS * KERNEL)).to(dtype)


def _reference_rows(stream, chunk, dtype=torch.float32):
    """Pre-activation conv for every row of ``chunk`` given its stream."""
    weight = _conv_weight(torch.float32)
    history = torch.zeros(STATE_LEN, CHANNELS, dtype=torch.float32)
    tail = stream[-STATE_LEN:] if stream.shape[0] else stream
    if tail.shape[0]:
        history[STATE_LEN - tail.shape[0] :] = tail
    seq = torch.cat((history, chunk.to(torch.float32)), dim=0)
    out = torch.zeros(chunk.shape[0], CHANNELS, dtype=torch.float32)
    for j in range(chunk.shape[0]):
        for k in range(KERNEL):
            out[j] += weight[:, k] * seq[j + k * DILATION]
    return out.to(dtype)


def _reference_outputs(stream, chunk, residual, outer, dtype=torch.float32):
    """Mirror of the eager post-processing order (SiLU -> residual -> outer)."""
    conv = _reference_rows(stream, chunk, dtype).to(dtype).float()
    y = conv * torch.sigmoid(conv)
    ple_output = (residual + y.to(dtype)).to(dtype)
    return (outer.float() + ple_output.float()).to(dtype)


def _fresh_state(num_slots, num_spec_tokens, dtype=torch.float32):
    return torch.zeros(num_slots, STATE_LEN + num_spec_tokens, CHANNELS, dtype=dtype)


def _seed_stream(conv_state, slot, num_spec_tokens, stream, dtype=torch.float32):
    """Seed a plain (non-spec) slot: the live window starts at offset 0.

    The reserved spec taps after ``STATE_LEN`` are not part of the window, so
    they are zeroed rather than filled with history.
    """
    capacity = STATE_LEN + num_spec_tokens
    conv_state[slot].zero_()
    live = stream[-STATE_LEN:] if stream.shape[0] else stream
    if live.shape[0]:
        conv_state[slot, : live.shape[0], :] = live.to(dtype)
    assert capacity == conv_state.shape[1]


POISON = 1e4


def _seed_accepted(conv_state, slot, num_spec_tokens, stream, accepted, dtype=torch.float32):
    """Seed a slot whose last real token sits where ``accepted`` says it does.

    A spec step reads ``window[slot_off : slot_off + STATE_LEN]`` with
    ``slot_off = accepted - 1``; everything outside that range is stale draft
    state and gets a poison value, so a read that ignores the rollback offset
    cannot pass.
    """
    capacity = STATE_LEN + num_spec_tokens
    slot_off = min(max(accepted - 1, 0), num_spec_tokens)
    live = stream[-STATE_LEN:] if stream.shape[0] else stream
    if live.shape[0] < STATE_LEN:
        live = torch.cat(
            (torch.zeros(STATE_LEN - live.shape[0], live.shape[1], dtype=live.dtype), live),
            dim=0,
        )
    conv_state[slot].fill_(POISON)
    conv_state[slot, slot_off : slot_off + live.shape[0], :] = live.to(dtype)
    # The rollback read window is ``slot_off .. slot_off + STATE_LEN - 1``; it
    # must stay inside the slot.
    assert slot_off + live.shape[0] <= capacity
    return live


def _drive_spec(harness, conv_state, chunk_rows, num_accepted, slot, qlen, num_rows):
    """Run one spec step for a single request and return the batch residual."""
    capacity = STATE_LEN + harness.num_spec_tokens
    inputs = torch.zeros(num_rows, CHANNELS, dtype=harness.dtype)
    inputs[:qlen] = chunk_rows
    residual = torch.zeros(num_rows, CHANNELS, dtype=harness.dtype)
    outer = torch.zeros(num_rows, CHANNELS, dtype=harness.dtype)
    metadata = _metadata(
        num_spec_decodes=1,
        num_actual_tokens=num_rows,
        spec_query_len=harness.num_spec_tokens + 1,
        spec_state_indices_tensor=torch.tensor([slot], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, qlen], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([num_accepted], dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
    )
    harness._short_conv_spec(
        inputs=inputs,
        residual=residual,
        outer_residual=outer,
        metadata=metadata,
        conv_state=conv_state,
        conv_weights=harness._conv_weight_2d,
    )
    assert capacity == conv_state.shape[1]
    return residual


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("accepted", [1, 3])
def test_spec_single_request_writes_back_to_the_accepted_token(dtype, accepted):
    """accepted=1 and accepted=num_spec both roll the window back."""
    num_spec = 3
    slot = 1
    conv_state = _fresh_state(3, num_spec, dtype)
    harness = Harness(conv_state, num_spec, dtype)
    # A prior stream exactly as long as the window the next step reads.
    stream = torch.arange(
        1, STATE_LEN * CHANNELS + 1, dtype=torch.float32
    ).reshape(-1, CHANNELS)
    _seed_accepted(conv_state, slot, num_spec, stream, accepted, dtype)

    chunk = torch.arange(1, num_spec + 2, dtype=torch.float32).reshape(-1, 1)
    chunk = chunk.expand(-1, CHANNELS) * 0.5
    out = _drive_spec(
        harness,
        conv_state,
        chunk,
        num_accepted=accepted,
        slot=slot,
        qlen=num_spec + 1,
        num_rows=num_spec + 1,
    )

    expected = _reference_outputs(stream, chunk, torch.zeros_like(chunk), torch.zeros_like(chunk), dtype)
    assert torch.equal(out[: num_spec + 1], expected), "draft rows must all get PLE output"

    # The live window rolls back to the last accepted token: positions
    # ``slot_off .. slot_off + STATE_LEN - 1`` hold exactly the stream tail
    # that the next step will read, and everything past ``do_write`` is a
    # stale draft tap that must have been left alone.
    capacity = STATE_LEN + num_spec
    slot_off = min(max(accepted - 1, 0), num_spec)
    do_write = min(STATE_LEN + chunk.shape[0] - 1, capacity)
    live = torch.cat((stream, chunk[:accepted].to(dtype)), dim=0)[-STATE_LEN:]
    got = conv_state[slot].to(dtype)
    assert torch.equal(
        got[slot_off : slot_off + STATE_LEN], live
    ), f"rollback window must start at accepted-1 (accepted={accepted})"
    # ``do_write - (STATE_LEN - 1)`` chunk rows were appended after the shift.
    appended = do_write - (STATE_LEN - 1)
    assert torch.equal(got[STATE_LEN - 1 : do_write], chunk[:appended].to(dtype))
    assert torch.equal(
        got[do_write:], torch.full((capacity - do_write, CHANNELS), POISON, dtype=dtype)
    ), "the unused state tail must survive"


def _simulate_spec_step(state, chunk, accepted, num_spec, weight, dtype):
    """Row-by-row model of the upstream ``mode="spec"`` read and write-back.

    Independent of the layer: it evaluates the convolution taps straight from
    the upstream indexing rule (``h = j + k * dilation``; state tap while
    ``h <= STATE_LEN - 1``, otherwise the chunk's own row) and then applies
    the documented write-back, so any divergence is an implementation defect
    rather than a difference between two pieces of test scaffolding.
    """
    slot_off = min(max(accepted - 1, 0), num_spec)
    qlen = chunk.shape[0]
    out = torch.zeros(qlen, CHANNELS, dtype=torch.float32)
    for j in range(qlen):
        acc = torch.zeros(CHANNELS, dtype=torch.float32)
        for k in range(KERNEL):
            h = j + DILATION * k
            if h <= STATE_LEN - 1:
                tap = state[slot_off + h].to(torch.float32)
            else:
                tap = chunk[h - STATE_LEN].to(torch.float32)
            acc += weight[:, k].to(torch.float32) * tap
        out[j] = acc
    conv = out.to(dtype).float()
    y = conv * torch.sigmoid(conv)
    outputs = y.to(dtype)

    capacity = STATE_LEN + num_spec
    do_write = min(STATE_LEN + qlen - 1, capacity)
    new_state = state.clone()
    for i in range(do_write):
        m = 1 + i
        new_state[i] = state[slot_off + m] if m <= STATE_LEN - 1 else chunk[m - STATE_LEN]
    return outputs, new_state


def test_spec_replay_matches_the_upstream_kernel_model():
    """Chained steps agree with an independent model of the spec kernel.

    Acceptance counts below ``SPEC_QUERY_LEN`` leave rejected draft rows in
    the state, and the next step's rollback offset decides which of them are
    read, so the chain only lines up if the offset and the shift agree.
    """
    num_spec = 3
    capacity = STATE_LEN + num_spec
    slot = 0
    conv_state = _fresh_state(1, num_spec)
    harness = Harness(conv_state, num_spec)
    weight = harness._conv_weight_2d
    model_state = torch.zeros(capacity, CHANNELS)

    for step, accepted in enumerate((1, 2, 1, 4)):
        chunk = torch.full((num_spec + 1, CHANNELS), float(step + 1))
        out = _drive_spec(
            harness,
            conv_state,
            chunk,
            num_accepted=accepted,
            slot=slot,
            qlen=num_spec + 1,
            num_rows=num_spec + 1,
        )
        expected, model_state = _simulate_spec_step(
            model_state, chunk, accepted, num_spec, weight, torch.float32
        )
        assert torch.equal(out[: num_spec + 1], expected), f"step {step}"
        assert torch.equal(
            conv_state[slot].to(torch.float32), model_state
        ), f"step {step}: state mismatch"

    assert model_state.shape[0] == capacity


def test_spec_short_chunk_preserves_the_unused_state_tail():
    """A chunk shorter than SPEC_QUERY_LEN must not clear the state tail."""
    num_spec = 3
    slot = 0
    capacity = STATE_LEN + num_spec
    conv_state = _fresh_state(2, num_spec)
    harness = Harness(conv_state, num_spec)
    stream = (
        torch.arange(1, STATE_LEN + 1, dtype=torch.float32)
        .reshape(-1, 1)
        .expand(-1, CHANNELS)
    )
    _seed_accepted(conv_state, slot, num_spec, stream, accepted=2)
    # A previous step already appended ``accepted - 1`` taps of history; the
    # tail it deliberately left alone is what a short chunk must not touch.
    conv_state[slot, capacity - 1 :, :] = 7.0
    before = conv_state[slot].clone()

    chunk = torch.full((2, CHANNELS), 3.0)
    _drive_spec(
        harness,
        conv_state,
        chunk,
        num_accepted=2,
        slot=slot,
        qlen=2,
        num_rows=num_spec + 1,
    )

    slot_off = min(max(2 - 1, 0), num_spec)
    do_write = min(STATE_LEN + 2 - 1, capacity)
    old_len = min(STATE_LEN - 1, do_write)
    # ``new_state[i] = old[slot_off + 1 + i]`` for the state taps, then the
    # chunk: relative to the rolled stream that is a shift by one.
    rolled = torch.cat((stream, chunk), dim=0)[-do_write:]
    assert torch.equal(
        conv_state[slot, :do_write].to(torch.float32), rolled
    ), "the written prefix must be the shifted window plus the chunk"
    assert torch.equal(
        conv_state[slot, do_write:].to(torch.float32), before[do_write:]
    ), "state beyond do_write belongs to a longer chunk and must survive"
    assert old_len == do_write - chunk.shape[0]
    assert capacity - do_write == 2


def test_spec_mixed_batch_keeps_row_order(monkeypatch):
    """Spec rows and non-spec rows share one batch without regrouping."""
    import vllm.forward_context as ctx_mod

    num_spec = 1
    capacity = STATE_LEN + num_spec
    conv_state = _fresh_state(4, num_spec)
    harness = Harness(conv_state, num_spec)

    spec_stream = (
        torch.arange(1, STATE_LEN + 1, dtype=torch.float32)
        .reshape(-1, 1)
        .expand(-1, CHANNELS)
    )
    _seed_accepted(conv_state, 0, num_spec, spec_stream, accepted=num_spec + 1)
    decode_stream = torch.full((STATE_LEN, CHANNELS), -2.0)
    _seed_stream(conv_state, 1, num_spec, decode_stream)
    prefill_stream = torch.full((STATE_LEN, CHANNELS), 5.0)
    _seed_stream(conv_state, 2, num_spec, prefill_stream)

    # Batch order after the runner reorder: spec first, then decode, prefill.
    spec_chunk = torch.full((num_spec + 1, CHANNELS), 0.25)
    decode_row = torch.full((1, CHANNELS), -0.5)
    prefill_rows = torch.full((3, CHANNELS), 1.5)
    inputs = torch.cat((spec_chunk, decode_row, prefill_rows), dim=0)
    residual = torch.zeros_like(inputs)
    outer = torch.zeros_like(inputs)

    metadata = _metadata(
        num_prefills=1,
        num_prefill_tokens=3,
        num_decodes=1,
        num_decode_tokens=1,
        num_reqs=3,
        num_spec_decodes=1,
        num_spec_decode_tokens=num_spec + 1,
        num_actual_tokens=inputs.shape[0],
        spec_query_len=num_spec + 1,
        spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, num_spec + 1], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([num_spec + 1], dtype=torch.int32),
        spec_sequence_masks=torch.tensor([True, False, False]),
        spec_token_indx=torch.arange(num_spec + 1, dtype=torch.int64),
        non_spec_token_indx=torch.arange(
            num_spec + 1, inputs.shape[0], dtype=torch.int64
        ),
        state_indices_tensor=torch.tensor([1, 2], dtype=torch.int32),
        has_initial_states_p=torch.tensor([True]),
        query_start_loc_p=torch.tensor([0, 3], dtype=torch.int32),
    )
    monkeypatch.setattr(
        ctx_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={harness.prefix: metadata}),
    )

    harness._short_conv(inputs.clone(), residual, outer)

    expected_spec = _reference_outputs(
        spec_stream, spec_chunk, torch.zeros_like(spec_chunk), torch.zeros_like(spec_chunk)
    )
    expected_decode = _reference_outputs(
        decode_stream, decode_row, torch.zeros_like(decode_row), torch.zeros_like(decode_row)
    )
    expected_prefill = _reference_outputs(
        prefill_stream, prefill_rows, torch.zeros_like(prefill_rows), torch.zeros_like(prefill_rows)
    )
    assert torch.equal(residual[: num_spec + 1], expected_spec)
    assert torch.equal(residual[num_spec + 1 : num_spec + 2], expected_decode)
    assert torch.equal(residual[num_spec + 2 :], expected_prefill)


def test_spec_null_slot_contributes_nothing():
    """A null (padded) spec row has no slot: no output row, no write."""
    num_spec = 1
    capacity = STATE_LEN + num_spec
    conv_state = _fresh_state(2, num_spec)
    harness = Harness(conv_state, num_spec)
    before = conv_state.clone()

    inputs = torch.ones(num_spec + 1, CHANNELS)
    residual = torch.zeros_like(inputs)
    metadata = _metadata(
        num_spec_decodes=1,
        num_actual_tokens=inputs.shape[0],
        spec_query_len=num_spec + 1,
        spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        # An empty chunk models the graph-padded row.
        spec_query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
    )
    harness._short_conv_spec(
        inputs=inputs,
        residual=residual,
        outer_residual=torch.ones_like(inputs),
        metadata=metadata,
        conv_state=conv_state,
        conv_weights=harness._conv_weight_2d,
    )
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(conv_state, before)


def test_spec_query_len_mismatch_raises():
    """A builder contract drift must not silently mis-size the window."""
    num_spec = 1
    conv_state = _fresh_state(1, num_spec)
    harness = Harness(conv_state, num_spec)
    metadata = _metadata(
        num_spec_decodes=1,
        num_actual_tokens=num_spec + 1,
        spec_query_len=num_spec + 2,
        spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, num_spec + 1], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
    )
    with pytest.raises(RuntimeError, match="spec_query_len"):
        harness._short_conv_spec(
            inputs=torch.zeros(num_spec + 1, CHANNELS),
            residual=torch.zeros(num_spec + 1, CHANNELS),
            outer_residual=torch.zeros(num_spec + 1, CHANNELS),
            metadata=metadata,
            conv_state=conv_state,
            conv_weights=harness._conv_weight_2d,
        )


def test_spec_chunk_wider_than_spec_query_len_raises():
    """A chunk longer than the reserved window is a hard error, not a clamp."""
    num_spec = 1
    conv_state = _fresh_state(1, num_spec)
    harness = Harness(conv_state, num_spec)
    metadata = _metadata(
        num_spec_decodes=1,
        num_actual_tokens=num_spec + 2,
        spec_query_len=num_spec + 1,
        spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, num_spec + 2], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
    )
    with pytest.raises(RuntimeError, match="exceeds spec_query_len"):
        harness._short_conv_spec(
            inputs=torch.zeros(num_spec + 2, CHANNELS),
            residual=torch.zeros(num_spec + 2, CHANNELS),
            outer_residual=torch.zeros(num_spec + 2, CHANNELS),
            metadata=metadata,
            conv_state=conv_state,
            conv_weights=harness._conv_weight_2d,
        )


def test_spec_without_token_map_uses_identity_rows():
    """All-spec batches carry no ``spec_token_indx``; rows are identity."""
    num_spec = 2
    slot = 0
    conv_state = _fresh_state(1, num_spec)
    harness = Harness(conv_state, num_spec)
    stream = torch.arange(1, STATE_LEN + 1, dtype=torch.float32).reshape(-1, 1)
    stream = stream.expand(-1, CHANNELS)
    accepted = num_spec + 1
    _seed_accepted(conv_state, slot, num_spec, stream, accepted=accepted)

    chunk = torch.full((num_spec + 1, CHANNELS), 0.75)
    inputs = chunk.clone()
    residual = torch.zeros_like(inputs)
    metadata = _metadata(
        num_spec_decodes=1,
        num_spec_decode_tokens=num_spec + 1,
        num_actual_tokens=num_spec + 1,
        spec_query_len=num_spec + 1,
        spec_state_indices_tensor=torch.tensor([slot], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, num_spec + 1], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([num_spec + 1], dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        spec_token_indx=None,
    )
    harness._short_conv_spec(
        inputs=inputs,
        residual=residual,
        outer_residual=torch.zeros_like(inputs),
        metadata=metadata,
        conv_state=conv_state,
        conv_weights=harness._conv_weight_2d,
    )
    expected = _reference_outputs(
        stream, chunk, torch.zeros_like(chunk), torch.zeros_like(chunk)
    )
    assert torch.equal(residual, expected)


def test_spec_ignores_graph_padding_beyond_num_actual_tokens():
    """Rows past ``num_actual_tokens`` are graph padding and must stay zero."""
    num_spec = 1
    slot = 0
    conv_state = _fresh_state(1, num_spec)
    harness = Harness(conv_state, num_spec)
    stream = torch.full((STATE_LEN, CHANNELS), 2.0)
    accepted = num_spec + 1
    _seed_accepted(conv_state, slot, num_spec, stream, accepted=accepted)

    padded_rows = 3
    inputs = torch.ones(num_spec + 1 + padded_rows, CHANNELS)
    residual = torch.zeros_like(inputs)
    metadata = _metadata(
        num_spec_decodes=1,
        num_spec_decode_tokens=num_spec + 1,
        num_actual_tokens=num_spec + 1,
        spec_query_len=num_spec + 1,
        spec_state_indices_tensor=torch.tensor([slot], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, num_spec + 1], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
    )
    harness._short_conv_spec(
        inputs=inputs,
        residual=residual,
        outer_residual=torch.ones_like(inputs),
        metadata=metadata,
        conv_state=conv_state,
        conv_weights=harness._conv_weight_2d,
    )
    assert torch.equal(residual[num_spec + 1 :], torch.zeros(padded_rows, CHANNELS))
    # The read window is the seeded stream rolled back by ``accepted - 1``.
    window = torch.cat((stream[:1], stream), dim=0)[accepted - 1 : accepted - 1 + STATE_LEN]
    assert torch.equal(residual[: num_spec + 1], _reference_outputs(
        window,
        inputs[: num_spec + 1],
        torch.zeros(num_spec + 1, CHANNELS),
        torch.ones(num_spec + 1, CHANNELS),
    ))


def test_non_spec_decode_rows_must_be_single_token(monkeypatch):
    """The weakened decode-row assertion still refuses a silent mismatch."""
    import vllm.forward_context as ctx_mod

    num_spec = 1
    conv_state = _fresh_state(2, num_spec)
    harness = Harness(conv_state, num_spec)
    metadata = _metadata(
        num_decodes=1,
        num_decode_tokens=2,
        num_actual_tokens=2,
        state_indices_tensor=torch.tensor([0], dtype=torch.int32),
        query_start_loc_p=None,
        has_initial_states_p=None,
    )
    inputs = torch.zeros(2, CHANNELS)
    residual = torch.zeros(2, CHANNELS)
    monkeypatch.setattr(
        ctx_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={harness.prefix: metadata}),
    )
    with pytest.raises(RuntimeError, match="one token each"):
        harness._short_conv(inputs, residual, torch.zeros_like(inputs))
