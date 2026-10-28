# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Ascend PLE (position-learning enhancement) layer for Qwen4Exp.

Only the compute path differs from the NVIDIA owner. Everything that is pure
bookkeeping (n-gram id construction, hash buffers, checkpoint-shard loading,
the ``MergedColumnParallelLinear`` projection, the grouped norms) is inherited
from the upstream layer classes, which are platform neutral:

* ``Qwen4ExpNGramEmbedding.compute_ngram_ids`` already falls back to its torch
  branch whenever ``input_ids`` is not a CUDA tensor, so the Triton n-gram
  kernel is never reached on NPU. ``--engram-config '{"cpu_offload": false}'``
  keeps the table device-resident, so no UVA host view is involved either.
* The PLE cache is a plain ``MambaSpec`` short-conv state, which Ascend already
  drives through the generic mamba manager patches.

What is replaced here:

* ``ple_gate`` (Triton) -> eager torch with the same fp32 rounding boundaries.
* The dilated short convolution (``ple_conv`` Triton) -> eager torch. The
  convolution is depthwise over ``hc_hidden_size=10240`` channels with
  ``dilation=ngram_size=3`` and a persistent ring window of
  ``conv_state_len=(kernel-1)*dilation`` tokens; Ascend has no dilated
  causal-conv1d kernel, and eager ``F.conv1d`` keeps the window explicit.

The short-conv metadata still comes from the upstream
``PleShortConvAttentionBackend`` builder (registered by the model), so the
request/state routing contract is unchanged.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpPLEGroupedNorm,
    Qwen4ExpPLELayer,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)

logger = init_logger(__name__)

__all__ = ["AscendQwen4ExpPLEGroupedNorm", "AscendQwen4ExpPLELayer", "ple_gate_torch"]


AscendQwen4ExpPLEGroupedNorm = Qwen4ExpPLEGroupedNorm


def ple_gate_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped-norm gate, eager equivalent of the fused ``ple_gate`` kernel.

    Per HC branch, over the H-wide slice:

        d = dot(norm(key), norm(query)) / sqrt(H)
        g = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))
        gated = g * value
        normed = norm(gated) with the norm_conv affine

    The kernel casts to the activation dtype at each materialized boundary and
    accumulates in fp32; the casts below reproduce that ordering, because
    dropping them changes the bf16 result by more than the tolerance used for
    layer-wise comparison.
    """
    num_tokens = hidden.shape[0]
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    dtype = hidden.dtype
    if key.dtype != dtype or value.dtype != dtype:
        raise ValueError("PLE gate requires key, value and hidden to share a dtype")

    def grouped_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Per-stream Gemma norm, kept as ``[T, HC, H]``.

        The stream axis has to survive until the gate dot product: each
        stream produces its own scalar gate from its own H-wide slice.
        """
        grouped = x.float().unflatten(-1, (hc, h))
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(variance + eps)
        return normalized * (1.0 + weight.float().unflatten(-1, (hc, h)))

    k_n = grouped_norm(key, norm_key_w).to(dtype)
    q_n = grouped_norm(hidden[..., : hc * h], norm_query_w).to(dtype)
    products = (k_n.float() * q_n.float()).to(dtype)
    dot = products.float().sum(dim=-1, keepdim=True).to(dtype).float()
    d = (dot / float(h) ** 0.5).to(dtype).float()
    sign = torch.sign(d)
    magnitude = torch.sqrt(torch.maximum(d.abs(), torch.full_like(d, 1e-6))).to(dtype)
    g = torch.sigmoid(sign * magnitude).to(dtype).float()

    # ``value`` is [T, H]; the per-token gate is broadcast over the HC streams
    # so every stream shares the same gated value.
    v = value.float()
    gated = (g * v.unsqueeze(1)).to(dtype)
    gf = gated.float()
    # ``norm_conv`` normalizes each HC stream independently and applies the
    # affine before the flat layout is restored.
    variance = gf.square().mean(dim=-1, keepdim=True)
    normed = gf * torch.rsqrt(variance + eps)
    normed = normed * (1.0 + norm_conv_w.float().unflatten(-1, (hc, h)))
    return gated.flatten(-2).contiguous(), normed.flatten(-2).to(dtype).contiguous()


class AscendQwen4ExpPLELayer(Qwen4ExpPLELayer):
    """Qwen4Exp PLE layer with eager gate and dilated short convolution."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            vllm_config=vllm_config,
            layer_idx=layer_idx,
            ple_dense_layer_id=ple_dense_layer_id,
            prefix=prefix,
        )
        # The conv weight is a depthwise kernel of shape [C, 1, K]; the
        # dilated path only needs [C, K].
        self._conv_weight_2d = self.conv1d.weight.squeeze(1).contiguous()

    # -- state contract (unchanged from upstream, restated for clarity) ----
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    # -- eager short convolution ------------------------------------------
    def _window_from_state(self, conv_state: torch.Tensor, state_capacity: int) -> torch.Tensor:
        """Return one slot's window as token-major ``[state_capacity, C]``.

        ``conv_state`` arrives as ``[slots, *, *]``. Which axis is the window
        depends on the SSM conv-state layout: ``SD`` (the default) is
        ``[slots, state_len, C]``, ``DS`` is ``[slots, C, state_len]``. The
        dilated convolution reads taps along time, so normalize to ``SD``
        here. Only the newest ``state_capacity`` taps are meaningful; older
        slots are graph padding.
        """
        slot = conv_state[0]
        window_axis = 1 if is_conv_state_dim_first() else 0
        if slot.shape[window_axis] < state_capacity:
            raise RuntimeError(
                "PLE short-conv cache is smaller than expected for dilated "
                f"convolution: got {slot.shape[window_axis]}, expect at least {state_capacity}"
            )
        window = slot.narrow(window_axis, slot.shape[window_axis] - state_capacity, state_capacity)
        return window if window_axis == 0 else window.transpose(0, 1)

    def _store_window(
        self,
        conv_state: torch.Tensor,
        slot: int,
        window: torch.Tensor,
        writeback_capacity: int,
    ) -> None:
        """Write the newest ``writeback_capacity`` taps of ``window`` at offset 0.

        ``window`` is token-major and holds ``history + chunk``; the live state
        is its tail, written to the start of the slot. The slot may be wider
        than the window when speculative decode reserves extra taps, and those
        taps are the stale draft tail, not the newest history.
        """
        newest = window[-writeback_capacity:]
        target = conv_state[slot]
        if is_conv_state_dim_first():
            target[:, :writeback_capacity] = newest.t()
        else:
            target[:writeback_capacity, :] = newest

    def _dilated_conv(
        self,
        tokens: torch.Tensor,
        history: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Dilated depthwise convolution of ``tokens`` ``[T, C]``.

        ``history`` holds the ``conv_state_len`` tokens immediately preceding
        this chunk (oldest first) and ``weight`` is ``[C, K]``. Returns the
        pre-activation convolution output, ``[T, C]`` in fp32.
        """
        kernel = weight.shape[1]
        dilation = self.short_conv_dilation
        state_len = self.conv_state_len
        seq = torch.cat((history, tokens), dim=0).float()
        num_tokens = tokens.shape[0]
        acc = torch.zeros(num_tokens, weight.shape[0], dtype=torch.float32, device=tokens.device)
        # Tap k of output row j reads sequence position j + k * dilation.
        for k in range(kernel):
            offset = k * dilation
            window = seq[offset : offset + num_tokens]
            acc += window * weight[:, k].float().unsqueeze(0)
        return acc

    def _slice_history(
        self,
        conv_state: torch.Tensor,
        state_index: int,
        state_capacity: int,
    ) -> torch.Tensor:
        """Token-major ``[state_capacity, C]`` window of one state slot.

        The live window starts at offset 0. The plain decode/prefill caller
        takes its leading ``conv_state_len`` taps; the spec caller slices a
        rollback offset on top because the reserved ``num_spec_tokens`` taps
        past ``conv_state_len`` belong to the draft chunk that is being
        re-read, not to a second copy of the history.
        """
        return self._window_from_state(conv_state[state_index : state_index + 1], state_capacity)

    def _short_conv_spec(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        """Speculative-decode short convolution (``mode="spec"`` upstream).

        Every one of the ``1 + K`` rows of a draft request reads the same
        pre-chunk window, so draft rows get a PLE output too (they feed the
        verifier). The state then advances by the *accepted* tokens only: the
        write-back shifts the live window by one and appends the whole chunk,
        and the next step re-reads it starting at ``num_accepted - 1``, so the
        old state plus that chunk reproduces exactly the accepted prefix.

        ``spec_token_indx`` maps logical spec rows back to the row order of
        the full batch, so a mixed spec/non-spec batch never has to be
        materialized into two groups.
        """
        num_spec_decodes = metadata.num_spec_decodes
        if not num_spec_decodes:
            return
        state_indices = metadata.spec_state_indices_tensor
        query_start_loc = metadata.spec_query_start_loc
        num_accepted = metadata.num_accepted_tokens
        if state_indices is None or query_start_loc is None or num_accepted is None:
            raise RuntimeError(
                "PLE spec short-conv requires spec_state_indices_tensor, "
                "spec_query_start_loc and num_accepted_tokens"
            )
        # ``num_actual_tokens`` already includes graph padding; every real spec
        # row lives below it.
        inputs = inputs[: metadata.num_actual_tokens]
        residual = residual[: metadata.num_actual_tokens]
        outer_residual = outer_residual[: metadata.num_actual_tokens]

        dtype = residual.dtype
        state_capacity = self.conv_state_len + self.num_spec_tokens
        spec_query_len = metadata.spec_query_len
        if spec_query_len != self.num_spec_tokens + 1:
            raise RuntimeError(
                "PLE spec short-conv expects spec_query_len == num_speculative_tokens + 1, "
                f"got {spec_query_len} for {self.num_spec_tokens} spec tokens"
            )

        rows = self._row_map(metadata, query_start_loc, num_spec_decodes)
        # No device sync *here*: these are host-side ints from the metadata
        # object, so the log itself costs nothing. The loop below does sync --
        # every ``.item()`` on ``query_start_loc`` / ``num_accepted_tokens`` /
        # ``spec_state_indices_tensor`` / ``rows`` is a per-request device->host
        # read (up to six per request). That is accepted rather than
        # accidental: the PLE layer appears once (``ple_layer_ids=[2]``), so the
        # total is small next to the attention stack. If this ever shows up in a
        # profile, batch the reads with one ``.tolist()`` per tensor instead of
        # per request.
        logger.info_once(
            "PLE spec short-conv active: %d spec requests, spec_query_len=%d, "
            "state window %d+%d",
            num_spec_decodes,
            spec_query_len,
            self.conv_state_len,
            self.num_spec_tokens,
        )

        def apply_group(
            group_inputs: torch.Tensor,
            group_residual: torch.Tensor,
            group_outer: torch.Tensor,
            history: torch.Tensor,
        ) -> torch.Tensor:
            conv = self._dilated_conv(group_inputs, history, conv_weights).to(dtype).float()
            y = conv * torch.sigmoid(conv)
            conv_output = y.to(dtype)
            ple_output = (group_residual + conv_output).to(dtype)
            return (group_outer.float() + ple_output.float()).to(dtype)

        for req in range(num_spec_decodes):
            q_begin = int(query_start_loc[req].item())
            q_end = int(query_start_loc[req + 1].item())
            qlen = q_end - q_begin
            if qlen <= 0:
                # Graph padding row: no slot, nothing to read or write.
                continue
            if qlen > spec_query_len:
                raise RuntimeError(
                    f"PLE spec chunk exceeds spec_query_len: {qlen} > {spec_query_len}"
                )
            accepted = int(num_accepted[req].item())
            # Roll the read window back to the last accepted token; a padded
            # or fully-rejected chunk still reads the newest live state.
            slot_off = min(max(accepted - 1, 0), spec_query_len - 1)
            slot = int(state_indices[req].item())
            if slot == NULL_BLOCK_ID:
                # No state slot for this row: skip rather than read/write slot
                # 0. Upstream's ``_ple_conv_kernel`` /
                # ``_ple_conv_writeback_kernel`` both guard on
                # ``sid != NULL_STATE_ID`` and the writeback returns early, so
                # treating a null slot as a real one would pollute another
                # request's convolution state silently.
                #
                # The builder currently keeps null slots out of
                # ``[:num_spec_decodes]``, so this is unreachable today; it is
                # the same protection the upstream kernel has, kept here so
                # that widening the loop bound cannot turn into a silent
                # cross-request write.
                continue

            begin = int(rows[q_begin].item())
            end = int(rows[q_end - 1].item()) + 1
            chunk = inputs[begin:end]
            if chunk.shape[0] != qlen:
                raise RuntimeError(
                    "PLE spec short-conv row map is not contiguous: "
                    f"{qlen} logical rows expand to {chunk.shape[0]} batch rows"
                )

            # ``history`` is only the accepted window: taps beyond it read the
            # chunk's own tokens, exactly as if the chunk followed the state.
            window = self._slice_history(conv_state, slot, state_capacity).to(dtype)
            history = window[slot_off : slot_off + self.conv_state_len]
            residual[begin:end] = apply_group(
                chunk,
                residual[begin:end],
                outer_residual[begin:end],
                history,
            )

            # Write-back: shift the live window by one, then append the chunk,
            # stopping before the state tail that this chunk never used.
            # Tap ``m = 1 + i`` comes from the state while ``m <= conv_state_len
            # - 1`` (independent of the rollback offset) and from the chunk
            # afterwards, so the state part is ``conv_state_len - 1`` taps and
            # the chunk contributes exactly ``qlen``.
            new_state = torch.zeros_like(window)
            do_write = min(self.conv_state_len + qlen - 1, state_capacity)
            old_len = min(self.conv_state_len - 1, do_write)
            new_state[:old_len] = window[slot_off + 1 : slot_off + 1 + old_len]
            new_state[old_len:do_write] = chunk[: do_write - old_len]
            self._store_window(conv_state, slot, new_state[:do_write], do_write)

    def _row_map(
        self,
        metadata: PleShortConvAttentionMetadata,
        query_start_loc: torch.Tensor,
        num_spec_decodes: int,
    ) -> torch.Tensor:
        """Logical spec row -> row in the batch token order."""
        if metadata.spec_token_indx is not None:
            return metadata.spec_token_indx
        total = int(query_start_loc[num_spec_decodes].item())
        return torch.arange(total, device=query_start_loc.device)

    def _short_conv_eager(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> None:
        """Non-spec short convolution: plain decode rows plus varlen prefills.

        ``inputs``/``residual``/``outer_residual`` are ``[T, C]`` in batch token
        order. Decode rows are the leading ``num_decode_tokens`` entries with
        one token per row; prefill rows follow, grouped by the per-request
        slices of ``query_start_loc``. Both groups map to conv-state slots
        through ``state_indices_tensor`` (decodes first, then prefills).
        """
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        num_actual_tokens = metadata.num_actual_tokens
        inputs = inputs[:num_actual_tokens]
        residual = residual[:num_actual_tokens]
        outer_residual = outer_residual[:num_actual_tokens]

        state_indices = metadata.state_indices_tensor
        if state_indices is None:
            raise RuntimeError("PLE short-conv requires per-request state indices")

        dtype = residual.dtype
        state_capacity = self.conv_state_len + self.num_spec_tokens
        writeback_capacity = self.conv_state_len

        # With spec-decode this group starts after the spec rows, so the
        # non-spec rows are addressed through ``non_spec_token_indx`` instead
        # of a fixed offset.
        non_spec_rows = metadata.non_spec_token_indx

        def non_spec_row(index: int) -> int:
            if non_spec_rows is None:
                return index
            return int(non_spec_rows[index].item())

        def apply_group(
            group_inputs: torch.Tensor,
            group_residual: torch.Tensor,
            group_outer: torch.Tensor,
            history: torch.Tensor,
        ) -> torch.Tensor:
            conv = self._dilated_conv(group_inputs, history, conv_weights).to(dtype).float()
            y = conv * torch.sigmoid(conv)
            conv_output = y.to(dtype)
            ple_output = (group_residual + conv_output).to(dtype)
            return (group_outer.float() + ple_output.float()).to(dtype)

        if num_decodes:
            # Non-spec decodes carry exactly one token each by construction
            # (``num_decodes`` counts requests with ``query_len == 1``); the
            # multi-token spec rows are handled by ``_short_conv_spec``.
            if num_decode_tokens != num_decodes:
                raise RuntimeError(
                    "PLE short-conv non-spec decode rows must hold one token each, "
                    f"got {num_decode_tokens} tokens for {num_decodes} decodes"
                )
            decode_states = state_indices[:num_decodes]
            for req in range(num_decodes):
                row = non_spec_row(req)
                slot = int(decode_states[req].item())
                history = self._slice_history(conv_state, slot, state_capacity)[
                    : self.conv_state_len
                ].to(dtype)
                residual[row : row + 1] = apply_group(
                    inputs[row : row + 1],
                    residual[row : row + 1],
                    outer_residual[row : row + 1],
                    history,
                )
                # Decode advances the window by one token and appends the input.
                new_window = torch.cat((history[1:], inputs[row : row + 1].to(dtype)), dim=0)
                self._store_window(conv_state, slot, new_window, writeback_capacity)

        if num_prefills:
            prefill_states = state_indices[num_decodes : num_decodes + num_prefills]
            query_start_loc = metadata.query_start_loc_p
            has_initial_states = metadata.has_initial_states_p
            if query_start_loc is None or has_initial_states is None:
                raise RuntimeError(
                    "PLE prefill short-conv requires query_start_loc_p and has_initial_states_p"
                )
            base = num_decode_tokens
            for req in range(num_prefills):
                begin = non_spec_row(int(query_start_loc[req].item()) + base)
                end = non_spec_row(int(query_start_loc[req + 1].item()) + base - 1) + 1
                if end <= begin:
                    continue
                slot = int(prefill_states[req].item())
                history = self._slice_history(conv_state, slot, state_capacity)[
                    : self.conv_state_len
                ].to(dtype)
                has_init = bool(has_initial_states[req].item())
                if not has_init:
                    history = torch.zeros_like(history)
                residual[begin:end] = apply_group(
                    inputs[begin:end],
                    residual[begin:end],
                    outer_residual[begin:end],
                    history,
                )
                # Every token of the chunk reads the pre-chunk window; the new
                # window is the tail of ``history + chunk``.
                chunk = inputs[begin:end].to(dtype)
                new_window = torch.cat((history, chunk), dim=0)
                self._store_window(conv_state, slot, new_window, writeback_capacity)

    def _short_conv(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
    ) -> None:
        """Route the current request layout into the eager short convolution.

        Profiling runs carry no attention metadata at all and only contribute
        the outer residual. Once metadata exists every other branch is a hard
        error: a missing entry or an unbound state slot means the layer can no
        longer produce its conv term, and pretending otherwise silently
        corrupts the output.
        """
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            residual.add_(outer_residual)
            return
        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}"
            )
        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            # A populated metadata dict without this layer's entry can only be
            # a registration/key mismatch; silently dropping the conv term
            # would corrupt the PLE output without a trace.
            raise RuntimeError(
                f"PLE layer '{self.prefix}' is missing from attention "
                f"metadata; available keys: {list(attn_metadata.keys())[:5]}"
            )
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got {type(layer_attn_metadata).__name__}"
            )
        conv_state = self.kv_cache[0]
        if conv_state.numel() == 0:
            # ``bind_kv_cache`` runs before any inference forward, so an empty
            # conv-state slot here means the state cache never got bound.
            raise RuntimeError(
                f"PLE layer '{self.prefix}' has an unbound (empty) conv state"
            )
        # Spec rows are handled first and the non-spec group keeps batch row
        # order through ``non_spec_token_indx``, so a mixed batch needs no
        # regrouping of inputs or residuals.
        if layer_attn_metadata.spec_sequence_masks is not None:
            self._short_conv_spec(
                inputs=inputs,
                residual=residual,
                outer_residual=outer_residual,
                metadata=layer_attn_metadata,
                conv_state=conv_state,
                conv_weights=self._conv_weight_2d,
            )
        self._short_conv_eager(
            inputs=inputs,
            residual=residual,
            outer_residual=outer_residual,
            metadata=layer_attn_metadata,
            conv_state=conv_state,
            conv_weights=self._conv_weight_2d,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same token "
                f"length, got {input_ids.shape[0]} and {hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
        )
        embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate_torch(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        self._short_conv(conv_input, gated_output, hidden_states)
        return gated_output


# Re-exported so the model file only depends on this module.
__all__ += ["Qwen4ExpNGramEmbedding"]
