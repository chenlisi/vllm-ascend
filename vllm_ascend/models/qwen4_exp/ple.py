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
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpPLEGroupedNorm,
    Qwen4ExpPLELayer,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)

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
    d = (dot / torch.sqrt(torch.tensor(float(h), device=dot.device, dtype=dot.dtype))).to(dtype).float()
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
        """Return the trailing ``state_capacity`` taps of one slot's window.

        ``conv_state`` arrives as ``[slots, channels, window]`` (already
        canonicalized by the caller). Only the newest ``conv_state_len`` taps
        are meaningful; anything older is graph padding.
        """
        if conv_state.shape[-1] < state_capacity:
            raise RuntimeError(
                "PLE short-conv cache is smaller than expected for dilated "
                f"convolution: got {conv_state.shape[-1]}, expect at least {state_capacity}"
            )
        return conv_state[..., -state_capacity:]

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
        return self._window_from_state(conv_state[state_index : state_index + 1], state_capacity)[0]

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
            rows = num_decode_tokens // num_decodes if num_decodes else 0
            if rows != 1:
                raise NotImplementedError(
                    "Stage 1 PLE short-conv handles one token per decode row, "
                    f"got {rows} (speculative decode belongs to Stage 3)"
                )
            decode_states = state_indices[:num_decodes]
            for req in range(num_decodes):
                slot = int(decode_states[req].item())
                history = self._slice_history(conv_state, slot, state_capacity).to(dtype)
                residual[:num_decodes][req:req + 1] = apply_group(
                    inputs[req : req + 1],
                    residual[req : req + 1],
                    outer_residual[req : req + 1],
                    history,
                )
                # Decode advances the window by one token and appends the input.
                new_window = torch.cat((history[:, 1:], inputs[req : req + 1].to(dtype)), dim=-1)
                conv_state[slot, :, :writeback_capacity] = new_window[:, -writeback_capacity:]

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
                begin = int(query_start_loc[req].item()) + base
                end = int(query_start_loc[req + 1].item()) + base
                if end <= begin:
                    continue
                slot = int(prefill_states[req].item())
                history = self._slice_history(conv_state, slot, state_capacity).to(dtype)
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
                new_window = torch.cat((history, chunk), dim=-1)
                conv_state[slot, :, :writeback_capacity] = new_window[:, -writeback_capacity:]

    def _short_conv(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor,
        outer_residual: torch.Tensor,
    ) -> None:
        """Route the current request layout into the eager short convolution.

        Mirrors the upstream entry point: profiling runs (no metadata) and
        layers without their own metadata entry only contribute the outer
        residual.
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
            residual.add_(outer_residual)
            return
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got {type(layer_attn_metadata).__name__}"
            )
        if layer_attn_metadata.spec_sequence_masks is not None:
            raise NotImplementedError(
                "PLE dilated short-conv with speculative decode is a Stage 3 feature"
            )

        conv_state = self.kv_cache[0]
        if conv_state.numel() == 0:
            residual.add_(outer_residual)
            return
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
