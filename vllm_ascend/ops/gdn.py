#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from einops import rearrange
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator, is_conv_state_dim_first
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.kda.kda import chunk_kda, fused_kda_gate, fused_recurrent_kda
from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.25.1"):
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd  # type: ignore[import-not-found]
else:
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd


class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    def _split_ba_for_tp(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "split_ba"):
            return self.split_ba(ba)
        return ba.chunk(2, dim=-1)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        return

    def _warmup_prefill_kernels_v0202(self, mixed_qkv: torch.Tensor) -> None:
        return

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor = None,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self._split_ba_for_tp(ba)
            b = b.contiguous()
            a = a.contiguous()
        else:
            if not self.gqa_interleaved_layout:
                mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
                num_tokens = mixed_qkvz.size(0)
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                ba, _ = self.in_proj_ba(hidden_states)
                b, a = self._split_ba_for_tp(ba)

                b = b.contiguous()
                a = a.contiguous()
            else:
                projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
                num_tokens = projected_states_qkvz.size(0)

                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                    projected_states_qkvz,
                    projected_states_ba,
                    triton.cdiv(self.num_k_heads, self.tp_size),
                    triton.cdiv(self.num_v_heads, self.tp_size),
                    self.head_k_dim,
                    self.head_v_dim,
                )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
            False,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        out, _ = self.out_proj(core_attn_out)
        if output is not None:
            output[:num_tokens] = out
        return out

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            spec_causal_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
            spec_query_start_loc_device = spec_causal_conv1d_meta.query_start_loc
            output_spec = torch.empty_like(mixed_qkv_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output_spec,
                mixed_qkv_spec,
                conv_weights_T,
                conv_state=self_kv_cache[0],
                bias_opt=self.conv1d.bias,
                query_start_loc_opt=spec_query_start_loc_device,
                cache_indices_opt=spec_causal_conv1d_meta.cache_indices,
                initial_state_mode_opt=None,
                num_accepted_tokens_opt=spec_causal_conv1d_meta.num_accepted_tokens,
                activation_mode=activation_num,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=1,
            )
            mixed_qkv_spec = output_spec

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            if mixed_qkv_non_spec is not None:
                non_spec_causal_conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
                query_start_loc_opt = non_spec_causal_conv1d_meta.query_start_loc
                cache_indices_opt = non_spec_causal_conv1d_meta.cache_indices
                initial_state_mode_opt = non_spec_causal_conv1d_meta.initial_state_mode
                if get_pcp_group().world_size > 1:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0
                    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
                    assert non_spec_query_start_loc is not None
                    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
                    width = conv_weights.shape[1]
                    state_len = width - 1
                    num_seqs = non_spec_query_start_loc.shape[0] - 1
                    prefill_seq_offset = max(0, num_seqs - attn_metadata.num_prefills)
                    prefill_cache_indices = non_spec_state_indices_tensor[prefill_seq_offset:]
                    mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
                    last_width_prefill_x = extract_last_width(
                        mixed_qkv_non_spec_T, non_spec_query_start_loc[prefill_seq_offset:], state_len
                    )
                    pcp_rank = get_pcp_group().rank_in_group
                    all_last_width_prefill_x = get_pcp_group().all_gather(
                        last_width_prefill_x.unsqueeze(0).contiguous(), 0
                    )
                    if pcp_rank > 0 and prefill_cache_indices.shape[0] > 0:
                        self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                            pcp_rank - 1, ...
                        ].transpose(-1, -2)
                    mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec_output,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=query_start_loc_opt,
                        cache_indices_opt=cache_indices_opt,
                        initial_state_mode_opt=initial_state_mode_opt,
                        num_accepted_tokens_opt=None,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
                    if prefill_cache_indices.shape[0] > 0:
                        self_kv_cache[0][prefill_cache_indices, :state_len, :] = all_last_width_prefill_x[
                            -1, ...
                        ].transpose(-1, -2)
                else:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0
                    mixed_qkv_non_spec_output = torch.empty_like(mixed_qkv_non_spec)
                    torch.ops._C_ascend.npu_causal_conv1d_custom(
                        mixed_qkv_non_spec_output,
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        conv_state=self_kv_cache[0],
                        bias_opt=self.conv1d.bias,
                        query_start_loc_opt=query_start_loc_opt,
                        cache_indices_opt=cache_indices_opt,
                        initial_state_mode_opt=initial_state_mode_opt,
                        num_accepted_tokens_opt=None,
                        activation_mode=activation_num,
                        pad_slot_id=PAD_SLOT_ID,
                        run_mode=0,
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
        elif attn_metadata.num_decodes > 0:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0
            non_spec_causal_conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
            non_spec_query_start_loc_device = non_spec_causal_conv1d_meta.query_start_loc
            output_non_spec = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output_non_spec,
                mixed_qkv_non_spec,
                conv_weights_T,
                conv_state=self_kv_cache[0],
                bias_opt=self.conv1d.bias,
                query_start_loc_opt=non_spec_query_start_loc_device,
                cache_indices_opt=non_spec_causal_conv1d_meta.cache_indices,
                initial_state_mode_opt=None,
                num_accepted_tokens_opt=None,
                activation_mode=activation_num,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=1,
            )
            mixed_qkv_non_spec = output_non_spec
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        # 2. Recurrent attention
        g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        split_non_spec = (
            spec_sequence_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            actual_seq_lengths = attn_metadata.spec_decode_metadata.actual_seq_lengths
            query_spec = l2norm_fwd(query_spec)
            key_spec = l2norm_fwd(key_spec)
            # Dispatches to the vllm-ascend AscendC custom operator
            # (csrc/recurrent_gated_delta_rule), NOT the built-in CANN operator.
            # The custom op extends dtype support (e.g. float32 state) and is
            # loaded at runtime via ASCEND_CUSTOM_OPP_PATH.
            core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_spec.squeeze(0),
                key=key_spec.squeeze(0),
                value=value_spec.squeeze(0),
                g=g_spec.squeeze(0),
                beta=beta_spec.squeeze(0),
                state=ssm_state,
                scale=key_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=spec_state_indices_tensor.flatten(),
                num_accepted_tokens=spec_causal_conv1d_meta.num_accepted_tokens.to(torch.int32),
            ).unsqueeze(0)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part in mixed non-spec batches
        if split_non_spec:
            assert mixed_qkv_non_spec is not None
            assert g_non_spec is not None
            assert beta_non_spec is not None
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(mixed_qkv_non_spec[:num_decode_tokens])
            actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
            query_decode = l2norm_fwd(query_decode)
            key_decode = l2norm_fwd(key_decode)
            core_attn_out_decode = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_decode.squeeze(0),
                key=key_decode.squeeze(0),
                value=value_decode.squeeze(0),
                g=g_non_spec[:, :num_decode_tokens].squeeze(0),
                beta=beta_non_spec[:, :num_decode_tokens].squeeze(0),
                state=ssm_state,
                scale=key_decode.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=non_spec_state_indices_tensor[: attn_metadata.num_decodes],
            ).unsqueeze(0)
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part
        if attn_metadata.num_prefills > 0:
            prefill_query_start_loc = attn_metadata.prefill_query_start_loc
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_query_start_loc is not None
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            assert g_non_spec is not None
            assert beta_non_spec is not None
            if split_non_spec:
                query_non_spec = query_non_spec[:, num_decode_tokens:]
                key_non_spec = key_non_spec[:, num_decode_tokens:]
                value_non_spec = value_non_spec[:, num_decode_tokens:]
                g_non_spec = g_non_spec[:, num_decode_tokens:]
                beta_non_spec = beta_non_spec[:, num_decode_tokens:]

            initial_state = ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()
            clear_ssm_states(initial_state, prefill_has_initial_state)
            (core_attn_out_non_spec, last_recurrent_state) = chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=prefill_query_start_loc,
                prebuilt_meta=attn_metadata.non_spec_prefill_metadata.chunk,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[prefill_state_indices] = last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
            if split_non_spec:
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec],
                    dim=1,
                )
        elif attn_metadata.num_decodes > 0:
            actual_seq_lengths = attn_metadata.non_spec_decode_metadata.actual_seq_lengths
            query_non_spec = l2norm_fwd(query_non_spec)
            key_non_spec = l2norm_fwd(key_non_spec)
            # Dispatches to the vllm-ascend AscendC custom operator
            # (csrc/recurrent_gated_delta_rule), NOT the built-in CANN operator.
            core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_non_spec.squeeze(0),
                key=key_non_spec.squeeze(0),
                value=value_non_spec.squeeze(0),
                g=g_non_spec.squeeze(0) if g_non_spec is not None else g_non_spec,
                beta=beta_non_spec.squeeze(0) if beta_non_spec is not None else beta_non_spec,
                state=ssm_state,
                scale=key_non_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=non_spec_state_indices_tensor,
            ).unsqueeze(0)
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
        maybe_save_kv_layer_to_connector("", [])


class AscendKimiGatedDeltaNetAttention(KimiGatedDeltaNetAttention):
    """Ascend OOT replacement for the Kimi-K3 KDA layer.

    The upstream class is a `PluggableLayer` registered under the name
    ``KimiGatedDeltaNetAttention``. Ascend had a registration only for the
    *base* ``GatedDeltaNetAttention`` key, so KDA layers fell through to the
    unpatched upstream implementation -- whose prefill path calls the upstream
    Triton ``causal_conv1d_fn`` (never replaced on this platform) and the
    upstream ``chunk_kda_with_fused_gate``, which has no Ascend counterpart.

    Only the platform-specific kernel call sites are overridden here; the
    projections, weight shapes and the `forward` skeleton are inherited
    unchanged, so the checkpoint's weight names and TP sharding are untouched.

    Two deliberate substitutions, both verified against the upstream semantics:

    1. **conv1d** -> CANN ``npu_causal_conv1d_custom`` (``run_mode=0`` prefill,
       ``run_mode=1`` decode). The layout differs from upstream's
       ``causal_conv1d_fn`` in three ways, all established by direct probe on
       A3 and by reading the operator's tiling validation:
         * ``x`` is ``(T, dim)`` for varlen prefill -- *not* upstream's
           ``(dim, T)``. Passing the upstream layout fails tiling outright
           ("dim must satisfy dim % 16 == 0 ... got dim=<T>"), since the
           operator reads the last axis as ``dim``;
         * ``weight`` must be ``(width, dim)``, i.e. the *transpose* of the
           ``(dim, width)`` view upstream uses;
         * ``convStates`` must be ``(num_cache_lines, state_len, dim)``. The
           SD layout this model allocates is ``(num_blocks, state_len,
           conv_dim)`` with the three q/k/v segments concatenated along
           ``conv_dim``, so the state is transposed to put dim last, split into
           three, and each piece transposed back -- matching upstream's
           "SD layout needs a transpose" comment.
       One tap ordering was confirmed one-hot: slot 0 is the *oldest* history
       entry and the final tap is the current token, so the semantics agree
       with the vendor convolution.
    2. **gate + chunk attention** -> the extern-gate two-step:
       ``fused_kda_gate`` (softplus / safe-gate activation ONLY) followed by
       ``chunk_kda`` (which performs its own chunk-local cumsum internally).
       The upstream ``chunk_kda_with_fused_gate`` folds the cumsum into the gate
       kernel; Ascend's copy of that symbol does not exist, so the split form is
       used instead. See `_kda_gate_activation` for the doubling hazard.

    `gate_lower_bound` comes from the checkpoint's
    ``text_config.linear_attn_config`` (``-5.0`` for Kimi-K3). The upstream
    implementation has no such parameter at all -- its ``**kwargs`` silently
    swallows it -- so the safe-gate activation is genuinely absent on that path.
    """

    def _gate_lower_bound(self) -> float | None:
        """Kimi-K3's `gate_lower_bound`, or None when the config omits it.

        Read through `hf_text_config` for the same reason the model-side state
        hooks do: the K3 checkpoint nests `linear_attn_config` under
        `text_config`, and the top-level config has no such field.
        """
        hf_text_config = getattr(self.model_config, "hf_text_config", None)
        kda_config = getattr(hf_text_config, "linear_attn_config", None) or {}
        return kda_config.get("gate_lower_bound", None)

    def _kda_gate_activation(self, g1: torch.Tensor) -> torch.Tensor:
        """Apply the KDA gate activation to the raw gate logits.

        IMPORTANT (double-cumsum hazard): `fused_kda_gate` performs the gate
        activation **only**. The chunk-local cumulative sum happens inside
        `chunk_kda` -> `chunk_kda_fwd`. Feeding an already-cumsummed gate into
        `chunk_kda` would accumulate twice -- silently, with no error, and with
        badly wrong numerics. Do not add a cumsum here.
        """
        return fused_kda_gate(
            rearrange(g1, "1 n h d -> n (h d)"),
            self.A_log,
            self.head_dim,
            g_bias=self.dt_bias,
            lower_bound=self._gate_lower_bound(),
        ).unsqueeze(0)

    def _conv_weight_for_op(self, weight: torch.Tensor, x_dtype: torch.dtype) -> torch.Tensor:
        """`(dim, 1, width)` -> `(width, dim)`, cast to the activation dtype.

        Two conversions in one place because both are properties of the CANN
        operator's contract rather than of the weights:
          * layout -- the operator takes `(width, dim)`, the transpose of the
            `(dim, width)` view its own kernels use upstream;
          * dtype -- the parameter is fp32 while the activations are bf16, and
            the operator rejects mixed dtypes.
        """
        w = weight.reshape(weight.size(0), weight.size(2)).transpose(0, 1)
        w = w.contiguous()
        return w if w.dtype == x_dtype else w.to(x_dtype)

    def _conv1d_prefill(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        has_initial_state: torch.Tensor | None,
        cache_indices: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
    ) -> torch.Tensor:
        """Causal conv over a varlen prefill batch via the CANN fused operator.

        `x` must be `(T, dim)` -- the *opposite* of upstream's
        `causal_conv1d_fn`, which takes `(dim, T)` varlen input. Probing the
        operator with the upstream layout fails tiling with
        "dim must satisfy dim % 16 == 0 ... got dim=<T>", i.e. it reads the last
        dimension as `dim`. The caller therefore must not apply upstream's
        `transpose(0, 1)` for this path.
        """
        # `initial_state_mode` tells the operator whether each sequence has a
        # valid incoming conv state; sequences without one must not consume the
        # (stale) block content. Upstream passes `has_initial_state` for exactly
        # this purpose.
        initial_state_mode = None
        if has_initial_state is not None:
            initial_state_mode = has_initial_state.to(torch.int32)

        out = torch.empty_like(x)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out,
            x,
            weight,
            conv_state=conv_state,
            bias_opt=bias,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=initial_state_mode,
            num_accepted_tokens_opt=None,
            activation_mode=1,  # silu, matching the upstream `activation="silu"`
            pad_slot_id=PAD_SLOT_ID,
            run_mode=0,
        )
        return out

    def _conv1d_decode(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        conv_state_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single-token causal conv via the CANN fused operator (`run_mode=1`).

        The operator updates `conv_state` in place for the rows named by
        `conv_state_indices`, matching `causal_conv1d_update`'s contract.
        """
        out = torch.empty_like(x)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out,
            x,
            weight,
            conv_state=conv_state,
            bias_opt=bias,
            query_start_loc_opt=None,
            cache_indices_opt=conv_state_indices,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=None,
            activation_mode=1,  # silu
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,
        )
        return out

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata_narrowed.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        constant_caches = self.kv_cache

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # Conv state layout, mirroring upstream's comment verbatim:
        #   "conv_state must be (..., dim, width-1) for the conv kernels.
        #    DS layout stores it that way directly; SD layout needs a transpose."
        # The default here is SD, i.e. (num_blocks, state_len, conv_dim); the CANN
        # operator takes (num_cache_lines, state_len, dim) with dim as the *last*
        # axis, which is the same convention as the (..., dim, state_len) form
        # after the segment split -- so transpose first, then chunk the dim axis
        # into q/k/v, then transpose each piece back.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_state_q, conv_state_k, conv_state_v = (
            c.transpose(-1, -2).contiguous() for c in conv_state.chunk(3, dim=-2)
        )

        # The conv weights are stored unsqueezed as (dim, 1, width). Upstream
        # reshapes to (dim, width) for its own kernel; the CANN operator wants
        # the transpose, (width, dim), so the weight is transposed here rather
        # than merely viewed. The values are also cast because the parameter is
        # declared `params_dtype=torch.float32` while the operator requires a
        # single dtype across x / weight / convStates (a mismatch is rejected
        # with EZ1009). Casting at the call site rather than at construction
        # keeps the parameter dtype -- and hence weight loading -- unchanged.
        q_conv_weights = self._conv_weight_for_op(self.q_conv1d.weight, x_dtype=q_proj_states.dtype)
        k_conv_weights = self._conv_weight_for_op(self.k_conv1d.weight, x_dtype=k_proj_states.dtype)
        v_conv_weights = self._conv_weight_for_op(self.v_conv1d.weight, x_dtype=v_proj_states.dtype)

        if attn_metadata_narrowed.num_prefills > 0:
            # Unlike upstream, do NOT transpose to (dim, T) here: the CANN
            # operator's varlen prefill mode takes (T, dim), and passing the
            # upstream layout fails tiling outright. The projections are
            # already (T, dim), so they are passed through unchanged and the
            # results feed straight into the rearrange below.
            q = self._conv1d_prefill(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                conv_state_q,
                has_initial_state,
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
            )
            k = self._conv1d_prefill(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                conv_state_k,
                has_initial_state,
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
            )
            v = self._conv1d_prefill(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                conv_state_v,
                has_initial_state,
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
            )
        else:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[: attn_metadata_narrowed.num_actual_tokens]
            q = self._conv1d_decode(q_proj_states, q_conv_weights, self.q_conv1d.bias, conv_state_q, decode_conv_indices)
            k = self._conv1d_decode(k_proj_states, k_conv_weights, self.k_conv1d.bias, conv_state_k, decode_conv_indices)
            v = self._conv1d_decode(v_proj_states, v_conv_weights, self.v_conv1d.bias, conv_state_v, decode_conv_indices)

        q, k, v = map(lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v))

        if attn_metadata_narrowed.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            # extern-gate equivalent of `chunk_kda_with_fused_gate`: see
            # `_kda_gate_activation` for why the cumsum must NOT happen here.
            g = self._kda_gate_activation(g1)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
            )
            # Init cache
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            assert non_spec_query_start_loc is not None
            g1 = self._kda_gate_activation(g1)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[: attn_metadata_narrowed.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[0, :num_actual_tokens]
