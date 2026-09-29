# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.


import torch
import torch_npu
from torch.nn.functional import pad
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp8_moe_available,
)
from vllm_ascend.models.kimi_k3.activation import situ_and_mul
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    dispose_tensor,
    enable_custom_op,
    get_ascend_device_type,
)

ASCEND_DEVICE_TYPE = get_ascend_device_type()


def _is_situ(activation) -> bool:
    """True when the activation is Kimi-K3's SiTU.

    SiTU is carried as a plain string (``"situ"``) rather than as a
    ``MoEActivation`` member: the upstream enum has no such member and extending
    it would mean patching three module-level tables
    (``_CUSTOM_OP_NAMES`` / ``_WITHOUT_MUL`` / the enum body) in the vLLM core
    for one model's activation. The Ascend MoE path already dispatches on the
    string for ``swigluoai_uninterleave``, so SiTU follows that precedent.
    """
    return getattr(activation, "value", activation) == "situ"


def _situ_betas(
    activation, *, swiglu_alpha: float = 1.0, swiglu_beta: float = 0.0
) -> tuple[float, float | None]:
    """Kimi-K3's SiTU betas, carried in on the per-layer activation scalars.

    ``activation`` is the first positional parameter and is **required**, while
    the two betas keep their defaults and are keyword-only. That asymmetry is
    deliberate: the betas defaulting is harmless (a caller that omits them gets
    a plain SiTU), but a caller that omitted ``activation`` would silently skip
    the borrow-conflict check below -- the assertion would never run and the
    protection would be gone with no symptom. A required argument turns that
    into a TypeError at the call site. Keyword-only betas close the same hole
    from the other side: with ``(activation, swiglu_alpha=..., swiglu_beta=...)``
    a caller writing ``_situ_betas(4.0, 25.0)`` binds ``4.0`` as ``activation``
    and passes the conflict check vacuously. The three in-tree callers pass all
    three by keyword.

    NOT read from the live vLLM config: this runs inside the model forward, and
    ``profile_run``'s ``_dummy_run`` executes outside any
    ``set_current_vllm_config`` context, so a config lookup here raises at
    startup. (That failure is what the dummy smoke caught -- the branch is
    reachable long before any real request.) The values therefore have to
    arrive as arguments, like every other per-layer activation scalar.

    The transport is ``swiglu_alpha`` / ``swiglu_beta``, which is the existing,
    already-plumbed channel for per-layer activation parameters:
    ``apply()`` -> ``MoEFusedExpertsInput`` -> ``MoEMlpComputeInput`` ->
    here. Borrowing them is safe because those two slots are consumed as
    ``glu_alpha`` / ``glu_bias`` only on the ``swigluoai_uninterleave`` branch,
    which ``_is_situ`` excludes -- the two can never both apply to one layer.
    The model-side attribute names stay explicit (``moe_situ_beta`` /
    ``moe_situ_linear_beta``) so the meaning is not lost at the source.

    ``linear_beta`` is returned as ``None`` when the caller left the slot at
    its default, which is how the vendor distinguishes "unset" from a value and
    is what makes ``situ_and_mul`` skip the ``up`` rescaling. ``beta`` follows
    the vendor's ``beta or 1.0`` so a stored 0.0 cannot divide by zero.
    """
    _assert_borrowed_slots_unambiguous(activation)
    beta = swiglu_alpha or 1.0
    linear_beta = swiglu_beta if swiglu_beta else None
    return beta, linear_beta


def _assert_borrowed_slots_unambiguous(activation) -> None:
    """Make the ``swiglu_alpha`` / ``swiglu_beta`` sharing an enforced invariant.

    Those two slots have two readers: ``glu_alpha`` / ``glu_bias`` on the
    ``swigluoai_uninterleave`` branch, and SiTU's ``beta`` / ``linear_beta``
    here. Reusing one slot-set for two meanings is only sound while no layer can
    satisfy both branches at once -- and that is currently a consequence of
    *assignment order* in the model (``swigluoai_uninterleave`` is set first,
    ``situ`` overwrites it), not of anything checked. Reading either consumer in
    isolation cannot detect the collision: each branch only sees the values it
    expects.

    This is the place that can. Every SiTU entry point passes the activation
    through, so if the two branches ever became simultaneously satisfiable --
    a config that sets both markers, a layer that keeps the OAI flag while
    acquiring the SiTU one -- the borrowed values would be interpreted as both
    ``glu_*`` and SiTU's betas, and the model would silently compute a third
    function with no error. Stopping here converts that into a startup failure.
    """
    act_name = getattr(activation, "value", activation)
    if act_name is not None and act_name == "swigluoai_uninterleave":
        raise AssertionError(
            "activation is 'swigluoai_uninterleave' but SiTU's betas were "
            "requested: the swiglu_alpha/swiglu_beta slots are being read as "
            "both glu_alpha/glu_bias and SiTU beta/linear_beta. The two "
            "branches must be mutually exclusive."
        )


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation=None):
    return (
        fusion
        and dynamic_eplb
        and getattr(activation, "value", activation) != "swigluoai_uninterleave"
        # SiTU is not SwiGLU: the fused op applies npu_swiglu unconditionally, so
        # admitting SiTU here would silently compute the wrong activation.
        and not _is_situ(activation)
        and enable_custom_op()
    )


def _gmm_swiglu_quant_fusion_enabled(use_mxfp_quant, fusion, dynamic_eplb, activation=None):
    return (use_mxfp_quant or (fusion and not dynamic_eplb)) and (
        getattr(activation, "value", activation) != "swigluoai_uninterleave"
        # Same reason as _custom_gmm_swiglu_enabled: this gate feeds the fused
        # SwiGLU+quant kernels, which have no SiTU variant.
        and not _is_situ(activation)
    )


def _mc2_head_block_applies(w1_scale_bias, w1_offset, is_mc2, is_gelu_activation, is_situ) -> bool:
    """Whether the fused MC2 head block above may handle this layer.

    Named rather than inlined so the SiTU guard inside the block's `else` arm
    can be reached from a UT: that guard exists to catch a *future* relaxation of
    this condition, so reproducing it requires being able to relax this
    condition, which an inline expression does not allow.
    """
    return (
        w1_scale_bias is None
        and w1_offset is None
        and is_mc2
        and not is_gelu_activation
        and not is_situ
    )


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    w2_scale: list[torch.Tensor] | torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    dynamic_eplb: bool = False,
    use_mxfp_quant: bool = False,
    mxfp_quant_dtype: QuantType | None = None,
    act_quant_type: torch.dtype = torch.float8_e4m3fn,
    weight_quant_type: torch.dtype | None = None,
    scale_type: torch.dtype | None = None,
    per_token_scale_type: torch.dtype | None = None,
    use_bf16: bool = True,
    activation: str | None = None,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    use_w4a8_per_channel_gmm_swiglu: bool = False,
) -> torch.Tensor:
    input_hidden_dtype = hidden_states.dtype
    act_name = getattr(activation, "value", activation)
    use_gmm_swiglu_quant_fusion = _gmm_swiglu_quant_fusion_enabled(
        use_mxfp_quant,
        fusion,
        dynamic_eplb,
        activation,
    )
    # GELU can't use the fused SwiGLU+quant ops below; fall back to the
    # non-fused GMM -> GELU -> (re)quant -> GMM2 path for GELU activations.
    is_gelu_activation = activation in (MoEActivation.GELU, MoEActivation.GELU_TANH)
    is_swigluoai_uninterleave = act_name == "swigluoai_uninterleave"
    is_situ = _is_situ(activation)

    if use_mxfp_quant:
        ensure_mxfp8_moe_available("MXFP MoE MLP path")

        if w1_scale_bias is not None or w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")
        if w1_offset is not None or w2_offset is not None:
            raise NotImplementedError("MXFP path does not support antiquant offset yet.")

    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif mxfp_quant_dtype == QuantType.W4A16MXFP:
        quantized_hidden_states = None
        pertoken_scale = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = (
            DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale) if use_mxfp_quant else dynamic_scale
        )
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype

    is_mc2 = _EXTRA_CTX.moe_comm_type == MoECommType.MC2
    # SiTU must stay out of this whole head block, not just out of the two
    # predicates inside it. The `else` arm below ends in
    # `npu_dequant_swiglu_quant`, which bakes SwiGLU into the kernel and has no
    # SiTU mode (`swiglu_mode=1` is the *clamped* variant -- clamp / alpha /
    # bias -- not a tanh-based gate). Gating only on the inner predicates would
    # therefore route SiTU into that arm and silently compute SwiGLU: the same
    # defect the predicates were added to fix, one layer further in.
    # Excluding SiTU here drops it through to the general path at the bottom of
    # this function, whose `is_situ` branch (see its own comment) applies the
    # real activation and requantizes for gmm2.
    if _mc2_head_block_applies(w1_scale_bias, w1_offset, is_mc2, is_gelu_activation, is_situ):
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                swiglu_limit=swiglu_limit,
            )
        elif use_gmm_swiglu_quant_fusion and activation != MoEActivation.SWIGLUSTEP:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=None,
                use_mxfp_quant=use_mxfp_quant,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                swiglu_limit=swiglu_limit,
                mxfp_quant_dtype=mxfp_quant_dtype,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        elif activation == MoEActivation.SWIGLUSTEP:
            # Step3.5/3.7 needs to clamp in swiglu: out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
            gmm1_kwargs = {
                "x": [hidden_states],
                "weight": w1 if isinstance(w1, list) else [w1],
                "scale": [w1_scale[0].to(w2_scale[0].dtype)] if isinstance(w1_scale, list) else [w1_scale],
                "bias": None,
                "per_token_scale": [pertoken_scale],
                "split_item": 2,
                "group_type": 0,
                "group_list": group_list,
                "output_dtype": torch.bfloat16,
            }
            if use_mxfp_quant:
                gmm1_kwargs.update(
                    {
                        "scale_dtype": FLOAT8_E8M0FNU_DTYPE,
                        "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
                    }
                )
            hidden_states = torch_npu.npu_grouped_matmul(**gmm1_kwargs)[0]
            hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=7.0)
            hidden_states, swiglu_out_scale = DeviceOperator.npu_dynamic_quant(
                hidden_states, act_quant_type=act_quant_type, use_mxfp_quant=use_mxfp_quant
            )
        else:
            # Depth guard for the invariant the head-block condition above relies
            # on. The condition already excludes SiTU, so in the current tree
            # SiTU cannot arrive here -- but that is enforced *only* by the outer
            # `if`, and this arm is a SwiGLU op end to end: `npu_dequant_swiglu_quant`
            # has no SiTU mode, so a relaxed condition (or a new caller entering
            # with MC2 + SiTU) would silently compute silu(gate)*up again, with no
            # runtime signal. That is the exact failure this whole dispatch was
            # fixed for, so it is worth an explicit stop rather than a comment.
            # Other activations are untouched: this branch fires for SiTU only.
            if is_situ:
                raise NotImplementedError(
                    "SiTU is not supported on the MC2 fused-experts path: "
                    "npu_dequant_swiglu_quant implements SwiGLU (and its clamped "
                    "variant), not SiTU. This branch must be unreachable for SiTU "
                    "-- reaching it means the head-block exclusion was relaxed."
                )
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            dequant_swiglu_kwargs = {
                "x": hidden_states,
                "weight_scale": w1_scale[0],
                "activation_scale": pertoken_scale,
                "bias": None,
                "quant_scale": None,
                "quant_offset": None,
                "group_index": cumsum_group_list(group_list, group_list_type, 1),
                "activate_left": True,
                "quant_mode": 1,
            }
            if is_swigluoai_uninterleave:
                dequant_swiglu_kwargs.update(
                    {
                        "swiglu_mode": 1,
                        "clamp_limit": swiglu_limit,
                        "glu_alpha": swiglu_alpha,
                        "glu_bias": swiglu_beta,
                    }
                )
            hidden_states, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(**dequant_swiglu_kwargs)
        before_gmm2_evt = torch.npu.current_stream().record_event()
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=None,
            fallback_output_dtype=w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype,
            mxfp_quant_dtype=mxfp_quant_dtype,
        )
    elif w1_offset is not None:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
        dispose_tensor(unquantized_hidden_states)
        # act_fn: swiglu
        if activation == MoEActivation.SWIGLUSTEP:
            hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
        elif is_gelu_activation:
            gate, up = hidden_states.chunk(2, dim=-1)
            approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
            hidden_states = torch.nn.functional.gelu(gate, approximate=approximate) * up
        elif is_swigluoai_uninterleave:
            hidden_states = torch_npu.npu_clipped_swiglu(
                hidden_states,
                interleaved=False,
                alpha=swiglu_alpha,
                limit=swiglu_limit,
                bias=swiglu_beta,
            )
        elif is_situ:
            # Kimi-K3 gate. NOT SwiGLU: this is the branch whose absence made the
            # routed experts silently compute silu(gate)*up.
            situ_beta, situ_linear_beta = _situ_betas(activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
            hidden_states = situ_and_mul(hidden_states, situ_beta, situ_linear_beta)
        else:
            hidden_states = torch_npu.npu_swiglu(hidden_states)
        before_gmm2_evt = torch.npu.current_stream().record_event()
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = w1_scale_bias
            bias2 = w2_scale_bias
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if (
            use_w4a8_per_channel_gmm_swiglu
            and enable_custom_op()
            and activation != MoEActivation.SWIGLUSTEP
            and not is_gelu_activation
            and not is_swigluoai_uninterleave
            and not is_situ
        ):
            hidden_states, swiglu_out_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale if isinstance(w1_scale, list) else [w1_scale],
                x_scale=pertoken_scale,
                group_list=group_list,
                weight_assist_matrix=bias1,
                dequant_mode=0,
                group_list_type=group_list_type,
                swiglu_limit=swiglu_limit,
            )
        elif (
            _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation)
            and not use_mxfp_quant
            and not is_gelu_activation
        ):
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                bias=bias1,
                swiglu_limit=swiglu_limit,
            )
        elif (
            use_gmm_swiglu_quant_fusion
            and activation != MoEActivation.SWIGLUSTEP
            and not is_gelu_activation
            and not is_situ
        ):
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=bias1,
                use_mxfp_quant=use_mxfp_quant,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                swiglu_limit=swiglu_limit,
                mxfp_quant_dtype=mxfp_quant_dtype,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            # gmm1: gate_up_proj
            gmm1_kwargs = {
                "x": [hidden_states],
                "weight": w1 if isinstance(w1, list) else [w1],
                "scale": [w1_scale[0].to(w2_scale[0].dtype)] if isinstance(w1_scale, list) else [w1_scale],
                "bias": bias1,
                "per_token_scale": [pertoken_scale],
                "split_item": 2,
                "group_type": 0,
                "group_list": group_list,
                "group_list_type": group_list_type,
                "output_dtype": _output_dtype,
            }
            if use_mxfp_quant:
                gmm1_kwargs.update(
                    {
                        "scale_dtype": FLOAT8_E8M0FNU_DTYPE,
                        "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
                        "output_dtype": torch.bfloat16,
                    }
                )
            hidden_states = torch_npu.npu_grouped_matmul(**gmm1_kwargs)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            if activation == MoEActivation.SWIGLUSTEP:
                hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
                hidden_states, swiglu_out_scale = DeviceOperator.npu_dynamic_quant(
                    hidden_states, act_quant_type=act_quant_type, use_mxfp_quant=use_mxfp_quant
                )
            elif is_gelu_activation:
                gate, up = hidden_states.chunk(2, dim=-1)
                approximate = "tanh" if activation == MoEActivation.GELU_TANH else "none"
                hidden_states = torch.nn.functional.gelu(gate, approximate=approximate) * up
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
            elif is_swigluoai_uninterleave:
                hidden_states = torch_npu.npu_clipped_swiglu(
                    hidden_states,
                    interleaved=False,
                    alpha=swiglu_alpha,
                    limit=swiglu_limit,
                    bias=swiglu_beta,
                )
                hidden_states, swiglu_out_scale = DeviceOperator.npu_dynamic_quant(
                    hidden_states, act_quant_type=act_quant_type, use_mxfp_quant=use_mxfp_quant
                )
            elif is_situ:
                # SiTU, then requantize for gmm2 -- same shape of step as the
                # interleave branch above. Must precede the HAS_TRITON branch:
                # `swiglu_quant` bakes in SwiGLU, so falling through would
                # silently compute the wrong activation.
                situ_beta, situ_linear_beta = _situ_betas(activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
                hidden_states = situ_and_mul(hidden_states, situ_beta, situ_linear_beta)
                hidden_states, swiglu_out_scale = DeviceOperator.npu_dynamic_quant(
                    hidden_states, act_quant_type=act_quant_type, use_mxfp_quant=use_mxfp_quant
                )
            elif HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states, group_list=group_list, group_list_type=group_list_type
                )
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
        before_gmm2_evt = torch.npu.current_stream().record_event()
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=bias2,
            fallback_output_dtype=_output_dtype,
            mxfp_quant_dtype=mxfp_quant_dtype,
        )
    return hidden_states, before_gmm2_evt


def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    lora_context=None,
    expanded_row_idx: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # MoE LoRA: only attempt injection when an adapter wraps this layer and
    # the comm method provided routing metadata in lora_context.
    # Two paths are supported:
    #   - AllGather: expanded_row_idx + topk_ids from npu_moe_init_routing
    #   - AlltoAll:  lora_context.exchanged_lora_indices + group_list after all_to_all
    lora_routing = None
    if lora_context is not None:  # LoRA applied
        from vllm_ascend.lora.fused_moe import (
            _recover_moe_lora_routing_all2all,
            _recover_moe_lora_routing_allgather,
            moe_lora_apply_w2,
            moe_lora_apply_w13,
        )

        if expanded_row_idx is not None and topk_ids is not None:
            # AllGather path: use npu_moe_init_routing's expanded_row_idx.
            lora_routing = _recover_moe_lora_routing_allgather(lora_context, expanded_row_idx, topk_ids)
        elif getattr(lora_context, "exchanged_lora_indices", None) is not None:
            # AlltoAll path: tokens already sorted by expert after exchange.
            # Build per-row (expert_id, lora_id) directly from group_list.
            lora_routing = _recover_moe_lora_routing_all2all(
                lora_context,
                group_list=group_list,
            )
        else:
            raise AssertionError(
                "MoE LoRA requires either expanded_row_idx+topk_ids "
                "(AllGather) or lora_context.exchanged_lora_indices "
                "(AlltoAll). Neither was provided."
            )

        moe_lora_apply_w13(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=hidden_states,
            lora_routing=lora_routing,
        )

    act_name = getattr(activation, "value", activation)
    if activation == MoEActivation.SWIGLUOAI:
        num_experts, _, hidden_size = w1.shape
        gate_up_out = AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out.view(-1, hidden_size))
    elif act_name == "swigluoai_uninterleave":
        gate_up_out = torch_npu.npu_clipped_swiglu(
            gate_up_out,
            interleaved=False,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            bias=swiglu_beta,
        )
    elif activation == MoEActivation.SWIGLUSTEP:
        gate_up_out = AscendSwigluStepAndMul.swiglustep_forward(gate_up_out, limit=swiglu_limit or 7.0)
    elif activation == MoEActivation.GELU:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate_up_out = torch.nn.functional.gelu(gate) * up
    elif activation == MoEActivation.GELU_TANH:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate_up_out = torch.nn.functional.gelu(gate, approximate="tanh") * up
    elif _is_situ(activation):
        # Kimi-K3's routed-expert gate. This is the site that matters most for
        # this model: without it the `else` branch below silently computed
        # silu(gate)*up, i.e. SwiGLU, with no runtime signal.
        situ_beta, situ_linear_beta = _situ_betas(activation, swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
        gate_up_out = situ_and_mul(gate_up_out, situ_beta, situ_linear_beta)
    else:
        if swiglu_limit > 0:
            gate, up = gate_up_out.chunk(2, dim=-1)
            gate.clamp_(max=swiglu_limit)
            up.clamp_(min=-swiglu_limit, max=swiglu_limit)
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # LoRA w2 delta: applied to the down-proj output, with the activation output
    # as the lora_a input. Reuses the per-row routing computed for w13.
    if lora_routing is not None:
        moe_lora_apply_w2(
            lora_context,
            down_out=hidden_states,
            silu_out=gate_up_out,
            lora_routing=lora_routing,
        )
    return hidden_states, None


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    """
    Unified MoE MLP entry.
    Quant path is dispatched by DeviceOperator with explicit typed kernel flags.
    """
    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    dynamic_scale = mlp_compute_input.dynamic_scale
    topk_scales = mlp_compute_input.topk_scales
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_bias = mlp_compute_input.weights.w1_bias
    w2_bias = mlp_compute_input.weights.w2_bias
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    w1_scale_bias = mlp_compute_input.weights.w1_scale_bias
    w2_scale_bias = mlp_compute_input.weights.w2_scale_bias
    w1_offset = mlp_compute_input.weights.w1_offset
    w2_offset = mlp_compute_input.weights.w2_offset
    activation = mlp_compute_input.activation
    need_trans = mlp_compute_input.need_trans
    dynamic_eplb = mlp_compute_input.dynamic_eplb
    fusion = mlp_compute_input.fusion
    swiglu_limit = mlp_compute_input.swiglu_limit
    swiglu_alpha = mlp_compute_input.swiglu_alpha
    swiglu_beta = mlp_compute_input.swiglu_beta

    if not mlp_compute_input.quant.is_quant:
        return unquant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            activation=activation,
            group_list=group_list,
            group_list_type=group_list_type,
            topk_scales=topk_scales,
            need_trans=need_trans,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            lora_context=mlp_compute_input.lora_context,
            expanded_row_idx=mlp_compute_input.expanded_row_idx,
            topk_ids=mlp_compute_input.topk_ids,
        )

    assert w1_scale is not None and w2_scale is not None
    act_quant_type = torch.int8 if mlp_compute_input.quant.is_int_quant else torch.float8_e4m3fn
    weight_quant_type = torch.float8_e4m3fn
    scale_type = None
    per_token_scale_type = None
    use_bf16 = hidden_states.dtype == torch.bfloat16
    use_mxfp_quant = mlp_compute_input.quant.is_mxfp
    mxfp_quant_dtype = mlp_compute_input.quant.quant_type

    if use_mxfp_quant:
        mxfp = mlp_compute_input.quant.mxfp
        assert mxfp is not None, "mlp_compute_input.quant.mxfp is required when quant_type is W8A8MXFP."
        act_quant_type = mxfp.act_quant_type or act_quant_type
        if mxfp_quant_dtype == QuantType.W4A16MXFP:
            act_quant_type = mxfp.act_quant_type
        weight_quant_type = mxfp.weight_quant_type or weight_quant_type
        if mxfp_quant_dtype in [QuantType.W4A8MXFP, QuantType.W4A16MXFP]:
            weight_quant_type = mxfp.weight_quant_type
        scale_type = mxfp.scale_dtype
        per_token_scale_type = mxfp.per_token_scale_dtype
        use_bf16 = mxfp.use_bf16

    return quant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group_list=group_list,
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        w1_offset=w1_offset,
        w2_offset=w2_offset,
        fusion=fusion,
        dynamic_eplb=dynamic_eplb,
        use_mxfp_quant=use_mxfp_quant,
        mxfp_quant_dtype=mxfp_quant_dtype,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_bf16=use_bf16,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        use_w4a8_per_channel_gmm_swiglu=mlp_compute_input.quant.use_w4a8_per_channel_gmm_swiglu,
    )
