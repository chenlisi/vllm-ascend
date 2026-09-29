# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
#
"""Kimi-K3 SiTU activation and cross-layer residual math.

Kept separate from ``model.py`` so the numerics can be unit-tested without
constructing a model (no NPU, no distributed init, no weights).
"""

import torch
from torch import nn

from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("SituAndMul")
class AscendSituAndMul(CustomOp):
    """SiTU activation used by Kimi-K3.

    Math (vendor ``modeling_kimi.SituAndMul.forward``, replicated verbatim)::

        d = x.shape[-1] // 2
        gate, up = x[..., :d], x[..., d:]           # gate/up split at half
        a = beta * tanh(gate / beta) * sigmoid(gate)
        up = linear_beta * tanh(up / linear_beta)   # only when linear_beta set
        return a * up

    Two precision requirements carried over from the vendor implementation:

    1. ``gate``/``up`` are promoted to float32 before any elementwise math.
       Computing in bf16 all the way through is the most common source of
       accuracy loss for this activation, because the tanh/sigmoid pair
       amplifies rounding in the exponent.
    2. The result is cast back to the input dtype at the very end.

    NOTE on ``activation_situ_beta`` / ``activation_situ_linear_beta``: the vendor
    helper ``_get_situ_activation_params`` returns ``beta or 1.0``, so a config
    value of 0.0 is indistinguishable from unset and falls back to 1.0. That is
    reproduced deliberately -- the weights were trained against it.

    Registration: the in-tree ``@CustomOp.register("SituAndMul")`` decorator is
    required because ``CustomOp.__init__`` reads ``cls.name`` during
    ``dispatch_forward``, and only the decorator sets it at class-definition
    time. The OOT mapping lives in ``REGISTERED_ASCEND_OPS``.
    """

    def __init__(
        self,
        beta: float = 1.0,
        linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return situ_and_mul(x, self.beta, self.linear_beta)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1 (golden baseline) deliberately runs the non-fused path:
        # correctness first, fusion later. The platform grouped-GEMM paths fuse
        # swiglu-family activations by default, so routing SiTU through them
        # would silently substitute SwiGLU -- an accuracy gap, not an error.
        return self.forward_native(x)


def situ_and_mul(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """The SiTU gate itself, as a free function.

    Split out of ``AscendSituAndMul`` so the MoE routed-expert path can apply
    exactly the same math without constructing a ``CustomOp`` (whose
    ``__init__`` requires a vLLM config context, which is not available where
    the grouped-GEMM activation is applied). One implementation, two callers --
    the dense/shared-expert path and the routed-expert path cannot drift.

    ``float32`` promotion of the intermediates is load-bearing, not stylistic:
    tanh and sigmoid both amplify rounding in the exponent, so computing the
    gate in bf16 end to end is the most common source of accuracy loss on this
    activation. The result is cast back to the input dtype.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def situ_params(config) -> tuple[float, float | None]:
    """Read SiTU betas the way the vendor helper does.

    Vendor ``_get_situ_activation_params`` uses ``beta or 1.0``, which makes a
    configured ``0.0`` indistinguishable from "unset". Reproduced verbatim so
    the activation matches what the weights were trained against.
    """
    beta = getattr(config, "activation_situ_beta", None)
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return beta or 1.0, linear_beta


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Module,
    norm: nn.Module,
) -> torch.Tensor:
    """RMSNorm + single-score softmax mix over the accumulating residual blocks.

    Copied verbatim from the vendor (``modeling_kimi._apply_attn_res``),
    including the fp32 promotion points: the whole computation runs in float32
    and is cast back at the end. Running this in bf16 end-to-end would be a
    silent accuracy regression.

    ``prefix_sum``: (num_tokens, hidden); ``block_residual``:
    (num_tokens, num_blocks, hidden). Returns (num_tokens, hidden).
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)
