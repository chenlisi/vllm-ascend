# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Ascend HyperConnection (gated residual) for Qwen4Exp.

Semantics come from the platform-neutral ``common/hyperconnection.py``
implementation (``mix``/``combine`` split). The NVIDIA owner delays each
combine to the next mix boundary only so its fused kernels can merge the two
launches; none of those kernels exist on Ascend, so the non-delayed form is
used. Both forms are line-equivalent in arithmetic and the unit tests compare
them against one transcription.

The parameter layout mirrors the NVIDIA owner, because it is what the
checkpoint's key mapping targets:

* ``input_mix_weight_down_block_inject`` -- one packed column-parallel linear
  whose output shards are ``[hc_lowrank, hc_count]`` (+ a 16-row pad when the
  pair is not already aligned). The checkpoint stores ``input_mix_weight_down``
  and ``block_inject_weight`` separately and the model's
  ``_EXTRA_WEIGHTS_MAPPER`` stacks them into shards 0 and 1 of this tensor.
* ``input_mix_weight_up`` -- a separate replicated linear ``(hc_lowrank ->
  HC*H)``.
* When the mixer does not combine (the final mixer), the down projection is a
  standalone replicated linear instead of shard 0 of the packed one.

All three are TP-replicated: the multi-stream hidden state is never sharded.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)

from vllm_ascend.utils import vllm_version_is

logger = init_logger(__name__)

__all__ = [
    "AscendGatedResidual",
    "AscendGroupedGemmaRMSNorm",
    "AscendHyperConnectionConfig",
]


@dataclass
class AscendHyperConnectionConfig:
    """Configuration for the Ascend gated residual (Qwen4Exp subset)."""

    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


class AscendGroupedGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm applied per HC branch.

    ``group_size == hidden_size`` normalizes each H-wide stream independently
    while keeping a single affine weight over the whole ``HC*H`` layout, which
    is the checkpoint's parameter shape. Statistics run in fp32.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.variance_epsilon)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


def _merged_linear(
    input_size: int,
    output_sizes: list[int],
    *,
    params_dtype: torch.dtype,
    prefix: str,
) -> MergedColumnParallelLinear:
    """TP-replicated packed linear (no bias, no sharding)."""
    kwargs: dict[str, object] = {
        "input_size": input_size,
        "output_sizes": output_sizes,
        "bias": False,
        "gather_output": False,
        "params_dtype": params_dtype,
        "quant_config": None,
        "prefix": prefix,
        "disable_tp": True,
    }
    if vllm_version_is("0.31.0"):
        kwargs["return_bias"] = True
    return MergedColumnParallelLinear(**kwargs)


def _replicated_linear(
    input_size: int,
    output_size: int,
    *,
    params_dtype: torch.dtype,
    prefix: str,
) -> ReplicatedLinear:
    kwargs: dict[str, object] = {
        "input_size": input_size,
        "output_size": output_size,
        "bias": False,
        "params_dtype": params_dtype,
        "quant_config": None,
        "prefix": prefix,
    }
    if vllm_version_is("0.31.0"):
        kwargs["return_bias"] = True
    return ReplicatedLinear(**kwargs)


def _call_linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call a vLLM linear and drop its bias return."""
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


class AscendGatedResidual(nn.Module):
    """Gated HyperConnection: low-rank gated mixing plus stream injection."""

    def __init__(
        self,
        config: AscendHyperConnectionConfig,
        *,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine
        self.hyper_hidden_size = config.hyper_hidden_size

        norm_size = self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        self.hc_norm = AscendGroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # The packed pair is padded to a 16-row multiple, matching the
        # checkpoint loader's `_input_mix_padding` slot (kept at zero).
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            output_sizes = [self.lora_rank, self.hc_count]
            if self.pad_size:
                output_sizes.append(self.pad_size)
            self.input_mix_weight_down_block_inject = _merged_linear(
                self.hyper_hidden_size,
                output_sizes,
                params_dtype=config.params_dtype,
                prefix=f"{prefix}.input_mix_weight_down_block_inject",
            )
        else:
            self.input_mix_weight_down = _replicated_linear(
                self.hyper_hidden_size,
                self.lora_rank,
                params_dtype=config.params_dtype,
                prefix=f"{prefix}.input_mix_weight_down",
            )
        self.input_mix_weight_up = _replicated_linear(
            self.lora_rank,
            self.hyper_hidden_size,
            params_dtype=config.params_dtype,
            prefix=f"{prefix}.input_mix_weight_up",
        )

    # -- helpers ----------------------------------------------------------
    def _normalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.hc_per_branch_norm:
            return self.hc_norm(hidden_states)
        return self.hc_norm(
            hidden_states.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)

    def _down_and_inject(
        self, normed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Down projection + SiLU, plus the injection logits when combining."""
        if not self.use_combine:
            lora = _call_linear(self.input_mix_weight_down, normed)
            return F.silu(lora / self.hc_count), None
        packed = _call_linear(self.input_mix_weight_down_block_inject, normed)
        # The trailing rows are the 16-row alignment pad; they carry no
        # meaning and are dropped here.
        lora, injection = packed.split(
            [self.lora_rank, self.hc_count, self.pad_size], dim=-1
        )[:2]
        return F.silu(lora / self.hc_count), injection

    def _mix_from_normed(
        self, normed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        lora, injection = self._down_and_inject(normed)
        gate = torch.sigmoid(_call_linear(self.input_mix_weight_up, lora)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        mixed = (gate * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        return mixed.to(normed.dtype), injection

    # -- public API -------------------------------------------------------
    def mix(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """RMSNorm -> low-rank gate -> gated mean over the HC streams.

        Returns ``(block_input, injection)``; ``injection`` is ``None`` for a
        mixer without ``use_combine`` (the block output then applies with unit
        weight).
        """
        if hyper_input.shape[-1] != self.hyper_hidden_size:
            raise ValueError(
                f"hyper input width {hyper_input.shape[-1]} != HC*H {self.hyper_hidden_size}"
            )
        return self._mix_from_normed(self._normalize(hyper_input))

    def combine(
        self,
        hyper_input: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add the block output to every stream, weighted by the injection."""
        if hyper_input.shape[-1] != self.hyper_hidden_size:
            raise ValueError(
                f"hyper input width {hyper_input.shape[-1]} != HC*H {self.hyper_hidden_size}"
            )
        if block_output.shape[-1] != self.hidden_size:
            raise ValueError(
                f"block output width {block_output.shape[-1]} != H {self.hidden_size}"
            )
        residual = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        if injection is None:
            # A missing injection applies the block output to every stream
            # with unit weight. That is not only the final mixer's contract:
            # the MTP drafter starts its first step with a pending combine
            # (``prev_block_output`` set, ``prev_injection`` None), and the
            # upstream fused ``hc_combine_norm`` takes the same branch when
            # ``injection_logits is None``. Raising here instead broke every
            # drafter step.
            #
            # That branch is legitimate, so the old unconditional
            # ``raise ValueError`` cannot come back. But it used to double as
            # the only alarm for "the previous layer's injection did not get
            # through" -- dropping it entirely would turn that wiring bug
            # into a silent output shift. Leave one trace per process
            # instead: a correct run emits it at most once (the drafter's
            # first step), so a second sighting still means the caller is
            # somewhere unexpected.
            logger.warning_once(
                "hyper-connection combine ran without an injection "
                "(unit-weight branch). Expected for the MTP drafter's first "
                "step only; if this mixer is not part of a drafter, the "
                "previous layer's injection was lost."
            )
            return (residual + block_output.unsqueeze(-2)).flatten(-2).to(
                hyper_input.dtype
            )
        injection_weight = 2.0 * torch.sigmoid(injection.float() / self.hc_count)
        output = residual + block_output.unsqueeze(-2) * injection_weight.to(
            block_output.dtype
        ).unsqueeze(-1)
        return output.flatten(-2).to(hyper_input.dtype)

    def combine_and_mix(
        self,
        hyper_input: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Materialize a pending combine, then start this module's mix.

        The non-delayed counterpart of the upstream fused entry point: two
        launches with identical arithmetic instead of one fused launch.
        """
        combined = self.combine(hyper_input, prev_block_output, prev_injection)
        block_input, injection = self.mix(combined)
        return combined, block_input, injection
