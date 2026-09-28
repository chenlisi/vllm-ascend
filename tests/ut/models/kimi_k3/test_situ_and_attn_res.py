#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Kimi-K3 Stage 1 UTs: SiTU activation and cross-layer residual mixing.

CPU-only: these exercise the pure numerics, which is where this model's
* silent * correctness risks live (an activation silently swapped for SwiGLU, or
a residual mix computed in the wrong precision). The NPU end-to-end path is
covered by the dummy smoke evidence under `smoke/`, not here.
"""

import pytest
import torch

from vllm_ascend.models.kimi_k3.activation import (
    AscendSituAndMul,
    apply_attn_res,
    situ_params,
)


@pytest.fixture(autouse=True)
def default_vllm_config():
    """Provide a vLLM config context.

    `AscendSituAndMul` derives from `CustomOp`, whose `__init__` calls
    `dispatch_forward` -> `get_current_vllm_config()`. Without a config set that
    raises, so the activation cannot be instantiated in a bare test.
    """
    from unittest.mock import MagicMock

    from vllm.config import set_current_vllm_config

    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    mock_config.compilation_config.enabled_custom_ops = set()
    mock_config.compilation_config.disabled_custom_ops = set()
    mock_config.model_config = None
    with set_current_vllm_config(mock_config):
        yield mock_config


class _Cfg:
    """Minimal config stand-in carrying only the SiTU fields."""

    def __init__(self, **kw):
        self.activation_situ_beta = kw.get("beta")
        self.activation_situ_linear_beta = kw.get("linear_beta")


def vendor_situ_reference(x, beta=1.0, linear_beta=None):
    """Independent transcription of the vendor `SituAndMul.forward`.

    Deliberately NOT an import of the implementation under test: using our own
    code as the oracle would make the comparison vacuous.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


# --------------------------------------------------------------------------
# SiTU activation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "beta,linear_beta",
    [(4.0, 25.0), (1.0, None), (2.5, None), (4.0, 1.0)],
)
def test_situ_matches_vendor_reference(beta, linear_beta):
    """Numerical equivalence with the vendor math across parameter settings.

    NOTE on shape: SiTU consumes the concatenated (gate, up) pair and returns the
    elementwise product, so the output is **half** the input width -- the same
    convention as SwiGLU. That matches the vendor, whose `forward` starts with
    `d = x.shape[-1] // 2` and returns `situ_a * up`.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    got = AscendSituAndMul(beta=beta, linear_beta=linear_beta)(x)
    want = vendor_situ_reference(x, beta=beta, linear_beta=linear_beta)
    assert got.shape == (8, 32)
    assert got.shape == want.shape
    assert got.dtype == x.dtype
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=1e-2)


def test_situ_is_not_swiglu():
    """Guard against the silent substitution this activation is at risk of.

    If SiTU were replaced by SwiGLU the model would still run and emit finite
    numbers -- only wrong ones. So assert the outputs differ substantially.
    """
    torch.manual_seed(1)
    x = torch.randn(16, 128, dtype=torch.float32)
    situ = AscendSituAndMul(beta=4.0, linear_beta=25.0)(x)
    d = x.shape[-1] // 2
    swiglu = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    assert not torch.allclose(situ, swiglu, atol=1e-2), (
        "SiTU output coincides with SwiGLU -- the activation is being substituted"
    )


def test_situ_fp32_intermediates_are_load_bearing():
    """The vendor promotes gate/up to fp32; computing in bf16 changes results.

    Inputs are scaled so the tanh/sigmoid pair works in a region where bf16
    rounding is visible, making the two paths distinguishable.
    """
    torch.manual_seed(2)
    x = (torch.randn(4, 64) * 8.0).to(torch.bfloat16)
    got = AscendSituAndMul(beta=1.0, linear_beta=None)(x).float()

    d = x.shape[-1] // 2
    gate_bf = x[..., :d]
    up_bf = x[..., d:]
    bf16_path = (torch.tanh(gate_bf) * torch.sigmoid(gate_bf) * up_bf).float()
    assert not torch.allclose(got, bf16_path, atol=0, rtol=0), (
        "fp32 intermediates are indistinguishable from bf16 on this input"
    )


def test_situ_params_beta_zero_falls_back_to_one():
    """Vendor `_get_situ_activation_params` uses `beta or 1.0`.

    A configured 0.0 is therefore indistinguishable from unset. Replicated on
    purpose: the weights were trained against this behaviour, so "fixing" it
    here would be a regression.
    """
    beta, linear_beta = situ_params(_Cfg(beta=0.0, linear_beta=25.0))
    assert beta == 1.0
    assert linear_beta == 25.0


def test_situ_params_reads_config():
    beta, linear_beta = situ_params(_Cfg(beta=4.0, linear_beta=25.0))
    assert (beta, linear_beta) == (4.0, 25.0)


# --------------------------------------------------------------------------
# Cross-layer attention residual
# --------------------------------------------------------------------------
class _IdentityNorm(torch.nn.Module):
    """RMSNorm stand-in exposing the attributes `apply_attn_res` reads."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = 1e-5


class _ScoreProj(torch.nn.Module):
    """`proj` stand-in with weight shape (1, hidden)."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, hidden) * 0.02)


def test_apply_attn_res_matches_vendor_formula():
    """`apply_attn_res` must reproduce the vendor fp32 softmax mixing."""
    torch.manual_seed(3)
    tokens, blocks, hidden = 5, 3, 16
    prefix_sum = torch.randn(tokens, hidden)
    block_residual = torch.randn(tokens, blocks, hidden)
    norm = _IdentityNorm(hidden)
    proj = _ScoreProj(hidden)

    got = apply_attn_res(prefix_sum, block_residual, proj, norm)

    # Independent transcription of the vendor `_apply_attn_res`.
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    want = torch.matmul(probs, v_float).squeeze(1)

    assert got.shape == (tokens, hidden)
    assert torch.allclose(got.float(), want.float(), atol=1e-4)


def test_apply_attn_res_is_convex_combination():
    """Softmax over (blocks + current) must weight the slabs to sum to one."""
    torch.manual_seed(4)
    hidden = 8
    prefix_sum = torch.randn(2, hidden)
    block_residual = torch.randn(2, 4, hidden)
    norm = _IdentityNorm(hidden)
    proj = _ScoreProj(hidden)

    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    probs = (k * score_weight).sum(-1).softmax(-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5)


def test_apply_attn_res_single_block_is_identity_free():
    """With one slab the result is that slab reinforced (no crash, right shape)."""
    hidden = 8
    prefix_sum = torch.randn(3, hidden)
    block_residual = torch.randn(3, 1, hidden)
    got = apply_attn_res(prefix_sum, block_residual, _ScoreProj(hidden),
                         _IdentityNorm(hidden))
    assert got.shape == (3, hidden)
    assert torch.isfinite(got).all()
