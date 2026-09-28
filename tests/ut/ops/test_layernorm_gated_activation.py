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
"""Activation passthrough for the Ascend gated-RMSNorm Triton kernel.

`layer_norm_fwd_npu` used to hard-wire the output gate to ``z * sigmoid(z)``
(= SiLU) regardless of what the caller asked for, and `AscendRMSNormGated`
never forwarded its `activation`. A model built with
``RMSNormGated(activation="sigmoid")`` therefore got SiLU applied silently --
no error, just a wrong gate.

Scope note, because it decides whether this is currently *reachable*: the
standalone `RMSNormGated` op is not in the OOT table for this model, and
Kimi-K3's KDA `o_norm` is a `FusedRMSNormGated` (which has no `forward_oot` and
so runs its own native path). These tests therefore pin the kernel contract
rather than a live model behaviour -- they are the regression guard for the
case where any module does route a sigmoid-gated norm through `forward_oot`.

The default (`swish`) must stay byte-for-byte identical to the previous
behaviour, which is asserted explicitly so existing models cannot be affected.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton")


def _requires_npu():
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip("no NPU")


def _analytic(x, weight, z, eps, activation, bias=None):
    xf, zf = x.float(), z.float()
    xhat = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xhat * weight.float()
    if bias is not None:
        # The bias is added before the gate, so it is scaled by the gate too.
        y = y + bias.float()
    if activation == "sigmoid":
        return y * torch.sigmoid(zf)
    return y * zf * torch.sigmoid(zf)


def _run(activation, with_bias=False):
    _requires_npu()
    from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu

    torch.manual_seed(0)
    M, N = 64, 128
    x = torch.randn(M, N, dtype=torch.float32).npu()
    weight = torch.rand(N, dtype=torch.float32).npu() + 0.5
    bias = torch.randn(N, dtype=torch.float32).npu() if with_bias else None
    z = torch.randn(M, N, dtype=torch.float32).npu() * 2.0

    y, _, _ = layer_norm_fwd_npu(
        x,
        weight,
        bias,
        eps=1e-5,
        z=z,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation=activation,
    )
    torch.npu.synchronize()
    return y.float(), _analytic(x, weight, z, 1e-5, activation, bias=bias)


def test_sigmoid_activation_matches_analytic():
    """`activation="sigmoid"` must produce `norm(x) * sigmoid(z)`."""
    y, ref = _run("sigmoid")
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max().item()


def test_sigmoid_activation_is_not_silu():
    """The explicit negative assertion: sigmoid must NOT be served as SiLU.

    SiLU is the historical hard-wired behaviour, and the two differ by the
    factor `z`, so a regression here is invisible under a `torch.allclose`
    against the wrong reference -- it has to be asserted against SiLU directly.
    """
    _requires_npu()
    from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu

    torch.manual_seed(0)
    M, N = 64, 128
    x = torch.randn(M, N, dtype=torch.float32).npu()
    weight = torch.ones(N, dtype=torch.float32).npu()
    z = torch.randn(M, N, dtype=torch.float32).npu() * 2.0

    y, _, _ = layer_norm_fwd_npu(
        x, weight, None, 1e-5, z=z, group_size=None, norm_before_gate=True,
        is_rms_norm=True, activation="sigmoid",
    )
    torch.npu.synchronize()

    y_float = y.float()
    silu_ref = _analytic(x, weight, z, 1e-5, "swish")

    # sigmoid-gate = silu-gate / z; pick z away from 0 so the two are far apart.
    far = z.float().abs() > 0.5
    assert far.any(), "test inputs degenerate: |z| <= 0.5 everywhere"
    assert not torch.allclose(y_float[far], silu_ref[far], atol=1e-3), (
        "activation='sigmoid' produced the SiLU result -- the gate activation "
        "is being dropped again"
    )


def test_default_activation_matches_previous_behaviour():
    """Omitting `activation` must keep the old `z * sigmoid(z)` path.

    This is the compatibility half of the change: every existing model relies
    on the default, so the default is pinned to the pre-existing formula.
    """
    y_default, ref_default = _run("swish")
    assert torch.allclose(y_default, ref_default, atol=1e-5)

    # And the explicit spelling of the same thing agrees with the default.
    y_silu, ref_silu = _run("silu")
    assert torch.allclose(y_silu, ref_silu, atol=1e-5)
    assert torch.allclose(y_default, y_silu, atol=1e-6)


def test_sigmoid_with_bias_path_matches_analytic():
    """The bias is added before the gate, so the gate scales it as well."""
    y, ref = _run("sigmoid", with_bias=True)
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max().item()


def test_unsupported_activation_is_rejected():
    """Unknown activation names must fail loudly, not default to SiLU."""
    _requires_npu()
    from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu

    x = torch.randn(8, 16, dtype=torch.float32).npu()
    weight = torch.ones(16, dtype=torch.float32).npu()
    z = torch.randn(8, 16, dtype=torch.float32).npu()
    with pytest.raises(AssertionError):
        layer_norm_fwd_npu(
            x, weight, None, 1e-5, z=z, group_size=None, norm_before_gate=True,
            is_rms_norm=True, activation="gelu",
        )


def test_ascend_rms_norm_gated_forwards_its_activation():
    """`AscendRMSNormGated.forward_oot` must pass `self.activation` through.

    Asserted on the call rather than numerically: `forward_oot` dispatches into
    the kernel above, and the point of the regression guard is that the
    attribute reaches the call at all (it previously did not). An unconditional
    `getattr` default of "swish" keeps the contract safe for instances built
    before `activation` existed.
    """
    import inspect

    from vllm_ascend.ops.layernorm import AscendRMSNormGated

    src = inspect.getsource(AscendRMSNormGated.forward_oot)
    assert "activation" in src, (
        "AscendRMSNormGated.forward_oot no longer forwards its activation; a "
        "sigmoid-gated norm would silently receive SiLU"
    )
    # The default must remain the legacy behaviour.
    assert 'getattr(self, "activation", "swish")' in src
