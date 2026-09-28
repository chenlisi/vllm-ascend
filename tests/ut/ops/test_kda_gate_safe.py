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
"""Kimi-K3 Stage 1 UT: the KDA gate's "safe gate" activation.

Kimi-K3's checkpoint sets ``linear_attn_config.gate_lower_bound = -5.0`` and the
vendor model passes it to the KDA kernel as ``safe_gate=True, lower_bound=-5.0``.
Upstream vLLM's KDA path has no such parameter -- its ``**kwargs`` swallows the
argument -- so the activation was silently different for this model.

The important part of this test is the **shape** of the substitution, not just
its numeric value: ``safe_gate`` *replaces* the activation

    g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))     # safe

rather than *clamping* the default one

    g = -exp(A_log) * softplus(g + dt_bias)                   # default

The two are not interchangeable, and a clamp is a plausible-looking but wrong
approximation, so it is asserted against explicitly below.

Reference formula: `fla`'s ``kda_gate_chunk_cumsum_vector_kernel`` (the two
branches guarded by ``USE_LOWER_BOUND``). `fla` is not importable in this
environment; the formula was read from a copy of that implementation on the
shared mount, so this test pins the *semantics*, not a vendor-installed package.
"""

import pytest
import torch
import torch.nn.functional as F


def _reference_default(g, A, bias=None, beta=1.0, threshold=20.0):
    """-exp(A) * softplus(g + bias); the pre-existing (non-safe) activation."""
    if bias is not None:
        g = g + bias
    scaled = g * beta
    softplus = torch.where(scaled > threshold, g, (1.0 / beta) * torch.log1p(torch.exp(scaled)))
    return -torch.exp(A) * softplus


def _reference_safe(g, A, lower_bound=-5.0, bias=None):
    """lower_bound * sigmoid(exp(A) * (g + bias)); the safe-gate activation."""
    if bias is not None:
        g = g + bias
    return lower_bound * torch.sigmoid(torch.exp(A) * g)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# --------------------------------------------------------------------------
# The formula itself
# --------------------------------------------------------------------------
def test_safe_gate_is_not_a_clamped_default():
    """Regression guard for the exact wrong approximation.

    Clamping the default activation to `lower_bound` produces a *different*
    curve: the default saturates only as `g -> +inf`, whereas the safe form has
    already reached `lower_bound` for moderately positive `g`, and it decays to
    0 much faster for negative `g`. A clamp therefore cannot express it.
    """
    A = torch.tensor([1.5])
    g = torch.linspace(-12.0, 12.0, 25)

    default = _reference_default(g, A)
    clamped = torch.clamp(default, min=-5.0)
    safe = _reference_safe(g, A)

    # Same at the extremes (both -> 0 for very negative g, both -> lower_bound
    # for very positive g), which is exactly why a clamp looks plausible.
    assert torch.allclose(clamped[:4], safe[:4], atol=1e-3)
    assert torch.allclose(clamped[-4:], safe[-4:], atol=1e-3)

    # But they diverge in the middle, where most of the mass lives.
    mid = slice(8, 17)
    assert (clamped[mid] - safe[mid]).abs().max() > 0.5, (
        "clamp and safe gate agreed everywhere; the distinction being tested "
        "would be vacuous"
    )


def test_safe_gate_is_bounded_below_by_lower_bound():
    """`lower_bound` is a floor on the activation, approached from above."""
    A = torch.tensor([1.5])
    g = torch.linspace(-20.0, 20.0, 200)
    safe = _reference_safe(g, A, lower_bound=-5.0)

    assert safe.min() >= -5.0 - 1e-5, "safe gate went below its own lower bound"
    # Approached but never exceeded, and reached for large positive g.
    assert safe[-1].item() == pytest.approx(-5.0, abs=1e-3)


def test_safe_gate_decays_faster_than_default_for_negative_g():
    """The behavioural difference that motivated the fix.

    For negative logits the safe form is much closer to 0 than the default
    (which keeps decaying roughly linearly in |g|). This changes the recurrent
    decay factor exp(g) per chunk -- the quantity that accumulates over a long
    prefill, and the reason a silent swap shows up as a G3 accuracy drift.
    """
    A = torch.tensor([1.5])
    g = torch.tensor([-4.0, -2.0])
    assert (_reference_safe(g, A) > _reference_default(g, A)).all()


def test_safe_gate_lower_bound_is_read_from_config():
    """The model-side accessor must pick up the checkpoint's -5.0.

    Reads through `hf_text_config` (the K3 checkpoint nests
    `linear_attn_config` under `text_config`); the accessor is exercised on a
    stand-in so this stays a CPU-only unit test.
    """
    from vllm_ascend.ops.gdn import AscendKimiGatedDeltaNetAttention

    class _ModelCfg:
        def __init__(self, text_cfg):
            self.hf_text_config = text_cfg

    obj = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)

    obj.model_config = _ModelCfg(_Cfg(linear_attn_config={"gate_lower_bound": -5.0}))
    assert obj._gate_lower_bound() == -5.0

    # Absent -> None, which selects the (unchanged) default activation.
    obj.model_config = _ModelCfg(_Cfg(linear_attn_config={}))
    assert obj._gate_lower_bound() is None

    # No linear_attn_config at all is also not an error.
    obj.model_config = _ModelCfg(_Cfg())
    assert obj._gate_lower_bound() is None


def test_fused_kda_gate_default_path_identity():
    """`lower_bound=None` must keep the default activation selected.

    The switch is a constexpr on the kernel; the Python side decides it by
    `lower_bound is None`. Asserted structurally so a future default change is
    caught here rather than by an accuracy regression on another model.
    """
    import inspect

    from vllm_ascend.ops.triton.kda import kda as kda_mod

    sig = inspect.signature(kda_mod.fused_kda_gate)
    assert sig.parameters["lower_bound"].default is None


def test_fused_kda_gate_rejects_out_of_range_lower_bound():
    """Mirrors the platform-side validation (lower_bound must be in [-5, 0)).

    Asserted on the Python guard rather than by calling the kernel: the kernel
    needs an NPU, and this specific contract is enforced before launch.
    """
    import inspect

    from vllm_ascend.ops.triton.kda import kda as kda_mod

    src = inspect.getsource(kda_mod.fused_kda_gate)
    assert "lower_bound < 0.0" in src, (
        "fused_kda_gate no longer validates lower_bound < 0; the safe-gate "
        "sigmoid would silently produce positive gates"
    )


# --------------------------------------------------------------------------
# Kernel-level equivalence (needs an NPU)
# --------------------------------------------------------------------------
def _npu_ready():
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.skip("no NPU")


def _run_gate(lower_bound, H=4, D=128, T=64, with_bias=True, seed=0):
    """Invoke the Ascend `fused_kda_gate` kernel and return (out, reference)."""
    _npu_ready()
    from vllm_ascend.utils import bootstrap_custom_op_env

    # The CANN vendor lib is what the real serve path loads; without it the op
    # table is incomplete on a bare import.
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.ops.triton.kda.kda import fused_kda_gate

    torch.manual_seed(seed)
    A = torch.log(torch.empty(H).uniform_(1, 16)).npu()
    g = (torch.randn(T, H * D) * 2.0).npu()
    bias = torch.randn(H * D).npu() if with_bias else None

    out = fused_kda_gate(g, A.reshape(-1, 1), D, g_bias=bias, lower_bound=lower_bound).reshape(T, H * D)

    a_col = torch.exp(A).repeat_interleave(D).unsqueeze(0)
    gg = g + bias if bias is not None else g
    if lower_bound is None:
        ref = (-a_col) * F.softplus(gg)
    else:
        ref = lower_bound * torch.sigmoid(a_col * gg)
    return out.float(), ref.float()


def test_kernel_default_branch_matches_reference():
    """The default (non-safe) branch is unchanged by this work.

    Pinned numerically so a future edit to the shared kernel cannot silently
    shift the activation for every other KDA/GDN model.
    """
    out, ref = _run_gate(lower_bound=None)
    assert (out - ref).abs().max().item() < 1e-4, (out - ref).abs().max().item()


def test_kernel_safe_branch_matches_reference():
    """`lower_bound=-5.0` produces the vendor safe-gate formula on device."""
    out, ref = _run_gate(lower_bound=-5.0)
    assert (out - ref).abs().max().item() < 1e-4, (out - ref).abs().max().item()


def test_kernel_safe_and_default_branches_differ_materially():
    """The two branches are genuinely different on hardware.

    Guards against the branches being wired to the same code path (e.g. both
    reading the default) -- a mistake that would still pass each branch's
    individual reference check if the reference were derived from the same
    (wrong) selector.
    """
    out_default, _ = _run_gate(lower_bound=None)
    out_safe, _ = _run_gate(lower_bound=-5.0)
    assert (out_default - out_safe).abs().max().item() > 0.5, (
        "safe and default gate activations agree on device; the constexpr "
        "selector is not taking effect"
    )


def test_kernel_safe_gate_respects_lower_bound_with_bias():
    """With a bias present the bound still holds (bias is added pre-activation)."""
    out, ref = _run_gate(lower_bound=-5.0, with_bias=True)
    assert (out - ref).abs().max().item() < 1e-4
    assert out.min().item() >= -5.0 - 1e-4
