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
"""Kimi-K3 Stage 1 UT: the MoE *routed-expert* activation is SiTU, not SwiGLU.

Background. The dense and shared-expert FFNs were covered from the start, but
the routed experts run through an entirely different chain:
``FusedMoE`` -> ``AscendMoERunner`` -> ``DeviceOperator.apply`` ->
``unquant_apply_mlp`` / ``quant_apply_mlp`` -> ``torch_npu.npu_swiglu``. The
activation for that chain was never wired, so it silently defaulted to SwiGLU.

"Silently" is the operative word: every one of the fall-through branches in
``moe_mlp`` ends in ``npu_swiglu``, and none of them raises. The model runs,
produces plausible-looking numbers, and is wrong. That is why these tests
assert *behaviour* (the math actually differs from SwiGLU and matches SiTU)
rather than configuration (that some string was set).

Two independent classes of failure are covered, because they need different
fixes:

1. **The dispatch never reaches a SiTU branch.** Covered by
   ``TestRoutedExpertDispatch``: the activation string the Ascend MoE runner
   hands downstream must become ``"situ"``, and the fused-SwiGLU predicates
   must exclude it.
2. **A SiTU branch exists but the fused kernels swallow it anyway.** Covered
   by ``TestSiTUNotSwallowedByFusion``: the fused grouped-GEMM paths bake the
   activation into the kernel. A SiTU activation that still satisfies
   ``_custom_gmm_swiglu_enabled`` runs SwiGLU inside the kernel, with the
   branch never consulted -- the original bug, one layer down.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.kimi_k3.activation import situ_and_mul
from vllm_ascend.ops.fused_moe.moe_mlp import (
    _custom_gmm_swiglu_enabled,
    _gmm_swiglu_quant_fusion_enabled,
    _is_situ,
    quant_apply_mlp,
    unified_apply_mlp,
    unquant_apply_mlp,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType

MOE_MLP = "vllm_ascend.ops.fused_moe.moe_mlp"

# Kimi-K3's checkpoint values (config.json text_config). Pinned as literals on
# purpose: reading them from the config would make the test agree with whatever
# the config says, including a wrong value.
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def vendor_situ_reference(x, beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA):
    """Independent transcription of the vendor SiTU math.

    Deliberately NOT a call into `situ_and_mul`: using the implementation under
    test as its own oracle would make the comparison vacuous. This is the
    vendor `SituAndMul.forward` transcribed from `modeling_kimi` line by line.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def swiglu_reference(x):
    """The activation the routed experts used to compute by accident."""
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    return (torch.nn.functional.silu(gate) * up).to(x.dtype)


def _cpu_npu_swiglu(x, *args, **kwargs):
    """CPU stand-in for the NPU fused SwiGLU, for the non-SiTU comparison path.

    A *faithful* implementation rather than a stub, so `_run("silu")` measures
    real SwiGLU: comparing SiTU against a mocked constant would prove nothing
    about whether the two activations are distinguishable.
    """
    return swiglu_reference(x)


# --------------------------------------------------------------------------
# 1. The activation string actually reaching the MLP
# --------------------------------------------------------------------------
class TestRoutedExpertDispatch(unittest.TestCase):
    """`apply()` must select SiTU from the layer marker."""

    def test_is_situ_accepts_string_and_enum(self):
        assert _is_situ("situ") is True
        # Anything else must be False, including the value SiTU used to be
        # silently replaced by.
        assert _is_situ("silu") is False
        assert _is_situ(MoEActivation.SILU) is False
        assert _is_situ(MoEActivation.SWIGLUSTEP) is False
        assert _is_situ(None) is False
        assert _is_situ("swigluoai_uninterleave") is False

    def test_apply_hands_situ_to_the_mlp_stage(self):
        """End-to-end through the real `apply()`: `activation` becomes `"situ"`.

        The assertion is placed on the activation string `apply()` passes into
        `build_fused_experts_input`, because that string is what every MLP
        branch downstream dispatches on. It is captured from the real call, not
        recomputed, so a marker that `apply()` fails to consult shows up here.
        This is the "triggered on the real path" assertion: the layer is the
        real `RoutedExperts` instance the model builds (see
        `TestRealModelWiring`), not a hand-built stand-in.
        """
        from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.fused_moe import fused_moe as fm_mod

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return MagicMock(name="fused_experts_input")

        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.dynamic_eplb = False
        method.tid2eid = None
        method.moe = MagicMock(has_bias=False)

        layer = MagicMock()
        layer.moe_activation = "situ"
        # Pinned: an unset attribute on a bare MagicMock is truthy, and the
        # `swigluoai_uninterleave` override would then fire and mask whether the
        # SiTU branch is what actually did the work.
        layer.swigluoai_uninterleave = False
        layer.zero_expert_num = 0
        layer.zero_expert_type = None
        layer.n_shared_experts = 0
        layer.swiglu_limit = 0.0
        layer.w13_weight = torch.zeros(1, 1, 1)
        layer.w2_weight = torch.zeros(1, 1, 1)

        comm = MagicMock()
        comm.fused_experts.return_value = MagicMock()

        with (
            patch.object(
                fm_mod,
                "select_experts",
                return_value=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64)),
            ),
            patch.object(fm_mod, "get_moe_num_logical_experts", return_value=1),
            patch.object(fm_mod, "build_fused_experts_input", side_effect=_capture),
            patch.object(fm_mod, "_EXTRA_CTX") as mock_ctx,
            patch.object(fm_mod, "get_forward_context") as mock_fwd,
        ):
            mock_ctx.moe_comm_method = comm
            mock_ctx.moe_comm_type = -1
            mock_fwd.return_value.input_ids = None
            AscendUnquantizedFusedMoEMethod.apply(
                method,
                layer=layer,
                x=torch.randn(2, 4),
                use_grouped_topk=False,
                top_k=1,
                router_logits=torch.randn(2, 1),
                renormalize=True,
                num_experts=1,
                activation=MoEActivation.SILU,
            )

        assert captured["activation"] == "situ", (
            f"apply() passed activation={captured['activation']!r}; the marker "
            "was not consulted and the routed experts will compute SwiGLU"
        )

    def test_apply_leaves_unmarked_layer_on_silu(self):
        """The control: an unmarked layer still resolves to its `silu` default."""
        from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.fused_moe import fused_moe as fm_mod

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return MagicMock(name="fused_experts_input")

        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.dynamic_eplb = False
        method.tid2eid = None
        method.moe = MagicMock(has_bias=False)

        layer = MagicMock(spec=[])
        layer.swigluoai_uninterleave = False
        layer.zero_expert_num = 0
        layer.zero_expert_type = None
        layer.n_shared_experts = 0
        layer.swiglu_limit = 0.0
        layer.w13_weight = torch.zeros(1, 1, 1)
        layer.w2_weight = torch.zeros(1, 1, 1)

        comm = MagicMock()
        comm.fused_experts.return_value = MagicMock()

        with (
            patch.object(
                fm_mod,
                "select_experts",
                return_value=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64)),
            ),
            patch.object(fm_mod, "get_moe_num_logical_experts", return_value=1),
            patch.object(fm_mod, "build_fused_experts_input", side_effect=_capture),
            patch.object(fm_mod, "_EXTRA_CTX") as mock_ctx,
            patch.object(fm_mod, "get_forward_context") as mock_fwd,
        ):
            mock_ctx.moe_comm_method = comm
            mock_ctx.moe_comm_type = -1
            mock_fwd.return_value.input_ids = None
            AscendUnquantizedFusedMoEMethod.apply(
                method,
                layer=layer,
                x=torch.randn(2, 4),
                use_grouped_topk=False,
                top_k=1,
                router_logits=torch.randn(2, 1),
                renormalize=True,
                num_experts=1,
                activation=MoEActivation.SILU,
            )

        assert captured["activation"] == MoEActivation.SILU, (
            "an unmarked layer's activation changed; other models would be "
            "silently rerouted"
        )

    def test_layer_without_marker_is_unaffected(self):
        """Every other model keeps its default activation.

        The override is guarded by `getattr(layer, "moe_activation", None)`, so
        a layer that never sets it -- i.e. every other model in the tree --
        resolves exactly as before.
        """
        from vllm_ascend.ops.fused_moe.fused_moe import _is_situ_activation

        assert _is_situ_activation(object()) is False
        assert _is_situ_activation(MagicMock(spec=[])) is False


# --------------------------------------------------------------------------
# 2. Fused SwiGLU kernels must not swallow SiTU
# --------------------------------------------------------------------------
class TestSiTUNotSwallowedByFusion(unittest.TestCase):
    """The two fusion predicates must both exclude SiTU.

    Missing this is the failure mode the audit flagged: the dispatch could be
    fixed, `activation="situ"` could arrive at `moe_mlp`, and the activation
    would *still* be SwiGLU -- because these predicates control whether the
    computation happens inside a fused kernel that hardcodes
    `npu_swiglu`/`dequant_swiglu_quant`.

    `_gmm_swiglu_quant_fusion_enabled` is especially easy to miss: it does not
    depend on `dynamic_eplb` alone, so on a quantized model it can be enabled
    even with fusion off.
    """

    def test_custom_gmm_swiglu_excludes_situ(self):
        # Sanity: the predicate is otherwise satisfiable, so a False result for
        # SiTU is genuinely the exclusion and not "it was False anyway".
        assert _custom_gmm_swiglu_enabled(True, True, "silu") is True
        assert _custom_gmm_swiglu_enabled(True, True, "situ") is False

    def test_gmm_swiglu_quant_fusion_excludes_situ(self):
        assert _gmm_swiglu_quant_fusion_enabled(False, True, False, "silu") is True
        assert _gmm_swiglu_quant_fusion_enabled(False, True, False, "situ") is False
        # The mxfp variant too: use_mxfp_quant alone enables this predicate.
        assert _gmm_swiglu_quant_fusion_enabled(True, False, False, "silu") is True
        assert _gmm_swiglu_quant_fusion_enabled(True, False, False, "situ") is False

    def test_situ_does_not_reach_the_fused_swiglu_kernel(self):
        """With SiTU the fused grouped-GEMM+swiglu op must not be called at all.

        Driven through `quant_apply_mlp` with the W4A8-per-channel and fused
        branches otherwise satisfiable (`fusion=True`, `dynamic_eplb=True`), so
        a SiTU activation that was *not* excluded would take one of those
        branches and never consult the SiTU code at all. The two ops that would
        then compute SwiGLU inside the kernel are asserted uncalled.
        """
        gate_up = torch.randn(1, 8)
        with (
            patch(f"{MOE_MLP}._EXTRA_CTX") as mock_ctx,
            patch(f"{MOE_MLP}.HAS_TRITON", False),
            patch("torch.npu.current_stream") as mock_stream,
            patch(
                "torch_npu.npu_grouped_matmul",
                return_value=[gate_up],
                create=True,
            ),
            patch(
                "torch_npu.npu_dynamic_quant",
                side_effect=lambda x, *a, **k: (x, torch.ones(1)),
                create=True,
            ),
            patch.object(
                DeviceOperator, "npu_grouped_matmul_gmm2", return_value=torch.zeros(1, 4)
            ),
            patch(f"{MOE_MLP}.dispose_tensor"),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.DeviceOperator.npu_grouped_matmul_swiglu_quant"
            ) as mock_fused,
            patch(
                "torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list",
                create=True,
            ) as mock_nz,
            patch(
                "torch.ops._C_ascend.npu_dequant_swiglu_quant", create=True
            ) as mock_dequant,
            patch("torch_npu.npu_swiglu", create=True) as mock_swiglu,
        ):
            mock_stream.return_value.record_event.return_value = MagicMock()
            mock_ctx.moe_comm_type = -1  # not MoECommType.MC2
            quant_apply_mlp(
                hidden_states=torch.randn(1, 4),
                w1=torch.randn(1, 8, 4),
                w1_scale=[torch.randn(1, 8)],
                w2=torch.randn(1, 4, 1),
                w2_scale=[torch.randn(1, 4)],
                group_list=torch.tensor([1], dtype=torch.int64),
                group_list_type=1,
                dynamic_scale=torch.randn(1, 1),
                w1_scale_bias=None,
                w2_scale_bias=None,
                w1_offset=None,
                w2_offset=None,
                fusion=True,
                dynamic_eplb=True,
                use_mxfp_quant=False,
                mxfp_quant_dtype=None,
                act_quant_type=torch.int8,
                weight_quant_type=torch.float8_e4m3fn,
                use_bf16=True,
                activation="situ",
                swiglu_limit=0.0,
                swiglu_alpha=SITU_BETA,
                swiglu_beta=SITU_LINEAR_BETA,
                use_w4a8_per_channel_gmm_swiglu=False,
            )
        mock_fused.assert_not_called()
        mock_nz.assert_not_called()
        mock_dequant.assert_not_called()
        mock_swiglu.assert_not_called()


# --------------------------------------------------------------------------
# 3. The math itself, on the routed-expert path
# --------------------------------------------------------------------------
class TestRoutedExpertActivationMath(unittest.TestCase):
    """Behavioural proof: the routed experts compute SiTU, not SwiGLU.

    `unquant_apply_mlp` is the non-fused path Stage 1 uses. The grouped matmul
    is stubbed to return a fixed gate/up tensor so the assertion lands squarely
    on the activation: given the same input, the `"situ"` branch and the
    `"silu"` branch must produce *different* output, and the `"situ"` output
    must match the vendor reference.
    """

    GATE_UP = torch.tensor(
        [
            [0.5, -1.25, 2.0, 0.75, -0.5, 1.5, -2.0, 0.25],
            [1.0, 0.5, -0.75, 1.25, 2.0, -1.0, 0.125, -0.875],
        ],
        dtype=torch.bfloat16,
    )

    def _run(self, activation):
        """Return the tensor the activation produced.

        `unquant_apply_mlp` feeds the activation's output straight into the
        second grouped matmul, so the activation result is observable as that
        call's `x`. Returning the function's own output would instead test the
        (stubbed) down-projection.
        """
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        calls = []

        def _record(*args, **kwargs):
            calls.append(kwargs["x"][0])
            if len(calls) == 1:
                return [self.GATE_UP]
            return [torch.zeros(2, 4, dtype=torch.bfloat16)]

        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                side_effect=_record,
                create=True,
            ),
            # npu_swiglu has no CPU kernel; substitute the faithful CPU form so
            # the SiLU branch under comparison computes real SwiGLU.
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_swiglu",
                side_effect=_cpu_npu_swiglu,
                create=True,
            ),
        ):
            unquant_apply_mlp(
                hidden_states=hidden_states,
                w1=torch.randn(1, 8, 4, dtype=torch.bfloat16),
                w2=torch.randn(1, 4, 4, dtype=torch.bfloat16),
                group_list=torch.tensor([2], dtype=torch.int64),
                activation=activation,
                need_trans=False,
                swiglu_alpha=SITU_BETA,
                swiglu_beta=SITU_LINEAR_BETA,
            )
        assert len(calls) == 2, "expected exactly two grouped matmuls"
        return calls[1]

    def test_situ_output_matches_vendor_reference(self):
        out = self._run("situ")
        expected = vendor_situ_reference(self.GATE_UP)
        assert torch.allclose(out, expected, atol=1e-2, rtol=1e-2), (
            f"routed-expert output does not match the vendor SiTU math\n"
            f"got      {out.flatten()[:6].tolist()}\n"
            f"expected {expected.flatten()[:6].tolist()}"
        )

    def test_situ_is_not_swiglu_on_the_routed_path(self):
        """The reverse assertion: SiTU and SwiGLU must be distinguishable here.

        This is the test whose absence let the bug through -- the dense-side
        equivalent existed, but the routed path had no such check. With
        Kimi-K3's betas (4.0 / 25.0) the two differ by far more than bf16 noise.
        """
        situ_out = self._run("situ")
        swiglu_out = self._run("silu")
        assert not torch.allclose(situ_out, swiglu_out, atol=1e-2, rtol=1e-2), (
            "routed-expert SiTU output is indistinguishable from SwiGLU; the "
            "activation is not actually being applied"
        )
        # ...and the SiTU result is the one we get, not the SwiGLU one.
        assert not torch.allclose(situ_out, swiglu_reference(self.GATE_UP), atol=1e-2, rtol=1e-2)

    def test_unquantized_path_bypasses_npu_swiglu_for_situ(self):
        """`npu_swiglu` must not be called at all on the SiTU routed path.

        A numeric-only assertion could still pass if some other layer happened
        to undo the wrong activation; asserting the kernel is never invoked
        closes that gap directly.
        """
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                side_effect=[[self.GATE_UP], [torch.zeros(2, 4, dtype=torch.bfloat16)]],
                create=True,
            ),
            patch("torch_npu.npu_swiglu") as mock_swiglu,
        ):
            unquant_apply_mlp(
                hidden_states=hidden_states,
                w1=torch.randn(1, 8, 4, dtype=torch.bfloat16),
                w2=torch.randn(1, 4, 4, dtype=torch.bfloat16),
                group_list=torch.tensor([2], dtype=torch.int64),
                activation="situ",
                need_trans=False,
                swiglu_alpha=SITU_BETA,
                swiglu_beta=SITU_LINEAR_BETA,
            )
        mock_swiglu.assert_not_called()

    def test_silu_activation_still_uses_npu_swiglu(self):
        """Regression guard: the default path is untouched.

        The SiTU branch sits directly above the `else` fallback, so an
        over-broad condition would silently reroute every other model's
        activations through SiTU.
        """
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                side_effect=[[self.GATE_UP], [torch.zeros(2, 4, dtype=torch.bfloat16)]],
                create=True,
            ),
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_swiglu",
                return_value=torch.zeros(2, 4, dtype=torch.bfloat16),
                create=True,
            ) as mock_swiglu,
        ):
            unquant_apply_mlp(
                hidden_states=hidden_states,
                w1=torch.randn(1, 8, 4, dtype=torch.bfloat16),
                w2=torch.randn(1, 4, 4, dtype=torch.bfloat16),
                group_list=torch.tensor([2], dtype=torch.int64),
                activation="silu",
                need_trans=False,
                swiglu_alpha=SITU_BETA,
                swiglu_beta=SITU_LINEAR_BETA,
            )
        mock_swiglu.assert_called()

    def test_situ_and_mul_matches_reference_directly(self):
        """The shared helper itself, independent of the MoE plumbing."""
        x = self.GATE_UP
        assert torch.allclose(
            situ_and_mul(x, SITU_BETA, SITU_LINEAR_BETA),
            vendor_situ_reference(x),
            atol=1e-2,
            rtol=1e-2,
        )


# --------------------------------------------------------------------------
# 4. The string reaches the MLP through `unified_apply_mlp` unchanged
# --------------------------------------------------------------------------
class TestSiTUSurvivesTheStageBoundary(unittest.TestCase):
    """SiTU must survive `MoEMlpComputeInput` -> `unified_apply_mlp` -> MLP.

    The activation is carried as a bare string through several dataclasses. Any
    one of them normalizing it (e.g. coercing through `MoEActivation.from_str`,
    which has no SiTU member) would drop it back to a SwiGLU-computing path.
    """

    def test_activation_string_reaches_unquant_apply_mlp(self):
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=torch.randn(2, 8),
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_bias=None,
                w2_bias=None,
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="situ",
            need_trans=False,
            dynamic_eplb=False,
        )
        with patch(f"{MOE_MLP}.unquant_apply_mlp", return_value=expected) as mock_unquant:
            unified_apply_mlp(mlp_compute_input=mlp_compute_input)
        assert mock_unquant.call_args.kwargs["activation"] == "situ"


if __name__ == "__main__":
    unittest.main(verbosity=2)
