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
"""Kimi-K3 Stage 1 UT: the KDA layer's Ascend OOT registration.

Background: upstream's KDA layer is a `PluggableLayer` *subclass* registered
under its own name (``kimi_gated_delta_net_attention``), not under the base
``GatedDeltaNetAttention`` name. Ascend registered only the base key, so every
KDA layer instantiated the unpatched upstream class and fell through to
upstream Triton kernels -- silently, with no error. These tests pin the key
that closes that hole.

The second half covers a *different* hazard with the same flavour: this model
has two independent entry points for the KDA state dtype/shape -- a class-level
config hook and an instance-level layer hook -- that are consumed by different
subsystems (block-size derivation vs. actual cache tensor allocation). If they
drift apart the page-size assertion still passes while the allocation is wrong,
so their agreement is asserted directly rather than inferred.
"""

import pytest


def _register():
    """Trigger Ascend custom-op registration and return the OOT registry."""
    from unittest.mock import MagicMock

    from vllm.config import set_current_vllm_config

    from vllm_ascend.utils import register_ascend_customop

    cfg = MagicMock()
    cfg.model_config.is_deepseek_mla = False
    with set_current_vllm_config(cfg):
        register_ascend_customop(cfg)

    from vllm.model_executor.custom_op import op_registry_oot

    return op_registry_oot


def test_kda_oot_key_registered():
    """`KimiGatedDeltaNetAttention` must have its own OOT entry.

    The base key alone is not enough: `PluggableLayer.__new__` looks up
    `cls.__name__`, and the KDA class is named `KimiGatedDeltaNetAttention`.
    """
    registry = _register()
    assert "KimiGatedDeltaNetAttention" in registry, (
        "KDA layers will silently fall through to the unpatched upstream "
        "implementation; the base 'GatedDeltaNetAttention' key does not cover "
        "this subclass"
    )
    assert "GatedDeltaNetAttention" in registry, "the base GDN key regressed"


def test_kda_oot_class_is_a_subclass_of_upstream():
    """The replacement must extend upstream, not reimplement it.

    Weight names, sharding and the `forward` skeleton are inherited; only the
    kernel call sites are overridden. A non-subclass would change the module's
    parameter names and break weight loading.
    """
    registry = _register()
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        KimiGatedDeltaNetAttention as Upstream,
    )

    ascend_cls = registry["KimiGatedDeltaNetAttention"]
    assert issubclass(ascend_cls, Upstream)
    # It must be a *different* class, i.e. a real override rather than a
    # re-registration of upstream under the same key.
    assert ascend_cls is not Upstream


def test_kda_oot_overrides_forward_only():
    """`_forward` is the override point; `forward` must stay upstream's.

    Upstream dispatches the core computation through
    `torch.ops.vllm.kda_attention`, which resolves `self._forward` off the
    instance. Overriding `forward` instead would duplicate the projection
    logic and risk drifting from upstream's weight layout.
    """
    registry = _register()
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        KimiGatedDeltaNetAttention as Upstream,
    )

    ascend_cls = registry["KimiGatedDeltaNetAttention"]
    assert "_forward" in ascend_cls.__dict__, "_forward is not overridden"
    assert "forward" not in ascend_cls.__dict__, (
        "forward should be inherited from upstream; overriding it duplicates "
        "the projection layout instead of only swapping kernels"
    )


def _referenced_names(obj) -> set[str]:
    """Names actually referenced in `obj`'s code, ignoring comments/docstrings.

    A plain `inspect.getsource` substring check would match the explanatory
    prose in the docstrings (which deliberately *names* the forbidden symbols to
    explain why they are not used), so the AST is walked instead.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
    return names


def test_kda_oot_does_not_call_missing_upstream_symbol():
    """The override must not reach for `chunk_kda_with_fused_gate`.

    That symbol exists upstream but has no Ascend counterpart, so importing or
    calling it would be an ImportError / AttributeError at first prefill. The
    extern-gate split (`fused_kda_gate` + `chunk_kda`) replaces it.
    """
    from vllm_ascend.ops.gdn import AscendKimiGatedDeltaNetAttention

    names = _referenced_names(AscendKimiGatedDeltaNetAttention._forward)
    names |= _referenced_names(AscendKimiGatedDeltaNetAttention._kda_gate_activation)
    assert "chunk_kda_with_fused_gate" not in names, (
        "the Ascend KDA override references an upstream-only symbol"
    )
    # The two symbols it should be using instead.
    assert "fused_kda_gate" in names
    assert "chunk_kda" in names


def test_kda_gate_activation_does_not_cumsum():
    """Double-cumsum guard (design's stated #1 misuse risk).

    `fused_kda_gate` applies the activation only; `chunk_kda` performs the
    chunk-local cumsum internally. If the activation helper ever grew a cumsum,
    the gate would accumulate twice -- silently, with no error, and with badly
    wrong numerics. Asserted structurally because a numeric test would need an
    NPU while the mistake is expressible in the source.
    """
    from vllm_ascend.ops.gdn import AscendKimiGatedDeltaNetAttention

    names = _referenced_names(AscendKimiGatedDeltaNetAttention._kda_gate_activation)
    for forbidden in ("cumsum", "chunk_local_cumsum", "chunk_kda_fwd", "chunk_kda"):
        assert forbidden not in names, (
            f"_kda_gate_activation calls {forbidden!r}; chunk_kda already "
            f"cumsums internally, so this would double-accumulate"
        )
    assert "fused_kda_gate" in names, "the helper must apply the gate activation"



# --------------------------------------------------------------------------
# Two entry points for state dtype / shape
# --------------------------------------------------------------------------
def _real_vllm_config():
    """Build a real VllmConfig from the run's dummy checkpoint config."""
    import os

    config_path = (
        "/home/c00618179/vllm-ascend/.day0/"
        "KimiK3ForConditionalGeneration_20260928_1/impl/dummy_config"
    )
    if not os.path.isdir(config_path):
        pytest.skip("dummy config not present (run-scoped artifact)")

    # A neighbouring UT module may leave Flash Comm v1 on; that flag makes MoE
    # config validation demand expert parallelism, so clear it rather than
    # depend on pytest module ordering.
    os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
    from vllm_ascend.utils import clear_enable_sp

    clear_enable_sp()

    from vllm.engine.arg_utils import EngineArgs

    from vllm_ascend import register_model

    register_model()
    engine_args = EngineArgs(
        model=config_path,
        tensor_parallel_size=8,
        max_model_len=4096,
        load_format="dummy",
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_expert_parallel=True,
    )
    return engine_args.create_engine_config()


def test_kda_state_two_entries_agree():
    """Entry A (class/config hook) and entry B (instance hook) must match.

    Entry A feeds `patch_mamba_config`, which derives the attention block size
    and `mamba_page_size_padded`. Entry B feeds `MambaSpec`, which allocates
    the actual cache tensors. If they disagree, the page-size assertion still
    passes (it only checks self-consistency with A) while the allocation uses
    B -- no fail-fast, just a wrong cache. So both are printed and compared
    pairwise here; a single dump would not distinguish them, since the two
    return values have identical shape and dtype.

    Entry B is an *instance* method taking no `vllm_config`; entry A is a
    *class* method taking one. They are deliberately called differently.
    """
    import torch

    vllm_config = _real_vllm_config()

    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )

    from vllm_ascend.models.kimi_k3.model import KimiK3ForConditionalGeneration
    from vllm_ascend.ops.gdn import AscendKimiGatedDeltaNetAttention

    # -- Entry A: class methods off the model class, as patch_mamba_config does.
    a_shapes = KimiK3ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)
    a_dtypes = KimiK3ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    # -- Entry B: the layer instance's own hooks, as MambaSpec does.
    # Built via __new__ to exercise the real overrides without an engine.
    layer = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    layer.tp_size = vllm_config.parallel_config.tensor_parallel_size
    hf_text_config = vllm_config.model_config.hf_text_config
    lac = hf_text_config.linear_attn_config
    layer.num_heads = lac["num_heads"]
    layer.head_dim = lac["head_dim"]
    layer.conv_size = lac["short_conv_kernel_size"]
    layer.model_config = vllm_config.model_config
    layer.cache_config = vllm_config.cache_config

    b_shapes = layer.get_state_shape()
    b_dtypes = layer.get_state_dtype()

    print("\nentry A shapes:", a_shapes)
    print("entry B shapes:", b_shapes)
    print("entry A dtypes:", a_dtypes)
    print("entry B dtypes:", b_dtypes)

    assert len(a_shapes) == len(b_shapes) == 2, "expected (conv, recurrent) pairs"
    for i, (sa, sb) in enumerate(zip(a_shapes, b_shapes)):
        assert tuple(sa) == tuple(sb), f"shape entry mismatch at index {i}: A={sa} B={sb}"
    assert tuple(a_dtypes) == tuple(b_dtypes), f"dtype entry mismatch: A={a_dtypes} B={b_dtypes}"

    # Pin the values the block-size derivation depends on, so a change in either
    # entry has to be a deliberate one.
    conv_shape, recurrent_shape = b_shapes
    assert tuple(recurrent_shape) == (12, 128, 128), recurrent_shape
    assert tuple(conv_shape) == (3, 4608), conv_shape
    assert tuple(b_dtypes) == (torch.bfloat16, torch.float32), (
        "the recurrent state is hard-coded float32 by kda_state_dtype; the "
        "block size (768) is derived from this and would otherwise be wrong"
    )

    # And re-derive it the way patch_mamba_config does, to keep the 768 link
    # visible from the test rather than only in the design docs.
    def _nbytes(shape, dtype):
        n = 1
        for d in shape:
            n *= d
        return n * torch.empty((), dtype=dtype).element_size()

    ssm = max(_nbytes(s, d) for s, d in zip(b_shapes, b_dtypes))
    conv = min(_nbytes(s, d) for s, d in zip(b_shapes, b_dtypes))
    assert ssm == 786432, ssm
    assert conv == 27648, conv
    assert 1024 * 768 == ssm, "attention block size alignment (768) broken"
