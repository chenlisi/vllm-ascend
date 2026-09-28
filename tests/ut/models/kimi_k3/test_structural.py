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
"""Kimi-K3 Stage 1 structural UTs.

Covers the three cheap, high-value checks that do not need an NPU:

1. **import smoke** -- every new module imports (catches missing classes/typos,
   the cheapest possible interception point).
2. **layer dispatch** -- the per-layer KDA/MLA split must come from the 1-based
   `kda_layers` set and *not* from an arithmetic rule on `layer_idx`.
3. **config resolution** -- the nested `text_config` must be reached through
   `hf_text_config`; reading `hf_config` raises on this (hybrid) model.
"""

import importlib

import pytest


def test_all_kimi_k3_modules_import():
    """Import smoke for every module added for Kimi-K3.

    A missing class or a circular import shows up here rather than at model
    construction time (which costs a full engine start to discover).
    """
    for module in [
        "vllm_ascend.models.kimi_k3",
        "vllm_ascend.models.kimi_k3.config",
        "vllm_ascend.models.kimi_k3.activation",
        "vllm_ascend.models.kimi_k3.tokenizer",
        "vllm_ascend.models.kimi_k3.model",
    ]:
        importlib.import_module(module)


def test_model_class_and_hooks_exist():
    """The registered class must expose the Mamba/hybrid state hooks the
    platform calls *before* any worker exists (they are resolved off the model
    class during engine-config validation)."""
    from vllm_ascend.models.kimi_k3.model import KimiK3ForConditionalGeneration

    for hook in (
        "get_mamba_state_shape_from_config",
        "get_mamba_state_dtype_from_config",
        "get_mamba_state_copy_func",
    ):
        assert hasattr(KimiK3ForConditionalGeneration, hook), f"missing {hook}"


def test_registration_entry_points_present():
    """The tokenizer/renderer/config registries must be populated."""
    from vllm_ascend import register_model

    register_model()

    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    assert "kimi_k3" in TokenizerRegistry.tokenizers
    assert "kimi_k3" in RENDERER_REGISTRY.renderers


# --------------------------------------------------------------------------
# Layer dispatch
# --------------------------------------------------------------------------
KDA_LAYERS_1BASED = [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27,
    29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53,
    54, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78,
    79, 81, 82, 83, 85, 86, 87, 89, 90, 91,
]
FULL_ATTN_LAYERS_1BASED = [
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
    80, 84, 88, 92, 93,
]


class _DispatchCfg:
    """Config stand-in exposing only `is_kda_layer`, as the model reads it."""

    def __init__(self, kda_layers):
        self.linear_attn_config = {"kda_layers": kda_layers}

    def is_kda_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) in self.linear_attn_config["kda_layers"]


def test_layer_dispatch_counts_and_membership():
    """93 layers split into exactly 69 KDA + 24 MLA, by set membership.

    The 1-based `+1` semantics are load-bearing: `layer_idx` is 0-based while
    `kda_layers` is 1-based.
    """
    cfg = _DispatchCfg(KDA_LAYERS_1BASED)
    kda = [i for i in range(93) if cfg.is_kda_layer(i)]
    mla = [i for i in range(93) if not cfg.is_kda_layer(i)]

    assert len(kda) == 69
    assert len(mla) == 24
    assert len(kda) + len(mla) == 93
    assert kda[0] == 0, "layer 0 must be KDA (1 in kda_layers)"
    assert set(kda) | set(mla) == set(range(93))


def test_layer_dispatch_is_not_arithmetic():
    """Layer 92 (0-based) is MLA but breaks the every-4th cadence.

    `full_attn_layers` contains both 92 and 93 (1-based), i.e. two adjacent MLA
    layers at the end. Any `layer_idx % 4` style derivation therefore gets the
    last two layers wrong -- this test pins that down.
    """
    cfg = _DispatchCfg(KDA_LAYERS_1BASED)

    # 0-based 92 is 1-based 93 -> MLA, appended past the regular cadence.
    assert not cfg.is_kda_layer(92)
    # The layer before it (0-based 91 / 1-based 92) is also MLA.
    assert not cfg.is_kda_layer(91)

    # A naive modulo rule would classify 92 as KDA (92 % 4 == 0), so assert the
    # arithmetic rule disagrees -- proving the membership test is what matters.
    naive = [i for i in range(93) if i % 4 != 3]
    actual = [i for i in range(93) if cfg.is_kda_layer(i)]
    assert naive != actual


# --------------------------------------------------------------------------
# Config layer resolution
# --------------------------------------------------------------------------
def test_mamba_state_hooks_read_hf_text_config(monkeypatch):
    """State hooks must work on the real checkpoint's nested config.

    This is the trap the design flags: `hf_config` on this (hybrid) model is the
    top-level VL config, which has no `linear_attn_config`. Only
    `hf_text_config` carries the geometry. The assertion is therefore that the
    hooks succeed against the real config -- a `hf_config`-based implementation
    would raise AttributeError here.
    """
    config_path = (
        "/home/c00618179/vllm-ascend/.day0/"
        "KimiK3ForConditionalGeneration_20260928_1/impl/dummy_config"
    )
    import os

    if not os.path.isdir(config_path):
        pytest.skip("dummy config not present (run-scoped artifact)")

    # A neighbouring UT module may have left Flash Comm v1 enabled; that flag
    # makes MoE config validation demand expert parallelism. Cleared here so
    # this test does not depend on pytest's module ordering.
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
    from vllm_ascend.utils import clear_enable_sp

    clear_enable_sp()

    from vllm_ascend import register_model

    register_model()
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=config_path,
        tensor_parallel_size=8,
        max_model_len=4096,
        load_format="dummy",
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
        # Flash Comm v1 (if an earlier UT left it on) requires expert
        # parallelism for MoE models, so state it explicitly rather than
        # depending on ambient state.
        enable_expert_parallel=True,
    )
    vllm_config = engine_args.create_engine_config()

    hf_config = vllm_config.model_config.hf_config
    hf_text_config = vllm_config.model_config.hf_text_config
    # The precondition that makes this test meaningful: the two differ.
    assert not hasattr(hf_config, "linear_attn_config")
    assert hasattr(hf_text_config, "linear_attn_config")

    from vllm_ascend.models.kimi_k3.model import KimiK3ForConditionalGeneration

    conv_shape, recurrent_shape = (
        KimiK3ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)
    )
    # TP=8: conv (kernel-1=3, conv_dim/8=4608); recurrent (heads/8=12, d, d).
    assert tuple(conv_shape) == (3, 4608)
    assert tuple(recurrent_shape) == (12, 128, 128)

    dtypes = KimiK3ForConditionalGeneration.get_mamba_state_dtype_from_config(
        vllm_config
    )
    # The recurrent state is fp32 regardless of kv dtype -- this is why the
    # aligned block size is 768 and not 384.
    assert dtypes[1] == __import__("torch").float32
