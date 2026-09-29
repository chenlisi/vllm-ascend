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
"""Kimi-K3 Stage 1 UT: the two new branches are reached by the *model*.

Both changes in this round are guarded by configuration that only the real
checkpoint sets:

* the MoE routed-expert SiTU marker (``moe_activation = "situ"``), and
* the KDA full-rank output gate (``linear_attn_config.use_full_rank_gate``).

A test that hand-builds the object and passes the flag proves the branch works
*if reached*. It does not prove the branch is reached -- and that gap is the
recurring failure mode on this model: the feature is implemented, unit-tested
in isolation, and never actually activated because the wiring between the
config and the construction site is missing.

So these tests construct the layers through their **real constructors**, with a
real ``VllmConfig`` obtained from ``EngineArgs`` over the run's config, and
assert on what came out. Nothing here is stubbed except the process groups,
which are initialized for real (``gloo``, world size 1) rather than faked.
"""

import os

import pytest

RUN_DIR = "/home/c00618179/vllm-ascend/.day0/KimiK3ForConditionalGeneration_20260928_1"
CONFIG_PATH = f"{RUN_DIR}/impl/dummy_config"


def _real_vllm_config(tp_size=1):
    """A real `VllmConfig` from the run's config, via the real parse path."""
    if not os.path.isdir(CONFIG_PATH):
        pytest.skip("run config not present (run-scoped artifact)")

    # A neighbouring UT module may leave Flash Comm v1 on; that flag makes MoE
    # config validation demand expert parallelism, so clear it rather than
    # depend on pytest module ordering.
    os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)

    from vllm_ascend.utils import clear_enable_sp

    clear_enable_sp()

    from vllm.engine.arg_utils import EngineArgs

    from vllm_ascend import register_model

    register_model()
    return EngineArgs(
        model=CONFIG_PATH,
        tensor_parallel_size=tp_size,
        max_model_len=4096,
        load_format="dummy",
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
    ).create_engine_config()


@pytest.fixture(scope="module")
def _initialized_dist():
    """Real (gloo) process-group init, world size 1, with a live vLLM config.

    The layers under test need TP/DP/EP/PCP group objects to exist. Using the
    real initializer keeps the construction path honest; stubbing the group
    accessors would make the test pass with shapes the engine never produces.

    The config context is entered *before* the group init, not after: Ascend's
    distributed patch reads the live config while building the process-group
    options (`calculate_dp_buffer_size`), so a config-less init fails there.
    """
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29597")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    vllm_config = _real_vllm_config()
    with set_current_vllm_config(vllm_config):
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", world_size=1, rank=0)
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield vllm_config


# --------------------------------------------------------------------------
# B-1: the routed-expert SiTU marker, on the real MoE block
# --------------------------------------------------------------------------
def test_real_moe_block_carries_the_situ_marker(_initialized_dist):
    """`KimiSparseMoeBlock` must mark its routed experts as SiTU.

    Without this marker the whole SiTU implementation is unreachable on the MoE
    path: `apply()` sees no marker, keeps the `silu` it was constructed with,
    and `moe_mlp` takes its `else: npu_swiglu` fallback. The model runs and is
    silently wrong. So the assertion is on the object the model actually built.
    """
    from vllm.config import set_current_vllm_config

    from vllm_ascend.models.kimi_k3.model import KimiSparseMoeBlock
    from vllm_ascend.ops.fused_moe.fused_moe import _is_situ_activation

    vllm_config = _initialized_dist
    with set_current_vllm_config(vllm_config):
        hf = vllm_config.model_config.hf_text_config
        # The premise: this checkpoint really is a SiTU model.
        assert hf.hidden_act == "situ"
        assert hf.linear_attn_config["use_full_rank_gate"] is True

        block = KimiSparseMoeBlock(hf, vllm_config, prefix="model.layers.1", layer_idx=1)
        routed_experts = block.experts.routed_experts

        assert getattr(routed_experts, "moe_activation", None) == "situ", (
            "the MoE block did not mark its routed experts; the SiTU branch "
            "implemented for the routed path is unreachable on the real model"
        )
        assert _is_situ_activation(routed_experts) is True, (
            "the marker is present but the dispatch predicate that apply() uses "
            "does not see it"
        )

        # The betas must travel on the layer too: reading them from the config
        # at the point of use fails, because the MLP branch runs inside the
        # model forward (`profile_run` included) where no vLLM config context is
        # set. Pin the transport, not just the value.
        assert getattr(routed_experts, "swiglu_alpha", None) == 4.0, (
            "SiTU's beta is not on the layer; the MLP branch has no way to see it"
        )
        assert getattr(routed_experts, "swiglu_beta", None) == 25.0

        # And it really is what the branch will unpack. The activation is passed
        # explicitly: it is a required argument, so a caller cannot skip the
        # borrow-conflict check by omitting it.
        from vllm_ascend.ops.fused_moe.moe_mlp import _situ_betas

        assert _situ_betas(
            "situ",
            swiglu_alpha=routed_experts.swiglu_alpha,
            swiglu_beta=routed_experts.swiglu_beta,
        ) == (4.0, 25.0)

        # The constructor argument stays the safe default -- SiTU is not a
        # MoEActivation member, so it cannot be passed there without raising.
        assert block.experts.moe_config.activation is not None
        assert block.experts.moe_config.activation.value == "silu"


def test_real_moe_block_activation_metadata_is_gated(_initialized_dist):
    """`is_act_and_mul` must be True for the `silu`-constructed SiTU layer.

    This is the load-bearing premise of the whole approach: SiTU and SwiGLU are
    both gated -- ``gate * act(up)`` -- so the shape metadata vLLM derives from
    the ``silu`` constructor argument is *also* correct for SiTU. The
    constructor therefore never needs to see the string ``"situ"`` (which it
    could not parse anyway). If the two were not both gated, the recorded shape
    assumptions would be wrong and no runtime check would catch it.

    The metadata is cross-checked against the real weight tensor rather than
    merely asserted: ``w13`` is ``(num_experts, 2 * intermediate, hidden)``, and
    that leading 2x is exactly the gated-activation contract.
    """
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_ascend.models.kimi_k3.model import KimiSparseMoeBlock

    vllm_config = _initialized_dist
    with set_current_vllm_config(vllm_config):
        hf = vllm_config.model_config.hf_text_config
        block = KimiSparseMoeBlock(hf, vllm_config, prefix="model.layers.2", layer_idx=2)

        assert block.experts.moe_config.is_act_and_mul is True
        # SiTU's gatedness, stated independently of the enum: the vendor math
        # splits the last axis in half and multiplies gate by act(up), which is
        # the same contract MoEActivation.SILU describes.
        assert MoEActivation.SILU.is_gated is True

        w13 = block.experts.routed_experts.w13_weight
        assert w13.shape[1] == 2 * block.experts.moe_config.intermediate_size, (
            f"w13's gate/up axis is {w13.shape[1]}, not 2x intermediate_size "
            f"({block.experts.moe_config.intermediate_size}); the gated layout "
            "implied by is_act_and_mul does not hold"
        )
        assert w13.shape[2] == block.moe_hidden_size


# --------------------------------------------------------------------------
# B-2: the KDA full-rank output gate, on the real layer
# --------------------------------------------------------------------------
def test_real_kda_layer_builds_full_rank_gate(_initialized_dist):
    """`AscendKimiGatedDeltaNetAttention` must build the single `g_proj`.

    Upstream's `__init__` unconditionally builds the two-stage low-rank gate,
    which this checkpoint does not use. Constructing the real layer shows which
    parameterisation the model actually gets.
    """
    from vllm.config import set_current_vllm_config

    from vllm_ascend.ops.gdn import AscendKimiGatedDeltaNetAttention

    vllm_config = _initialized_dist
    with set_current_vllm_config(vllm_config):
        hf = vllm_config.model_config.hf_text_config
        assert hf.linear_attn_config["use_full_rank_gate"] is True

        layer = AscendKimiGatedDeltaNetAttention(
            hf, vllm_config, prefix="model.layers.1.self_attn"
        )

        assert type(layer) is AscendKimiGatedDeltaNetAttention, (
            "the layer did not resolve through the Ascend OOT registration"
        )
        assert layer.use_full_rank_gate is True
        assert hasattr(layer, "g_proj"), "full-rank g_proj was not built"
        assert not hasattr(layer, "g_a_proj"), (
            "the low-rank g_a_proj is still registered; the loader hard-fails "
            "on parameters absent from the checkpoint, so a full-rank "
            "checkpoint would not load"
        )
        assert not hasattr(layer, "g_b_proj"), "the low-rank g_b_proj is still registered"

        # Geometry: hidden -> projection_size, i.e. the single-stage form.
        projection_size = hf.linear_attn_config["head_dim"] * hf.linear_attn_config["num_heads"]
        assert tuple(layer.g_proj.weight.shape) == (projection_size, hf.hidden_size), (
            f"g_proj shape {tuple(layer.g_proj.weight.shape)} is not "
            f"({projection_size}, {hf.hidden_size})"
        )

        gate_params = sorted(n for n, _ in layer.named_parameters() if "g_proj" in n)
        assert gate_params == ["g_proj.weight"], gate_params


def test_real_kda_layer_has_no_low_rank_gate_parameters(_initialized_dist):
    """The two forms must be mutually exclusive in `named_parameters()`.

    The loader raises when a registered parameter is missing from the
    checkpoint. A layer that kept both forms registered would therefore fail to
    load *either* checkpoint kind, so exclusivity is the invariant to pin.
    """
    from vllm.config import set_current_vllm_config

    from vllm_ascend.ops.gdn import AscendKimiGatedDeltaNetAttention

    vllm_config = _initialized_dist
    with set_current_vllm_config(vllm_config):
        hf = vllm_config.model_config.hf_text_config
        layer = AscendKimiGatedDeltaNetAttention(
            hf, vllm_config, prefix="model.layers.2.self_attn"
        )
        names = {n for n, _ in layer.named_parameters()}
        assert "g_proj.weight" in names
        assert "g_a_proj.weight" not in names
        assert "g_b_proj.weight" not in names
