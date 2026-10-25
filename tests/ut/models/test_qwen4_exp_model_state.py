# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Runner-state wiring for Qwen4Exp PLE inputs.

``init_asecnd_model_state`` prefers a model-provided ``get_model_state_cls()``
over its platform default, so the Ascend entry point must both (a) be chosen by
the factory and (b) inject ``ngram_context`` and the padded
``query_start_loc``. Losing the second part raises
``RuntimeError("PLE inputs were not prepared")`` at the first PLE layer, which
is much later than this test.
"""

import os

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs

import vllm_ascend  # noqa: F401
from vllm_ascend.models import register_model
from vllm_ascend.models.qwen4_exp.model import (
    AscendQwen4ExpForCausalLM,
    AscendQwen4ExpForConditionalGeneration,
)
from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state

MODEL = "/mnt/weight/Qwen3.8-Flash-Next"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL), reason=f"Qwen4Exp checkpoint not present at {MODEL}"
)


@pytest.fixture(scope="module")
def engine_config():
    register_model()
    args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=8,
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
        max_model_len=32768,
        max_num_seqs=8,
        load_format="dummy",
        engram_config={"cpu_offload": False},
    )
    return args.create_engine_config()


class _StateModel:
    """Model stand-in that only answers the factory's dispatch question.

    The factory calls ``get_model_state_cls()`` and then constructs the class
    with ``(vllm_config, model, encoder_cache, device)``; nothing else about
    the model is read before the state's own assertions run.
    """

    def __init__(self, cls):
        self._cls = cls

    def get_model_state_cls(self):
        return self._cls


def test_model_state_cls_is_declared_by_both_architectures():
    """Both top-level models expose the Ascend state class."""
    for model_cls in (AscendQwen4ExpForCausalLM, AscendQwen4ExpForConditionalGeneration):
        assert model_cls.get_model_state_cls() is AscendQwen4ExpModelState


def test_factory_prefers_the_model_state_class(engine_config):
    """``init_asecnd_model_state`` returns the model-provided state class.

    Without this the factory falls through to ``AscendMambaHybridModelState``
    and the PLE inputs are silently dropped.
    """
    import torch.nn as nn

    model = _StateModel(AscendQwen4ExpModelState)
    state = init_asecnd_model_state(
        engine_config, model, encoder_cache=None, device=torch.device("cpu")
    )
    assert isinstance(state, AscendQwen4ExpModelState)
    assert state.uses_ngram_embedding is True
    assert state.ngram_context_len == 2  # ngram_size - 1
    text_config = engine_config.model_config.hf_text_config
    assert state.ngram_eos_token_id == int(text_config.eos_token_id)
    assert state.ngram_context.shape == (state.max_num_reqs, 2)
    assert state.ngram_context.dtype == torch.int32
    assert (state.ngram_context == state.ngram_eos_token_id).all()


def test_dummy_inputs_carry_ngram_context_and_query_start_loc(engine_config):
    """``prepare_dummy_inputs`` exposes both PLE inputs with the right shapes."""
    model = _StateModel(AscendQwen4ExpModelState)
    state = init_asecnd_model_state(
        engine_config, model, encoder_cache=None, device=torch.device("cpu")
    )
    num_reqs, num_tokens = 3, 7
    model_inputs = state.prepare_dummy_inputs(num_reqs=num_reqs, num_tokens=num_tokens)

    assert "ngram_context" in model_inputs
    assert "query_start_loc" in model_inputs
    ngram_context = model_inputs["ngram_context"]
    assert ngram_context.shape[0] >= num_reqs
    assert ngram_context.shape[1] == state.ngram_context_len

    query_start_loc = model_inputs["query_start_loc"]
    # 7 tokens over 3 requests -> [0, 2, 4, 7], padding rows repeat the total.
    assert query_start_loc[: num_reqs + 1].tolist() == [0, 2, 4, 7]
    assert int(query_start_loc[-1].item()) == num_tokens
    assert (query_start_loc[num_reqs + 1 :] == num_tokens).all()


def test_pp_gt_1_is_rejected(engine_config):
    """PLE reads raw ``input_ids``, which non-first PP ranks never receive."""
    import copy

    config = copy.deepcopy(engine_config)
    config.parallel_config.pipeline_parallel_size = 2
    model = _StateModel(AscendQwen4ExpModelState)
    with pytest.raises(RuntimeError, match="pipeline_parallel_size=1"):
        AscendQwen4ExpModelState(
            config, model, encoder_cache=None, device=torch.device("cpu")
        )
