# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Ascend Qwen4Exp (`Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration`).

Registered through ``ModelRegistry.register_model`` under the upstream
architecture names, so the upstream package's platform dispatch (whose
``else`` branch selects the NVIDIA implementation) is never entered.

Layout:

* ``model.py``            -- decoder stack, HC state machine, top-level models
* ``qsa.py``              -- indexed sparse attention + eager indexer selection
* ``ple.py``              -- n-gram PLE layer with an eager dilated short-conv
* ``hyperconnection.py``  -- gated residual over the checkpoint's packed linear
* ``model_state.py``      -- runner state carrying ``ngram_context``
"""

from vllm_ascend.models.qwen4_exp.model import (
    AscendQwen4ExpDecoderLayer,
    AscendQwen4ExpForCausalLM,
    AscendQwen4ExpForConditionalGeneration,
    AscendQwen4ExpMixtureOfExperts,
    AscendQwen4ExpModel,
)

__all__ = [
    "AscendQwen4ExpDecoderLayer",
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpMixtureOfExperts",
    "AscendQwen4ExpModel",
]
