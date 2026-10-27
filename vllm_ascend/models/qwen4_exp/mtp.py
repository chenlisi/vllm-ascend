# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Inference-only Qwen4Exp MTP drafter for Ascend NPU.

The checkpoint ships 31 ``mtp.*`` tensors and the upstream
``Qwen4ExpMultiTokenPredictor`` already implements the whole draft contract
(``fc_embedding``/``fc_hidden`` fusion, weight remapping, HC multi-stream
feedback).  Only the *owners* of two components are platform-bound and must be
swapped:

============================  ===================================
upstream                      here
============================  ===================================
``Qwen4ExpDecoderLayer``      ``AscendQwen4ExpDecoderLayer``
``GatedResidual``             ``AscendGatedResidual``
============================  ===================================

The upstream draft layer builds ``Qwen4ExpQSAAttention``, whose constructor
raises ``NotImplementedError("Qwen4Exp QSA requires FlashAttention")`` when
``is_flash_attn_varlen_func_available()`` is false -- i.e. always on NPU.  So
``Qwen4ExpMTP`` cannot be used as-is even though its weights are present.

Everything else is inherited, not copied: ``Qwen4ExpMultiTokenPredictor``'s
``forward`` (scheme-A two-stream feedback), ``load_weights`` /
``_remap_mtp_weight_name`` (checkpoint -> draft renaming), ``set_skip_topk`` /
``compact_topk_indices`` and ``Qwen4ExpMTP``'s head/EPLB plumbing.  Only
``__init__`` is restated, because that is where the two hard-coded owners are
constructed.

PLE stays off without any code change: the draft layer sits at
``layer_idx = mtp_start_layer_idx + idx = num_hidden_layers + idx`` (48 with
this checkpoint) while ``ple_layer_ids == [2]``, so the ``(layer_idx + 1) in
ple_layer_ids`` test in ``AscendQwen4ExpDecoderLayer`` is false.  ``_build``
asserts that rather than trusting it.
"""

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.qwen4_exp.nvidia.mtp import (
    Qwen4ExpMTP,
    Qwen4ExpMultiTokenPredictor,
    _make_draft_vllm_config,
)
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.qwen4_exp.hyperconnection import (
    AscendGatedResidual,
    AscendHyperConnectionConfig,
)
from vllm_ascend.models.qwen4_exp.model import (
    AscendQwen4ExpDecoderLayer,
    AscendQwen4ExpMixtureOfExperts,
)

__all__ = ["AscendQwen4ExpMTP", "AscendQwen4ExpMultiTokenPredictor"]



def _align_draft_rope_with_target(vllm_config: VllmConfig) -> Qwen4ExpTextConfig:
    """Make the drafter use the same RoPE rank the target ended up with.

    ``Qwen4ExpMTPConfig`` keeps MRoPE whenever the draft config still carries a
    ``vision_config`` -- which it always does, because the draft config is
    derived from the multimodal checkpoint config. With
    ``--language-model-only`` the target nevertheless runs 1D RoPE and strips
    ``mrope_section`` / ``mrope_interleaved`` from its own text config, so the
    two models would disagree: the drafter's QSA ring stores three int64 axis
    positions per key (head width 140 instead of 128).

    That width is not cosmetic. The ring page is
    ``capacity * head_size * 2`` bytes (2240 with MRoPE, 2048 without), and
    the KV-cache pool unifies every page to the largest one (409600). 2240
    does not divide 409600, so the ring view cannot be built and KV-cache
    initialization fails; 2048 divides it exactly. Mirroring the target keeps
    the draft geometry consistent with the contract the design specifies
    ("draft ``rope_theta`` independent, but 1D because
    ``--language-model-only`` already stripped mrope").

    This is not a blanket downgrade: a target that genuinely runs MRoPE
    (vision enabled) is refused with ``NotImplementedError`` rather than
    silently downgraded, because the draft would then keep width 140 and its
    ring page would not divide the unified page. See the raise below.

    The draft config is taken from ``speculative_config.draft_model_config``,
    never from ``vllm_config.model_config``. The drafter is loaded through
    ``load_eagle_model`` (``vllm/v1/worker/gpu/spec_decode/eagle/utils.py``),
    which calls ``get_model(vllm_config=..., model_config=draft_model_config)``:
    ``get_model`` forwards the *target* ``vllm_config`` untouched and only uses
    the separate ``model_config`` argument for dtype and construction, so
    ``vllm_config.model_config`` inside the draft's ``__init__`` is still the
    target. Reading it there stripped the target's text config (already 1D, so
    nothing to remove and no log line) and left the draft at width 140.
    ``_make_draft_vllm_config`` -- which builds the config the draft layers
    actually read -- derives from ``speculative_config.draft_model_config``, so
    mutating that object is what reaches ``AscendQSAKeyStateCache``.
    """
    draft_model_config = vllm_config.speculative_config.draft_model_config
    if draft_model_config is None:
        raise ValueError("MTP drafter requires speculative_config.draft_model_config")

    target_model_config = vllm_config.speculative_config.target_model_config
    if target_model_config is not None and target_model_config.uses_mrope:
        # A genuinely multimodal target keeps MRoPE, and so would the draft.
        # The draft ring page is then ``capacity * (index_head_dim + 3 * axes)
        # * dtype`` = ``8 * 140 * 2`` = 2240 bytes, which does not divide the
        # unified page (409600) that the shared KV pool rounds every page up
        # to, so ``_reshape_kv_cache_v2`` fails at KV-cache initialization
        # with a shape error that reads like the width-140 bug this function
        # exists to fix. Refuse up front: VL + MTP has never been validated,
        # and padding the draft ring up to the unified page is a separate
        # design question, not something to guess at here.
        raise NotImplementedError(
            "Qwen4Exp MTP + multimodal target is not supported: a real VL "
            "target's draft ring geometry (uses_mrope=True, page "
            "8*140*2=2240) is incompatible with the unified KV page (409600) "
            "and has not been verified. Run the drafter on the text path "
            "(--language-model-only) instead; that path is unaffected."
        )

    stripped = []
    for owner in (
        getattr(draft_model_config, "hf_config", None),
        draft_model_config.hf_text_config,
    ):
        if owner is None:
            continue
        # ``Qwen4ExpConfig`` keeps the rope parameters on its nested
        # ``text_config``; the outer config's own attribute is ``None``.
        for holder in (owner, getattr(owner, "text_config", None)):
            rope_parameters = getattr(holder, "rope_parameters", None)
            if rope_parameters is None:
                continue
            for key in ("mrope_section", "mrope_interleaved"):
                if rope_parameters.pop(key, None) is not None:
                    stripped.append(f"{type(holder).__name__}.{key}")
    # ``uses_mrope`` is derived from the rope parameters on every read, so the
    # pop above is the whole state change -- there is no cached flag to reset.
    if not stripped:
        # Already aligned (the drafter may be constructed against a config the
        # caller aligned earlier). Returning quietly is only safe because the
        # draft really is 1D here; a draft that still reports MRoPE is a
        # misalignment this function failed to fix, so it must not pass.
        if draft_model_config.uses_mrope:
            raise RuntimeError(
                "MTP drafter still reports MRoPE but no mrope_section was found "
                "on the draft text config; refusing to continue with an "
                "unaligned ring width "
                f"(draft uses_mrope={draft_model_config.uses_mrope})"
            )
        return draft_model_config.hf_text_config
    logger.info(
        "Qwen4Exp MTP draft aligned to the target's 1D RoPE (removed %s)",
        ", ".join(stripped),
    )
    if draft_model_config.uses_mrope:
        raise RuntimeError(
            "MTP draft still reports uses_mrope after stripping "
            f"{', '.join(stripped)}"
        )
    return draft_model_config.hf_text_config


class AscendQwen4ExpMultiTokenPredictor(Qwen4ExpMultiTokenPredictor):
    """The upstream predictor with Ascend attention / HC owners.

    ``forward``, ``load_weights``, ``embed_input_ids`` and the MTP-index
    helpers are inherited unchanged; only the constructor is restated so the
    draft layer and the final mixer are built by the Ascend classes.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Deliberately not calling ``super().__init__``: it builds the two
        # owners this class exists to replace (and the NVIDIA draft layer
        # raises on NPU). The body below mirrors it line for line otherwise.
        nn.Module.__init__(self)

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion, verbatim from the upstream owner.
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                AscendQwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="qwen_sparse_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                )
                for idx in range(self.num_mtp_layers)
            )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen4ExpSparseMoeBlock,
            "mlp",
        )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = AscendHyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = AscendGatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

        self._assert_draft_layer_contract()

    def _assert_draft_layer_contract(self) -> None:
        """Fail loudly if the draft silently drifts off the QSA-only contract.

        Three properties are load-bearing and none of them is enforced by the
        constructors:

        * the draft must not carry a PLE layer (its ``layer_idx`` is past the
          checkpoint's ``ple_layer_ids``; a PLE here would need ``input_ids`` /
          ``query_start_loc`` / ``ngram_context`` that the draft never passes);
        * every draft layer's attention must be the Ascend QSA owner, not the
          upstream flash-attention one;
        * the HC mixer must be combine-free (the checkpoint ships no
          ``hyper_connection_mixer.block_inject_weight``).
        """
        from vllm_ascend.models.qwen4_exp.qsa import AscendQwen4ExpQSAAttention

        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, AscendQwen4ExpDecoderLayer):
                raise TypeError(
                    f"draft layer {idx} is {type(layer).__name__}, expected "
                    "AscendQwen4ExpDecoderLayer"
                )
            if layer.ple is not None:
                raise RuntimeError(
                    f"draft layer {idx} (layer_idx={layer.layer_idx}) built a PLE "
                    f"layer; ple_layer_ids={self.config.ple_layer_ids}"
                )
            if not isinstance(layer.self_attn, AscendQwen4ExpQSAAttention):
                raise TypeError(
                    f"draft layer {idx} attention owner is "
                    f"{type(layer.self_attn).__name__}, expected the Ascend QSA"
                )
        if self.hyper_connection_mixer.use_combine:
            raise RuntimeError("draft HC final mixer must not combine")


class AscendQwen4ExpMTP(Qwen4ExpMTP, AscendQwen4ExpMixtureOfExperts):
    """Ascend Qwen4Exp MTP head (draft model for ``method="mtp"``).

    ``forward`` / ``compute_logits`` / ``load_weights`` /
    ``packed_modules_mapping`` are inherited from the upstream head; the
    constructor is restated only to build the Ascend predictor and to expose
    the draft MoE layers through the Ascend EPLB mixin.

    The upstream ``__init__`` calls ``self.set_moe_parameters(self.model.layers)``,
    whose ``isinstance(layer, Qwen4ExpDecoderLayer)`` test matches nothing here
    (the Ascend layer is an independent ``nn.Module``), which would silently
    report zero MoE layers. Hence the explicit call below.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        self.config = _align_draft_rope_with_target(vllm_config)
        self.model = AscendQwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Explicit Ascend owner: the inherited body would test the upstream
        # decoder-layer class and find no MoE layer.
        AscendQwen4ExpMixtureOfExperts.set_moe_parameters(self, self.model.layers)
        # ``enable_qwen4_exp_low_latency_gemm`` is the NVIDIA skinny-GEMM plan
        # table keyed by device capability; there is no NPU equivalent.

    def set_moe_parameters(self, layers) -> None:
        """Route every caller to the Ascend EPLB owner.

        ``AscendQwen4ExpMTP`` inherits both upstream ``Qwen4ExpMixtureOfExperts``
        (via ``Qwen4ExpMTP``) and ``AscendQwen4ExpMixtureOfExperts``, and C3
        linearisation puts the upstream one first. Without this override the
        worker's EPLB re-derivation would run the upstream ``isinstance`` test,
        find zero MoE layers and silently disable expert-parallel bookkeeping.
        """
        AscendQwen4ExpMixtureOfExperts.set_moe_parameters(self, layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # ``load_weights`` is inherited from ``Qwen4ExpMTP`` unchanged (weight-name
    # remapping plus the auto-weight loader); only the module tree beneath it
    # differs, and that is what the loader walks.
    # ``enable_qwen4_exp_low_latency_gemm`` is the NVIDIA skinny-GEMM plan
    # table keyed by device capability; there is no NPU equivalent, so the
    # inherited call is intentionally not made.
