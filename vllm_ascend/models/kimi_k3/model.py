# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
#
"""Kimi-K3 text-path model for Ascend (Stage 1 golden baseline).

Scope
-----
Text path only. The checkpoint's top-level arch is a VL shell
(``KimiK3ForConditionalGeneration``) but Stage 1 runs text-only, so the wrapper
here owns the language model and nothing else -- no vision tower, no projector,
no vision preprocessing.

Why the upstream language-side implementation is not reused as-is
-----------------------------------------------------------------
``vllm/model_executor/models/kimi_linear.py`` is the structural baseline, but it
hard-asserts the *absence* of features this checkpoint enables::

    assert self.use_nope is True
    assert self.q_lora_rank is None        # checkpoint ships q_lora_rank=1536

and it has no notion of latent MoE, cross-layer attention residuals, or MLA
output gating. So the language side is reimplemented here: upstream primitives
(RMSNorm, parallel linear family, VocabParallelEmbedding, ParallelLMHead,
FusedMoE, MultiHeadLatentAttentionWrapper) are reused where semantics match, and
only the differing parts are written fresh.

The five things that make this model different
----------------------------------------------
1. ``hidden_act="situ"`` -- upstream raises on anything but silu. Handled by
   ``AscendSituAndMul``.
2. **Latent MoE** (``routed_expert_hidden_size=3584``): routed experts run in a
   narrower space bracketed by down/up projections, with an optional RMSNorm
   applied to the *expert output* -- after the experts, **before** ``up_proj``.
   The ordering is accuracy-only; getting it wrong raises nothing.
3. **Cross-layer attention residuals** (``attn_res_block_size=12``): an
   accumulating ``(tokens, num_blocks, hidden)`` tensor threaded through every
   layer. All intermediate math is fp32, copying the vendor promotion points
   verbatim.
4. **MLA output gate** (``mla_use_output_gate=true``): attention output is
   scaled by ``sigmoid(g_proj(hidden_states))`` *before* ``o_proj``.
5. **Per-layer heterogeneous dispatch**: 69 KDA + 24 MLA layers, selected by
   membership in the 1-based ``kda_layers`` list -- never by an arithmetic rule
   on ``layer_idx`` (layer 93 is an extra MLA breaking the every-4th pattern).
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.kimi_k3.activation import (
    AscendSituAndMul,
    apply_attn_res as _apply_attn_res,
    situ_params as _situ_params,
)
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention

logger = init_logger(__name__)


class KimiMLP(nn.Module):
    """Dense FFN and shared-expert FFN.

    ``hidden_act`` is validated defensively: anything but ``situ`` raises rather
    than falling back to ``ACT2FN``. A silent substitution to SwiGLU would be an
    accuracy bug carrying no runtime signal, which is exactly the failure mode
    the design flags as the highest risk.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "situ":
            raise NotImplementedError(
                f"Unsupported activation for Kimi-K3 MLP: {hidden_act!r}. Only "
                "'situ' is implemented; refusing to silently substitute "
                "another activation whose runtime behaviour would be wrong."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = AscendSituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class KimiSparseMoeBlock(nn.Module):
    """MoE block with optional latent (reduced-width) expert computation.

    Vendor data flow (``KimiSparseMoeBlock.forward``), reproduced exactly::

        identity = hidden_states
        h = gate(hidden_states)                       # raw logits
        if latent: h = routed_expert_down_proj(h)     # hidden -> moe_hidden
        y = experts(h, topk)                          # runs at moe_hidden
        if latent:
            if latent_moe_use_norm: y = routed_expert_norm(y)   # <-- BEFORE up
            y = routed_expert_up_proj(y)              # moe_hidden -> hidden
        if shared_experts: y = y + shared_experts(identity)

    The norm placement is the trap: it applies to the expert *output* and must
    precede ``up_proj``. Both orders run fine; only one is correct.
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        quant_config = vllm_config.quant_config
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts

        if config.hidden_act != "situ":
            raise NotImplementedError(
                f"Unsupported MoE activation: {config.hidden_act!r}. Only 'situ' "
                "is implemented for Kimi-K3."
            )
        self.hidden_act = config.hidden_act
        self.situ_beta, self.situ_linear_beta = _situ_params(config)

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size if self.use_latent_moe else hidden_size
        )
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx

        # Gate runs at full precision upstream; the correction bias is a
        # separate parameter consumed by FusedMoE's routing.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.block_sparse_moe.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))

        if self.num_shared_experts is not None:
            shared_intermediate = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=hidden_size,
                intermediate_size=shared_intermediate,
                hidden_act=config.hidden_act,
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                quant_config=quant_config,
                # Shared-expert output is added to the routed output before any
                # further reduction, so it must NOT all-reduce on its own
                # (double-reduce would scale the result).
                reduce_results=False,
                prefix=f"{prefix}.block_sparse_moe.shared_experts",
            )
        else:
            self.shared_experts = None

        # `FusedMoE` is constructed with the default `activation="silu"` on
        # purpose. SiTU is not a member of the upstream `MoEActivation` enum, so
        # passing it here would raise in `MoEActivation.from_str` -- and passing
        # nothing would leave the routed experts silently computing SwiGLU.
        # Instead the activation is selected downstream, in the Ascend MoE
        # dispatch (`vllm_ascend/ops/fused_moe/fused_moe.py`), which reads the
        # `moe_activation` marker set below. `"silu"` is the correct constructor
        # argument because SiTU and SwiGLU are both gated activations, so the
        # `is_act_and_mul` shape metadata the constructor derives from it is the
        # same for both; only the elementwise math differs, and that is chosen
        # by the marker.
        #
        # The marker goes on `self.experts.routed_experts` rather than on the
        # runner: `AscendMoERunner` forwards `layer=self.routed_experts` into
        # `apply()`, and that object is what the dispatch inspects.
        self.experts = FusedMoE(
            shared_experts=None,
            num_experts=num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.block_sparse_moe.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        self.experts.routed_experts.moe_activation = "situ"
        # SiTU's two betas ride the existing per-layer activation-scalar channel
        # (`swiglu_alpha` / `swiglu_beta`), which `apply()` already forwards all
        # the way into the MLP stage -- the same values arrive at the SiTU branch
        # as `swiglu_alpha` / `swiglu_beta`. They cannot be read from the config
        # at the point of use: that code runs inside the model forward,
        # `profile_run` included, and `profile_run` executes outside any
        # `set_current_vllm_config` context (the smoke run failed exactly there).
        #
        # Safe because those two slots are consumed as `glu_alpha` / `glu_bias`
        # only on the `swigluoai_uninterleave` branch, which a SiTU-marked layer
        # never takes (`_is_situ` excludes it) -- see `_situ_betas` in
        # `ops/fused_moe/moe_mlp.py`. The explicit aliases alongside them keep
        # the meaning readable from the model side.
        self.experts.routed_experts.swiglu_alpha = self.situ_beta
        self.experts.routed_experts.swiglu_beta = self.situ_linear_beta or 0.0
        self.experts.routed_experts.moe_situ_beta = self.situ_beta
        self.experts.routed_experts.moe_situ_linear_beta = self.situ_linear_beta

        if self.use_latent_moe:
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe.routed_expert_down_proj",
            )
            self.routed_expert_up_proj = ReplicatedLinear(
                self.moe_hidden_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe.routed_expert_up_proj",
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = RMSNorm(
                    self.moe_hidden_size, eps=config.rms_norm_eps
                )
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_up_proj = None
            self.routed_expert_norm = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        router_logits, _ = self.gate(hidden_states)

        if self.routed_expert_down_proj is not None:
            hidden_states = self.routed_expert_down_proj(hidden_states)[0]

        y = self.experts(hidden_states=hidden_states, router_logits=router_logits)

        if self.routed_expert_up_proj is not None:
            # Norm precedes up_proj -- see class docstring.
            if self.routed_expert_norm is not None:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)[0]

        y = y.view(num_tokens, hidden_size)

        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class _GatedOutputProjection(nn.Module):
    """``o_proj`` with the Kimi-K3 output gate folded in front of it.

    Vendor order (``modeling_kimi.KimiMLAAttention.forward``)::

        g = self.g_proj(hidden_states).sigmoid()
        attn_output = attn_output * g
        attn_output = self.o_proj(attn_output)

    i.e. the gate multiplies the *attention output* and is applied **before**
    ``o_proj``. Applying it after ``o_proj`` is not equivalent (``o_proj`` is
    linear with no bias, so the two differ by the projection's mixing of the
    per-head gate), and applying it to the module's final output would silently
    change the result.

    The Ascend MLA implementation owns ``o_proj`` and calls it directly, so the
    gate is installed by substituting this wrapper for ``o_proj``. ``o_proj``
    itself is kept as a real submodule so weight loading and TP sharding behave
    exactly as before -- only the call is interposed.

    ``gate_input`` is set per forward by the owning attention module; if it is
    unset the wrapper degrades to a plain ``o_proj`` call (so a missing gate
    input raises no error but also applies no gate -- guarded by an explicit
    assertion in the owning module's forward).
    """

    def __init__(self, o_proj: nn.Module, g_proj: nn.Module | None) -> None:
        super().__init__()
        self.o_proj = o_proj
        self.g_proj = g_proj
        self.gate_input: torch.Tensor | None = None

    def __getattr__(self, name: str):
        # Delegate anything we do not define (e.g. `weight`, `quant_method`,
        # `process_weights_after_loading`) to the wrapped projection so the
        # wrapper is transparent to weight loading and quant bookkeeping.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.o_proj, name)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.g_proj is not None:
            if self.gate_input is None:
                raise RuntimeError(
                    "Kimi-K3 MLA output gate is enabled but no gate input was "
                    "provided; the gate would be silently skipped."
                )
            # ColumnParallelLinear returns (output, bias); we only need output.
            g = self.g_proj(self.gate_input)[0].sigmoid()
            x = x * g
        out = self.o_proj(x, *args, **kwargs)
        return out


class KimiMLAAttention(nn.Module):
    """MLA with q LoRA path, NoPE, and optional output gate.

    Vendor reference (``modeling_kimi.KimiMLAAttention``):
      * q path: ``q_a_proj -> q_a_layernorm -> q_b_proj`` when ``q_lora_rank``
        is set (this checkpoint: 1536). Upstream kimi_linear asserts this away,
        so the q-LoRA assembly here follows the deepseek_v2 family shape.
      * NoPE: the vendor sets ``self.rotary_emb = None`` and asserts
        ``use_nope``. The Ascend MLA path applies RoPE unconditionally from a
        module-global cache (it never reads ``rotary_emb`` -- see
        ``impl/gap_analysis.md`` G-3), so NoPE is realised by installing an
        identity rotary embedding rather than by passing ``None``.
      * Output gate: when enabled, ``g = sigmoid(g_proj(hidden_states))``
        multiplies the attention output **before** ``o_proj``.

    The gate has to be applied before ``o_proj``, and ``o_proj`` is owned by the
    MLA implementation, so the gate is installed by substituting a
    ``_GatedOutputProjection`` wrapper for ``o_proj`` in the ``MLAModules``
    bundle. When no gate is configured the wrapper is not used at all and the
    implementation sees the plain ``o_proj`` -- behaviour unchanged.
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.qk_head_dim**-0.5
        self.use_nope = config.mla_use_nope
        self.use_output_gate = getattr(config, "mla_use_output_gate", False)

        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = num_heads // tp_size
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_heads}) must be divisible by TP ({tp_size})"
            )

        # ---- q/kv LoRA path ----------------------------------------------
        # The Ascend MLA implementation consumes a *single fused* qkv_a
        # projection (`fused_qkv_a_proj`) and splits it into `q_c` and
        # `kv_no_split` itself. Supplying separate `q_a_proj` /
        # `kv_a_proj_with_mqa` modules does not work: the impl only falls back
        # to `kv_a_proj_with_mqa` and then feeds `hidden_states` straight into
        # `q_b_proj`, which is a k-axis mismatch (hidden vs q_lora_rank).
        # So the fused form is built here and the two logical halves are sliced
        # out of it, mirroring deepseek_v2's assembly.
        if q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                hidden_size,
                [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            self.q_a_proj = None
            self.kv_a_proj_with_mqa = None
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
        else:
            self.fused_qkv_a_proj = None
            self.q_a_proj = None
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                hidden_size,
                kv_lora_rank + qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        # Column-parallel: output is per-head (qk_nope + v), so it is sharded.
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # ---- output gate -------------------------------------------------
        # The gate multiplies the attention output *before* o_proj, so it is
        # installed as a wrapper around o_proj rather than applied to the
        # module's output (see _GatedOutputProjection).
        if self.use_output_gate:
            projection_size = num_heads * v_head_dim
            # Column-parallel, NOT replicated: the gate multiplies the
            # *per-rank* attention output (the MLA impl produces
            # num_heads/tp x v_head_dim per rank, and o_proj is the matching
            # RowParallelLinear). A replicated gate would emit the full
            # num_heads * v_head_dim and fail with a shape mismatch at the
            # multiply. Sharding here keeps gate and o_proj consistent.
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            mla_o_proj = _GatedOutputProjection(self.o_proj, self.g_proj)
        else:
            self.g_proj = None
            mla_o_proj = self.o_proj
        self._gated_o_proj = mla_o_proj if self.use_output_gate else None

        # ---- NoPE ----------------------------------------------------------
        # Kimi-K3 is NoPE, so there is no rotary module -- matching the vendor,
        # which sets `self.rotary_emb = None`. Note that passing None here is
        # NOT what makes the model skip RoPE: the Ascend MLA path reads its
        # cos/sin tables from a module-global cache and never consults
        # `rotary_emb` at all. The identity table that actually realises NoPE is
        # installed in `vllm_ascend/ops/rotary_embedding.py::set_cos_and_sin`.
        rotary_emb = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=rotary_emb,
            o_proj=mla_o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        self.mla_attn = AscendMultiHeadLatentAttention(
            hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix=f"{prefix}.mla",
        )
        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Hand the gate its input (the layer input, i.e. the post-input-norm
        # hidden states -- the vendor computes `g_proj(hidden_states)` from the
        # same tensor it feeds the attention). Cleared afterwards so a stale
        # tensor cannot leak into a later call.
        if self.g_proj is not None:
            self._gated_o_proj.gate_input = hidden_states
        try:
            output[:] = self.mla_attn(positions, hidden_states)
        finally:
            if self.g_proj is not None:
                self._gated_o_proj.gate_input = None


class KimiDecoderLayer(nn.Module):
    """Layer assembly with heterogeneous dispatch and cross-layer residuals.

    Dispatch uses ``is_kda_layer`` (1-based membership test), matching both the
    vendor config and upstream. An arithmetic rule such as ``layer_idx % 4``
    would misclassify layer 92 (0-based), which is an MLA appended past the
    regular every-4th cadence.
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_idx = int(prefix.rsplit(".", 1)[1])
        self.layer_idx = layer_idx

        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiGatedDeltaNetAttention(
                config, vllm_config, prefix=f"{prefix}.self_attn"
            )
            self.is_linear_attn = True
        else:
            self.self_attn = KimiMLAAttention(
                config, vllm_config, prefix=f"{prefix}.self_attn", layer_idx=layer_idx
            )
            self.is_linear_attn = False

        if (
            config.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(
                config, vllm_config, prefix=prefix, layer_idx=layer_idx
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                situ_beta=_situ_params(config)[0],
                situ_linear_beta=_situ_params(config)[1],
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size, 1, bias=False, quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size, 1, bias=False, quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        block_residual: torch.Tensor | None = None,
        **kwargs,
    ):
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                positions, hidden_states, block_residual
            )
        # Non-attn-res path is retained for completeness; this checkpoint always
        # takes the attn-res path (attn_res_block_size is set).
        raise NotImplementedError(
            "Kimi-K3 requires the cross-layer attention-residual path "
            "(attn_res_block_size is set); the plain residual path is unused."
        )

    def _forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor | None,
    ):
        num_tokens, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(num_tokens, hidden_size)

        # Block boundary: snapshot the running sum into block_residual.
        if self.layer_idx % self.attn_res_block_size == 0:
            if block_residual is None:
                block_residual = prefix_sum.view(-1, hidden_size).unsqueeze(1)
            else:
                block_residual = torch.cat(
                    [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)],
                    dim=1,
                )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)

        attn_output = torch.empty_like(hidden_states)
        if self.is_linear_attn:
            self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                output=attn_output,
            )
        else:
            self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                output=attn_output,
            )
        hidden_states = attn_output

        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(num_tokens, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            prefix_sum = prefix_sum + hidden_states

        return prefix_sum, block_residual


class KimiLinearModel(nn.Module):
    """Backbone: embedding, 93 layers, block-residual chain, final mix."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: KimiDecoderLayer(config, vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.output_attn_res_proj = ReplicatedLinear(
                config.hidden_size, 1, bias=False, quant_config=None,
                prefix=f"{prefix}.output_attn_res_proj",
            )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by tensor parallel world size"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        block_residual = None
        if self.use_attn_residuals:
            block_residual = hidden_states.new_zeros(
                hidden_states.shape[0],
                0,
                hidden_states.shape[-1],
            )

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, block_residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=None,
                block_residual=block_residual,
            )

        if self.use_attn_residuals:
            hidden_states = _apply_attn_res(
                hidden_states.view(-1, hidden_states.shape[-1]),
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
            ).view_as(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = (
            fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
            if self.config.is_moe
            else []
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                matched_expert = False
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    matched_expert = True
                    break
                if matched_expert:
                    loaded_params.add(name)
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class KimiK3ForConditionalGeneration(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    """Text-path wrapper for Kimi-K3.

    Registered as the top-level arch. The checkpoint is a VL model, but Stage 1
    is text-only: this class owns the language model and exposes no vision
    tower / projector, and its weight loader skips the vision prefixes
    explicitly rather than ignoring all unknown names.
    """

    #: Vision-side checkpoint prefixes, skipped deliberately (text-path scope).
    _VISION_PREFIXES = ("vision_tower.", "mm_projector.")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = KimiLinearModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # -- Mamba / hybrid state hooks ---------------------------------------
    # These are consulted by the platform's hybrid-config validation before
    # any worker exists, so they must read through `hf_text_config`: the K3
    # checkpoint nests every geometry field inside `text_config`, and the
    # top-level `hf_config` (KimiK3Config) has no `linear_attn_config` at all.
    # Upstream's equivalent methods read `hf_config` directly, which works only
    # because upstream registers the *language-side* arch where the two are the
    # same object. Copying that here would raise AttributeError.
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        hf_text_config = vllm_config.model_config.hf_text_config
        linear_attn_config = hf_text_config.linear_attn_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            linear_attn_config["num_heads"],
            linear_attn_config["head_dim"],
            conv_kernel_size=linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )

        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint nests the language model under `language_model.`;
        # strip that prefix and drop vision-side tensors (text-path scope).
        def _strip(name: str) -> str | None:
            if name.startswith("language_model."):
                return name[len("language_model.") :]
            for pfx in self._VISION_PREFIXES:
                if name.startswith(pfx):
                    return None
            return name

        stripped = []
        skipped_vision = 0
        for name, weight in weights:
            new_name = _strip(name)
            if new_name is None:
                skipped_vision += 1
                continue
            stripped.append((new_name, weight))
        if skipped_vision:
            logger.info_once(
                "Kimi-K3: skipped %d vision-side checkpoint tensors (text-path scope).",
                skipped_vision,
            )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["lm_head."] if self.config.tie_word_embeddings else None,
        )
        return loader.load_weights(stripped)
