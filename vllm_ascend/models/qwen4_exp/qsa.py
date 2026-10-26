# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
"""Ascend QSA (indexed sparse attention) for Qwen4Exp.

A QSA layer is a 256-wide GQA main attention restricted to a per-token top-k
selection of *compressed* keys, plus the indexer that produces that selection.
This module mirrors the upstream structure and replaces only the kernel-host
pieces:

================================  ==============================  ==========================
upstream (nvidia)                  here (Ascend)                  why
================================  ==============================  ==========================
``Qwen4ExpQSAFlashAttentionImpl``  ``AscendQwen4ExpQSAAttnImpl``  ``npu_sparse_flash_attention``
``ops/qsa_indexer.py`` Triton       eager torch select             small-operator composition
``ops/qsa.py`` compress/store       eager torch                    small-operator composition
``CircularBufferSpec``              ``AscendCircularBufferSpec``   Ascend KV zeroing semantics
================================  ==============================  ==========================

Everything else is inherited from the shared, platform-neutral owners so the
contract stays single-sourced: ``QSAIndexer`` (projections, side caches,
metadata plumbing), ``QSAMetadataBuilder``/``QSAForwardMetadata``,
``QSAStateBackend``, ``apply_qsa_rope`` and ``canonical_qsa_rope_positions``.
The eager kernels below are semantic ports of the Triton originals and are
unit-tested against the same closed-form reference.

Index units (the ``sparse_indices`` contract)
---------------------------------------------
The selection buffer uses the upstream *packed* layout
``[tokens, token_topk + compress_ratio]``: leading columns are
request-relative indices with ``-1`` padding, the trailing column is the row's
valid-entry count. Each selected compressed key ``b`` covers the token span
``[b*ratio, b*ratio + ratio)`` and the causal tail covers the tokens after the
last complete group.

``npu_sparse_flash_attention`` indexes the logical token axis of the paged
cache (``sparse_block_size=1`` makes every block one token), so a compressed
index is expanded to its token span before the call. That reading is the
handoff S-2 risk and is asserted by a standalone operator case in the unit
tests rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import mrope_positions_factor
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.qwen4_exp.common.qsa_cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
    QSAKeyStateCache,
    QSAStateBackend,
    _build_qsa_metadata_torch,
    canonical_qsa_rope_positions,
)
from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendCircularBufferSpec,
    AscendMLAAttentionSpec,
)

__all__ = [
    "AscendQSAIndexer",
    "AscendQSAKeyStateCache",
    "AscendQSARingSpec",
    "AscendQSAMetadataBuilder",
    "AscendQSAStateBackend",
    "AscendQwen4ExpQSAAttention",
    "AscendQwen4ExpQSAAttnImpl",
    "AscendQwen4ExpQSABackend",
    "expand_qsa_indices_torch",
    "qsa_compress_groups_torch",
    "qsa_select_torch",
    "qsa_store_rows_torch",
]

PAD = -1
# The sparse-attention operator's only implemented tiling mode is MLA, whose
# qk head width must be 512. QSA's main cache is narrower, so that path is
# reserved for the exact-width case and scored in torch otherwise.
_MLA_HEAD_SIZE = 512


def _qsa_layout_preference() -> tuple:
    """Only layer-compact layouts: the QSA cache cannot be block-outer.

    Each QSA layer owns three page families with independent sizes (main K/V,
    compressed keys at ``tokens_per_state``, the raw ring). The Ascend hybrid
    allocator materialises one contiguous region per layer and slices each
    page at its own size, which requires the layer axis to be outermost. The
    other backends in this model (GDN, PLE short-conv) are SSM backends and
    declare no preference, so this is the only constraint on the intersection.
    """
    from vllm.v1.kv_cache_interface import KVCacheLayout

    return (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC)


_QSA_LAYOUT_PREFERENCE = _qsa_layout_preference()


# ---------------------------------------------------------------------------
# Side caches
# ---------------------------------------------------------------------------
class AscendQSAKeyStateCache(QSAKeyStateCache):
    """Raw-key ring backed by the Ascend circular-buffer manager.

    ``block_size`` on this spec carries the ring *capacity* (4), which is
    orthogonal to the scheduler block size. ``AscendCircularBufferManager``
    keeps the two apart (the same logical/physical split
    ``AscendIndexerKPoolTailSpec`` relies on) and zeroes recycled physical
    pages, which the ring requires: a slot must never be read with a previous
    owner's bytes.

    The backend override matters as much as the spec: the base class points at
    the upstream state backend, whose layout tuple is block-outer and would
    otherwise win the intersection for this model.
    """

    # Auxiliary QSA caches keep their own page geometry. Padding them up to the
    # shared hybrid page would leave the dense per-layer view unable to fill the
    # slot, which the worker's reshape guard rejects. Same contract as
    # ``Glm5NextIndexerCache``.
    align_kv_cache_with_mamba = False

    def bind_kv_cache(self, kv_cache) -> None:
        """Unwrap the single state view the circular branch binds.

        The worker's circular-cache branch publishes ``(state_view,)`` because
        the cache holds one component; the base class expects the bare
        ``[blocks, 1, states, width]`` tensor.
        """
        if isinstance(kv_cache, (tuple, list)):
            if len(kv_cache) != 1:
                raise ValueError("QSA ring cache binds exactly one state view")
            kv_cache = kv_cache[0]
        super().bind_kv_cache(kv_cache)

    def get_attn_backend(self):
        return AscendQSARingBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        from vllm.utils.math_utils import cdiv

        span = self.compress_ratio + vllm_config.num_speculative_tokens
        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)
        assert self.cache_config.block_size % capacity == 0, (
            f"QSA ring capacity {capacity} must divide the attention block "
            f"size {self.cache_config.block_size}"
        )
        return AscendQSARingSpec(
            block_size=capacity,
            num_kv_heads=1,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self.dtype,
        )


class AscendQSAIndexer(QSAIndexer):
    """QSA indexer whose selection runs as eager torch operators."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.indexer_kv_dtype != "bf16":
            raise NotImplementedError(
                "Ascend QSA indexer supports bf16 index keys only, got "
                f"{self.indexer_kv_dtype!r} (fp8 needs a validated fast path)"
            )
        # Rebuild only the raw-key ring with the Ascend spec owner. The base
        # already registered its instance under this prefix in the static
        # forward context; replacing that entry keeps the layer-name -> cache
        # mapping intact (the duplicate-name guard would otherwise fire).
        vllm_config = kwargs["vllm_config"]
        old_ring = self.raw_key_cache
        cache_prefix = self.prefix + "." if self.prefix else ""
        static_context = vllm_config.compilation_config.static_forward_context
        static_context.pop(old_ring.prefix, None)
        self.raw_key_cache = AscendQSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=vllm_config.cache_config,
            compress_ratio=self.compress_ratio,
            vllm_config=vllm_config,
        )
        # Both side caches are built by the base as the upstream state classes;
        # retarget them so the Ascend ring spec and the layer-compact layout
        # preference apply to the layers the worker actually queries.
        self.raw_key_cache.__class__ = AscendQSAKeyStateCache
        self.compressed_key_cache.__class__ = AscendQSACompressedKeyStateCache

    @torch.no_grad()
    def forward(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        attn: "AscendQwen4ExpQSAAttention",
        qkv: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Update the side caches, then select token indices.

        Returns ``(packed_selection, None)``; the second element is the shape
        the upstream fused-prepare path uses and is always ``None`` here
        because the main K/V cache write happens in the attention owner.
        """
        del attn, qkv, slot_mapping
        metadata = self._metadata()
        if metadata is None:
            if self.skip_topk and out is not None:
                return out, None
            result = torch.full(
                (projected_qk.shape[0], self.packed_output_width),
                PAD,
                dtype=torch.int32,
                device=projected_qk.device,
            )
            # Inert rows carry a zero valid count (empty loop bound), not -1.
            result[:, -1] = 0
            if out is not None:
                out.copy_(result)
                return out, None
            return result, None

        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        projected_qk = projected_qk[:num_tokens]
        positions = positions[..., :num_tokens]

        projected_q, raw_keys = projected_qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        raw_key_state_cache = self.raw_key_cache
        compressed_key_cache = self.compressed_key_cache.kv_cache
        if isinstance(compressed_key_cache, (tuple, list)):
            compressed_key_cache = compressed_key_cache[0]
        ring_cache = raw_key_state_cache.kv_cache
        if isinstance(ring_cache, (tuple, list)):
            ring_cache = ring_cache[0]
        raw_key_state_cache.kv_cache = ring_cache

        q = self._prepare_query(projected_q, positions)
        self._write_side_caches(
            raw_keys,
            positions,
            raw_key_state_cache,
            raw_metadata,
            compressed_key_cache,
            compressed_metadata,
        )

        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out, None

        if out is None:
            out = torch.empty(
                num_tokens,
                self.packed_output_width,
                dtype=torch.int32,
                device=q.device,
            )
        elif out.shape != (num_tokens, self.packed_output_width):
            raise ValueError("QSA selection output has an invalid shape")

        block_indices = torch.full(
            (num_tokens, self.token_topk // self.compress_ratio),
            PAD,
            dtype=torch.int32,
            device=q.device,
        )
        visible_blocks = compressed_metadata.visible_blocks[:num_tokens]
        qsa_select_torch(
            q,
            compressed_key_cache,
            compressed_metadata.block_table,
            compressed_metadata.token_to_req,
            compressed_metadata.query_start_loc,
            visible_blocks,
            self.token_topk,
            block_indices,
        )
        expand_qsa_indices_torch(
            block_indices,
            compressed_metadata.logical_positions[:num_tokens],
            visible_blocks,
            self.compress_ratio,
            self.token_topk,
            out,
        )
        return out, None

    # -- query path -------------------------------------------------------
    def _prepare_query(self, projected_q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Norm the indexer query and apply the main attention's RoPE."""
        q = gemma_rmsnorm_torch(
            projected_q.reshape(-1, self.index_head_dim),
            self.q_layernorm.weight,
            float(self.q_layernorm.variance_epsilon),
        ).reshape(-1, self.index_n_heads, self.index_head_dim)
        q = apply_qsa_rope_torch(self.rotary_emb, positions, q)
        return q.to(self.indexer_dtype)

    # -- side-cache writes ------------------------------------------------
    def _write_side_caches(
        self,
        raw_keys: torch.Tensor,
        positions: torch.Tensor,
        raw_key_state_cache,
        raw_metadata: QSAForwardMetadata,
        compressed_key_cache: torch.Tensor,
        compressed_metadata: QSAForwardMetadata,
    ) -> None:
        """Store raw keys, pool completed groups, store compressed keys."""
        rope_position_cache = raw_key_state_cache.rope_position_cache
        if rope_position_cache is None:
            position_rows = raw_metadata.logical_positions.view(-1, 1, 1).expand(-1, 1, 3)
        else:
            position_rows = canonical_qsa_rope_positions(positions).to(
                device=raw_key_state_cache.kv_cache.device
            )

        pooled, first_positions = qsa_compress_groups_torch(
            raw_keys.reshape(-1, 1, self.index_head_dim),
            position_rows,
            raw_key_state_cache.kv_cache,
            raw_metadata.block_table,
            raw_metadata.token_to_req,
            raw_metadata.query_start_loc,
            raw_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.compress_ratio,
            rope_position_cache,
        )
        compressed_keys = gemma_rmsnorm_torch(
            pooled.reshape(-1, self.index_head_dim),
            self.k_layernorm.weight,
            float(self.k_layernorm.variance_epsilon),
        ).reshape(-1, 1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            first_positions = first_positions.transpose(0, 1)
        else:
            first_positions = first_positions[:, 0]
        compressed_keys = apply_qsa_rope_torch(self.rotary_emb, first_positions, compressed_keys)
        qsa_store_rows_torch(
            compressed_key_cache,
            compressed_metadata.slot_mapping,
            compressed_keys,
        )
        qsa_store_rows_torch(
            raw_key_state_cache.kv_cache,
            raw_metadata.slot_mapping,
            raw_keys,
        )
        if rope_position_cache is not None:
            # The rope tail is an int64 view of the trailing bf16 lanes; the
            # row store works in bf16 units, so reinterpret before copying.
            packed = (
                position_rows.reshape(-1, 3)
                .to(torch.int64)
                .contiguous()
                .view(torch.bfloat16)
                .reshape(-1, 1, 3 * 4)
            )
            qsa_store_rows_torch(
                rope_position_cache,
                raw_metadata.slot_mapping,
                packed,
            )


class AscendQSACompressedKeyStateCache(QSACompressedKeyCache):
    """Compressed-key cache with an Ascend spec and the Ascend layout pair.

    The spec class is load-bearing. One page holds
    ``block_size // tokens_per_state`` keys, but the Ascend reshape path
    derives the kernel block size from
    ``kv_cache_interface.get_storage_block_size(spec)``, which only divides by
    ``tokens_per_state`` for :class:`AscendMLAAttentionSpec`; the plain
    upstream spec reports the full token count. The resulting view then claims
    ``tokens_per_state`` times the elements the page actually owns and the
    reshape guard rejects it ("does not exactly fill the small slot").
    """

    # Keep the compressed page at its own size; see ``AscendQSAKeyStateCache``.
    align_kv_cache_with_mamba = False

    def bind_kv_cache(self, kv_cache) -> None:
        """Unwrap the single compressed view the worker binds.

        The compressed indexer page holds keys only, so the reshape path
        publishes ``(view,)``; downstream code expects the bare tensor.
        """
        if isinstance(kv_cache, (tuple, list)):
            if len(kv_cache) != 1:
                raise ValueError("QSA compressed-key cache binds one view")
            kv_cache = kv_cache[0]
        super().bind_kv_cache(kv_cache)

    def get_attn_backend(self):
        return AscendQSAStateBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return AscendMLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
        )


class AscendQSARingSpec(AscendCircularBufferSpec):
    """Raw-key ring with Ascend zeroing semantics and a key-only page.

    The page holds keys and no values, so the worker's combined K/V path cannot
    express it; ``_reshape_kv_cache_v2`` has a dedicated circular branch that
    publishes the single state view instead.
    """


class AscendQSAStateBackend(QSAStateBackend):
    """Metadata-only backend for the QSA side caches (no attention impl).

    Layout preference is the load-bearing part. The QSA stack has three page
    families per layer (main K/V 2*D, compressed D at ``tokens_per_state``,
    ring ``capacity``), so the Ascend hybrid allocator must see *layer-compact*
    pages: it builds one contiguous per-layer region and slices each page at
    its own size. Adopting the upstream block-outer tuple would ask for
    mixed-page blocks, which the Ascend hybrid view cannot express.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_QWEN4_EXP_QSA_STATE"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # QSA state pages store one key per state slot: ``block_size`` here is
        # already the spec's state count (the ring's capacity, or a
        # compressed page's ``block_size // tokens_per_state``), so the row is
        # ``[blocks, 1, states, width]`` exactly as ``_QSAStateCache`` expects.
        return (num_blocks, 1, block_size, head_size)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple:
        from vllm.v1.kv_cache_interface import KVCacheLayout

        return _QSA_LAYOUT_PREFERENCE

    @staticmethod
    def get_builder_cls():
        # The base class resolves to the Triton metadata builder, which has not
        # been validated on NPU; use the torch port for the side caches too.
        return AscendQSAMetadataBuilder


class AscendQSARingBackend(AscendQSAStateBackend):
    """Ring-only backend whose pages hold keys and no values.

    ``get_kv_cache_shape`` stays the state shape
    ``[blocks, 1, states, width]``; a key-only page has no second half for the
    worker's combined K/V path to split off.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_QWEN4_EXP_QSA_RING"


class AscendQSAMetadataBuilder(QSAStateBackend.get_builder_cls()):  # type: ignore[misc]
    """Side-cache metadata builder pinned to the torch metadata path.

    ``common/qsa_cache.py`` binds ``build_qsa_metadata`` at import time to the
    Triton variant whenever ``HAS_TRITON`` is true -- which it is on Ascend.
    That kernel has never been validated on NPU, so this builder routes every
    build through ``_build_qsa_metadata_torch`` and keeps the rest of the
    contract (splits, buffer shapes, reorder threshold) from the base.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> QSAForwardMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        from vllm.v1.attention.backends.utils import split_decodes_and_prefills

        decode_threshold = self.reorder_batch_threshold
        assert decode_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=decode_threshold,
                require_uniform=True,
            )
        )
        if num_decodes + num_prefills != common_attn_metadata.num_reqs:
            raise RuntimeError("QSA metadata request split does not cover the batch")
        if num_decode_tokens + num_prefill_tokens != num_tokens:
            raise RuntimeError("QSA metadata token split does not cover the batch")

        decode_query_len = 0
        if num_decodes > 0:
            query_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )
            nonzero = query_lens_cpu[query_lens_cpu > 0]
            if nonzero.numel() > 0:
                decode_query_len = int(nonzero[0].item())
                if not bool(torch.all(nonzero == decode_query_len)):
                    raise RuntimeError("QSA decode rows must share one query length")

        build_k_work = not self.is_circular_buffer and self.compress_ratio != 1
        k_work_metadata = self.k_work_metadata_buffer
        request_capacity = None
        if build_k_work:
            num_requests = common_attn_metadata.query_start_loc.shape[0] - 1
            request_capacity = self.request_capacity
            max_num_work = (
                num_tokens + (self.compress_ratio - 1) * num_requests
            ) // self.compress_ratio
            k_work_metadata = self.k_work_metadata_buffer[:max_num_work]

        token_to_req, logical_positions, visible_blocks, slot_mapping = _build_qsa_metadata_torch(
            common_attn_metadata,
            self.token_to_req_buffer,
            self.logical_positions_buffer,
            self.visible_blocks_buffer,
            self.slot_mapping_buffer,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            circular_buffer_size=(
                self.kv_cache_spec.block_size if self.is_circular_buffer else 0
            ),
            k_work_metadata_buffer=k_work_metadata if build_k_work else None,
            request_capacity=request_capacity,
        )
        return QSAForwardMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc,
            token_to_req=token_to_req,
            logical_positions=logical_positions,
            visible_blocks=visible_blocks,
            k_work_metadata=k_work_metadata,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            decode_query_len=decode_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )


# ---------------------------------------------------------------------------
# Eager semantic ports of the Triton kernels
# ---------------------------------------------------------------------------
def gemma_rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm: fp32 statistics, ``(1 + w)`` affine, cast back."""
    dtype = x.dtype
    xf = x.float()
    variance = xf.square().mean(dim=-1, keepdim=True)
    normalized = xf * torch.rsqrt(variance + eps)
    return (normalized * (1.0 + weight.float())).to(dtype)


def apply_qsa_rope_torch(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's RoPE composition to QSA heads in torch.

    Mirrors the upstream helper, which calls the CUDA flash-attention kernel
    (``apply_rotary_emb.forward_cuda``) that does not exist on Ascend. The
    rotation is a plain gather-and-rotate over the first ``rotary_dim`` lanes,
    so it is expressed directly; the trailing lanes pass through unchanged.
    """
    if positions.ndim != 1:
        raise NotImplementedError(
            "QSA on Ascend applies 1D RoPE only; multi-axis positions need the "
            "validated MRoPE path (--language-model-only strips them upstream)"
        )
    _, _, head_dim = tensor.shape
    rotary_dim = int(rotary_emb.rotary_dim)
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    head = tensor[..., :rotary_dim]
    if rotary_emb.is_neox_style:
        first, second = head[..., : rotary_dim // 2], head[..., rotary_dim // 2 :]
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    else:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        pairs = head.reshape(*head.shape[:-1], rotary_dim // 2, 2)
        rotated = torch.stack(
            (pairs[..., 0] * cos[..., ::2] - pairs[..., 1] * sin[..., ::2],
             pairs[..., 1] * cos[..., ::2] + pairs[..., 0] * sin[..., ::2]),
            dim=-1,
        ).reshape(head.shape)
    if rotary_dim == head_dim:
        return rotated
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


@torch.no_grad()
def qsa_store_rows_torch(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Scatter fixed-width rows into a ``[pages, page_size, 1, width]`` cache.

    ``PAD_SLOT_ID`` entries are skipped: they address the null block, which
    must never be written.
    """
    if isinstance(cache, (tuple, list)):
        # Key-only QSA pages bind as a one-component tuple.
        if len(cache) != 1:
            raise ValueError("QSA cache must expose exactly one component")
        cache = cache[0]
    if cache.ndim == 4 and cache.shape[1] == 1 and cache.shape[2] != 1:
        # The generic allocator publishes ``[pages, 1, states, width]``; the
        # state caches keep the transposed layout the base class binds.
        cache = cache.transpose(1, 2)
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError(
            f"QSA cache rows {tuple(rows.shape)} and slots "
            f"{slot_mapping.numel()} have incompatible shapes"
        )
    if rows.shape[0] == 0:
        return
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    slots = slot_mapping[valid].long()
    # Pages are block-strided views, so address them by block and offset
    # instead of flattening a non-contiguous tensor, and write in place: the
    # assignment form would materialize a temporary the size of the page plane.
    block_ids = torch.div(slots, cache.shape[1], rounding_mode="floor")
    offsets = torch.remainder(slots, cache.shape[1])
    _scatter_page_rows(cache, block_ids, offsets, rows[valid].to(cache.dtype).unsqueeze(1))


@torch.no_grad()
def qsa_compress_groups_torch(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_position_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed compression groups; return ``(pooled, first_positions)``.

    A group ending at logical position ``p`` (``p % ratio == ratio - 1``)
    averages the raw keys at ``p-ratio+1 .. p``. Members at or after this
    chunk's first position come from ``raw_keys``; older members come from the
    request's ring. ``first_positions`` is the three-axis position of the
    group's first member (used for the compressed key's RoPE).
    """
    rows = token_to_req.numel()
    head_dim = raw_keys.shape[-1]
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if isinstance(compressor_state_cache, (tuple, list)):
        # The worker publishes a key-only circular page as a one-element
        # component tuple; the ring itself is the only component.
        if len(compressor_state_cache) != 1:
            raise ValueError("QSA raw-key ring must be a single state view")
        compressor_state_cache = compressor_state_cache[0]
    if (
        compressor_state_cache.ndim == 4
        and compressor_state_cache.shape[1] == 1
        and compressor_state_cache.shape[2] != 1
    ):
        compressor_state_cache = compressor_state_cache.transpose(1, 2)
    pooled = torch.zeros((rows, 1, head_dim), dtype=torch.float32, device=raw_keys.device)
    first_positions = torch.zeros((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if rows == 0:
        return pooled.to(raw_keys.dtype), first_positions

    capacity = compressor_state_cache.shape[1]
    ratio = compress_ratio
    for row in range(rows):
        request = int(token_to_req[row].item())
        end_position = int(logical_positions[row].item())
        if request < 0 or end_position < ratio - 1:
            continue
        if int(compressed_slots[row].item()) < 0:
            continue
        query_row_start = int(query_start_loc[request].item())
        query_row_end = int(query_start_loc[request + 1].item())
        if not (query_row_start <= row < query_row_end):
            continue
        chunk_start_position = end_position - (row - query_row_start)
        block = int(compressor_state_block_table[request, 0].item())
        accumulator = torch.zeros(head_dim, dtype=torch.float32, device=raw_keys.device)
        for group_offset in range(ratio):
            position = end_position - (ratio - 1 - group_offset)
            use_raw = position >= chunk_start_position
            raw_row = query_row_start + position - chunk_start_position
            if use_raw and query_row_start <= raw_row < min(query_row_end, rows):
                accumulator += raw_keys[raw_row, 0].float()
            elif 0 <= block < compressor_state_cache.shape[0]:
                accumulator += compressor_state_cache[block, position % capacity, 0].float()
        pooled[row, 0] = accumulator / ratio

        first_position = end_position - ratio + 1
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        if (
            rope_position_cache is not None
            and first_from_raw
            and query_row_start <= raw_first_row < min(query_row_end, rows)
        ):
            first_positions[row] = raw_positions[raw_first_row, 0]
        elif rope_position_cache is not None and 0 <= block < rope_position_cache.shape[0]:
            first_positions[row] = rope_position_cache[block, first_position % capacity, 0]
        else:
            first_positions[row] = raw_positions[row, 0]
    return pooled.to(raw_keys.dtype), first_positions


@torch.no_grad()
def qsa_select_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """Pick the ``token_topk`` best compressed keys for every query row.

    The compressed keys are already RMSNormed and RoPE'd, so the score is a
    dot product of the row's indexer query against each candidate key, clamped
    at zero, summed over the row's indexer query heads. Only *complete* groups
    (``visible_blocks``) are candidates. ``out`` receives *key indices*
    ``[rows, token_topk // compress_ratio]``, ``-1`` padded.

    Ranking detail that matters for parity: the Triton kernel scans candidate
    keys in ascending index and keeps the first ``k`` maxima, so ties resolve
    to the lower key index. ``torch.topk`` is stable for the sorted=False path
    on NPU, which yields the same selection.
    """
    num_rows = q.shape[0]
    if num_rows == 0:
        return
    num_heads = q.shape[1]
    key_slots = out.shape[-1]
    page_size = k_cache.shape[1]

    for row in range(num_rows):
        visible = int(visible_blocks[row].item())
        request = int(token_to_req[row].item())
        if request < 0 or visible <= 0:
            out[row].fill_(PAD)
            continue
        num_pages = (visible + page_size - 1) // page_size
        pages = block_table[request, :num_pages].long()
        if bool((pages < 0).any()):
            out[row].fill_(PAD)
            continue
        # One logical key is ``key_heads`` head-sized rows. The reference
        # kernel dots each query head against its matching key row, clamps at
        # zero and sums over the heads; with a single indexer KV head every
        # query head sees the same row.
        flat = k_cache.index_select(0, pages).reshape(-1, k_cache.shape[-1])[:visible]
        key_head_dim = q.shape[-1]
        if k_cache.shape[-1] % key_head_dim:
            raise ValueError(
                f"QSA key width {k_cache.shape[-1]} is not a multiple of the "
                f"query head size {key_head_dim}"
            )
        key_heads = k_cache.shape[-1] // key_head_dim
        keys = flat.reshape(visible, key_heads, key_head_dim).float()
        if key_heads == 1:
            head_scores = keys[:, 0] @ q[row].float().T
        elif key_heads == num_heads:
            head_scores = (keys * q[row].float().unsqueeze(0)).sum(dim=-1)
        elif num_heads % key_heads == 0:
            # Several query heads share one key head; group them in order.
            grouped = keys.repeat_interleave(num_heads // key_heads, dim=1)
            head_scores = (grouped * q[row].float().unsqueeze(0)).sum(dim=-1)
        else:
            raise NotImplementedError(
                "QSA selection needs one key row per query head or a single "
                f"shared key row, got {key_heads} key heads for {num_heads} "
                "query heads"
            )
        scores = torch.clamp(head_scores, min=0.0).sum(dim=-1)
        keep = min(key_slots, visible)
        # Stable sort keeps the lower index for equal scores, matching the
        # kernel's ascending scan.
        order = torch.argsort(scores, descending=True, stable=True)[:keep]
        out[row, :keep] = order.to(out.dtype)
        out[row, keep:] = PAD


@torch.no_grad()
def expand_qsa_indices_torch(
    block_indices: torch.Tensor,
    positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """Expand selected compressed keys into token indices and write the count.

    Each selected key ``b`` covers ``[b*ratio, (b+1)*ratio)``; the causal tail
    covers the tokens after the last complete group. The trailing column holds
    the row's valid-entry count and is never a token index.
    """
    output_width = token_topk + compress_ratio - 1
    if out.shape[1] != output_width + 1:
        raise ValueError(
            f"QSA packed buffer must be {output_width + 1} wide, got {out.shape[1]}"
        )
    block_topk = token_topk // compress_ratio
    if block_indices.shape[1] != block_topk:
        raise ValueError(
            f"QSA block-index buffer must be {block_topk} wide "
            f"({token_topk} token top-k / {compress_ratio}), "
            f"got {block_indices.shape[1]}"
        )
    num_rows = block_indices.shape[0]
    for row in range(num_rows):
        position = int(positions[row].item())
        visible = int(visible_blocks[row].item())
        if position < 0:
            out[row].fill_(PAD)
            out[row, -1] = 0
            continue
        # ``visible_blocks`` counts compressed keys, so for sequences longer
        # than ``token_topk`` tokens it outgrows the block-index columns. The
        # scan is bounded by BLOCK_TOPK (the column count), matching the
        # reference kernel's ``complete_blocks = min(visible, BLOCK_TOPK)``.
        complete = min(visible, block_topk)
        expanded_count = complete * compress_ratio
        tail_start = ((position + 1) // compress_ratio) * compress_ratio
        # The open group after ``tail_start`` holds (position + 1) % ratio
        # tokens, already < ratio, so no clamp is needed here either.
        tail_count = (position + 1) - tail_start

        tokens: list[int] = []
        for rank in range(complete):
            key = int(block_indices[row, rank].item())
            if key < 0:
                expanded_count = rank * compress_ratio
                break
            tokens.extend(
                key * compress_ratio + offset for offset in range(compress_ratio)
            )
        tokens.extend(tail_start + offset for offset in range(tail_count))

        valid = tokens[:output_width]
        out[row, : len(valid)] = torch.tensor(valid, dtype=out.dtype, device=out.device)
        out[row, len(valid) : output_width] = PAD
        out[row, -1] = len(valid)


# ---------------------------------------------------------------------------
# Main attention backend
# ---------------------------------------------------------------------------
class AscendQwen4ExpQSABackend(AttentionBackend):
    """Backend descriptor for the QSA main attention."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "ASCEND_QWEN4_EXP_QSA"

    @staticmethod
    def get_builder_cls():
        return AscendQSAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type:
        return AscendQwen4ExpQSAAttnImpl

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA reads manager pages directly and does not run a paged FA kernel.
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        """Shape of the packed K/V page, K and V interleaved on the leading axis.

        The QSA main cache is one raw allocation holding both K and V, so the
        worker's combined path needs the leading ``2`` axis:
        ``[2, blocks, block_size, heads, dim]`` in the layer-compact layout and
        ``[2, blocks, heads, block_size, dim]`` in the block-outer one.
        """
        from vllm.envs import VLLM_KV_CACHE_LAYOUT

        if VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple:
        # See AscendQSAStateBackend: the QSA stack needs layer-compact pages,
        # so the layer-outer layouts are preferred; the block-outer pair stays
        # accepted because the other backends in this model only offer it, and
        # the resolution picks the layout the most backends put first.
        return _QSA_LAYOUT_PREFERENCE


def _page_rows(tensor: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Reshape one step's rows to the page's ``[heads, width]`` cell layout."""
    return tensor.reshape(tensor.shape[0], *cache.shape[2:])


def _page_groups(
    block_ids: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, list[int], list[tuple[int, int]]]:
    """Group ``(block, offset)`` pairs by page, preserving within-page order.

    Returns the permutation that sorts the pairs by page, the distinct page
    ids in ascending order, and one ``(start, end)`` span of that permutation
    per page. Only the pages actually touched are visited, so the work is
    O(pages addressed) rather than O(pages in the pool).
    """
    order = torch.argsort(block_ids, stable=True)
    pages, counts = torch.unique_consecutive(block_ids[order], return_counts=True)
    pages_list = pages.tolist()
    counts_list = counts.tolist()
    spans: list[tuple[int, int]] = []
    start = 0
    for count in counts_list:
        spans.append((start, start + count))
        start += count
    return order, pages_list, spans


def _scatter_page_rows(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    offsets: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Write ``rows`` into ``cache`` at ``(block_ids, offsets)`` page cells.

    ``cache`` is a block-strided view of the shared KV pool, so the obvious
    ``cache[block_ids, offsets] = rows`` is a whole-tensor advanced-index
    assignment: torch materializes a temporary the size of the *destination*
    view before scattering, which on the real pool is one whole layer plane
    (measured 6.26 GiB, independent of the number of rows selected) and OOMs at
    default utilization.

    ``index_copy_`` writes in place, and since a step's slots are unique it is
    the exact same operation. Rows are grouped per page so the scatter touches
    each page once; the only temporaries are O(rows in that page).
    """
    if rows.shape[0] == 0:
        return
    order, pages, spans = _page_groups(block_ids, offsets)
    ordered_offsets = offsets[order]
    ordered_rows = rows[order]
    for page, (start, end) in zip(pages, spans):
        cache[page].index_copy_(0, ordered_offsets[start:end], ordered_rows[start:end])


def _gather_page_rows(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Read ``cache`` at ``(block_ids, offsets)`` into ``[rows, *cell]``.

    The read-side twin of :func:`_scatter_page_rows`, and for the same reason:
    ``cache[block_ids, offsets]`` on a block-strided view materializes one
    whole layer plane regardless of how few rows are selected (measured 2.441
    GiB for a 6400-page pool at sel=8 and sel=57 alike), which OOMs the first
    real request at default utilization. Gathering one page at a time bounds
    the temporaries to the rows addressed in that page.
    """
    count = block_ids.numel()
    out = cache.new_empty((count, *cache.shape[2:]))
    if count == 0:
        return out
    order, pages, spans = _page_groups(block_ids, offsets)
    ordered_offsets = offsets[order]
    for page, (start, end) in zip(pages, spans):
        out.index_copy_(
            0,
            order[start:end],
            cache[page].index_select(0, ordered_offsets[start:end]),
        )
    return out


def _sparse_gqa_torch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    token_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend over the selected tokens of a paged GQA cache in torch.

    ``token_indices`` holds logical token positions (``-1`` padded) per query
    row, i.e. the same unit the operator's ``sparse_indices`` uses. Selected
    positions are gathered through the page table and softmaxed causally.
    """
    num_tokens, num_heads, head_dim = query.shape
    page_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    out = torch.zeros_like(query)
    if num_tokens == 0:
        return out
    if num_heads % num_kv_heads:
        raise ValueError(
            f"QSA GQA needs query heads divisible by KV heads, got {num_heads} "
            f"and {num_kv_heads}"
        )
    repeat = num_heads // num_kv_heads
    for row in range(num_tokens):
        request = int(token_to_req[row].item())
        if request < 0:
            continue
        valid = token_indices[row] >= 0
        selected = token_indices[row][valid].long()
        if selected.numel() == 0:
            continue
        pages = block_table[request, torch.div(selected, page_size, rounding_mode="floor").long()]
        offsets = torch.remainder(selected, page_size)
        # One gathered cell is a whole page slot (all KV heads). The caches are
        # block-strided views, so the rows are gathered page by page: a single
        # ``key_cache[pages, offsets]`` here would materialize one whole layer
        # plane no matter how few tokens were selected.
        keys = _gather_page_rows(key_cache, pages, offsets).float()
        values = _gather_page_rows(value_cache, pages, offsets).float()
        if repeat != 1:
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)
        q = query[row].float()
        scores = torch.einsum("hd,nhd->hn", q, keys) * scale
        weights = torch.softmax(scores, dim=-1)
        out[row] = torch.einsum("hn,nhd->hd", weights, values).to(out.dtype)
    return out


class AscendQwen4ExpQSAAttnImpl:
    """Sparse paged GQA over the QSA main cache.

    Deliberately not an ``AttentionImpl`` subclass: QSA never runs the vLLM
    attention dispatcher (there is no ``Attention`` child), so the only
    contract that matters is what ``AscendQwen4ExpQSAAttention`` calls.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

    @torch.no_grad()
    def do_kv_cache_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter this step's K and V into their paged caches."""
        key_cache, value_cache = kv_cache
        if key_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")
        num_tokens = key.shape[0]
        if num_tokens == 0:
            return
        slots = slot_mapping[:num_tokens].long()
        valid = slots >= 0
        if not bool(valid.any()):
            return
        # The pages are block-strided views, so they are addressed by block and
        # offset rather than flattened into one contiguous row space.
        slot_values = slots[valid]
        block_ids = torch.div(slot_values, key_cache.shape[1], rounding_mode="floor")
        offsets = torch.remainder(slot_values, key_cache.shape[1])
        _scatter_page_rows(key_cache, block_ids, offsets, _page_rows(key[valid], key_cache))
        _scatter_page_rows(
            value_cache, block_ids, offsets, _page_rows(value[valid], value_cache)
        )

    @torch.no_grad()
    def forward_qsa(
        self,
        layer: nn.Module,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata,
        output: torch.Tensor,
        selection: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selected sparse attention and apply the sigmoid output gate.

        ``selection`` is the packed ``[tokens, token_topk + ratio]`` buffer:
        token indices in the leading columns, valid count last. The operator
        consumes ``-1`` padding directly, so the count column is dropped.
        """
        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output
        if layer.alibi_slopes is not None or layer.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if layer.sliding_window not in (None, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")
        if layer.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("QSA requires a BF16 main KV cache")

        key_cache, value_cache = kv_cache
        if key_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")
        # TND sparse indices are ``[Q_T, KV_N, sparse_size]``; QSA has one
        # indexer KV head, so every row carries its token indices under a
        # singleton head axis.
        token_indices = selection[:num_tokens, :-1].contiguous()

        if key_cache.shape[-1] == _MLA_HEAD_SIZE:
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=query[:num_tokens].contiguous(),
                key=key_cache,
                value=value_cache,
                sparse_indices=token_indices.unsqueeze(1),
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=attn_metadata.block_table,
                actual_seq_lengths_query=attn_metadata.query_start_loc[1:].to(torch.int32),
                actual_seq_lengths_kv=attn_metadata.seq_lens.to(torch.int32),
                query_rope=None,
                key_rope=None,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=False,
            )
            attn = result[0] if isinstance(result, tuple) else result
        else:
            # The operator's tiling only implements its MLA mode, which
            # hard-requires 512-wide heads; QSA's main cache is 256-wide GQA,
            # so select the same token set and score it directly.
            attn = _sparse_gqa_torch(
                query[:num_tokens],
                key_cache,
                value_cache,
                token_indices,
                attn_metadata.block_table,
                attn_metadata.token_to_req,
                self.scale,
            )
        num_tokens_output_heads = attn.shape[1]
        head_dim = attn.shape[2]
        if output_gate.shape[-1] != num_tokens_output_heads * head_dim:
            raise ValueError(
                "QSA output gate must hold one scalar per (head, lane): got "
                f"{output_gate.shape[-1]} for {num_tokens_output_heads} heads "
                f"of width {head_dim}"
            )
        head_gate = output_gate[:num_tokens].reshape(
            num_tokens, num_tokens_output_heads, head_dim
        )
        output[:num_tokens] = attn * torch.sigmoid(head_gate)
        return output


# ---------------------------------------------------------------------------
# The QSA owner layer
# ---------------------------------------------------------------------------
class AscendQwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Qwen4Exp full-attention layer with a QSA index side branch.

    Structurally identical to the upstream owner (same projections, same
    ``qkv_proj`` with the indexer's ``index_qk_proj`` as its fourth shard, same
    rope/norm construction) but backed by the Ascend impl above. The upstream
    owner cannot be subclassed: its ``__init__`` builds a flash-attn impl that
    raises on non-CUDA platforms.
    """

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Ascend Qwen4Exp QSA requires a BF16 main KV cache, got "
                f"{cache_config.cache_dtype}"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError("Qwen4Exp QSA does not support context parallelism")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        # 2 KV heads on 8 ranks: each rank owns one *replicated* head, so the
        # cache is replicated rather than sharded.
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # The checkpoint always packs a sigmoid output gate next to Q.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            mrope_positions_factor=mrope_positions_factor(vllm_config),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # No fused QK-norm/RoPE/gate kernel on NPU: the eager split path runs.
        self.use_fused_qk_norm_rope_gate = False

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError("Ascend Qwen4Exp QSA requires a BF16 main KV cache")
        self.kv_sharing_target_layer_name = None
        # Read by the impl; QSA supports neither ALiBi, sinks nor a window.
        self.alibi_slopes = None
        self.sinks = None
        self.sliding_window = -1
        # The main cache is bound as a (k, v) tuple by the Ascend allocator.
        self.kv_cache: tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([]))
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0

        self.attn_backend = AscendQwen4ExpQSABackend
        self.impl = AscendQwen4ExpQSAAttnImpl(
            self.num_heads, self.head_dim, self.scaling, self.num_kv_heads
        )
        self.indexer = AscendQSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # Fold the indexer projection into ``qkv_proj`` as a fourth shard so
        # the checkpoint's separate ``indexer.index_qk_proj`` rows load
        # through one merged GEMM.
        if vllm_config.lora_config is not None:
            raise NotImplementedError(
                "Ascend Qwen4Exp QSA requires the fused indexer projection"
            )
        self.fuse_indexer_projection = True
        self.index_qk_size = self.indexer.index_qk_proj.output_size
        self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
            self.qkv_proj, self.index_qk_size, quant_config
        )
        del self.indexer.index_qk_proj

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(max_tokens, self.indexer.packed_output_width, dtype=torch.int32),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    # -- loader -----------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(
            orig_to_new_stacked={"indexer.index_qk_proj.": ("qkv_proj.", 3)}
        )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        qkv, projected_qk = qkv.split(
            [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
        )
        num_tokens = hidden_states.shape[0]
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        if gate is None:
            raise RuntimeError("Qwen4Exp QSA requires the packed output gate")
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        self._run_qsa(projected_qk, positions, query, key, value, attn_output, gate)
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output

    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            # Profiling / memory-sizing runs carry no attention metadata at
            # all (``_dummy_run`` leaves it ``None`` unless it is forced).
            output.zero_()
            return
        main_metadata = metadata.get(self.layer_name)
        if main_metadata is None:
            # A populated metadata dict without this layer's entry can only be
            # a registration/key mismatch. Failing loudly here beats zeroing
            # twelve QSA layers and emitting fluent nonsense.
            raise RuntimeError(
                f"QSA layer '{self.layer_name}' is missing from attention "
                f"metadata; available keys: {list(metadata.keys())[:5]}"
            )
        if self.kv_cache[0].numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata, metadata[self.indexer.raw_key_cache.prefix]
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")

        selection, _ = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
        )
        if selection.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")

        self.impl.do_kv_cache_update(
            key, value, self.kv_cache, main_metadata.slot_mapping
        )
        self.impl.forward_qsa(
            self,
            query,
            self.kv_cache,
            main_metadata,
            output,
            selection=selection,
            output_gate=output_gate,
        )


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM.

    Ported from the upstream owner because the shard-rank remap is part of the
    weight contract: shard 3 (the indexer projection) is replicated to every
    TP rank, shards 1/2 (K/V) follow the KV-head replica rank, and shard 0
    (Q/gate) follows the plain TP rank.
    """

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(self, loader, param, loaded_weight, loaded_shard_id) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader_v2, param, loaded_weight, loaded_shard_id)

    def load_weights(self, weights):
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())
