# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Latent MQA core attention for hybrid MLA layers, with DSA.

``hybrid_mla_attention`` selects which core attention the
``csa_compress_ratios == -2`` (MLA) layers of a ``dsv4_hybrid`` model run. This
module owns the two modes that attend to the **absorbed KV latent**, i.e. the
ones that consume a sorted candidate set:

* ``"mqa_full_causal"`` -> ``_forward_full_causal``. Latent MQA with the indexer
  dropped, attending to the full per-document causal set. Mathematically
  identical to the dense MHA phase, so it isolates the absorption from the
  sparsity for equivalence experiments; ``O(s^2)`` in index memory and therefore
  not a production mode.
* ``"mqa_dsa"`` + ``dsa_indexer_use_sparse_loss=True`` -> ``_forward_sparse``
  (phase 3/4). A forced local window plus Lightning-indexer top-k, i.e. DeepSeek
  Sparse Attention on the KV latent, with the KL restricted to that same
  selected set. This is the only phase that reads ``index_topk``.

The other two modes are *not* this module, and the shared predicate
``hybrid_mla_indexer.latent_mqa_enabled`` is what keeps the spec dispatch and
``MLASelfAttention.mqa_latent`` in step about that:

* ``"mha"`` -- unchanged dense MLA (MHA).
* ``"mqa_dsa"`` + ``dsa_indexer_use_sparse_loss=False`` (phase 2, DSA warmup,
  paired with ``train_indexer_only``) -- also dense MHA, plus the indexer, in
  ``mha_dsa_warmup_attention.MHADSAWarmupAttention``. That phase has no top-k on
  either side, so absorbing here would buy nothing and cost a zero-sparsity
  ``[b, s, s]`` index table pushed through the block-sparse kernel.

The indexer reuses the model-wide ``index_n_heads`` / ``index_head_dim``.

Note the ``mqa_*`` modes here are *latent* MQA (this module). A
``csa_compress_ratios`` entry of ``-1`` is *CSA full-causal MQA*, a different
layer kind handled by ``csa_attention.py``.

``MLASelfAttention`` performs the activation-level absorption (see its
``mqa_latent`` flag), so this module receives

    query [b, s, h, kv_lora_rank + qk_rope_head_dim]
    key   [b, s, 1, kv_lora_rank + qk_rope_head_dim]

and de-absorbs the value side with ``v_b_proj_weight``
(``[kv_lora_rank, h, v_head_dim]``). Every parameter stays byte-identical to
the MHA layout, so an MHA checkpoint loads into an MQA run unchanged.

``add_full_attention_sink_bias`` (or ``softmax_type``) adds one learnable
per-head sink logit as ``softmax_offset``, built by the same
``build_softmax_offset`` helper ``DotProductAttention`` uses, so the parameter
name matches the dense phase. It is fed to the block-sparse kernel as its
``attn_sink``, which then enables the finite-sink LSE correction and the
analytic sink gradient. The indexer KL target does not see it: that target
normalises over the indexer's own candidate set, and the sink -- like the forced
window -- is outside that set by construction, not by cancellation. See
``_attn_target`` for the measured size of the alternative definition.

Multi-document equivalence: RoPE/YaRN scores depend only on ``pos_q - pos_k``,
the YaRN ``mscale`` is a constant and the Hadamard ``rotate_activation`` is
orthogonal, so no per-document position reset is needed. Equivalence to running
every document on its own therefore reduces to index correctness, which is what
``_derive_csa_doc_boundaries`` plus the index builders below guarantee.

Context parallel (``contiguous_allgather`` only, as the HCA layers of the same
model require at ``dsv4_hybrid_attention.py:607-611``) follows the same
"Miles pattern" as ``CSASelfAttention._forward_cp``: the query slice stays
sharded, only the low-dimensional KV latent is all-gathered to the global
length, and the output comes back sharded with no reduce-scatter. The index
tables are built over the *global* sequence and then row-sliced to this rank,
which is exactly right because ``_derive_csa_doc_boundaries`` and the builders
are ``seqlen``-agnostic and their column values are global token ids -- the same
space the all-gathered KV is indexed in. ``attn_mask_startend_row_indices``
already arrives at the global length under CP
(``dsv4_hybrid_attention.py:634``), while ``query`` / ``key`` / ``x`` / ``qr``
arrive local but RoPE'd with global positions
(``multi_latent_attention.py:1485-1545``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.cp_utils import all_gather_cp
from paddlefleet.transformer.csa_attention import (
    TileLangCSAIndexerLossAutoScaler,
    _build_mqa_causal_topk_idxs_from_doc_bounds,
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
    _validate_csa_docmask_shape,
)
from paddlefleet.transformer.dot_product_attention import build_softmax_offset
from paddlefleet.transformer.dsa_attention import (
    DSAIndexerLossLoggingHelper,
)
from paddlefleet.transformer.hybrid_mla_indexer import HybridMLAIndexerMixin
from paddlefleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddlefleet.transformer.enums import AttnMaskType

# Working set of the phase-3 KL-target gather, as a ``rows x slots`` budget:
# 256 rows x 512 slots x 576 dims is ~150MB of gathered bf16 keys, transient and
# freed every iteration. Measured at s=8192/h=64/topk=512/dk=576 on one B30Z:
# 15.9ms at 128 rows, 13.3ms at 256, 12.4ms at 512, 12.4ms at 1024 -- past 256
# the curve is flat, so larger chunks only buy peak memory.
_TARGET_ROW_SLOTS = 256 * 512
# ``lse_indexer`` widths ``flash_mla_sparse_fwd`` implements; anything else is
# rejected outright (FlashMLA ``sparse_fwd.h:160``). Narrower budgets keep the
# Python target path.
_LSE_INDEXER_TOPKS = (512, 1024, 2048)
# Narrowest query-head count the score-recompute kernel's MMA ``M`` tile
# handles; see ``_attn_target_cudnn``.
_TARGET_QHEAD_MIN = 16
_NEG_INF = -1e30
_EPS = 1e-10


class _HashableTensor(paddle.Tensor):
    """``paddle.Tensor`` with hashable ``shape`` / ``stride()``.

    The cuDNN score-recompute wrapper keys its kernel cache on
    ``(dtype, shape, stride(), ...)``, and Paddle returns those as lists, which
    are unhashable.
    """

    @property
    def shape(self):
        return tuple(super().shape)

    def stride(self, dim=None):
        if dim is None:
            return tuple(super().stride())
        return super().stride(dim)


@dataclass
class MQALatentAttentionSublayersSpec:
    """Sublayers spec for :class:`MQALatentAttention`.

    Args:
        indexer: ``DSAIndexer`` spec. ``hybrid_mla_attention="mqa_dsa"`` provides
            one; ``"mqa_full_causal"`` leaves it ``None`` and the layer attends
            to the full per-document causal set (dense MHA equivalent). Also
            ``None`` in the absorption equivalence unit tests.
    """

    indexer: LayerSpec | type = None


class MQALatentAttention(HybridMLAIndexerMixin, FleetLayer):
    """Sparse attention on the absorbed MLA KV latent (``core_attention``).

    Consumes the pre-absorbed ``query`` / ``key`` produced by
    ``MLASelfAttention`` and returns ``[b, s, h * v_head_dim]``, so the MLA
    output tail (gate, ``o_proj``) is unchanged.
    """

    def __init__(
        self,
        config,
        sublayers_spec: MQALatentAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        is_mtp_layer: bool = False,
        is_swa: bool = False,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)

        DSAIndexerLossLoggingHelper.register_total_num_layers(config)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self._init_hybrid_mla_cp_state(config, pg_collection)

        # ``k_channels`` is the MHA q_head_dim (qk_nope + qk_rope), NOT the 576
        # latent width: absorption is exactly score-preserving, so the MHA
        # softmax scale must be kept.
        if softmax_scale is None:
            k_ch = k_channels if k_channels is not None else config.head_dim
            self.softmax_scale = float(k_ch**-0.5)
        else:
            self.softmax_scale = float(softmax_scale)

        self.window_size = int(config.csa_window_size)
        self.indexer = (
            build_spec_layer(
                sublayers_spec.indexer,
                config=config,
                layer_number=layer_number,
                pg_collection=pg_collection,
            )
            if sublayers_spec.indexer is not None
            else None
        )
        self.indexer_loss_coeff = float(
            getattr(config, "dsa_indexer_loss_coeff", 0.0) or 0.0
        )
        # Phase selector, read by ``_phase()`` -- see its docstring. Stored raw
        # rather than as a derived phase so that flipping it on a live module
        # (tests do) cannot leave a stale phase behind.
        # ``transformer_config.__post_init__`` pins it against
        # ``train_indexer_only`` so the two cannot disagree.
        self.indexer_use_sparse_loss = bool(
            getattr(config, "dsa_indexer_use_sparse_loss", False)
        )
        # Learnable per-head attention-sink logit, from the model-wide
        # ``add_full_attention_sink_bias`` / ``softmax_type``. Built by the same
        # helper ``DotProductAttention`` uses, so the state_dict name
        # (``core_attention.softmax_offset``) and the switch are shared with the
        # dense MHA phase. ``None`` keeps the kernel on its sinkless ``-1e30``
        # path, bit-for-bit unchanged.
        self.softmax_offset = build_softmax_offset(
            self,
            config,
            num_attention_heads
            if num_attention_heads is not None
            else config.num_attention_heads,
            is_swa,
        )
        # Shares ``mqa_split_kv_b_proj`` with the query-side
        # absorption: when set, ``MLASelfAttention`` passes its standalone
        # ``v_b_proj`` parameter instead of a view of ``kv_b_proj.weight``, laid
        # out as ``[h, v_head_dim, kv_lora_rank]`` for ``fused_grouped_matmul``
        # rather than the ``[kv_lora_rank, h, v_head_dim]`` the einsum wants.
        self.split_kv_b = bool(getattr(config, "mqa_split_kv_b_proj", False))
        # Latent width, i.e. the value width the sparse kernel sees and the
        # contraction dim of the de-absorption weight. This layer only exists on
        # the ``dsv4_hybrid`` path, so the hybrid field is the authoritative one.
        kv_lora_rank = getattr(config, "hybrid_mla_kv_lora_rank", None)
        if kv_lora_rank is None:
            kv_lora_rank = config.kv_lora_rank
        self.kv_lora_rank = int(kv_lora_rank)
        # Fused Triton epilogue for the analytic sink gradient. Only reachable
        # when ``softmax_offset`` exists; a sinkless layer has no sink gradient.
        self.sink_grad_fusion = getattr(config, "dsa_sink_grad_fusion", False)

    def _phase(self) -> str:
        """Which of the two phases this layer runs.

        The single place the phase is decided, so the attention candidate set
        and the indexer-loss shape cannot disagree:

        * ``"full_causal"`` -- no indexer at all
          (``hybrid_mla_attention="mqa_full_causal"``, and the
          absorption-equivalence unit tests). Per-document full causal
          attention, no indexer loss.
        * ``"sparse"`` -- phase 3/4. Attention consumes window + top-k and the
          KL is restricted to that same selected set.

        There is deliberately no warmup state here. Phase 2 has no top-k on
        either side, so routing a zero-sparsity candidate set through the
        block-sparse kernel would cost a full ``[b, s, s]`` index table for
        nothing; it runs the dense MHA of phase 1 instead, in
        ``mha_dsa_warmup_attention.MHADSAWarmupAttention``, and
        ``hybrid_mla_indexer.latent_mqa_enabled`` is what keeps the spec
        dispatch and ``MLASelfAttention.mqa_latent`` from ever building this
        class for it.

        Read live rather than cached in ``__init__``: a test flipping
        ``indexer_use_sparse_loss`` on a live module must not be able to
        desynchronise the two.
        """
        if self.indexer is None:
            return "full_causal"
        if not self.indexer_use_sparse_loss:
            raise ValueError(
                "MQALatentAttention has an indexer but "
                "dsa_indexer_use_sparse_loss=False; that is the DSA warmup "
                "phase, which runs dense MHA in MHADSAWarmupAttention. "
                "latent_mqa_enabled() should have dispatched there."
            )
        return "sparse"

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        attn_mask_startend_row_indices: Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params=None,
        use_rr_flash_attention: bool = False,
        past_key_values=None,
        layer_idx=None,
        use_cache: bool = False,
        x: Tensor | None = None,
        qr: Tensor | None = None,
        kv_compressed: Tensor | None = None,
        k_pos_emb: Tensor | None = None,
        q_absorbed: Tensor | None = None,
        v_b_proj_weight: Tensor | None = None,
        input_ids: Tensor | None = None,
    ) -> Tensor:
        """Absorbed-MQA forward.

        Args:
            query: ``[b, s, h, kv_lora_rank + qk_rope_head_dim]`` absorbed query.
            key:   ``[b, s, 1, kv_lora_rank + qk_rope_head_dim]`` shared latent.
            value: unused (``None``); the V side lives inside ``key``.
            attn_mask_startend_row_indices: ``[b, 1, s, 1]`` exclusive per-token
                document end rows, at the **global** sequence length under CP.
                ``None`` means a single document.
            x / qr: hidden states / q latent, inputs of the DSA indexer.
            v_b_proj_weight: de-absorption weight. ``[kv_lora_rank, h,
                v_head_dim]`` (the V slice of ``kv_b_proj``) by default, or the
                standalone ``v_b_proj`` parameter as ``[h, v_head_dim,
                kv_lora_rank]`` under
                ``mqa_split_kv_b_proj``.
            input_ids: ``[b, s]`` token ids, only used to build the indexer-loss
                row mask (``!= pad_token_id``). ``None`` falls back to the plain
                row mean, as CSA does at ``csa_attention.py:1306``.

        Returns:
            ``[b, s, h * v_head_dim]`` (this rank's query slice under CP)
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "latent MQA does not support packed_seq_params; "
                "document masking is driven by "
                "attn_mask_startend_row_indices."
            )
        if v_b_proj_weight is None:
            raise ValueError(
                "MQALatentAttention requires v_b_proj_weight; it is only valid "
                "as the core_attention of an absorbed MLA layer."
            )

        b, s = int(query.shape[0]), int(query.shape[1])
        if b != 1:
            raise NotImplementedError(
                "latent MQA requires micro batch size 1 (documents "
                f"are packed along the sequence), got b={b}."
            )
        # Under CP this rank owns global query positions
        # [position_offset, position_offset + s). Everything index-related is
        # derived at s_global and row-sliced; the KV is all-gathered so that the
        # global column ids in those tables address it directly.
        s_global = s * self.cp_size
        position_offset = self.cp_rank * s
        kv = key.squeeze(2).contiguous()  # [b, s, kv_lora + qk_rope]
        kv = all_gather_cp(
            kv, dim=1, group=self.cp_group
        )  # -> [b, s_global, .]
        # Not derivable from ``v_b_proj_weight.shape[0]``: that only holds for the
        # einsum layout ``[l, h, v]``. The grouped-matmul layout is
        # ``[h, v, l]``, so the rank has to come from the config.
        kv_lora_rank = self.kv_lora_rank
        # The de-absorption weight's contraction dim sits last in the
        # grouped-matmul layout and first in the einsum one. Checking it here
        # turns a silently-wrong ``[G, R, D]`` / ``[l, h, v]`` mix-up (both
        # reshape fine) into an error at the first forward.
        # The rank is checked before the contraction dim: a folded 2-D
        # parameter that was never reshaped back would otherwise pass the
        # contraction test and fail deeper down, on an unpacking or reshape
        # whose message says nothing about the layout.
        if len(v_b_proj_weight.shape) != 3:
            expected = (
                "[h, v_head_dim, kv_lora_rank]"
                if self.split_kv_b
                else "[kv_lora_rank, h, v_head_dim]"
            )
            raise ValueError(
                f"v_b_proj_weight must be 3-D {expected}, got shape "
                f"{v_b_proj_weight.shape} with "
                f"mqa_split_kv_b_proj={self.split_kv_b}."
            )
        contraction = v_b_proj_weight.shape[-1 if self.split_kv_b else 0]
        if int(contraction) != kv_lora_rank:
            raise ValueError(
                "v_b_proj_weight layout mismatch: expected contraction dim "
                f"kv_lora_rank={kv_lora_rank}, got shape "
                f"{v_b_proj_weight.shape} with "
                f"mqa_split_kv_b_proj={self.split_kv_b}."
            )

        with paddle.no_grad():
            row_end = attn_mask_startend_row_indices
            if row_end is None:
                row_end = paddle.full(
                    [b, 1, s_global, 1], s_global, dtype="int32"
                )
            _validate_csa_docmask_shape(row_end, b, s_global)
            doc_start, doc_len, is_valid, doc_lens, _ = (
                _derive_csa_doc_boundaries(row_end, s_global)
            )

        phase = self._phase()
        if phase == "full_causal":
            return self._forward_full_causal(
                query,
                kv,
                v_b_proj_weight,
                doc_start,
                is_valid,
                kv_lora_rank,
                position_offset,
                s,
                s_global,
            )

        return self._forward_sparse(
            query,
            kv,
            x,
            qr,
            v_b_proj_weight,
            doc_start,
            doc_len,
            is_valid,
            doc_lens,
            kv_lora_rank,
            input_ids,
            position_offset,
        )

    # ------------------------------------------------------------------
    # full_causal
    # ------------------------------------------------------------------
    def _forward_full_causal(
        self,
        query: Tensor,
        kv: Tensor,
        v_b_proj_weight: Tensor,
        doc_start: Tensor,
        is_valid: Tensor,
        kv_lora_rank: int,
        position_offset: int,
        s_local: int,
        s_global: int,
    ) -> Tensor:
        """Per-document full-causal attention on the absorbed latent.

        No indexer is involved: the column set is decided by the document
        boundaries alone, so this output is mathematically identical to the
        dense MHA phase and is bit-identical across repeated calls (nothing
        here depends on a top-k tie-break).

        Used by ``hybrid_mla_attention="mqa_full_causal"`` (the MHA -> MQA
        equivalence isolation) and by the absorption unit tests. The DSA warmup
        phase, which also needs a full-causal attention, does *not* come here:
        it keeps the dense MHA backend instead, because at zero sparsity the
        ``[b, s, s]`` index table below buys nothing.
        """
        b = int(query.shape[0])
        token_indices = self._build_full_causal_indices(
            b, s_global, doc_start, is_valid, position_offset, s_local
        )
        core_out = self._sparse_attn(
            query, kv, token_indices, self.softmax_scale, kv_lora_rank
        )
        return self._deabsorb(core_out, v_b_proj_weight, self.split_kv_b)

    # ------------------------------------------------------------------
    # index construction / kernel plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _build_full_causal_indices(
        b, s_global, doc_start, is_valid, position_offset=0, s_local=None
    ) -> Tensor:
        """Per-document full-causal ``[b, s_local, s_global]`` int32 (``-1`` pad).

        Built over the global sequence and row-sliced to this CP rank, so the
        column values stay global token ids. ``O(s_global^2)`` while building,
        which is why this mode is an equivalence experiment, not production.
        """
        with paddle.no_grad():
            indices, _ = _build_mqa_causal_topk_idxs_from_doc_bounds(
                b, s_global, doc_start, is_valid
            )
            if s_local is not None and s_local != s_global:
                indices = indices[
                    :, position_offset : position_offset + s_local
                ]
            indices = indices.contiguous()
        indices.stop_gradient = True
        return indices

    def _sparse_attn(
        self, query, kv, token_indices, sm_scale, d_v, indexer_topk=0
    ):
        """Sparse MQA over the absorbed latent, via the shared cudnn backend.

        Same FlashMLA sparse forward + cuDNN DSA backward pair that the CSA/HCA
        layers use; the absorbed layout only differs in ``d_v`` (512 value dims
        out of a 576-wide query/key) and in the sink being optional --
        ``softmax_offset`` is ``None`` when ``add_full_attention_sink_bias`` is
        off, which the backend turns into a sinkless softmax. Query-head padding
        to the DSA-fixed ``h_q == 64`` is the backend's job.

        ``indexer_topk > 0`` additionally returns the LSE over the first
        ``indexer_topk`` columns, which is the indexer-loss target's normalizer.
        """
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        return mqa_sparse_attn(
            query,
            kv,
            token_indices,
            sm_scale,
            d_v,
            attn_sink=self.softmax_offset,
            indexer_topk=indexer_topk,
            sink_grad_fusion=self.sink_grad_fusion,
        )

    @staticmethod
    def _deabsorb(core_out, v_b_proj_weight, split_kv_b=False) -> Tensor:
        """``[b, s, h * kv_lora_rank]`` -> ``[b, s, h * v_head_dim]``."""
        b, s, _ = core_out.shape
        if split_kv_b:
            # ``v_b_proj``: [h, v_head_dim, kv_lora_rank], the grouped-matmul
            # ``[G, R, D]`` contract -- one Triton GEMM, no transpose.
            from paddlefleet.triton_ops import fused_grouped_matmul

            h, v_head_dim, kv_lora_rank = v_b_proj_weight.shape
            out = fused_grouped_matmul(
                core_out.reshape([b, s, h, kv_lora_rank]), v_b_proj_weight
            )
        else:
            kv_lora_rank, h, v_head_dim = v_b_proj_weight.shape
            out = core_out.reshape([b, s, h, kv_lora_rank])
            out = paddle.einsum("bshl,lhv->bshv", out, v_b_proj_weight)
        out = out.reshape([b, s, h * v_head_dim])
        return out

    # ------------------------------------------------------------------
    # sparse (phase 3)
    # ------------------------------------------------------------------
    def _forward_sparse(
        self,
        query,
        kv,
        x,
        qr,
        v_b_proj_weight,
        doc_start,
        doc_len,
        is_valid,
        doc_lens,
        kv_lora_rank,
        input_ids=None,
        position_offset=0,
    ) -> Tensor:
        """Phase 3: attention consumes window + top-k, KL on that same set.

        The indexer is trained enough to steer attention, so both sides narrow to
        the selected set. Reached only when ``_phase() == "sparse"``, i.e.
        ``dsa_indexer_use_sparse_loss=True``; the warmup phase is a different
        class entirely (``mha_dsa_warmup_attention``) and shares no loss code
        with this one.

        Recompute: the loss is attached on the grad-enabled forward only, so under
        full recompute it is counted once, on the second pass. Both passes see
        identical inputs and reselect the same columns. (The top-k tie-break is
        not bit-stable in every shape -- one document spanning the whole sequence
        drifts by ~2% of the emitted slots between identical calls -- but the
        selected *set* is, and any residual drift would only change which columns
        the backward differentiates.)
        """
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        b, s = int(query.shape[0]), int(query.shape[1])
        s_global = s * self.cp_size
        need_loss = self._needs_indexer_loss()

        with paddle.no_grad():
            window_idxs = _build_window_topk_idxs_from_doc_bounds(
                b, s_global, self.window_size, doc_start, is_valid
            ).cast("int32")
            if self.cp_enabled:
                window_idxs = window_idxs[
                    :, position_offset : position_offset + s
                ]
            valid_range, row_empty = self._indexer_valid_range(
                s_global,
                doc_start,
                doc_len,
                is_valid,
                self.window_size,
                position_offset,
                s,
            )

        q_idx, k_idx, w_idx = self._indexer_projections(
            x, qr, position_offset, grad_enabled=need_loss
        )
        # ``DSAIndexer`` pre-bakes ``head_dim**-0.5`` into the weights, but both
        # cuDNN indexer kernels apply ``dim**-0.5`` themselves (the backward one
        # hardcodes it). Undo the pre-bake once so forward and backward agree.
        w_idx = w_idx * (float(self.indexer.head_dim) ** 0.5)

        # The cuDNN top-k backward (``indexer_backward_sm100.__init__``) asserts
        # ``topk % block_I == 0`` with ``block_I = 128``, so keep the configured
        # budget instead of clamping it to the sequence length. Short rows come
        # back ``-1`` padded. One width, consumed by attention and by the KL --
        # that identity *is* this phase.
        topk = int(self.indexer.index_topk)
        # The THD/varlen fast path builds ``cu_seqlens_k`` from ``doc_lens``,
        # i.e. it assumes a document-compacted K buffer. At ratio 1 the K buffer
        # is the raw token sequence, so the two only coincide when the documents
        # exactly tile the sequence; otherwise fall back to the dense path.
        # ``doc_lens`` stays global: ``_make_cu_seqlens`` is CP-aware and uses
        # ``seq_offset`` to pick out the documents this rank queries
        # (csa_indexer_fwd_cudnn.py:407-466).
        doc_lens_arg = (
            doc_lens.tolist() if int(doc_lens.sum()) == s_global else None
        )

        with paddle.no_grad():
            selected, _, *scores_out = cudnn_indexer_topk_fwd(
                q_idx.detach(),
                k_idx.detach(),
                w_idx.detach(),
                ratio=1,
                topk_effective=topk,
                valid_range=valid_range,
                doc_lens=doc_lens_arg,
                seq_offset=position_offset,
                return_topk_scores=need_loss,
            )
            topk_indices = paddle.where(
                row_empty, paddle.full_like(selected, -1), selected
            )
            # ``[indexer topk, window]``, not the other way round: the kernel's
            # ``lse_indexer`` covers the *first* ``indexer_topk`` columns
            # (``flash_mla_sparse_fwd``), and that restricted per-head LSE is
            # exactly the normalizer the loss target needs -- with it every head
            # contributes mass 1 over the selected set, matching the per-head
            # softmax of ``_attn_target_python``. Feeding the *attention* LSE
            # instead (window + sink included) would turn the target into a
            # head-mass-weighted mixture: a different objective, invisible in the
            # forward output.
            #
            # Attention itself is a softmax over a set, so the order changes only
            # the accumulation order (hence the last bits), not the value. The
            # order is unconditional on purpose: gating it on ``need_loss`` would
            # make the two forwards of a full-recompute step disagree on the
            # table layout and on ``topk_length``.
            token_indices = paddle.concat(
                [topk_indices, window_idxs], axis=-1
            ).contiguous()
        token_indices.stop_gradient = True

        # The kernel only implements a handful of ``lse_indexer`` widths; other
        # budgets fall back to the Python target path.
        if topk in _LSE_INDEXER_TOPKS:
            core_out, lse_indexer = self._sparse_attn(
                query,
                kv,
                token_indices,
                self.softmax_scale,
                kv_lora_rank,
                indexer_topk=topk,
            )
        else:
            lse_indexer = None
            core_out = self._sparse_attn(
                query, kv, token_indices, self.softmax_scale, kv_lora_rank
            )
        output = self._deabsorb(core_out, v_b_proj_weight, self.split_kv_b)
        if not need_loss:
            return output

        with paddle.no_grad():
            (topk_scores,) = scores_out
            valid = topk_indices >= 0
            scores = paddle.where(
                valid,
                topk_scores.cast("float32"),
                paddle.full(topk_indices.shape, _NEG_INF, dtype="float32"),
            )
            topk_probs = F.softmax(scores, axis=-1)
            topk_probs = paddle.where(
                valid, topk_probs, paddle.zeros_like(topk_probs)
            )
            # The window is force-selected, so it is outside the indexer's
            # decision space and outside the KL: only the top-k columns.
            target = self._attn_target(
                query.detach(), kv, topk_indices, lse_indexer
            )
            kl = target * (
                paddle.log(target + _EPS) - paddle.log(topk_probs + _EPS)
            )
            # Masked reduction, same as csa_attention.py:1302-1306, over the same
            # row mask csa_attention.py:2411-2443 builds. It has to come from
            # ``input_ids``, not from the document metadata: a packed sequence's
            # trailing padding is folded into the last document's row range, so
            # ``attn_mask_startend_row_indices`` -- and therefore ``is_valid`` --
            # still marks those rows valid. ``input_ids != pad_token_id`` is the
            # backstop that catches them, keeping the padding out of both the
            # logged loss and the gradient denominator.
            loss_mask, valid_rows = self._indexer_loss_mask(input_ids, b, s)
            # Reduction shape follows ``csa_attention._compute_fused_indexer_target``
            # (:2325-2330) with one deliberate difference: the ``/cp_size`` of the
            # unmasked branch is folded into ``loss_coeff``, i.e. it also reaches
            # the **backward**, instead of being applied to the logged scalar
            # alone the way CSA does it at :2868-2869.
            #
            # That difference is load-bearing, not cosmetic. CP sums parameter
            # gradients across the group, so every rank must normalise by the
            # *global* row count. The masked branch gets that for free
            # (``valid_rows`` is global, and it is what ``num_rows_override``
            # hands the backward); the unmasked branch has no mask, so the
            # backward falls back to the kernel's own ``1/(B*Sq)`` -- a *local*
            # denominator -- and the only place left to correct it is the
            # coefficient. Copying CSA's placement here was measured: CP=2
            # ``test_4_indexer_loss_cp_normalisation`` fails on
            # ``indexer.wq_b.linear.weight`` with relative error ``1.000`` against
            # CP=1 (exactly a factor of ``cp_size``), in both phases, while the
            # logged scalar still matched. CSA has the same latent gap on its
            # unmasked CP path; production always supplies ``input_ids``, so it is
            # the masked branch that runs.
            loss_coeff = (
                self.indexer_loss_coeff
                if loss_mask is not None
                else self.indexer_loss_coeff / self.cp_size
            )
            kl_per_pos = kl.sum(axis=-1)
            if loss_mask is None:
                loss = kl_per_pos.mean() * loss_coeff
            else:
                loss = (kl_per_pos * loss_mask).sum() / valid_rows * loss_coeff

        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                self.config
            ),
        )
        # Argument order follows ``TileLangCSAIndexerLossAutoScaler.forward``
        # (csa_attention.py): ``output, target, index_q, weights, index_k_comp,
        # topk_indices, topk_probs, loss_coeff, indexer_backend,
        # num_rows_override, loss_mask``. ``target`` sits second, right after
        # ``output`` -- the CSA call sites spell that as
        # ``apply(output, target, *state)``.
        return TileLangCSAIndexerLossAutoScaler.apply(
            output,
            target,
            q_idx,
            w_idx,
            k_idx,
            topk_indices,
            topk_probs,
            loss_coeff,
            "cudnn",
            # ``num_rows_override`` + ``loss_mask``: the backward zeroes the pad
            # rows and rescales the cuDNN kernel's built-in ``1/(B*Sq)`` into
            # ``1/valid_rows``, matching the forward reduction above. Both
            # ``None`` (no ``input_ids``) leaves the kernel's own mean in place.
            valid_rows,
            loss_mask,
        )

    def _attn_target(self, query, kv, kl_columns, lse_indexer=None) -> Tensor:
        """KL target: head-summed attention probs over the indexer's own columns.

        The denominator is **the KL's candidate set and nothing else** -- not the
        forced window, not the sink. That is the definition of this loss, not an
        oversight: the indexer is only ever asked to rank the columns it can
        choose, and ``_indexer_valid_range`` explicitly clamps the window out of
        its candidate range, so neither the window nor the sink is in its decision
        space. Upstream does the same for the CSA layers --
        ``dsa_attention._compute_dsa_indexer_loss`` adds ``index_mask`` to
        ``attention_scores`` *before* the softmax, i.e. it normalises over the
        selected set.

        Worth recording because it was measured and is easy to misread as a bug:
        normalising over the full attention row instead (window + top-k + sink)
        gives a *different* target -- a head mixture weighted by each head's mass
        on the candidate set, ``Σ_h c_h·softmax_K(l^h) / Σ_h c_h``, rather than
        the uniform ``(1/H)·Σ_h softmax_K(l^h)`` here. The gap is 4.4e-3 at the
        production sink initialisation, 1.7e-2 at ``sink=+3``, and 1.9e-2 for the
        window term alone. Those are two objectives, not right and wrong; this one
        is the intended one.

        Args:
            query: ``[1, s, h, dk]`` detached absorbed query (local rows).
            kv: ``[1, s_global, dk]`` latent keys (all-gathered under CP).
            kl_columns: ``[1, s, w]`` int32 global column ids the KL scores,
                ``-1`` for empty slots. Column *order* is irrelevant -- the
                indexer's score-ordered table can be passed straight in.
            lse_indexer: ``[1, s, 64]`` float32 per-head LSE over exactly
                ``kl_columns`` (``mqa_sparse_attn(indexer_topk=...)``). When
                present the cuDNN score-recompute kernel does the whole thing in
                one launch. ``None`` when the budget is not one of
                ``_LSE_INDEXER_TOPKS``, so the Python path stays as the reference
                and the fallback.

        Returns:
            ``[1, s, w]`` float32 rows summing to 1 (0 for empty rows).
        """
        if lse_indexer is not None:
            return self._attn_target_cudnn(query, kv, kl_columns, lse_indexer)
        return self._attn_target_python(query, kv, kl_columns)

    def _attn_target_cudnn(self, query, kv, kl_columns, lse_indexer) -> Tensor:
        """``_attn_target`` via the cuDNN DSA score-recompute kernel.

        The kernel computes ``sum_h exp(Q_h·K_i*scale - LSE_h)`` L1-normalised
        over the selected columns. With ``LSE_h`` restricted to those same
        columns each head contributes mass 1, which is what the per-head softmax
        of :meth:`_attn_target_python` produces.

        Two different head paddings meet here, and they pull in opposite
        directions:

        * The *attention* kernel pads the query to its fixed ``h_q == 64``, and
          those pad heads must stay out of the head sum. Hence the LSE is sliced
          down to the layer's real ``h`` -- the query itself is already the
          unpadded one.
        * The *target* kernel uses the query-head count as its MMA ``M`` tile
          (``_dispatch_sparse_attn_tile_params``: ``m = qhead_per_kv_head``), and
          only powers of two from 16 up work: 24/40/48/80 raise, and ``h == 8``
          -- the unit fixture's width -- silently returns an all-zero target,
          which would make the KL target uniform-after-renormalisation with no
          error anywhere. So pad back up to ``_TARGET_QHEAD_MIN`` when the layer
          is narrower.

        The pad heads carry an **infinite** LSE, so ``exp(score - inf) == 0``
        keeps them out of the head sum exactly rather than approximately (the
        query pad rows are zeros, whose score is a finite 0). Measured against
        :meth:`_attn_target_python` at ``h=8 -> 16``: 7.5e-4 max abs error, i.e.
        bf16 matmul noise.
        """
        from paddlefleet_ops.cudnn.deepseek_sparse_attention import (
            sparse_attn_score_recompute_wrapper,
        )

        b, s, h, dk = (int(dim) for dim in query.shape)
        idx = kl_columns.cast("int32").contiguous()
        lse = lse_indexer[:, :, :h].cast("float32")
        h_padded = max(_TARGET_QHEAD_MIN, 1 << (h - 1).bit_length())
        if h_padded != h:
            pad = h_padded - h
            query = paddle.concat(
                [query, paddle.zeros([b, s, pad, dk], dtype=query.dtype)],
                axis=2,
            )
            lse = paddle.concat(
                [lse, paddle.full([b, s, pad], float("inf"), dtype="float32")],
                axis=2,
            )
        target = sparse_attn_score_recompute_wrapper(
            _HashableTensor(query.contiguous()),
            _HashableTensor(kv.contiguous()),
            _HashableTensor(lse.contiguous()),
            _HashableTensor(idx),
            self.softmax_scale,
        )["target"]

        # Empty slots / all-empty rows: the kernel is not contracted to return
        # zeros there, and a row of zeros must stay a row of zeros (the KL
        # reduction divides by the valid-row count, not by the row sum).
        valid = idx >= 0
        target = paddle.where(valid, target, paddle.zeros_like(target))
        return target / target.sum(axis=-1, keepdim=True).clip(min=_EPS)

    def _attn_target_python(self, query, kv, kl_columns) -> Tensor:
        """Reference ``_attn_target``: per-head softmax over gathered keys.

        The tilelang ``csa_attn_target_reducesum`` kernel is not usable here (it
        requires a power-of-two head dim; the latent is 576) and the dense
        ``_compute_attn_target_on_selected_set`` materialises ``[b, h, s, s]``, so
        gather the selected keys in query-row chunks instead: ``s*h*w*dk`` MACs,
        no ``s*s`` tensor. The matmul runs in the input dtype (bf16) with fp32
        accumulation, as the tilelang kernel does internally for the CSA layers;
        the softmax and the L1 normalisation are fp32.
        """
        s, width = int(query.shape[1]), int(kl_columns.shape[-1])
        dk = int(query.shape[-1])
        chunk = max(1, _TARGET_ROW_SLOTS // width)
        q0, kv0, idx0 = query[0], kv[0], kl_columns[0]
        parts = []
        for start in range(0, s, chunk):
            end = min(start + chunk, s)
            idx_c = idx0[start:end].cast("int64")
            valid = idx_c >= 0
            safe = paddle.where(valid, idx_c, paddle.zeros_like(idx_c))
            k_sel = paddle.gather(kv0, safe.flatten(), axis=0).reshape(
                [end - start, width, dk]
            )
            scores = (
                paddle.matmul(q0[start:end], k_sel, transpose_y=True).cast(
                    "float32"
                )
                * self.softmax_scale
            )
            scores = paddle.where(
                valid.unsqueeze(1), scores, paddle.full_like(scores, _NEG_INF)
            )
            probs = F.softmax(scores, axis=-1).sum(axis=1)  # head-sum [c, w]
            parts.append(paddle.where(valid, probs, paddle.zeros_like(probs)))
        target = paddle.concat(parts, axis=0)
        target = target / target.sum(axis=-1, keepdim=True).clip(min=_EPS)
        return target.unsqueeze(0)
