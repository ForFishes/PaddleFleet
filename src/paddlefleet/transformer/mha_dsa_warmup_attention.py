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

"""Phase 2 (DSA warmup) core attention: the pretraining dense MHA, plus an
indexer trained over the full causal candidate set.

Phase 2 freezes the backbone (``train_indexer_only``) and has **no top-k on
either side**: the attention must see exactly the activations phase 1
pretrained, and the indexer is supervised over every causal column so it cannot
reinforce its own random initial ranking.

Attention therefore has nothing to gain from the absorbed latent MQA of phase 3
and a great deal to lose: routing a zero-sparsity candidate set through the
block-sparse kernel means materialising a per-document causal index table
(``[b, s, s]`` int32, 256MB at s=8192, built from an ``[s, s]`` int64
intermediate twice that size -- ``csa_attention._build_mqa_causal_topk_idxs_
from_doc_bounds``) and then having the kernel walk all ``s`` columns anyway.
Dense flashmask does the same maths with no ``s x s`` tensor at all. So this
class subclasses :class:`DotProductAttention` and delegates the whole attention
half to ``super().forward``; only the indexer loss is new.

Consequences of that choice, all deliberate:

* ``MLASelfAttention.mqa_latent`` is False in this phase
  (``hybrid_mla_indexer.latent_mqa_enabled``), so ``kv_b_proj`` materialises
  per-head K/V and the layer is bit-for-bit the phase-1 layer. Including its
  constraints: a dense MLA attention sink needs
  ``FLAGS_flash_attn_version in (3, 4)`` (checked at construction in
  ``multi_latent_attention.py``). "Phase 2 runs wherever phase 1 runs" is the
  point.
* The sink is *not* in the KL target, exactly as in phase 3: the target
  normalises over the indexer's own candidate set, and the sink is outside it by
  construction. See ``mqa_latent_attention.MQALatentAttention._attn_target``
  for the measured size of the alternative definition.
* ``softmax_offset`` is built by the shared ``build_softmax_offset`` (inherited
  from :class:`DotProductAttention`) and the indexer lives at
  ``core_attention.indexer.*``, so the phase-1 -> 2 -> 3 parameter names line up
  and an HF checkpoint moves between the phases with no rename mapping.

What is *not* removed here: ``target`` / ``probs`` / ``columns`` are still
``[1, s, s_global]`` (three 256MB transients at s=8192). That width is the
full-candidate KL objective itself -- ``csa_indexer_topk_fwd`` in its documented
"full-candidate selection" mode returns one slot per candidate -- not an
artefact of the attention backend.
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
    _derive_csa_doc_boundaries,
    _validate_csa_docmask_shape,
)
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.hybrid_mla_indexer import HybridMLAIndexerMixin

if TYPE_CHECKING:
    from paddlefleet.transformer.enums import AttnMaskType

# Row budget of the KL-target scoring loop, as ``rows x candidate slots``.
# The peak tensor is the fp32 score block ``[chunk, h, s_global]``; at
# s_global=8192 / h=64 this budget gives chunk=16, i.e. 33.5MB.
_TARGET_ROW_SLOTS = 256 * 512
_NEG_INF = -1e30
_EPS = 1e-10


@dataclass
class MHADSAWarmupAttentionSublayersSpec:
    """Sublayers spec for :class:`MHADSAWarmupAttention`.

    Args:
        indexer: ``DSAIndexer`` spec. Always provided in this phase -- a warmup
            layer without an indexer would just be the phase-1 dense layer, and
            ``gpt_layer_specs`` builds that one instead.
    """

    indexer: LayerSpec | type = None


class MHADSAWarmupAttention(HybridMLAIndexerMixin, DotProductAttention):
    """Dense MHA attention (phase 1's, unchanged) with the DSA indexer loss."""

    def __init__(
        self,
        config,
        sublayers_spec: MHADSAWarmupAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        pg_collection: ProcessGroupCollection | None = None,
        **kwargs,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            pg_collection=pg_collection,
            **kwargs,
        )

        DSAIndexerLossLoggingHelper.register_total_num_layers(config)
        self._init_hybrid_mla_cp_state(config, pg_collection)
        if sublayers_spec.indexer is None:
            raise ValueError(
                "MHADSAWarmupAttention requires an indexer; a warmup layer "
                "without one is just the dense phase-1 layer."
            )
        self.indexer = build_spec_layer(
            sublayers_spec.indexer,
            config=config,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )
        self.indexer_loss_coeff = float(
            getattr(config, "dsa_indexer_loss_coeff", 0.0) or 0.0
        )

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
        """Dense MHA forward, with the indexer loss attached to its output.

        Every attention-side argument is forwarded to
        :meth:`DotProductAttention.forward` untouched, so this phase inherits
        the whole dense dispatch (flashmask / flashmask-CP / SDPA / eager /
        varlen / refined recompute / KV cache) rather than re-implementing any
        of it. ``x`` / ``qr`` reach the indexer instead of being ignored, and
        ``input_ids`` (which the base class does not accept) only builds the
        indexer-loss row mask.

        Returns:
            ``[b, s, h * v_head_dim]`` -- this rank's query slice under CP.
        """
        output = super().forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            use_rr_flash_attention=use_rr_flash_attention,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            use_cache=use_cache,
            x=x,
            qr=qr,
            kv_compressed=kv_compressed,
            k_pos_emb=k_pos_emb,
            q_absorbed=q_absorbed,
            v_b_proj_weight=v_b_proj_weight,
        )
        if not self._needs_indexer_loss():
            return output
        return self._attach_indexer_loss(
            output,
            query,
            key,
            x,
            qr,
            attn_mask_startend_row_indices,
            packed_seq_params,
            input_ids,
        )

    def _attach_indexer_loss(
        self,
        output: Tensor,
        query: Tensor,
        key: Tensor,
        x: Tensor,
        qr: Tensor,
        row_end: Tensor | None,
        packed_seq_params,
        input_ids: Tensor | None,
    ) -> Tensor:
        """Full-candidate indexer KL, attached to ``output``'s gradient.

        One ``csa_indexer_topk_fwd`` call in its documented "full-candidate
        selection" mode (``ratio=1``, ``topk_effective=s_global``) gives both the
        candidate columns and the indexer's softmax over them; the head dimension
        never leaves the kernel. The backward is upstream's ``csa_indexer_bwd``
        via :class:`TileLangCSAIndexerLossAutoScaler`, whose tilelang branch
        computes exactly ``(P - Q) * coeff / valid_rows``.

        Recompute: the loss is attached on the grad-enabled forward only
        (``_needs_indexer_loss``), so it is counted once, and the no-grad forward
        skips the indexer entirely instead of computing and discarding it.

        CP: ``index_k`` is all-gathered to ``s_global`` inside
        ``forward_before_topk`` and the per-head ``key`` here, ``valid_range`` is
        built over the global sequence and row-sliced, and ``valid_rows`` is the
        global valid-row count -- so the per-rank losses sum to the single-rank
        one.
        """
        from paddlefleet.tilelang_ops import csa_indexer_topk_fwd

        if packed_seq_params is not None:
            raise NotImplementedError(
                "the DSA warmup indexer loss does not support "
                "packed_seq_params; document masking is driven by "
                "attn_mask_startend_row_indices."
            )
        b, s_local = int(query.shape[0]), int(query.shape[1])
        if b != 1:
            raise NotImplementedError(
                "the DSA warmup indexer loss requires micro batch size 1 "
                f"(documents are packed along the sequence), got b={b}."
            )
        s_global = s_local * self.cp_size
        position_offset = self.cp_rank * s_local

        self._check_tilelang_indexer_support()
        index_q, index_k, weights = self._indexer_projections(
            x, qr, position_offset, grad_enabled=True
        )
        # ``DSAIndexer`` pre-bakes ``head_dim**-0.5`` into the weights and the
        # tilelang indexer kernels apply ``dim**-0.5`` themselves, so undo the
        # pre-bake once -- before both the forward call and the weights handed to
        # the backward, so the two agree. Getting this wrong is silent: measured
        # against a plain-paddle reference the un-baked weights match to max_abs
        # 3.0e-8 / cosine 1-1.5e-13, unscaled they give max_abs 7.5e-1 /
        # cosine 0.62.
        weights = weights * (float(self.indexer.head_dim) ** 0.5)

        with paddle.no_grad():
            if row_end is None:
                row_end = paddle.full(
                    [b, 1, s_global, 1], s_global, dtype="int32"
                )
            _validate_csa_docmask_shape(row_end, b, s_global)
            doc_start, doc_len, is_valid, _, _ = _derive_csa_doc_boundaries(
                row_end, s_global
            )
            # No forced window in this phase, so the candidate range is the
            # whole per-document causal span.
            valid_range, row_empty = self._indexer_valid_range(
                s_global,
                doc_start,
                doc_len,
                is_valid,
                0,
                position_offset,
                s_local,
            )
            columns, probs = csa_indexer_topk_fwd(
                index_q.detach(),
                index_k.detach(),
                weights.detach(),
                ratio=1,
                topk_effective=s_global,
                seq_offset=position_offset,
                valid_range=valid_range,
            )
            columns = paddle.where(
                row_empty, paddle.full_like(columns, -1), columns
            )
            probs = paddle.where(columns >= 0, probs, paddle.zeros_like(probs))
            target = self._dense_attn_target(
                query.detach(),
                all_gather_cp(key.detach(), dim=1, group=self.cp_group),
                columns,
                doc_start,
                is_valid,
                position_offset,
                s_local,
                s_global,
            )
            loss_mask, valid_rows = self._indexer_loss_mask(
                input_ids, b, s_local
            )
            # The unmasked branch's ``/cp_size`` has to sit in ``loss_coeff``
            # (and therefore reach the backward), not only in the logged scalar
            # the way ``csa_attention`` places it -- see the long comment at
            # ``mqa_latent_attention._forward_sparse``.
            loss_coeff = (
                self.indexer_loss_coeff
                if loss_mask is not None
                else self.indexer_loss_coeff / self.cp_size
            )
            kl = (
                target * (paddle.log(target + _EPS) - paddle.log(probs + _EPS))
            ).sum(axis=-1)
            if loss_mask is None:
                loss = kl.mean() * loss_coeff
            else:
                loss = (kl * loss_mask).sum() / valid_rows * loss_coeff

        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                self.config
            ),
        )
        # ``TileLangCSAIndexerLossAutoScaler`` returns its first argument
        # unchanged when that argument needs a gradient, and Paddle records a
        # PyLayer returning one of its inputs as an inplace write on it
        # (version 0 -> 1). The dense attention backward *saves its own output*,
        # so the version bump makes the attention backward -- not ours -- raise
        # ``PermissionDenied: Tensor ... modified by an inplace operation``
        # (``tensor_wrapper.h:268``). Hand the scaler a fresh tensor to bump
        # instead; ``clone`` is a gradient identity. The copy is confined to
        # this phase: phase 3 passes a fresh matmul result
        # (``mqa_latent_attention._deabsorb``) and the CSA call sites a fresh
        # kernel output, and the scaler already clones when the backbone is
        # frozen, which is why the production ``train_indexer_only`` run never
        # hit this.
        if not output.stop_gradient:
            output = output.clone()
        return TileLangCSAIndexerLossAutoScaler.apply(
            output,
            target,
            index_q,
            weights,
            index_k,
            columns,
            probs,
            loss_coeff,
            "tilelang",
            valid_rows,
            loss_mask,
        )

    def _check_tilelang_indexer_support(self) -> None:
        """Fail loudly on the one tilelang indexer constraint we cannot absorb.

        The candidate *width* needs no check: the wrappers round
        ``topk_effective`` up to a power-of-two multiple of their block and crop
        the result back (``csa_indexer_fwd.py:430-462``,
        ``csa_indexer_bwd.py:617-638``), so any causal span from 1 upwards is
        served -- measured at s = 1/2/4/8/16/32/300/384/512/8192.

        The head count is different: ``index_n_heads`` other than 64 trips the
        kernel's warp tiling with a bare
        ``Check failed: (m_warp * n_warp == num_warps)`` from inside tilelang
        (measured with 8). Reject that here rather than at the launch. It is not
        checked at config time on purpose -- that would make every
        small-geometry unit fixture unrepresentable.
        """
        heads = int(self.indexer.n_heads)
        if heads != 64:
            raise ValueError(
                "the tilelang indexer's warp tiling requires "
                f"index_n_heads == 64 (measured: 8 fails inside the kernel), "
                f"got {heads}."
            )

    def _dense_attn_target(
        self,
        query: Tensor,
        key: Tensor,
        columns: Tensor,
        doc_start: Tensor,
        is_valid: Tensor,
        position_offset: int,
        s_local: int,
        s_global: int,
    ) -> Tensor:
        """KL target: head-summed attention probs, in the kernel's column order.

        The per-head layout forbids the phase-3 trick of gathering the selected
        keys (``[chunk, width, h, dk]`` is 3.2GB at chunk=16 / width=8192 /
        h=64 / dk=256), so score in **natural column order** instead -- the full
        causal row is the candidate set in this phase anyway -- and permute
        afterwards with ``take_along_axis``. That is exact, not an
        approximation: ``columns`` holds global token ids and the natural-order
        index *is* the global token id.

        Scored in query-row chunks, ``[chunk, h, s_global]`` fp32 at a time. The
        matmul runs in the input dtype (bf16) with fp32 accumulation, as the
        tilelang kernel does internally for the CSA layers; the softmax and the
        L1 normalisation are fp32.

        Args:
            query: ``[1, s_local, h, dk]`` detached per-head query (local rows).
            key: ``[1, s_global, h, dk]`` per-head key (all-gathered under CP).
            columns: ``[1, s_local, s_global]`` int32 candidate ids, ``-1`` for
                empty slots.
            doc_start / is_valid: ``[s_global]``, from
                ``_derive_csa_doc_boundaries``.

        Returns:
            ``[1, s_local, s_global]`` float32, rows summing to 1 (0 for empty
            rows).
        """
        h = int(query.shape[2])
        chunk = max(1, _TARGET_ROW_SLOTS // s_global)
        # Head-major once, rather than per chunk: [h, s, dk].
        q_all = query[0].transpose([1, 0, 2])
        k_all = key[0].transpose([1, 0, 2])
        cols = paddle.arange(s_global, dtype="int64")
        parts = []
        for start in range(0, s_local, chunk):
            end = min(start + chunk, s_local)
            lo, hi = position_offset + start, position_offset + end
            rows = paddle.arange(lo, hi, dtype="int64").unsqueeze(1)
            # Per-document causal, i.e. exactly the range
            # ``_indexer_valid_range(window=0)`` handed the kernel.
            allowed = (
                (cols.unsqueeze(0) >= doc_start[lo:hi].unsqueeze(1))
                & (cols.unsqueeze(0) <= rows)
                & is_valid[lo:hi].unsqueeze(1)
            )
            scores = (
                paddle.matmul(
                    q_all[:, start:end], k_all, transpose_y=True
                ).cast("float32")
                * self.softmax_scale
            )
            scores = paddle.where(
                allowed.unsqueeze(0), scores, paddle.full_like(scores, _NEG_INF)
            )
            # Each head contributes mass 1; an all-masked (padding) row would
            # give a uniform softmax, so zero it explicitly -- a row of zeros
            # must stay a row of zeros, because the KL reduction divides by the
            # valid-row count, not by the row sum.
            probs = F.softmax(scores, axis=-1).sum(axis=0)
            parts.append(paddle.where(allowed, probs, paddle.zeros_like(probs)))
        target = paddle.concat(parts, axis=0)
        target = target / target.sum(axis=-1, keepdim=True).clip(min=_EPS)

        idx = columns[0].cast("int64")
        valid = idx >= 0
        target = paddle.take_along_axis(
            target, paddle.where(valid, idx, paddle.zeros_like(idx)), axis=-1
        )
        target = paddle.where(valid, target, paddle.zeros_like(target))
        return target.unsqueeze(0)
