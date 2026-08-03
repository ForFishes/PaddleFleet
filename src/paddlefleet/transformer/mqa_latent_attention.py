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

"""Non-absorbed MQA core attention for hybrid MLA layers, with DSA.

``non_absorbed_mqa`` selects which core attention the
``csa_compress_ratios == -2`` (MLA) layers of a ``dsv4_hybrid`` model run:

* ``false`` -- unchanged dense MLA (MHA); this module is not used.
* ``true``  -- :class:`MQALatentAttention` with a forced local window plus
  Lightning-indexer top-k, i.e. DeepSeek Sparse Attention on the KV latent.
  The indexer reuses the model-wide ``index_n_heads`` / ``index_head_dim`` /
  ``index_topk``.

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
analytic sink gradient. The indexer KL target is unaffected: it is renormalised
over the selected set, where the sink mass cancels.

Multi-document equivalence: RoPE/YaRN scores depend only on ``pos_q - pos_k``,
the YaRN ``mscale`` is a constant and the Hadamard ``rotate_activation`` is
orthogonal, so no per-document position reset is needed. Equivalence to running
every document on its own therefore reduces to index correctness, which is what
``_derive_csa_doc_boundaries`` plus the index builders below guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.parallel_state import get_context_parallel_world_size
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.csa_attention import (
    TileLangCSAIndexerLossAutoScaler,
    _build_mqa_causal_topk_idxs_from_doc_bounds,
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
    _validate_csa_docmask_shape,
)
from paddlefleet.transformer.dot_product_attention import build_softmax_offset
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddlefleet.transformer.enums import AttnMaskType

# Query-row chunk used when materialising the KL target on the selected set.
# 128 rows x 512 slots x 576 dims is ~75MB of gathered bf16 keys, ~150MB more
# for the fp32 copy the matmul runs on. Transient, freed every iteration.
_TARGET_CHUNK = 128
_NEG_INF = -1e30
_EPS = 1e-10


@dataclass
class MQALatentAttentionSublayersSpec:
    """Sublayers spec for :class:`MQALatentAttention`.

    Args:
        indexer: ``DSAIndexer`` spec. ``non_absorbed_mqa`` always provides one;
            ``None`` (dense per-document causal attention, mathematically equal
            to MHA) exists only for the absorption equivalence unit tests.
    """

    indexer: LayerSpec | type = None


class MQALatentAttention(FleetLayer):
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
    ) -> Tensor:
        """Absorbed-MQA forward.

        Args:
            query: ``[b, s, h, kv_lora_rank + qk_rope_head_dim]`` absorbed query.
            key:   ``[b, s, 1, kv_lora_rank + qk_rope_head_dim]`` shared latent.
            value: unused (``None``); the V side lives inside ``key``.
            attn_mask_startend_row_indices: ``[b, 1, s, 1]`` exclusive per-token
                document end rows. ``None`` means a single document.
            x / qr: hidden states / q latent, inputs of the DSA indexer.
            v_b_proj_weight: ``[kv_lora_rank, h, v_head_dim]`` de-absorption
                weight (the V slice of ``kv_b_proj``).

        Returns:
            ``[b, s, h * v_head_dim]``
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "non_absorbed_mqa=True does not support packed_seq_params; "
                "document masking is driven by "
                "attn_mask_startend_row_indices."
            )
        if get_context_parallel_world_size() > 1:
            raise NotImplementedError(
                "non_absorbed_mqa=True does not support context parallel yet: "
                "the document metadata below is derived in local token space, "
                "while a CP rank only holds a query slice of the globally "
                "all-gathered KV. Keep non_absorbed_mqa=false for CP runs (the "
                "dense MHA path supports contiguous_allgather)."
            )
        if v_b_proj_weight is None:
            raise ValueError(
                "MQALatentAttention requires v_b_proj_weight; it is only valid "
                "as the core_attention of an absorbed MLA layer."
            )

        b, s = int(query.shape[0]), int(query.shape[1])
        if b != 1:
            raise NotImplementedError(
                "non_absorbed_mqa=True requires micro batch size 1 (documents "
                f"are packed along the sequence), got b={b}."
            )
        kv = key.squeeze(2).contiguous()  # [b, s, kv_lora + qk_rope]
        kv_lora_rank = int(v_b_proj_weight.shape[0])

        with paddle.no_grad():
            row_end = attn_mask_startend_row_indices
            if row_end is None:
                row_end = paddle.full([b, 1, s, 1], s, dtype="int32")
            _validate_csa_docmask_shape(row_end, b, s)
            doc_start, doc_len, is_valid, doc_lens, _ = (
                _derive_csa_doc_boundaries(row_end, s)
            )

        if self.indexer is None:
            # Absorption-equivalence path (unit tests only): per-document full
            # causal attention, mathematically identical to dense MHA.
            token_indices = self._build_full_causal_indices(
                b, s, doc_start, is_valid
            )
            core_out = self._sparse_attn(
                query, kv, token_indices, self.softmax_scale, kv_lora_rank
            )
            return self._deabsorb(core_out, v_b_proj_weight)

        return self._forward_dsa(
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
        )

    # ------------------------------------------------------------------
    # index construction / kernel plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _build_full_causal_indices(b, s, doc_start, is_valid) -> Tensor:
        """Per-document full-causal ``[b, s, s]`` int32 table (``-1`` padded)."""
        with paddle.no_grad():
            indices, _ = _build_mqa_causal_topk_idxs_from_doc_bounds(
                b, s, doc_start, is_valid
            )
            indices = indices.contiguous()
        indices.stop_gradient = True
        return indices

    def _indexer_valid_range(self, s, doc_start, doc_len, is_valid):
        """Non-local candidate range per query, in **global token** space.

        The forced local window already covers the last ``window_size`` causal
        tokens, so the indexer must only rank what lies *before* it. Clamping
        the right edge to ``doc_start + causal_len - window_size`` removes every
        duplicate while leaving the full top-k budget for distant tokens.
        Because the clamped end never exceeds the kernel's own causal limit, no
        masked ``-inf`` column can enter the top-k.

        Returns:
            ``(valid_range [1, s, 2] int32, row_empty [1, s, 1] bool)``.
        """
        positions = paddle.arange(s, dtype="int64")
        causal_avail = paddle.minimum(positions - doc_start + 1, doc_len)
        n_avail = paddle.clip(causal_avail - self.window_size, min=0)
        n_avail = paddle.where(is_valid, n_avail, paddle.zeros_like(n_avail))
        valid_range = (
            paddle.stack([doc_start, doc_start + n_avail], axis=-1)
            .cast("int32")
            .unsqueeze(0)
        )
        return valid_range, (n_avail == 0).reshape([1, s, 1])

    def _sparse_attn(self, query, kv, token_indices, sm_scale, d_v):
        """Sparse MQA over the absorbed latent, via the shared cudnn backend.

        Same FlashMLA sparse forward + cuDNN DSA backward pair that the CSA/HCA
        layers use; the absorbed layout only differs in ``d_v`` (512 value dims
        out of a 576-wide query/key) and in the sink being optional --
        ``softmax_offset`` is ``None`` when ``add_full_attention_sink_bias`` is
        off, which the backend turns into a sinkless softmax. Query-head padding
        to the DSA-fixed ``h_q == 64`` is the backend's job.
        """
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        return mqa_sparse_attn(
            query,
            kv,
            token_indices,
            sm_scale,
            d_v,
            attn_sink=self.softmax_offset,
        )

    @staticmethod
    def _deabsorb(core_out, v_b_proj_weight) -> Tensor:
        """``[b, s, h * kv_lora_rank]`` -> ``[b, s, h * v_head_dim]``."""
        b, s, _ = core_out.shape
        kv_lora_rank, h, v_head_dim = v_b_proj_weight.shape
        out = core_out.reshape([b, s, h, kv_lora_rank])
        out = paddle.einsum("bshl,lhv->bshv", out, v_b_proj_weight)
        return out.reshape([b, s, h * v_head_dim])

    # ------------------------------------------------------------------
    # mqa_dsa
    # ------------------------------------------------------------------
    def _forward_dsa(
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
    ) -> Tensor:
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        b, s = int(query.shape[0]), int(query.shape[1])
        # The indexer loss is only attached on the grad-enabled forward. Under
        # full-layer recompute the first forward runs under no_grad and must
        # only materialise indices; the top-k is deterministic, so both forwards
        # select the same columns.
        need_loss = (
            self.training
            and paddle.is_grad_enabled()
            and self.indexer_loss_coeff > 0
        )

        with paddle.no_grad():
            window_idxs = _build_window_topk_idxs_from_doc_bounds(
                b, s, self.window_size, doc_start, is_valid
            ).cast("int32")
            valid_range, row_empty = self._indexer_valid_range(
                s, doc_start, doc_len, is_valid
            )

        x_det, qr_det = x.detach(), qr.detach()
        if need_loss:
            x_det.stop_gradient = False
            qr_det.stop_gradient = False
            q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                x_det, qr_det
            )
        else:
            with paddle.no_grad():
                q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                    x_det, qr_det
                )
        # ``DSAIndexer`` pre-bakes ``head_dim**-0.5`` into the weights, but both
        # cuDNN indexer kernels apply ``dim**-0.5`` themselves (the backward one
        # hardcodes it). Undo the pre-bake once so forward and backward agree.
        w_idx = w_idx * (float(self.indexer.head_dim) ** 0.5)

        # Both forward top-k paths return a table of exactly ``topk_effective``
        # columns (short rows are ``-1`` padded), and the backward kernel
        # ``indexer_backward_sm100.__init__`` asserts ``topk % block_I == 0``
        # with ``block_I=128``. So keep the configured budget instead of
        # clamping it to the sequence length, which would break that assert.
        topk_eff = int(self.indexer.index_topk)
        # The THD/varlen fast path builds ``cu_seqlens_k`` from ``doc_lens``,
        # i.e. it assumes a document-compacted K buffer. At ratio 1 the K buffer
        # is the raw token sequence, so the two only coincide when the documents
        # exactly tile the sequence; otherwise fall back to the dense path.
        doc_lens_arg = doc_lens.tolist() if int(doc_lens.sum()) == s else None
        with paddle.no_grad():
            topk_out = cudnn_indexer_topk_fwd(
                q_idx.detach(),
                k_idx.detach(),
                w_idx.detach(),
                ratio=1,
                topk_effective=topk_eff,
                valid_range=valid_range,
                doc_lens=doc_lens_arg,
                return_topk_scores=need_loss,
            )
            topk_indices = topk_out[0]
            topk_indices = paddle.where(
                row_empty, paddle.full_like(topk_indices, -1), topk_indices
            )
            token_indices = paddle.concat(
                [window_idxs, topk_indices], axis=-1
            ).contiguous()
        token_indices.stop_gradient = True

        core_out = self._sparse_attn(
            query, kv, token_indices, self.softmax_scale, kv_lora_rank
        )
        output = self._deabsorb(core_out, v_b_proj_weight)
        if not need_loss:
            return output

        with paddle.no_grad():
            valid = topk_indices >= 0
            scores = paddle.where(
                valid,
                topk_out[2].cast("float32"),
                paddle.full(topk_indices.shape, _NEG_INF, dtype="float32"),
            )
            topk_probs = F.softmax(scores, axis=-1)
            topk_probs = paddle.where(
                valid, topk_probs, paddle.zeros_like(topk_probs)
            )
            target = self._attn_target(query.detach(), kv, topk_indices)
            kl = target * (
                paddle.log(target + _EPS) - paddle.log(topk_probs + _EPS)
            )
            loss = kl.sum(axis=-1).mean() * self.indexer_loss_coeff

        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                self.config
            ),
        )
        return TileLangCSAIndexerLossAutoScaler.apply(
            output,
            q_idx,
            w_idx,
            k_idx,
            topk_indices,
            topk_probs,
            target,
            self.indexer_loss_coeff,
            "cudnn",
        )

    def _attn_target(self, query, kv, topk_indices) -> Tensor:
        """KL target: head-summed attention probs restricted to the top-k set.

        The tilelang ``csa_attn_target_reducesum`` kernel requires a
        power-of-two head dim, which the 576-wide latent is not, and the dense
        ``_compute_attn_target_on_selected_set`` materialises ``[b, h, s, s]``.
        So gather the selected keys in query-row chunks instead.

        Args:
            query: ``[1, s, h, dk]`` detached absorbed query.
            kv: ``[1, s, dk]`` latent keys.
            topk_indices: ``[1, s, topk]`` int32, ``-1`` for empty slots.

        Returns:
            ``[1, s, topk]`` float32 rows summing to 1 (0 for empty rows).
        """
        s, topk = int(query.shape[1]), int(topk_indices.shape[-1])
        dk = int(query.shape[-1])
        q0, kv0, idx0 = query[0], kv[0], topk_indices[0]
        parts = []
        for start in range(0, s, _TARGET_CHUNK):
            end = min(start + _TARGET_CHUNK, s)
            idx_c = idx0[start:end].cast("int64")
            valid = idx_c >= 0
            safe = paddle.where(valid, idx_c, paddle.zeros_like(idx_c))
            k_sel = paddle.gather(kv0, safe.flatten(), axis=0).reshape(
                [end - start, topk, dk]
            )
            # fp32 *before* the matmul, matching the reference target
            # (``_compute_attn_target_on_selected_set``); a bf16 matmul here
            # perturbs the KL target by ~1e-4.
            scores = (
                paddle.matmul(
                    q0[start:end].cast("float32"),
                    k_sel.cast("float32"),
                    transpose_y=True,
                )
                * self.softmax_scale
            )
            scores = paddle.where(
                valid.unsqueeze(1), scores, paddle.full_like(scores, _NEG_INF)
            )
            probs = F.softmax(scores, axis=-1).sum(axis=1)  # head-sum [c, topk]
            parts.append(paddle.where(valid, probs, paddle.zeros_like(probs)))
        target = paddle.concat(parts, axis=0)
        target = target / target.sum(axis=-1, keepdim=True).clip(min=_EPS)
        return target.unsqueeze(0)
