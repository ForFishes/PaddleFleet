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

"""Shared DSA-indexer plumbing for the hybrid MLA (``csa_compress_ratios == -2``)
layers, plus the single predicate that decides whether those layers run the
absorbed latent MQA or the dense MHA of the pretraining phase.

Two core attentions own a ``DSAIndexer`` on those layers and they run different
*attention* backends, so the indexer-side pieces they do share live here rather
than in either of them:

* ``MHADSAWarmupAttention`` (phase 2, ``mha_dsa_warmup_attention.py``) -- dense
  flashmask attention, exactly the pretraining phase, with a full-candidate KL.
* ``MQALatentAttention`` (phase 3/4, ``mqa_latent_attention.py``) -- block-sparse
  attention on the KV latent, with the KL restricted to the selected set.
"""

from __future__ import annotations

import paddle
from paddle import Tensor

from paddlefleet.context_parallel_utils import ContextParallelGatherOp


def latent_mqa_enabled(config) -> bool:
    """Whether the hybrid MLA layers run *absorbed latent MQA* (not dense MHA).

    The single judgement behind both the spec dispatch
    (``gpt_layer_specs.py``) and ``MLASelfAttention.mqa_latent``, which decides
    whether ``kv_b_proj`` materialises per-head K/V at all. Splitting it would
    let the spec build a dense core attention while the enclosing MLA layer
    feeds it absorbed activations.

    * ``"mqa_full_causal"`` -- latent MQA with no indexer (equivalence isolation).
    * ``"mqa_dsa"`` -- latent MQA **only in the sparse phase**. The warmup phase
      (``dsa_indexer_use_sparse_loss=False``) has no top-k on either side, so
      absorption would buy nothing and cost a full ``[b, s, s]`` index table fed
      to a block-sparse kernel at zero sparsity; it runs the dense MHA of phase 1
      instead, with the indexer bolted on.
    * anything else (``"mha"``, non-``dsv4_hybrid`` models) -- dense MHA.
    """
    if getattr(config, "experimental_attention_variant", None) != "dsv4_hybrid":
        return False
    mode = getattr(config, "hybrid_mla_attention", "mha")
    if mode == "mqa_full_causal":
        return True
    if mode == "mqa_dsa":
        return bool(getattr(config, "dsa_indexer_use_sparse_loss", False))
    return False


class HybridMLAIndexerMixin:
    """Indexer-side helpers shared by the two DSA core attentions.

    Expects the host layer to have set ``self.config``, ``self.indexer`` and the
    CP state of :meth:`_init_hybrid_mla_cp_state`.
    """

    # Read by ``MLASelfAttention`` to decide whether to forward ``input_ids``
    # (needed for the indexer-loss row mask, since the packed sequence's trailing
    # padding is invisible to ``attn_mask_startend_row_indices``). A capability
    # of the core attention rather than a property of the phase, so no core
    # attention can be handed a kwarg it does not accept.
    accepts_input_ids = True

    def _init_hybrid_mla_cp_state(self, config, pg_collection) -> None:
        """Set ``cp_group`` / ``cp_size`` / ``cp_rank`` / ``cp_enabled``.

        Same derivation as ``csa_attention.py:2079-2090``. Deliberately asserts
        the *same* ``contiguous_allgather`` constraint the HCA layers of this
        model assert (``dsv4_hybrid_attention.py:607-611``), not a weaker one:
        the contiguous layout is what makes "build the index tables over the
        global sequence, then row-slice this rank's queries" correct, and what
        makes the all-gathered KV land in natural global order.
        """
        cp_pg = pg_collection.cp if pg_collection is not None else None
        if cp_pg is not None and getattr(cp_pg, "nranks", 1) > 1:
            self.cp_group = cp_pg
            self.cp_size = cp_pg.nranks
            self.cp_rank = cp_pg.rank
            self.cp_enabled = True
            if (
                getattr(config, "cp_balance_mode", None)
                != "contiguous_allgather"
            ):
                raise NotImplementedError(
                    f"{type(self).__name__} under context parallel requires "
                    "cp_balance_mode='contiguous_allgather' (the same mode the "
                    "hybrid model's HCA layers require), got "
                    f"{getattr(config, 'cp_balance_mode', None)!r}."
                )
        else:
            self.cp_group = None
            self.cp_size = 1
            self.cp_rank = 0
            self.cp_enabled = False

    def _needs_indexer_loss(self) -> bool:
        """Whether this forward should build and attach the indexer loss.

        ``paddle.is_grad_enabled()`` is what makes the loss count exactly once
        under recompute: the first (no-grad) forward only produces the attention
        output, the second one attaches the loss.
        """
        return (
            self.training
            and paddle.is_grad_enabled()
            and self.indexer_loss_coeff > 0
        )

    def _indexer_projections(self, x, qr, position_offset, grad_enabled):
        """``(index_q, index_k, weights)`` from the DSA indexer.

        ``x`` / ``qr`` are always detached first: the indexer loss must never
        flow back into the backbone, independently of whether the backbone
        parameters are frozen. ``index_k`` comes back all-gathered to
        ``s_global`` when CP is on (the indexer gathers the 128-wide key rather
        than the hidden states, which is ~32x less traffic).

        ``weights`` is returned exactly as ``DSAIndexer.forward_before_topk``
        produced it, i.e. carrying ``n_heads**-0.5 * head_dim**-0.5``. Every
        kernel-backed caller must undo the ``head_dim`` half itself, because both
        the cuDNN and the tilelang indexer kernels re-apply ``dim**-0.5``
        internally; only a pure-paddle evaluation of the score (as in
        ``dsa_attention.FusedDSAIndexerLoss``) uses it unscaled.
        """
        x_det, qr_det = x.detach(), qr.detach()
        if grad_enabled:
            x_det.stop_gradient = False
            qr_det.stop_gradient = False
            return self.indexer.forward_before_topk(
                x_det, qr_det, position_offset, self.cp_group
            )
        with paddle.no_grad():
            return self.indexer.forward_before_topk(
                x_det, qr_det, position_offset, self.cp_group
            )

    def _indexer_valid_range(
        self,
        s_global,
        doc_start,
        doc_len,
        is_valid,
        window,
        position_offset=0,
        s_local=None,
    ):
        """Candidate range per query, in **global token** space.

        ``window`` is how many trailing causal tokens to exclude, i.e. the
        forced local window the sparse phase adds separately: clamping the right
        edge to ``doc_start + causal_len - window`` removes every duplicate while
        leaving the full top-k budget for distant tokens. Because the clamped end
        never exceeds the kernel's own causal limit, no masked ``-inf`` column can
        enter the top-k. The warmup phase passes ``0``: it has no forced window,
        its candidate set is the whole per-document causal span.

        Built over the global sequence and row-sliced to this CP rank; the two
        columns stay global token ids, which is what the kernel's
        ``seq_offset``-aware causal bound expects.

        Returns:
            ``(valid_range [1, s_local, 2] int32, row_empty [1, s_local, 1])``.
        """
        positions = paddle.arange(s_global, dtype="int64")
        causal_avail = paddle.minimum(positions - doc_start + 1, doc_len)
        n_avail = paddle.clip(causal_avail - window, min=0)
        n_avail = paddle.where(is_valid, n_avail, paddle.zeros_like(n_avail))
        valid_range = paddle.stack(
            [doc_start, doc_start + n_avail], axis=-1
        ).cast("int32")
        if s_local is not None and s_local != s_global:
            valid_range = valid_range[
                position_offset : position_offset + s_local
            ]
            n_avail = n_avail[position_offset : position_offset + s_local]
        rows = int(valid_range.shape[0])
        return valid_range.unsqueeze(0), (n_avail == 0).reshape([1, rows, 1])

    def _indexer_loss_mask(self, input_ids: Tensor | None, b: int, s: int):
        """``([b, s] float32 row mask, its row count)`` from ``input_ids``.

        ``(None, None)`` when no ``input_ids`` reached this layer (inference and
        the direct-construction unit tests), which keeps the plain row mean.

        Under CP the mask is this rank's row slice but the denominator is the
        **global** valid-row count, so summing the per-rank losses reproduces the
        single-rank reduction. ``input_ids`` arrives sharded unless
        ``experimental_dataflow``, exactly as at ``csa_attention.py:2419-2428``.
        """
        if input_ids is None:
            return None, None
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        assert pad_token_id is not None, (
            "pad_token_id must be set in config when input_ids is provided"
        )
        if self.cp_enabled:
            if not getattr(self.config, "experimental_dataflow", False):
                input_ids = ContextParallelGatherOp.apply(
                    input_ids, axis=1, mode=self.config.cp_balance_mode
                )
            loss_mask_global = (
                input_ids.reshape([b, self.cp_size * s]) != pad_token_id
            ).astype(paddle.float32)
            valid_rows = max(float(loss_mask_global.sum()), 1.0)
            offset = self.cp_rank * s
            return loss_mask_global[:, offset : offset + s], valid_rows
        loss_mask = (input_ids.reshape([b, s]) != pad_token_id).astype(
            paddle.float32
        )
        return loss_mask, max(float(loss_mask.sum()), 1.0)
