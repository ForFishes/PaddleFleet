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

"""Unit tests for :mod:`paddlefleet.transformer.mqa_latent_attention`.

``hybrid_mla_attention`` decides which core attention the hybrid MLA
(``csa_compress_ratios == -2``) layers of a ``dsv4_hybrid`` model run.
:class:`MQALatentAttention` (latent MQA) owns exactly the two modes that attend
to the **absorbed KV latent**, i.e. the ones that consume a sorted candidate
set, and it picks between them from the sublayers spec rather than from any
config string:

* ``MQALatentAttentionSublayersSpec(indexer=None)`` -- per-document full-causal
  attention on the latent, mathematically equal to MHA. This is what production
  builds for ``"mqa_full_causal"``. The absorption-equivalence tests here drive
  it by constructing the layer directly with ``indexer=None``.
* an indexer spec **plus** ``dsa_indexer_use_sparse_loss=True`` (phase 3/4) --
  forced local window + Lightning-indexer top-k, i.e. DSA on the KV latent.

The other two modes are not this class, and
``hybrid_mla_indexer.latent_mqa_enabled`` is the single predicate that keeps the
spec dispatch and ``MLASelfAttention.mqa_latent`` in step about it: ``"mha"``
and ``"mqa_dsa"`` + ``dsa_indexer_use_sparse_loss=False`` (phase 2, the DSA
warmup) both run dense per-head attention, the latter in
``mha_dsa_warmup_attention.MHADSAWarmupAttention``. An indexer on this class
with the sparse loss off is therefore an **error state**, not a phase, and
``_phase()`` raises for it.

Coverage:
  1. Guards -- unsupported configurations fail loudly (no GPU needed).
  2. Index construction over adversarial multi-document layouts: the forced
     128-window and the indexer candidate range are disjoint yet jointly equal
     the per-document causal set (no duplicate column, no lost window column).
  3. The indexer-less full-causal path equals a dense fp32 reference, because
     the activation-level absorption is exactly score preserving.
  4. Packed multi-document equals independent per-document runs, bit exact.
  5. The DSA (indexer) path: a saturated budget reproduces the dense
     reference, a sparse budget stays causal/duplicate-free, backward yields
     finite gradients, and the recompute double-forward selects identical
     columns while attaching the indexer loss on the grad-enabled pass only.
  6. The model-wide learnable per-head sink (``add_full_attention_sink_bias`` /
     ``softmax_type``, built by the shared ``build_softmax_offset`` helper)
     equals one extra value-less softmax column (on both the dense and the DSA
     path) and takes a finite non-zero fp32 gradient. There is a single sink
     switch now, so the config no longer rejects any combination.
  7. The phase-2 (warmup) shape of ``"mqa_dsa"``, selected by
     ``dsa_indexer_use_sparse_loss=False``: it is the **dense** backend
     ``MHADSAWarmupAttention``, bit-identical to phase 1's attention, building
     no index table at all while the indexer's KL spans every causal column.
     Plus the two guards that keep the split honest: ``latent_mqa_enabled`` over
     all four config combinations, and ``MQALatentAttention`` refusing to run
     that phase itself.
  8. Migration: the renamed config keys (``non_absorbed_mqa*``,
     ``csa_train_indexer_only``, ``csa_indexer_init_from_scratch``) ship without
     an alias, so a stale config must raise rather than be absorbed into a
     silent default.
  9. The fused indexer-loss target's plumbing with both kernels stubbed out:
     the ``_attn_target`` dispatch, ``_attn_target_cudnn``'s call contract and
     empty-slot handling, ``mqa_sparse_attn``'s ``lse_indexer`` side channel and
     the ``_forward_sparse`` branch that asks for it. The kernels themselves need
     SM100+ (see 5 and ``TestMQADSACudnnTarget``); everything around them is
     plain Python and stays checked on machines that lack them.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle

from paddlefleet.transformer.csa_attention import (
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.hybrid_mla_indexer import (
    HybridMLAIndexerMixin,
    latent_mqa_enabled,
)
from paddlefleet.transformer.mha_dsa_warmup_attention import (
    MHADSAWarmupAttention,
)
from paddlefleet.transformer.mqa_latent_attention import (
    _LSE_INDEXER_TOPKS,
    MQALatentAttention,
    MQALatentAttentionSublayersSpec,
    _HashableTensor,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

from .hybrid_mla_utils import (
    _CAPTURED,
    _GPU,
    _WARMUP_TARGETS,
    DK,
    DV,
    HIDDEN,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    K_CHANNELS,
    Q_LORA,
    V_HEAD_DIM,
    WINDOW,
    H,
    _build_module,
    _build_phase1_dense_module,
    _check_index_invariants,
    _create_mqa_config,
    _dense_mha_reference,
    _dense_reference,
    _indexer_layer_spec,
    _make_dense_inputs,
    _make_inputs,
    _rel,
    _row_end,
)

# Adversarial document layouts: shorter than / equal to / longer than the
# forced window, single-token documents, and a document overrunning the buffer.
_LAYOUTS = [
    [1, 2, 3],
    [5, 7],
    [WINDOW],
    [WINDOW + 1],
    [WINDOW - 1, 2],
    [WINDOW + 2, 1],
    [8],
    [1, 1, 1, 1, 1, 1],
    [3, WINDOW, WINDOW + 1, 1],
    [300],
]


def _full_causal_table(layout, seqlen):
    """The per-document full-causal ``[1, s, s]`` table, from the production
    builder itself -- it is a pure integer function of the document bounds.

    Still the attention table of the indexer-less latent path; for phase 2 it is
    now only the *analytic* per-document causal set (that phase materialises no
    table at all), which is what its indexer KL must span.
    """
    row_end = _row_end(layout, seqlen)
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
    table = MQALatentAttention._build_full_causal_indices(
        1, seqlen, doc_start, is_valid
    )
    return table.numpy()


def _fp32(tensor):
    """bf16 -> fp32 numpy; the widening is exact, so bit equality survives."""
    return tensor.cast("float32").numpy()


def _build_latent_warmup_module(loss_coeff=0.01):
    """A ``MQALatentAttention`` in the (now illegal) phase-2 combination.

    ``_build_module`` cannot produce this: it dispatches on the production
    predicate ``latent_mqa_enabled``, which sends ``"mqa_dsa"`` +
    ``dsa_indexer_use_sparse_loss=False`` to ``MHADSAWarmupAttention``. Reaching
    the guard therefore means building the latent class by hand -- which is
    exactly the situation the guard is for: a spec change that forgets the
    predicate must fail loudly instead of quietly running a zero-sparsity
    block-sparse kernel.

    Local helper on purpose: ``hybrid_mla_utils`` is shared with other suites.
    """
    config = _create_mqa_config(
        "mqa_dsa", loss_coeff=loss_coeff, sparse_loss=False
    )
    return MQALatentAttention(
        config=config,
        sublayers_spec=MQALatentAttentionSublayersSpec(
            indexer=_indexer_layer_spec()
        ),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        k_channels=K_CHANNELS,
    )


class TestMQAGuards(unittest.TestCase):
    """Unsupported configurations must fail loudly, not silently mis-compute."""

    def setUp(self):
        self.module = _build_module(_create_mqa_config("mqa"))

    @staticmethod
    def _args(b=1, s=8):
        query = paddle.zeros([b, s, H, DK], dtype="bfloat16")
        key = paddle.zeros([b, s, 1, DK], dtype="bfloat16")
        w_v = paddle.zeros([DV, H, V_HEAD_DIM], dtype="bfloat16")
        return query, key, w_v

    def test_packed_seq_params_rejected(self):
        query, key, w_v = self._args()
        with self.assertRaises(NotImplementedError):
            self.module(
                query,
                key,
                None,
                None,
                packed_seq_params=object(),
                v_b_proj_weight=w_v,
            )

    def test_missing_v_b_proj_weight_rejected(self):
        query, key, _ = self._args()
        with self.assertRaises(ValueError):
            self.module(query, key, None, None)

    def test_batch_size_gt_one_rejected(self):
        query, key, w_v = self._args(b=2)
        with self.assertRaises(NotImplementedError):
            self.module(query, key, None, None, v_b_proj_weight=w_v)

    def test_v_b_proj_weight_layout_mismatch_rejected(self):
        # Both layouts reshape fine, so a ``[h, v, l]`` weight handed to the
        # einsum path (or the reverse) would silently mis-compute. The
        # contraction dim is checked against the config rank instead.
        query, key, _ = self._args()
        w_v = paddle.zeros([H, V_HEAD_DIM, DV], dtype="bfloat16")
        with self.assertRaises(ValueError):
            self.module(query, key, None, None, v_b_proj_weight=w_v)

    def test_v_b_proj_weight_rank_mismatch_rejected(self):
        # A folded 2-D parameter that was never reshaped back must be named as
        # such, not fail later on an unpacking whose message hides the cause.
        query, key, _ = self._args()
        w_v = paddle.zeros([H * V_HEAD_DIM, DV], dtype="bfloat16")
        with self.assertRaisesRegex(ValueError, "must be 3-D"):
            self.module(query, key, None, None, v_b_proj_weight=w_v)

    def test_kv_lora_rank_comes_from_the_hybrid_field(self):
        # The rank is not derivable from ``v_b_proj_weight.shape[0]`` once the
        # grouped-matmul layout is in play, so the layer reads it from the
        # config: the hybrid field when set, the model-wide one otherwise.
        self.assertEqual(self.module.kv_lora_rank, DV)
        config = _create_mqa_config("mqa")
        config.hybrid_mla_kv_lora_rank = None
        config.kv_lora_rank = DV
        self.assertEqual(_build_module(config).kv_lora_rank, DV)

    def test_softmax_scale_is_the_mha_scale(self):
        # Absorption is exactly score preserving, so the scale must stay the MHA
        # q_head_dim one (256**-0.5), never the 576-wide latent one.
        self.assertAlmostEqual(
            self.module.softmax_scale, K_CHANNELS**-0.5, places=12
        )
        self.assertAlmostEqual(self.module.softmax_scale, 0.0625, places=12)


class TestMQAIndexRanges(unittest.TestCase):
    """The forced window and the indexer candidate range partition the causal
    set: no overlap (would double-count) and no gap (would waste budget).

    ``_indexer_valid_range`` now lives on the shared
    ``HybridMLAIndexerMixin``, because both DSA phases build a candidate range
    from it, and its ``window`` argument became a **required positional** one
    placed before ``position_offset`` (the warmup phase passes ``0``). It is
    reached here through the latent class, which is one of its two callers.
    """

    def setUp(self):
        self.module = _build_module(_create_mqa_config("mqa"))

    def test_the_range_builder_is_the_shared_mixins(self):
        """Retargeted, not deleted: the method moved out of the latent class.

        Both DSA phases must clamp the candidate range identically, so a copy
        per class would be a silent divergence risk. Pin the ownership, and pin
        that ``window`` is mandatory -- it used to default to
        ``self.window_size``, so an updated caller that forgets it would
        otherwise keep working while a phase-2 caller silently subtracted a
        128-wide window it does not have.
        """
        self.assertIs(
            MQALatentAttention._indexer_valid_range,
            HybridMLAIndexerMixin._indexer_valid_range,
        )
        self.assertIs(
            MHADSAWarmupAttention._indexer_valid_range,
            HybridMLAIndexerMixin._indexer_valid_range,
        )
        seqlen = 32
        args = self._doc_bounds([seqlen], seqlen)
        with self.assertRaises(TypeError):
            self.module._indexer_valid_range(seqlen, *args)
        # ``window=0`` (what the warmup phase passes) is the whole per-document
        # causal span, diagonal included; the sparse phase's ``WINDOW`` cuts the
        # trailing window off the same range.
        valid_range, row_empty = self.module._indexer_valid_range(
            seqlen, *args, 0
        )
        vr = valid_range.numpy()[0]
        for q in range(seqlen):
            self.assertEqual((int(vr[q, 0]), int(vr[q, 1])), (0, q + 1))
        self.assertFalse(bool(row_empty.numpy().any()))

    @staticmethod
    def _doc_bounds(layout, seqlen):
        """``(doc_start, doc_len, is_valid)`` for one layout."""
        row_end = _row_end(layout, seqlen)
        doc_start, doc_len, is_valid, _, _ = _derive_csa_doc_boundaries(
            row_end, seqlen
        )
        return doc_start, doc_len, is_valid

    def test_window_and_indexer_range_partition_causal_set(self):
        seqlen = 256
        for layout in _LAYOUTS:
            with self.subTest(layout=layout):
                row_end = _row_end(layout, seqlen)
                doc_start, doc_len, is_valid, _, _ = _derive_csa_doc_boundaries(
                    row_end, seqlen
                )
                window = _build_window_topk_idxs_from_doc_bounds(
                    1, seqlen, WINDOW, doc_start, is_valid
                ).numpy()
                valid_range, row_empty = self.module._indexer_valid_range(
                    seqlen, doc_start, doc_len, is_valid, WINDOW
                )
                self._assert_partition(
                    window,
                    valid_range.numpy()[0],
                    row_empty.numpy().reshape([seqlen]),
                    doc_start.numpy(),
                    is_valid.numpy(),
                    seqlen,
                    layout,
                )

    def _assert_partition(
        self, window, vr, row_empty, doc_start, is_valid, seqlen, layout
    ):
        for q in range(seqlen):
            win = {int(c) for c in window[0, q] if c >= 0}
            cand = set(range(int(vr[q, 0]), int(vr[q, 1])))
            tag = f"{layout} row {q}"
            if not is_valid[q]:
                self.assertEqual(win, set(), tag)
                self.assertEqual(cand, set(), tag)
                self.assertTrue(bool(row_empty[q]), tag)
                continue
            start = int(doc_start[q])
            self.assertEqual(
                win, set(range(max(start, q - WINDOW + 1), q + 1)), tag
            )
            self.assertEqual(win & cand, set(), f"{tag}: overlap")
            self.assertEqual(
                win | cand, set(range(start, q + 1)), f"{tag}: incomplete"
            )
            self.assertEqual(bool(row_empty[q]), not cand, tag)


class TestHybridMLAConfig(unittest.TestCase):
    """The hybrid MLA config surface after the ``hybrid_mla_attention`` refactor.

    The old 3-state ``hybrid_mla_attn_mode`` and the ``hybrid_mla_attn_sink``
    switch (with its mutual-exclusion ValueError against the model-wide sinks)
    are gone. There is now a single enum ``hybrid_mla_attention`` (``"mha"`` /
    ``"mqa_dsa"`` / ``"mqa_full_causal"``) and a single sink switch
    (``add_full_attention_sink_bias`` / ``softmax_type``), so the two can no
    longer conflict. Under ``"mqa_dsa"`` the -2 layers run a cuDNN DSA indexer,
    so the config validates the model-wide ``dsa_index_*`` fields
    (index_n_heads / index_head_dim / index_topk).
    """

    @staticmethod
    def _kwargs(**overrides):
        kwargs = {
            "num_hidden_layers": 2,
            "hidden_size": HIDDEN,
            "num_attention_heads": H,
            "experimental_attention_variant": "dsv4_hybrid",
            "csa_compress_ratios": [-2, -2],
            "hybrid_mla_q_lora_rank": Q_LORA,
            "hybrid_mla_kv_lora_rank": DV,
            "hybrid_mla_qk_nope_head_dim": 192,
            "hybrid_mla_qk_rope_head_dim": 64,
            "hybrid_mla_v_head_dim": V_HEAD_DIM,
            "hybrid_mla_num_attention_heads": H,
            "hybrid_mla_num_key_value_heads": H,
        }
        kwargs.update(overrides)
        return kwargs

    @classmethod
    def _mqa_dsa_kwargs(cls, **overrides):
        # ``hybrid_mla_attention="mqa_dsa"`` triggers the DSA-indexer
        # validation, so a valid baseline must carry the model-wide index dims.
        base = {
            "hybrid_mla_attention": "mqa_dsa",
            "dsa_index_n_heads": INDEX_HEADS,
            "dsa_index_head_dim": INDEX_HEAD_DIM,
            "dsa_index_topk": INDEX_TOPK,
        }
        base.update(overrides)
        return cls._kwargs(**base)

    def test_hybrid_mla_attention_defaults_to_mha(self):
        config = TransformerConfig(**self._kwargs())
        self.assertEqual(config.hybrid_mla_attention, "mha")

    def test_mqa_dsa_accepted_with_valid_index_dims(self):
        config = TransformerConfig(**self._mqa_dsa_kwargs())
        self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")
        self.assertEqual(config.dsa_index_head_dim, 128)

    def test_sink_coexists_with_latent_mqa(self):
        # The old mutual-exclusion ValueError is gone: one sink switch only, so
        # enabling a model-wide sink alongside latent MQA must be accepted.
        for sink in (
            {"add_full_attention_sink_bias": True},
            {"softmax_type": "learnable"},
        ):
            with self.subTest(sink=sink):
                config = TransformerConfig(**self._mqa_dsa_kwargs(**sink))
                self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")

    def test_index_dims_are_validated(self):
        # Keyed on (field, value), not on the message: topk=100 (not a multiple
        # of 128) and topk=2176 (>2048) both mention "index_topk" but hit
        # different branches.
        #
        # ``index_topk`` is validated in the sparse phase only, because that is
        # the only phase that reads it -- the warmup KL spans the whole causal
        # set and runs no top-k -- so the topk cases must ask for that phase.
        for field, value, msg in (
            ("dsa_index_head_dim", 64, "index_head_dim"),
            ("dsa_index_topk", 100, "index_topk"),
            ("dsa_index_topk", 2048 + 128, "index_topk"),
            ("dsa_index_n_heads", None, "index_n_heads"),
        ):
            extra = (
                {"dsa_indexer_use_sparse_loss": True}
                if field == "dsa_index_topk"
                else {}
            )
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(ValueError, msg),
            ):
                TransformerConfig(
                    **self._mqa_dsa_kwargs(**{field: value}, **extra)
                )

    def test_warmup_phase_needs_no_index_topk(self):
        """Phase 2 must not be forced to carry a top-k budget.

        ``MHADSAWarmupAttention`` never selects a top-k on either side, so a
        kernel-illegal (or simply absent, hence default) ``index_topk`` must not
        block startup --
        while the sparse phase still rejects it. The production phase-2
        ``model_config.json`` relies on this: it ships no ``index_topk`` at all.
        """
        for topk in (100, 2048 + 128):
            with self.subTest(dsa_index_topk=topk):
                config = TransformerConfig(
                    **self._mqa_dsa_kwargs(
                        dsa_index_topk=topk,
                        dsa_indexer_use_sparse_loss=False,
                    )
                )
                self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")
                self.assertFalse(config.dsa_indexer_use_sparse_loss)
                # ...and the same value is still rejected in the sparse phase,
                # so this is a phase gate, not a dropped check.
                with self.assertRaisesRegex(ValueError, "index_topk"):
                    TransformerConfig(
                        **self._mqa_dsa_kwargs(
                            dsa_index_topk=topk,
                            dsa_indexer_use_sparse_loss=True,
                        )
                    )

    def test_illegal_hybrid_mla_attention_configs_are_rejected(self):
        """The enum makes the old ``(dense=True, mqa=False)`` state
        unrepresentable, so what is left to reject is (a) a value outside the
        enum and (b) a latent-MQA mode on a config that owns no MLA (``-2``)
        layer -- which used to be a silent no-op. Both must be config errors.
        """
        # (a) out-of-enum values, including near-misses and the old bool.
        for value in ("mqa", "MHA", "dense", "", None, True):
            with (
                self.subTest(hybrid_mla_attention=value),
                self.assertRaisesRegex(ValueError, "is invalid"),
            ):
                TransformerConfig(**self._kwargs(hybrid_mla_attention=value))
        # (b) a latent-MQA mode with no -2 layer to apply to. Ratio -1 is CSA
        # full-causal MQA, a different layer kind, so it must not satisfy the
        # check either; nor may a non-dsv4_hybrid variant.
        for mode in ("mqa_dsa", "mqa_full_causal"):
            for ratios, variant in (
                ([128, 128], "dsv4_hybrid"),
                ([-1, -1], "dsv4_hybrid"),
                (None, "dsv4_hybrid"),
                ([-2, -2], None),
            ):
                with (
                    self.subTest(mode=mode, ratios=ratios, variant=variant),
                    self.assertRaisesRegex(
                        ValueError, "only applies to MLA layers"
                    ),
                ):
                    TransformerConfig(
                        **self._kwargs(
                            hybrid_mla_attention=mode,
                            csa_compress_ratios=ratios,
                            experimental_attention_variant=variant,
                            dsa_index_n_heads=INDEX_HEADS,
                            dsa_index_head_dim=INDEX_HEAD_DIM,
                            dsa_index_topk=INDEX_TOPK,
                        )
                    )

    def test_split_kv_b_proj_only_means_anything_for_latent_mqa(self):
        # The switch splits latent MQA's kv_b_proj into standalone k_b_proj /
        # v_b_proj absorption parameters. On the dense MHA path there is nothing
        # to split, so silently accepting it would hide a mis-set config.
        with self.assertRaisesRegex(ValueError, "only means"):
            TransformerConfig(**self._kwargs(mqa_split_kv_b_proj=True))
        # The DSA warmup phase is one of those dense paths: it keeps kv_b_proj,
        # so accepting the flag there would silently change the parameter set
        # at the warmup -> sparse switch instead of at a config edit.
        with self.assertRaisesRegex(ValueError, "only means"):
            TransformerConfig(
                **self._mqa_dsa_kwargs(
                    mqa_split_kv_b_proj=True,
                    dsa_indexer_use_sparse_loss=False,
                )
            )
        for mode, extra in (
            ("mqa_dsa", {"dsa_indexer_use_sparse_loss": True}),
            ("mqa_full_causal", {}),
        ):
            with self.subTest(mode=mode):
                config = TransformerConfig(
                    **self._mqa_dsa_kwargs(
                        hybrid_mla_attention=mode,
                        mqa_split_kv_b_proj=True,
                        **extra,
                    )
                )
                self.assertTrue(config.mqa_split_kv_b_proj)

    def test_split_kv_b_proj_rejects_hy_sparse_attention(self):
        # HySparse swaps the layer class for MQASelfAttention, whose forward and
        # decode paths still absorb against kv_b_proj.weight -- the parameter the
        # split removes. Accepting the combination would fail on a None
        # attribute deep in the forward.
        with self.assertRaisesRegex(ValueError, "enable_hy_sparse_attention"):
            TransformerConfig(
                **self._mqa_dsa_kwargs(
                    hybrid_mla_attention="mqa_dsa",
                    dsa_indexer_use_sparse_loss=True,
                    mqa_split_kv_b_proj=True,
                    enable_hy_sparse_attention=True,
                )
            )

    def test_mqa_full_causal_does_not_require_index_dims(self):
        # No indexer is built, so the index_* validation must be skipped -- these
        # kwargs deliberately omit dsa_index_* and would otherwise be rejected.
        config = TransformerConfig(
            **self._kwargs(hybrid_mla_attention="mqa_full_causal")
        )
        self.assertEqual(config.hybrid_mla_attention, "mqa_full_causal")

    def test_index_dims_unvalidated_when_hybrid_mla_attention_is_mha(self):
        """With the mode left at ``"mha"`` the -2 layers are dense per-head
        attention, so the indexer fields are unused and must not be validated.

        Asserting only that a *default* config builds is near-tautological: the
        defaults are ``None``, which is exactly what
        ``test_index_dims_are_validated`` shows the ``"mqa_dsa"`` path rejects,
        but nothing pins the other three rejections. So feed the exact values
        that test proves are rejected under ``"mqa_dsa"`` -- head_dim != 128,
        topk not a multiple of 128, topk > 2048 -- and assert each one builds
        and survives onto the config unchanged.
        """
        bad = {
            "dsa_index_n_heads": None,
            "dsa_index_head_dim": 64,
            "dsa_index_topk": 100,
        }
        for field, value in [*bad.items(), ("dsa_index_topk", 2048 + 128)]:
            with self.subTest(field=field, value=value):
                config = TransformerConfig(
                    **self._kwargs(hybrid_mla_attention="mha", **{field: value})
                )
                self.assertEqual(config.hybrid_mla_attention, "mha")
                self.assertEqual(getattr(config, field), value)
        # ... and all of them together, still no raise.
        config = TransformerConfig(
            **self._kwargs(hybrid_mla_attention="mha", **bad)
        )
        self.assertEqual(config.hybrid_mla_attention, "mha")

    def test_train_indexer_only_is_pinned_to_the_wide_indexer_loss(self):
        """The two phases are fixed pairs, not four independent modes.

        On the ``-2`` layers ``dsa_indexer_use_sparse_loss`` decides the
        attention candidate set as well as the KL width, so
        ``train_indexer_only=True`` (frozen backbone, warmup) only makes sense
        with the wide loss. The mixed pair is rejected; the other mix (trainable
        backbone + wide loss) is merely unusual and only warns, so it must still
        build.

        Constructed from scratch every time: ``__post_init__`` is not reentrant,
        so mutating an already-normalised config and re-validating would trip
        unrelated ``first_k_dense_replace`` / ``moe_layer_freq`` checks.
        """
        # ``train_indexer_only`` additionally demands a positive loss coeff, so
        # the legal pair carries one; the illegal pair is rejected before that
        # check is even reached (transformer_config.py:1540 vs :1708).
        with self.assertRaisesRegex(ValueError, "is not a valid phase"):
            TransformerConfig(
                **self._mqa_dsa_kwargs(
                    train_indexer_only=True,
                    dsa_indexer_use_sparse_loss=True,
                    dsa_indexer_loss_coeff=0.01,
                )
            )
        for indexer_only, sparse_loss in ((True, False), (False, True)):
            with self.subTest(
                train_indexer_only=indexer_only, sparse_loss=sparse_loss
            ):
                config = TransformerConfig(
                    **self._mqa_dsa_kwargs(
                        train_indexer_only=indexer_only,
                        dsa_indexer_use_sparse_loss=sparse_loss,
                        dsa_indexer_loss_coeff=0.01,
                    )
                )
                self.assertEqual(config.train_indexer_only, indexer_only)
                self.assertEqual(
                    config.dsa_indexer_use_sparse_loss, sparse_loss
                )

    def test_renamed_config_keys_are_rejected_not_absorbed(self):
        """A stale config key must fail loudly instead of turning into a no-op.

        The renames here ship without a compatibility alias (the repo's habit --
        see ``sonicmoe_quant_format``), so the only question is whether a config
        that still carries the old key is *told*. Two paths, two mechanisms:

        * direct construction -- the dataclass ``__init__`` already raises
          ``TypeError`` on an unknown kwarg, so nothing was needed;
        * :meth:`TransformerConfig.from_config` -- ``_process_attribute``'s
          fallback is a bare ``setattr``, so the stale key used to be absorbed
          as a dead attribute. The switch it was meant to flip stayed at its
          default and nothing complained: ``non_absorbed_mqa=True`` silently
          became ``hybrid_mla_attention="mha"``. That is the hole this pins.
        """
        legacy_to_new = {
            "non_absorbed_mqa": "hybrid_mla_attention",
            "non_absorbed_mqa_dense": "hybrid_mla_attention",
            "csa_train_indexer_only": "train_indexer_only",
            "csa_indexer_init_from_scratch": "indexer_init_from_scratch",
        }
        for legacy, replacement in legacy_to_new.items():
            # ``False`` must be rejected too: a stale key is a stale config even
            # when its value happens to agree with the new field's default, and
            # accepting it would leave the writer thinking the key still works.
            for value in (True, False):
                with self.subTest(legacy=legacy, value=value):
                    stale = SimpleNamespace(**self._kwargs(**{legacy: value}))
                    with self.assertRaises(ValueError) as raised:
                        TransformerConfig.from_config(stale)
                    message = str(raised.exception)
                    self.assertIn(f"{legacy} was renamed", message)
                    # The message has to name the replacement, otherwise the
                    # reader has to go read the diff to migrate.
                    self.assertIn(replacement, message)
                    with self.assertRaises(TypeError):
                        TransformerConfig(**self._kwargs(**{legacy: value}))

    def test_from_config_accepts_the_current_key_names(self):
        """Control for the test above: the rejection is keyed on the old names
        only, so the new ones must survive the same ``from_config`` path.
        """
        fresh = SimpleNamespace(
            **self._mqa_dsa_kwargs(
                train_indexer_only=True,
                dsa_indexer_use_sparse_loss=False,
                dsa_indexer_loss_coeff=0.01,
                indexer_init_from_scratch=True,
            )
        )
        config = TransformerConfig.from_config(fresh)
        self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")
        self.assertTrue(config.train_indexer_only)
        self.assertTrue(config.indexer_init_from_scratch)


class TestLatentMqaEnabledPredicate(unittest.TestCase):
    """``latent_mqa_enabled`` (``hybrid_mla_indexer.py:37-61``) decides alone.

    Both the spec dispatch (``gpt_layer_specs.py``) and
    ``MLASelfAttention.mqa_latent`` read this one predicate, so if it drifts the
    spec builds one core attention while the enclosing layer feeds it the other
    one's activations -- absorbed latents into dense MHA, or per-head K/V into
    the block-sparse kernel. Pinned over every combination that reaches it.
    """

    @staticmethod
    def _dsv4(**overrides):
        return TransformerConfig(**TestHybridMLAConfig._kwargs(**overrides))

    def test_non_dsv4_models_never_run_latent_mqa(self):
        config = TransformerConfig(
            num_hidden_layers=2, hidden_size=HIDDEN, num_attention_heads=H
        )
        self.assertEqual(config.hybrid_mla_attention, "mha")
        self.assertIs(latent_mqa_enabled(config), False)
        # The variant gate is checked before the mode, so it holds even for a
        # mode that would otherwise say True. ``__post_init__`` rejects that
        # pair outright (transformer_config.py:1628-1649), hence the
        # post-construction assignment: defence in depth, not a live config.
        config.hybrid_mla_attention = "mqa_full_causal"
        self.assertIs(latent_mqa_enabled(config), False)

    def test_mha_mode_is_dense_even_on_dsv4(self):
        self.assertIs(latent_mqa_enabled(self._dsv4()), False)

    def test_mqa_full_causal_is_latent(self):
        config = self._dsv4(hybrid_mla_attention="mqa_full_causal")
        self.assertIs(latent_mqa_enabled(config), True)
        # No indexer in this mode, so the sparse-loss switch is irrelevant.
        config.dsa_indexer_use_sparse_loss = False
        self.assertIs(latent_mqa_enabled(config), True)

    def test_mqa_dsa_is_latent_only_in_the_sparse_phase(self):
        for sparse_loss in (False, True):
            with self.subTest(dsa_indexer_use_sparse_loss=sparse_loss):
                config = TransformerConfig(
                    **TestHybridMLAConfig._mqa_dsa_kwargs(
                        dsa_indexer_use_sparse_loss=sparse_loss
                    )
                )
                self.assertIs(latent_mqa_enabled(config), sparse_loss)
                # The unit fixture must agree with the production config or
                # every phase-2 test below would exercise the wrong backend.
                self.assertIs(
                    latent_mqa_enabled(
                        _create_mqa_config("mqa_dsa", sparse_loss=sparse_loss)
                    ),
                    sparse_loss,
                )


class TestLatentMqaRefusesTheWarmupPhase(unittest.TestCase):
    """An indexer with the sparse loss off is an error state, not a phase.

    ``MQALatentAttention`` used to *implement* phase 2 (``_forward_warmup``).
    Now that phase runs dense MHA in ``MHADSAWarmupAttention``, so this class
    seeing that combination means the dispatch predicate was bypassed -- which
    must fail loudly rather than quietly build a ``[b, s, s]`` index table and
    feed it to the block-sparse kernel at zero sparsity
    (``mqa_latent_attention.py:279-288``).
    """

    S = 64

    def _assert_message(self, message):
        for fragment in (
            "dsa_indexer_use_sparse_loss=False",
            "DSA warmup phase",
            "dense MHA",
            "MHADSAWarmupAttention",
        ):
            self.assertIn(fragment, message)

    def test_phase_raises_and_names_the_dense_backend(self):
        module = _build_latent_warmup_module()
        self.assertIsNotNone(module.indexer)
        self.assertFalse(module.indexer_use_sparse_loss)
        with self.assertRaises(ValueError) as raised:
            module._phase()
        self._assert_message(str(raised.exception))

    def test_the_forward_refuses_before_touching_a_kernel(self):
        """The guard sits in front of the whole sparse path, not inside it.

        ``_phase`` is consulted after the document bounds and before any index
        table or kernel launch (``mqa_latent_attention.py:372``), so this needs
        no GPU: a well-formed call that would previously have run the warmup
        forward now raises instead.
        """
        module = _build_latent_warmup_module()
        module.eval()
        query = paddle.zeros([1, self.S, H, DK], dtype="float32")
        key = paddle.zeros([1, self.S, 1, DK], dtype="float32")
        w_v = paddle.zeros([DV, H, V_HEAD_DIM], dtype="float32")
        with self.assertRaises(ValueError) as raised:
            module(
                query,
                key,
                None,
                None,
                _row_end([self.S], self.S),
                v_b_proj_weight=w_v,
                x=paddle.zeros([1, self.S, HIDDEN], dtype="float32"),
                qr=paddle.zeros([1, self.S, Q_LORA], dtype="float32"),
            )
        self._assert_message(str(raised.exception))

    def test_the_two_surviving_phases_still_resolve(self):
        # The guard must not have swallowed the legal states: no indexer is
        # full causal, sparse loss on is the sparse phase.
        latent = _build_module(_create_mqa_config("mqa"))
        self.assertIsNone(latent.indexer)
        self.assertEqual(latent._phase(), "full_causal")
        sparse = _build_module(_create_mqa_config("mqa_dsa", loss_coeff=0.01))
        self.assertIsNotNone(sparse.indexer)
        self.assertEqual(sparse._phase(), "sparse")


@_GPU
class TestMQAEquivalence(unittest.TestCase):
    """The indexer-less full-causal path is mathematically identical to MHA."""

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        self.module = _build_module(_create_mqa_config("mqa"))

    def _run(self, seqlen, layout):
        query, key, w_v = _make_inputs(seqlen)
        row_end = None if layout is None else _row_end(layout, seqlen)
        out = self.module(query, key, None, None, row_end, v_b_proj_weight=w_v)
        ref = _dense_reference(
            query,
            key,
            w_v,
            _row_end([seqlen], seqlen) if row_end is None else row_end,
            self.module.softmax_scale,
        )
        return out, ref

    def test_single_document_matches_dense(self):
        out, ref = self._run(256, None)
        # W7 measured _rel 2.20e-3..2.63e-3 over seeds 0-4 (bf16, cudnn
        # deterministic); tightened 5e-3 -> 3.5e-3 (1.3x headroom over worst).
        self.assertLess(_rel(out, ref), 3.5e-3)

    def test_multi_document_layouts_match_dense(self):
        seqlen = 256
        for layout in _LAYOUTS:
            with self.subTest(layout=layout):
                _CAPTURED.clear()
                out, ref = self._run(seqlen, layout)
                # W7 measured max _rel 2.63e-3 over 10 layouts x seeds 0-4;
                # tightened 5e-3 -> 3.5e-3.
                self.assertLess(_rel(out, ref), 3.5e-3)
                _check_index_invariants(
                    self,
                    _CAPTURED[-1],
                    _row_end(layout, seqlen),
                    seqlen,
                    expect_full=True,
                )

    def test_attention_sink_matches_dense_reference(self):
        """The learnable sink is one extra value-less softmax column."""
        seqlen, layout = 256, [40, 88, 128]
        sink = np.linspace(1.0, 3.0, H)
        module = _build_module(_create_mqa_config("mqa"), sink=sink)
        self.assertEqual(module.softmax_offset.dtype, paddle.float32)
        self.assertEqual(list(module.softmax_offset.shape), [H])
        query, key, w_v = _make_inputs(seqlen)
        row_end = _row_end(layout, seqlen)
        out = module(query, key, None, None, row_end, v_b_proj_weight=w_v)
        sink_used = module.softmax_offset.astype("float32").numpy()
        ref = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale, sink=sink_used
        )
        # W7 measured _rel 2.74e-3 over seeds 0-2 (dense + sink); 5e-3 -> 3.5e-3.
        self.assertLess(_rel(out, ref), 3.5e-3)
        # ... and the sink genuinely changed the result: a positive logit drains
        # probability mass, so the sinkless reference is far away.
        sinkless = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale
        )
        # W7 measured sinkless _rel ~0.985-1.02 (the sink dominates this layout);
        # lower-bound tightened 5e-2 -> 0.5 (2x headroom below worst observed).
        self.assertGreater(_rel(sinkless, ref), 0.5)


@_GPU
class TestMQADSA(unittest.TestCase):
    """The DSA (indexer) path: forced window + Lightning-indexer top-k."""

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01), bf16=True
        )

    @staticmethod
    def _inputs(seqlen, seed=0):
        return _make_inputs(seqlen, seed=seed, with_hidden=True)

    def _forward(self, seqlen, layout, seed=0):
        """Inference-mode forward, for checking numerics and index invariants.

        ``eval()`` skips the indexer KL loss, which is irrelevant here and
        cannot be attached to an all-detached graph: the autoscaler PyLayer
        returns its ``output`` argument unchanged, and Paddle rejects that
        "inplace" return for a leaf tensor. Real training always feeds a
        differentiable ``query``; the loss path is covered by the backward and
        recompute tests below.
        """
        query, key, w_v, x, qr = self._inputs(seqlen, seed=seed)
        self.module.eval()
        row_end = _row_end(layout, seqlen)
        out = self.module(
            query,
            key,
            None,
            None,
            row_end,
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        return out, (query, key, w_v, x, qr), row_end

    def test_saturated_budget_reproduces_dense(self):
        # window(128) + topk(128) covers every row's causal length at s=256, so
        # the selected set is exactly the full causal set and DSA must be exact.
        seqlen = WINDOW + INDEX_TOPK
        for layout in ([seqlen], [40, 88, 128], [3, WINDOW, WINDOW + 1, 1]):
            with self.subTest(layout=layout):
                _CAPTURED.clear()
                out, tensors, row_end = self._forward(seqlen, layout)
                query, key, w_v = tensors[0], tensors[1], tensors[2]
                ref = _dense_reference(
                    query, key, w_v, row_end, self.module.softmax_scale
                )
                # W7 measured max _rel 2.62e-3 over 3 layouts x seeds 0-2
                # (DSA saturated budget); 5e-3 -> 3.5e-3.
                self.assertLess(_rel(out, ref), 3.5e-3)
                self.assertEqual(_CAPTURED[-1].shape[-1], WINDOW + INDEX_TOPK)
                _check_index_invariants(
                    self, _CAPTURED[-1], row_end, seqlen, expect_full=True
                )

    def test_sparse_budget_indices_are_sound(self):
        seqlen = 512  # window + topk = 256 < 512 => genuinely sparse
        _CAPTURED.clear()
        out, _, row_end = self._forward(seqlen, [200, 312])
        self.assertTrue(bool(paddle.isfinite(out.cast("float32")).all()))
        self.assertEqual(_CAPTURED[-1].shape[-1], WINDOW + INDEX_TOPK)
        _check_index_invariants(self, _CAPTURED[-1], row_end, seqlen)

    def test_backward_produces_finite_grads_and_reports_loss(self):
        seqlen = WINDOW + INDEX_TOPK
        query, key, w_v, x, qr = self._inputs(seqlen)
        for tensor in (query, key, x, qr):
            tensor.stop_gradient = False
        self.module.train()
        out = self.module(
            query,
            key,
            None,
            None,
            _row_end([seqlen], seqlen),
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        out.cast("float32").sum().backward()
        for name, tensor in (("query", query), ("key", key)):
            self.assertIsNotNone(tensor.grad, f"{name} has no gradient")
            self.assertTrue(
                bool(paddle.isfinite(tensor.grad.cast("float32")).all()),
                f"{name} gradient is not finite",
            )
        # The indexer inputs are deliberately detached from the backbone (same
        # contract as DSAttention/CSA): the indexer learns from its own KL loss
        # only, so ``x``/``qr`` must stay gradient-free while the indexer
        # projections still receive gradients.
        self.assertIsNone(x.grad)
        self.assertIsNone(qr.grad)
        indexer_params = {
            "wq_b": self.module.indexer.wq_b.linear.weight,
            "wk": self.module.indexer.wk.linear.weight,
            "weights_proj": self.module.indexer.weights_proj.linear.weight,
        }
        for name, param in indexer_params.items():
            self.assertIsNotNone(param.grad, f"indexer.{name} has no gradient")
            self.assertTrue(
                bool(paddle.isfinite(param.grad.cast("float32")).all()),
                f"indexer.{name} gradient is not finite",
            )
            self.assertGreater(
                float(param.grad.cast("float32").abs().max()),
                0.0,
                f"indexer.{name} gradient is all zero",
            )
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

    def test_indexer_loss_mask_comes_from_input_ids(self):
        """Padding rows must not dilute the KL loss, and only ``input_ids`` sees them.

        Same masked reduction and the same mask construction as
        ``csa_attention.py:1302-1306`` / ``:2411-2443``. The document metadata
        cannot stand in for it: ``attn_mask_startend_row_indices`` has no way to
        say "these trailing rows are padding", so the tail here is folded into a
        second document and every row reports ``is_valid == True``. Only
        ``input_ids != pad_token_id`` separates them.

        Two runs share the same 384-token document and differ only in how much
        trailing padding follows it. With the mask both must report the same KL;
        without it (a plain ``mean()``, or a mask derived from ``is_valid``) the
        pad rows enter both numerator and denominator and the two diverge.
        """
        doc = 384  # > WINDOW, so later rows have a non-empty top-k set
        query, key, w_v, x, qr = self._inputs(512, seed=3)

        def run(seqlen):
            DSAIndexerLossLoggingHelper.tracker.clear()
            tensors = [t[:, :seqlen].clone() for t in (query, key, x, qr)]
            for tensor in tensors:
                tensor.stop_gradient = False
            # 1 = real token, 0 = pad (``config.pad_token_id`` defaults to 0).
            input_ids = paddle.concat(
                [
                    paddle.ones([1, doc], dtype="int64"),
                    paddle.zeros([1, seqlen - doc], dtype="int64"),
                ],
                axis=-1,
            )
            self.module.train()
            out = self.module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([doc, seqlen - doc], seqlen),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
                input_ids=input_ids,
            )
            out.cast("float32").sum().backward()
            return float(DSAIndexerLossLoggingHelper.tracker["values"][0])

        loss_448, loss_512 = run(448), run(512)
        self.assertGreater(loss_448, 0.0)
        # W7 measured loss_512/loss_448 - 1 == 0.0 exactly over seeds
        # 0/3/4/7/11 (two-doc layout is bit-reproducible); delta 2e-3 -> 5e-4.
        self.assertAlmostEqual(loss_512 / loss_448, 1.0, delta=5e-4)

    def test_use_sparse_loss_switches_both_attention_and_loss_width(self):
        """``dsa_indexer_use_sparse_loss`` picks the whole training phase.

        On these uncompressed ``-2`` layers the switch is one decision with two
        effects (``MQALatentAttention._phase``), not just the KL width it picks
        for the CSA layers of the same model
        (``_resolve_csa_indexer_loss_topk_effective``):

        * ``True`` -- phase 3 (``_forward_sparse``). Attention consumes
          ``window + index_topk`` and the KL is restricted to that same set, so
          ``_attn_target`` is called once per step at exactly ``index_topk``.
        * ``False`` -- phase 2, which is no longer this class's phase: it runs
          dense MHA in ``MHADSAWarmupAttention`` (see
          ``TestMQADSAWarmupPhase``). Flipping the switch on a live latent
          module therefore *raises* instead of widening the attention table to
          the full per-document causal set. That inversion is the point: the
          ``[b, s, s]`` table the old ``False`` branch built for a zero-sparsity
          kernel does not exist any more.

        The ``True`` path stays statistical, which is the pre-existing measured
        fact this test still records: on a single full-length document neither
        the output bits nor the index table are reproducible across identical
        calls -- 2.4-2.8% of the table's slots move between two *identical*
        eval-mode calls, and ~1e-4 on the output. (Splitting the same 512 rows
        into two documents is exactly reproducible, which is why
        ``test_recompute_double_forward_is_consistent`` can assert equality --
        it uses ``[200, 312]``.)
        """
        seqlen = 512
        query, key, w_v, x, qr = self._inputs(seqlen, seed=5)
        loss_widths = []
        inner_target = self.module._attn_target

        def recording_target(query_, kv_, kl_columns, lse_indexer=None):
            # The KL's column set is the indexer's candidate set, i.e. the
            # top-k. The forced window is never in it.
            loss_widths.append(int(kl_columns.shape[-1]))
            return inner_target(query_, kv_, kl_columns, lse_indexer)

        self.module._attn_target = recording_target

        def run():
            _CAPTURED.clear()
            DSAIndexerLossLoggingHelper.tracker.clear()
            tensors = [t.clone() for t in (query, key, x, qr)]
            for tensor in tensors:
                tensor.stop_gradient = False
            self.module.train()
            out = self.module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([seqlen], seqlen),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
            )
            out.cast("float32").sum().backward()
            return (
                _CAPTURED[-1].copy(),
                float(DSAIndexerLossLoggingHelper.tracker["values"][0]),
            )

        idx_a, loss_sparse = run()
        idx_b, _ = run()

        # The KL column set is exactly ``index_topk`` wide...
        self.assertEqual(loss_widths, [INDEX_TOPK, INDEX_TOPK])
        # ...and never covers the forced window, i.e. it is the indexer's own
        # candidate budget, not ``WINDOW + INDEX_TOPK``.
        self.assertNotIn(WINDOW + INDEX_TOPK, loss_widths)

        # The attention table is window + top-k.
        for table in (idx_a, idx_b):
            self.assertEqual(int(table.shape[-1]), WINDOW + INDEX_TOPK)

        # The measured identical-call drift of the table, kept as the reason its
        # width -- not its contents -- is what gets asserted.
        drift = float((idx_a != idx_b).mean())
        self.assertLess(drift, 0.05)
        self.assertGreater(loss_sparse, 0.0)

        # The other half of the switch: no wider table, a refusal. ``_phase`` is
        # read live, so the flip takes effect on this same module.
        self.module.indexer_use_sparse_loss = False
        _CAPTURED.clear()
        with self.assertRaisesRegex(ValueError, "MHADSAWarmupAttention"):
            run()
        self.assertEqual(_CAPTURED, [], "an index table was built anyway")
        self.assertEqual(loss_widths, [INDEX_TOPK, INDEX_TOPK])
        self.module.indexer_use_sparse_loss = True

    def test_recompute_double_forward_is_consistent(self):
        """Reentrant recompute runs the layer twice: pass 1 under ``no_grad``.

        The top-k must be deterministic across the two passes (otherwise the
        backward would differentiate a different sparsity pattern), and the
        indexer loss must be attached on the grad-enabled pass only.
        """
        seqlen = 512
        query, key, w_v, x, qr = self._inputs(seqlen)
        query.stop_gradient = False
        self.module.train()
        row_end = _row_end([200, 312], seqlen)
        kwargs = {"v_b_proj_weight": w_v, "x": x, "qr": qr}

        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        with paddle.no_grad():
            self.module(query, key, None, None, row_end, **kwargs)
        first = _CAPTURED[-1]
        self.assertNotIn(
            "values",
            DSAIndexerLossLoggingHelper.tracker,
            "indexer loss must not be attached on the no_grad pass",
        )

        self.module(query, key, None, None, row_end, **kwargs)
        second = _CAPTURED[-1]
        np.testing.assert_array_equal(first, second)
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

    def test_sink_saturated_budget_matches_dense(self):
        """DSA + sink: the finite-sink LSE correction must be exact.

        ``d_qk``(576) != ``d_v``(512) here, so the kernel takes its finite-sink
        correction path; a saturated budget makes the selected set the full
        causal set, hence the dense sink reference applies exactly.
        """
        seqlen = WINDOW + INDEX_TOPK
        sink = np.linspace(1.0, 3.0, H)
        module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01),
            bf16=True,
            sink=sink,
        )
        module.eval()
        query, key, w_v, x, qr = self._inputs(seqlen)
        row_end = _row_end([40, 88, 128], seqlen)
        out = module(
            query,
            key,
            None,
            None,
            row_end,
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        sink_used = module.softmax_offset.astype("float32").numpy()
        ref = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale, sink=sink_used
        )
        # W7 measured _rel 2.75e-3 over seeds 0-1 (DSA + finite-sink LSE
        # correction, saturated budget); 5e-3 -> 3.5e-3.
        self.assertLess(_rel(out, ref), 3.5e-3)
        sinkless = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale
        )
        # W7 measured sinkless _rel ~0.985; lower-bound 5e-2 -> 0.5.
        self.assertGreater(_rel(sinkless, ref), 0.5)

    def test_sink_receives_finite_nonzero_fp32_gradient(self):
        """The sink gradient is computed analytically (the kernel returns 0)."""
        seqlen = WINDOW + INDEX_TOPK
        module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01),
            bf16=True,
            sink=np.linspace(1.0, 3.0, H),
        )
        module.train()
        query, key, w_v, x, qr = self._inputs(seqlen)
        for tensor in (query, key, x, qr):
            tensor.stop_gradient = False
        out = module(
            query,
            key,
            None,
            None,
            _row_end([seqlen], seqlen),
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        out.cast("float32").sum().backward()
        grad = module.softmax_offset.grad
        self.assertIsNotNone(grad, "attention sink has no gradient")
        # The gradient must come back in the parameter's dtype (bf16 here, the
        # production ``params_dtype``); an fp32 grad on a bf16 parameter would
        # be a dtype mismatch on accumulation.
        self.assertEqual(grad.dtype, module.softmax_offset.dtype)
        self.assertEqual(list(grad.shape), [H])
        self.assertTrue(
            bool(paddle.isfinite(grad.astype("float32")).all()),
            "sink gradient is not finite",
        )
        self.assertGreater(
            float(grad.astype("float32").abs().max()),
            0.0,
            "sink gradient is all zero",
        )
        # The backbone must keep flowing with the sink enabled.
        self.assertIsNotNone(query.grad)
        self.assertTrue(bool(paddle.isfinite(query.grad.cast("float32")).all()))


@_GPU
class TestMQADSAWarmupPhase(unittest.TestCase):
    """Phase 2 of ``"mqa_dsa"``: ``dsa_indexer_use_sparse_loss=False``.

    The indexer is still being learned, so attention must not consume its
    ranking -- and with no top-k on either side there is nothing for absorbed
    latent MQA to save. So this phase runs **phase 1's dense MHA** with the
    indexer bolted on (``mha_dsa_warmup_attention.MHADSAWarmupAttention``),
    which ``latent_mqa_enabled`` selects (``TestLatentMqaEnabledPredicate``)
    and which ``MQALatentAttention`` now refuses to impersonate
    (``TestLatentMqaRefusesTheWarmupPhase``). ``TestMQADSA`` covers phase 3,
    where attention does consume ``window + index_topk``.

    Kept in this file although the backend moved: what these tests are about is
    the phase boundary, and the phase-3 fixtures they contrast with live here.
    Two class-wide inversions of the old assertions, both consequences of the
    dense backend:

    * ``_CAPTURED`` (the block-sparse kernel's ``token_indices``) must stay
      **empty** -- no ``[b, s, s]`` index table is built at all, where the old
      warmup built one and then walked all ``s`` columns at zero sparsity;
    * the reference is ``_build_phase1_dense_module``, a plain
      ``DotProductAttention``, and agreement with it is **bit** equality.
    """

    SEQLEN = 256
    # Two documents, the second longer than the forced window, so the indexer's
    # candidate range is non-empty on the late rows yet still excludes them.
    LAYOUT = [40, 216]

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        _WARMUP_TARGETS.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.config = _create_mqa_config(
            "mqa_dsa", loss_coeff=0.01, sparse_loss=False
        )
        self.module = self._build_warmup(self.config)
        self.row_end = _row_end(self.LAYOUT, self.SEQLEN)

    @staticmethod
    def _build_warmup(config):
        """The dense phase-2 backend, picked by the production predicate."""
        module = _build_module(config, bf16=True)
        assert isinstance(module, MHADSAWarmupAttention), type(module)
        assert not isinstance(module, MQALatentAttention)
        assert module.indexer is not None
        return module

    def _inputs(self, seed=0):
        return _make_dense_inputs(self.SEQLEN, seed=seed)

    def _ids(self):
        """All-valid ``input_ids``; ``pad_token_id`` defaults to 0."""
        return paddle.ones([1, self.SEQLEN], dtype="int64")

    def _call(
        self, module, tensors, training, differentiable=False, input_ids=None
    ):
        """The phase-2 call shape: per-head q/k/v plus the indexer's inputs.

        ``attn_mask_type`` is explicit because ``DotProductAttention`` leaves
        ``is_causal`` False when it is omitted, and the document mask alone
        would then leave the upper triangle unmasked. ``input_ids`` is the only
        kwarg the phase-1 reference does not take, so it is passed only when
        given -- which is how ``MLASelfAttention`` forwards it too, gated on
        ``accepts_input_ids``.
        """
        module.train() if training else module.eval()
        query, key, value, x, qr = tensors
        if differentiable:
            for tensor in tensors:
                tensor.stop_gradient = False
        extra = {} if input_ids is None else {"input_ids": input_ids}
        return module(
            query,
            key,
            value,
            None,
            self.row_end,
            attn_mask_type=AttnMaskType.causal,
            x=x,
            qr=qr,
            **extra,
        )

    def test_attention_output_is_bit_identical_to_phase_1(self):
        """The core invariant, inverted: phase 2 *is* phase 1's attention.

        The old assertion compared against the indexer-less **latent** path.
        The reference is now the dense ``DotProductAttention`` of phase 1
        itself, and the claim is stronger: the whole attention half is
        ``super().forward`` (``mha_dsa_warmup_attention.py:179-198``), so the
        indexer loss is the only new thing. Nothing needs weight copying --
        attention consumes no module parameter here (q/k/v are inputs and
        ``softmax_offset`` is ``None`` in both) -- so a difference could only
        come from the backend. Measured maxabs 0.0 on SM103 / FA4.

        Asserted in eval and in train mode, i.e. with the indexer branch both
        skipped (``mha_dsa_warmup_attention.py:199-200``) and taken.
        """
        tensors = self._inputs()
        reference = _build_phase1_dense_module(self.config, bf16=True)
        self.assertIsNone(reference.softmax_offset)
        self.assertIsNone(self.module.softmax_offset)
        self.assertEqual(self.module.softmax_scale, reference.softmax_scale)
        out_ref = _fp32(self._call(reference, tensors, training=False))
        for training in (False, True):
            with self.subTest(training=training):
                DSAIndexerLossLoggingHelper.tracker.clear()
                clones = [t.clone() for t in tensors]
                out = self._call(
                    self.module,
                    clones,
                    training=training,
                    differentiable=training,
                    input_ids=self._ids(),
                )
                np.testing.assert_array_equal(_fp32(out), out_ref)
        self.assertEqual(_CAPTURED, [], "the block-sparse kernel was reached")

    def test_output_matches_the_fp32_dense_mha_reference(self):
        """Independent check that the delegated half really is per-document
        causal MHA, rather than merely equal to another copy of itself.

        The document layouts are varied here rather than in the bit-identity
        test because masking is what this one is about -- and no layout may
        produce an index table.
        """
        tensors = self._inputs(seed=3)
        query, key, value = tensors[0], tensors[1], tensors[2]
        for layout in ([self.SEQLEN], self.LAYOUT, [3, WINDOW, WINDOW + 1, 1]):
            with self.subTest(layout=layout):
                self.row_end = _row_end(layout, self.SEQLEN)
                _CAPTURED.clear()
                out = self._call(self.module, tensors, training=False)
                ref = _dense_mha_reference(
                    query,
                    key,
                    value,
                    self.row_end,
                    self.module.softmax_scale,
                )
                # bf16 flashmask vs the fp32 reference: measured rel 1.969e-3
                # at [100, 156]; 3.5e-3 keeps the phase-3 tests' margin.
                self.assertLess(_rel(out, ref), 3.5e-3)
                self.assertEqual(_CAPTURED, [], "an index table was built")

    def test_warmup_undoes_the_indexer_weight_prebake_for_tilelang(self):
        """The tilelang indexer re-applies ``head_dim**-0.5``, so the pre-bake
        must be undone -- exactly as the cuDNN pair needs in phase 3.

        This is a regression test for a bug the test suite was blind to: warmup
        first shipped passing ``weights`` through unscaled, on the (wrong)
        reasoning that the ``head_dim**0.5`` fixup was cuDNN-specific. Nothing
        crashed and every existing assertion still passed -- the indexer was just
        trained against a distribution flattened by ``1/sqrt(128)``. The precision
        audit caught it by comparing against a plain-paddle reference:
        un-baked weights match to ``max|d|=3.0e-8 / cosine 1-1.5e-13``, unscaled
        ones are off by ``max|d|=7.5e-1 / cosine 0.62``.

        So the discriminator has to be numeric. ``probs`` from the kernel is
        compared against the reference expression evaluated with ``weights`` **as
        ``forward_before_topk`` returns them**, which is the intended scale.
        """
        import paddle.nn.functional as F

        import paddlefleet.tilelang_ops as tl_mod

        seen = {}
        inner_tl = tl_mod.csa_indexer_topk_fwd
        inner_proj = self.module._indexer_projections

        def recording_proj(*args, **kwargs):
            q, k, w = inner_proj(*args, **kwargs)
            seen["w_as_returned"] = w.detach().cast("float32").numpy().copy()
            seen["q"] = q.detach().cast("float32").numpy().copy()
            seen["k"] = k.detach().cast("float32").numpy().copy()
            return q, k, w

        def recording_tl(*args, **kwargs):
            seen["w_passed"] = args[2].detach().cast("float32").numpy().copy()
            columns, probs = inner_tl(*args, **kwargs)
            seen["columns"] = columns.numpy().copy()
            seen["probs"] = probs.cast("float32").numpy().copy()
            return columns, probs

        tensors = [t.clone() for t in self._inputs()]
        self.module._indexer_projections = recording_proj
        tl_mod.csa_indexer_topk_fwd = recording_tl
        try:
            self._call(
                self.module,
                tensors,
                training=True,
                differentiable=True,
                input_ids=self._ids(),
            )
        finally:
            self.module._indexer_projections = inner_proj
            tl_mod.csa_indexer_topk_fwd = inner_tl

        # The pre-bake is undone exactly once. ``weights`` is bf16, so the
        # product carries bf16 rounding (~2.6e-3 relative measured); the factor
        # being separated here is sqrt(128) ~ 11.3 against 1.0, so a loose
        # tolerance still discriminates it by three orders of magnitude.
        root_d = float(self.module.indexer.head_dim) ** 0.5
        np.testing.assert_allclose(
            seen["w_passed"],
            seen["w_as_returned"] * root_d,
            rtol=5e-3,
            atol=1e-6,
        )

        # ...and that is the scale which reproduces the reference distribution.
        q = paddle.to_tensor(seen["q"])
        k = paddle.to_tensor(seen["k"])
        w = paddle.to_tensor(seen["w_as_returned"])
        scores = paddle.einsum("bshd,btd->bsht", q, k)
        logits = (F.relu(scores) * w.unsqueeze(-1)).sum(axis=2)
        rows = paddle.arange(self.SEQLEN, dtype="int64").unsqueeze(-1)
        cols = paddle.arange(self.SEQLEN, dtype="int64").unsqueeze(0)
        doc_start = []
        start = 0
        for length in self.LAYOUT:
            doc_start += [start] * length
            start += length
        doc_start = paddle.to_tensor(doc_start, dtype="int64")
        keep = (cols <= rows) & (cols >= doc_start.unsqueeze(-1))
        logits = logits + paddle.where(
            keep.unsqueeze(0),
            paddle.zeros([1, self.SEQLEN, self.SEQLEN], dtype="float32"),
            paddle.full([1, self.SEQLEN, self.SEQLEN], -1e30, dtype="float32"),
        )
        ref = F.softmax(logits, axis=-1).numpy()

        # The kernel emits columns in score order, so gather the reference onto
        # the same permutation before comparing.
        cols_seen = seen["columns"][0]
        got = seen["probs"][0]
        valid = cols_seen >= 0
        safe = np.where(valid, cols_seen, 0)
        ref_perm = np.take_along_axis(ref[0], safe, axis=-1)
        ref_perm = np.where(valid, ref_perm, 0.0)
        max_abs = float(np.abs(got - ref_perm).max())
        # bf16 end to end (production dtype), so the reference rebuilt from the
        # rounded weights lands ~2.3e-5 away; the audit's fp32 run gets 3.0e-8.
        # The wrong scale is 7.5e-1, i.e. this threshold still discriminates by
        # nearly three orders of magnitude.
        self.assertLess(
            max_abs,
            1e-3,
            f"warmup probs do not match the reference scale: max|d|={max_abs:.3e}",
        )

    def test_warmup_scores_every_causal_column_via_tilelang(self):
        """Phase 2 scores the whole causal span, in one tilelang call.

        Three things are pinned. First, the **cuDNN** top-k kernel -- phase 3's
        selector -- is called zero times: this phase reads no ``index_topk``, no
        window and no clamped candidate range. Second, the tilelang indexer is
        called exactly once at ``topk_effective == s``, its documented
        "full-candidate selection" mode. Third, the columns it comes back with
        are exactly the per-document causal set, diagonal included -- the very
        column the old clamped candidate range could never return.

        The causal set is now only *analytic* (``_full_causal_table``): with the
        dense backend there is no attention table to compare against, which is
        itself asserted -- ``_CAPTURED`` stays empty while the KL still spans
        every causal column, i.e. the width came without the ``[b, s, s]``
        transient.

        Before the phase split this test demanded one *cuDNN* call for a widened
        loss table. That widening was the bug: at the production
        ``index_topk=2048`` it capped back to the phase-3 width, so the KL scored
        the same columns attention would have picked.
        """
        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod
        import paddlefleet.tilelang_ops as tl_mod

        cudnn_calls = []
        tl_widths = []
        tl_columns = []
        inner_cudnn = fwd_mod.cudnn_indexer_topk_fwd
        inner_tl = tl_mod.csa_indexer_topk_fwd

        def recording_cudnn(*args, **kwargs):
            cudnn_calls.append(int(kwargs["topk_effective"]))
            return inner_cudnn(*args, **kwargs)

        def recording_tl(*args, **kwargs):
            tl_widths.append(int(kwargs["topk_effective"]))
            columns, probs = inner_tl(*args, **kwargs)
            tl_columns.append(columns.numpy().copy())
            return columns, probs

        tensors = [t.clone() for t in self._inputs()]
        fwd_mod.cudnn_indexer_topk_fwd = recording_cudnn
        tl_mod.csa_indexer_topk_fwd = recording_tl
        try:
            out = self._call(
                self.module,
                tensors,
                training=True,
                differentiable=True,
                input_ids=self._ids(),
            )
            out.cast("float32").sum().backward()
        finally:
            fwd_mod.cudnn_indexer_topk_fwd = inner_cudnn
            tl_mod.csa_indexer_topk_fwd = inner_tl

        self.assertEqual(
            cudnn_calls, [], "warmup called the cuDNN indexer top-k kernel"
        )
        self.assertEqual(tl_widths, [self.SEQLEN])
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)
        self.assertEqual(_CAPTURED, [], "the block-sparse kernel was reached")

        causal = _full_causal_table(self.LAYOUT, self.SEQLEN)
        kl_columns = tl_columns[0]
        for row in range(self.SEQLEN):
            causal_cols = causal[0, row]
            kl_cols = kl_columns[0, row]
            self.assertEqual(
                set(kl_cols[kl_cols >= 0].tolist()),
                set(causal_cols[causal_cols >= 0].tolist()),
                f"row {row}: KL and causal column sets differ",
            )
        last = self.SEQLEN - 1
        self.assertIn(last, set(kl_columns[0, last].tolist()))
        # The KL target is built over that same width, once.
        self.assertEqual(
            [t.shape for t in _WARMUP_TARGETS],
            [(1, self.SEQLEN, self.SEQLEN)],
        )

    def test_eval_early_exit_matches_the_training_forward(self):
        """The no-loss forward skips the indexer entirely.

        ``_needs_indexer_loss`` gates the whole second half
        (``mha_dsa_warmup_attention.py:199-200``), so with nothing to learn this
        step there is nothing to compute -- and because the attention half is
        the same ``super().forward`` either way, the output must be
        bit-identical to the training forward, which does run the indexer.
        """
        tensors = self._inputs(seed=2)
        calls = []
        inner = self.module.indexer.forward_before_topk

        def recording(*args, **kwargs):
            calls.append(len(calls))
            return inner(*args, **kwargs)

        self.module.indexer.forward_before_topk = recording

        out_train = _fp32(
            self._call(
                self.module,
                [t.clone() for t in tensors],
                training=True,
                differentiable=True,
                input_ids=self._ids(),
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)
        self.assertEqual(len(_WARMUP_TARGETS), 1)

        DSAIndexerLossLoggingHelper.tracker.clear()
        out_eval = _fp32(
            self._call(
                self.module, tensors, training=False, input_ids=self._ids()
            )
        )
        self.assertEqual(len(calls), 1, "eval must not run the indexer at all")
        self.assertEqual(len(_WARMUP_TARGETS), 1, "eval built a KL target")
        self.assertNotIn("values", DSAIndexerLossLoggingHelper.tracker)
        np.testing.assert_array_equal(out_train, out_eval)

    def test_indexer_gradients_flow_in_the_warmup_phase(self):
        """Every indexer parameter keeps a finite non-zero gradient.

        Phase 2 is where the indexer does all of its learning (the backbone is
        frozen by the trainer), so a silently gradient-free indexer parameter
        would waste the entire phase. Same contract as
        ``TestMQADSA.test_backward_produces_finite_grads_and_reports_loss``,
        widened to the whole parameter set and driven through the dense backend,
        whose attention half does not touch the indexer at all -- the gradients
        can only come from the KL attached to the output
        (``mha_dsa_warmup_attention.py:342-354``). Measured range on this
        fixture: 9.3e-8 .. 2.2e-7 over the 8 parameters.
        """
        tensors = self._inputs()
        query, x, qr = tensors[0], tensors[3], tensors[4]
        out = self._call(
            self.module,
            tensors,
            training=True,
            differentiable=True,
            input_ids=self._ids(),
        )
        out.cast("float32").sum().backward()
        indexer = self.module.indexer
        # Every parameter, not a hand-listed subset: the measured set is 8
        # (weight+bias of wq_b / wk / k_norm / weights_proj), and a new one
        # appearing gradient-free would otherwise go unnoticed.
        named = dict(indexer.named_parameters())
        self.assertGreaterEqual(len(named), 5)
        for expected in ("wq_b", "wk", "k_norm", "weights_proj"):
            self.assertTrue(
                any(expected in name for name in named),
                f"indexer has no {expected} parameter any more",
            )
        for name, param in named.items():
            self.assertIsNotNone(param.grad, f"indexer.{name} has no gradient")
            self.assertTrue(
                bool(paddle.isfinite(param.grad.cast("float32")).all()),
                f"indexer.{name} gradient is not finite",
            )
            self.assertGreater(
                float(param.grad.cast("float32").abs().max()),
                0.0,
                f"indexer.{name} gradient is all zero",
            )
        # Unchanged contract: the indexer learns from its own KL only, so its
        # inputs stay detached while the backbone query/key still flow.
        self.assertIsNone(x.grad)
        self.assertIsNone(qr.grad)
        self.assertIsNotNone(query.grad)
        self.assertTrue(bool(paddle.isfinite(query.grad.cast("float32")).all()))
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

    def test_recompute_double_forward_attaches_the_loss_once(self):
        """Reentrant recompute runs the layer twice: pass 1 under ``no_grad``.

        The loss must be attached on the grad-enabled pass only -- otherwise it
        would be counted twice -- and, since the attention half is dense
        flashmask on a fixed mask, the two passes must produce the same output
        bit for bit. The old form of this test asserted the same thing about the
        index table; there is none any more, so the KL target takes its place:
        it is built once, on the differentiable pass.

        Single document on purpose: the phase-3 equivalent
        (``TestMQADSA.test_recompute_double_forward_is_consistent``) has to
        avoid that layout because the top-k kernel's emitted order drifts on it.
        Phase 2 has no top-k, so the hard layout is available.
        """
        self.row_end = _row_end([self.SEQLEN], self.SEQLEN)
        tensors = self._inputs()
        tensors[0].stop_gradient = False

        with paddle.no_grad():
            first = _fp32(
                self._call(
                    self.module,
                    tensors,
                    training=True,
                    input_ids=self._ids(),
                )
            )
        self.assertEqual(_WARMUP_TARGETS, [])
        self.assertNotIn(
            "values",
            DSAIndexerLossLoggingHelper.tracker,
            "indexer loss must not be attached on the no_grad pass",
        )

        second = _fp32(
            self._call(
                self.module, tensors, training=True, input_ids=self._ids()
            )
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(_WARMUP_TARGETS), 1)
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)
        self.assertEqual(_CAPTURED, [], "the block-sparse kernel was reached")


class TestHashableTensor(unittest.TestCase):
    """``_HashableTensor`` exists only to make the kernel cache key hashable.

    The cuDNN score-recompute wrapper hashes ``(dtype, shape, stride(), ...)``;
    Paddle returns both as lists, which ``dict`` rejects. No GPU needed -- the
    contract is purely about the container types.
    """

    def test_shape_and_stride_are_hashable_tuples(self):
        tensor = _HashableTensor(paddle.zeros([2, 3, 4], dtype="float32"))
        self.assertIsInstance(tensor.shape, tuple)
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertIsInstance(tensor.stride(), tuple)
        self.assertEqual(tensor.stride(), (12, 4, 1))
        # Per-dim form: the wrapper does not use it, but it is the half of the
        # override that would silently return a plain int if dropped.
        self.assertEqual(tensor.stride(0), 12)
        hash((tensor.shape, tensor.stride()))


@_GPU
class TestMQADSACudnnTarget(unittest.TestCase):
    """The fused indexer-loss target (``_attn_target_cudnn``).

    The kernel needs an LSE taken over exactly the scored column set, which
    exists only when attention and loss share one table (phase 3,
    ``dsa_indexer_use_sparse_loss=True``) and the budget is a width
    ``flash_mla_sparse_fwd`` implements (``_LSE_INDEXER_TOPKS``). The
    module-wide fixture runs ``INDEX_TOPK = 128``, which is not one of them, so
    every test here raises the budget to 512 -- the narrowest supported width.
    """

    TOPK = 512
    SEQLEN = 768  # > WINDOW + TOPK, so the table stays genuinely sparse

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def _build(self, topk):
        config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
        config.dsa_index_topk = topk
        module = _build_module(config, bf16=True)
        self.assertEqual(int(module.indexer.index_topk), topk)
        return module

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.module = self._build(self.TOPK)

    def _train_step(self, module, seed=0):
        """One grad-enabled forward + backward, capturing the target's inputs.

        Returns the ``_attn_target`` arguments so a test can re-run the Python
        reference on the *same* columns; recomputing them from a second forward
        would not work, the top-k table drifts between identical calls on a
        single full-length document (``TestMQADSA``
        ``test_use_sparse_loss_switches_both_attention_and_loss_width``).
        """
        query, key, w_v, x, qr = _make_inputs(
            self.SEQLEN, seed=seed, with_hidden=True
        )
        captured = {}
        inner = module._attn_target

        def spy(query_, kv_, topk_indices, lse_indexer=None):
            captured["args"] = (query_, kv_, topk_indices)
            captured["lse_indexer"] = lse_indexer
            target = inner(query_, kv_, topk_indices, lse_indexer)
            captured["target"] = target
            return target

        module._attn_target = spy
        try:
            tensors = [t.clone() for t in (query, key, x, qr)]
            for tensor in tensors:
                tensor.stop_gradient = False
            module.train()
            out = module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([self.SEQLEN], self.SEQLEN),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
            )
            out.cast("float32").sum().backward()
        finally:
            module._attn_target = inner
        return captured

    def test_supported_budget_takes_the_fused_path(self):
        """A 512 budget routes the target through the kernel, and the KL lands.

        Asserts the plumbing end to end: ``mqa_sparse_attn`` returns the second
        output at all (the ``indexer_topk > 0`` signature), the LSE has the
        kernel's fixed ``h_q == 64`` head count rather than the layer's 8, and
        the loss built on top of it is finite and non-zero.

        The LSE is finite exactly on the rows that have a candidate: it is a
        log-sum-exp over the ``indexer_topk`` prefix of the table, so the rows
        whose prefix is all ``-1`` -- the first ``window_size`` tokens of a
        document, which ``_indexer_valid_range`` leaves without candidates --
        come back ``+inf``. Those rows have to end up as a zero target row, not
        as a NaN one, which is what the KL's valid-row denominator assumes.
        """
        captured = self._train_step(self.module)
        lse = captured["lse_indexer"]
        self.assertIsNotNone(lse, "the fused path did not receive an LSE")
        self.assertEqual(list(lse.shape), [1, self.SEQLEN, 64])
        self.assertEqual(lse.dtype, paddle.float32)
        has_candidate = (captured["args"][2] >= 0).any(axis=-1)
        self.assertTrue(
            bool(has_candidate.any()) and not bool(has_candidate.all())
        )
        self.assertTrue(bool(paddle.isfinite(lse[has_candidate]).all()))
        self.assertTrue(bool(paddle.isinf(lse[~has_candidate]).all()))

        target = captured["target"]
        self.assertEqual(list(target.shape), [1, self.SEQLEN, self.TOPK])
        self.assertTrue(bool(paddle.isfinite(target).all()))
        self.assertEqual(float(target[~has_candidate].abs().max()), 0.0)
        np.testing.assert_allclose(
            target.sum(axis=-1)[has_candidate].numpy(), 1.0, atol=1e-5
        )
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_fused_target_matches_the_python_reference(self):
        """The kernel and ``_attn_target_python`` must agree on the same table.

        Both produce ``sum_h softmax_h`` over the selected columns, L1
        normalised, so this is the correctness contract of the whole change: the
        head-sum's normalizer is the LSE restricted to those columns, and
        getting that wrong (e.g. feeding the attention LSE, which also covers
        the window and the sink) changes the objective without changing the
        forward output.
        """
        captured = self._train_step(self.module)
        query, kv, topk_indices = captured["args"]
        fused = captured["target"]
        reference = self.module._attn_target_python(query, kv, topk_indices)

        self.assertLess(_rel(fused, reference), 5e-3)
        # Every non-empty row is a distribution; empty slots stay exactly zero
        # (the KL divides by the valid-row count, not by the row sum).
        valid_rows = (topk_indices >= 0).any(axis=-1)
        row_sums = fused.sum(axis=-1)
        np.testing.assert_allclose(row_sums[valid_rows].numpy(), 1.0, atol=1e-5)
        empty_slots = topk_indices < 0
        self.assertEqual(float(fused[empty_slots].abs().max()), 0.0)

    def test_unsupported_budget_falls_back_to_python(self):
        """384 is a legal ``index_topk`` the kernel does not implement.

        ``dsa_index_topk`` only has to be a multiple of 128 and at most 2048
        (``transformer_config.py``), while ``indexer_topk`` accepts
        0/512/1024/2048 only. Without the ``_LSE_INDEXER_TOPKS`` guard the
        illegal width would reach the kernel; with it the target silently uses
        the Python reference instead.
        """
        self.assertNotIn(384, _LSE_INDEXER_TOPKS)
        module = self._build(384)
        captured = self._train_step(module)
        self.assertIsNone(captured["lse_indexer"])
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))


def _fake_score_recompute(fn):
    """Put ``fn`` in place of the cuDNN score-recompute wrapper.

    ``_attn_target_cudnn`` imports it lazily from
    ``paddlefleet_ops.cudnn.deepseek_sparse_attention``, so swapping the module
    in ``sys.modules`` intercepts the call without importing (or owning a GPU
    able to run) the real op.
    """
    module = types.ModuleType("paddlefleet_ops.cudnn.deepseek_sparse_attention")
    module.sparse_attn_score_recompute_wrapper = fn
    return mock.patch.dict(
        sys.modules,
        {"paddlefleet_ops.cudnn.deepseek_sparse_attention": module},
    )


class TestAttnTargetDispatch(unittest.TestCase):
    """``_attn_target`` picks its implementation from ``lse_indexer`` alone.

    No GPU: the dispatch is what decides whether the loss target comes from the
    kernel or from the reference, and it must not consult anything else (the
    caller has already resolved the phase and the budget).
    """

    def setUp(self):
        self.calls = []
        self.stub = SimpleNamespace(
            _attn_target_cudnn=lambda *a: self.calls.append("cudnn") or "cudnn",
            _attn_target_python=lambda *a: self.calls.append("python")
            or "python",
        )

    def test_lse_present_selects_the_kernel(self):
        got = MQALatentAttention._attn_target(
            self.stub, "q", "kv", "idx", "lse"
        )
        self.assertEqual((got, self.calls), ("cudnn", ["cudnn"]))

    def test_lse_absent_selects_the_reference(self):
        got = MQALatentAttention._attn_target(self.stub, "q", "kv", "idx", None)
        self.assertEqual((got, self.calls), ("python", ["python"]))
        # The default is the fallback too. No production caller omits the LSE
        # any more (phase 2 used to), but the parameter's default is what makes
        # the reference reachable from a bare three-argument call, which is how
        # every reference-vs-kernel comparison in this file invokes it.
        self.assertEqual(
            MQALatentAttention._attn_target(self.stub, "q", "kv", "idx"),
            "python",
        )


class TestAttnTargetCudnnMocked(unittest.TestCase):
    """``_attn_target_cudnn``'s call contract, kernel mocked out.

    The kernel itself needs SM100+ (covered by ``TestMQADSACudnnTarget``); what
    is checked here is everything the wrapper is responsible for and the kernel
    is not: hashable cache-key metadata, the LSE sliced down to the real head
    count and then both it and the query padded back up to the kernel's
    narrowest MMA tile, int32 indices, and the empty-slot handling on the way
    out.
    """

    H_Q, S, TOPK, DK_ = 4, 3, 4, 8
    SCALE = 0.25

    def _inputs(self, h_q=None):
        h_q = self.H_Q if h_q is None else h_q
        query = paddle.zeros([1, self.S, h_q, self.DK_], dtype="bfloat16")
        kv = paddle.zeros([1, self.S, self.DK_], dtype="bfloat16")
        # Row 0 has two valid columns, row 1 one, row 2 none (a fully padded
        # query row, which a short document's first token produces).
        idx = paddle.to_tensor(
            [[[0, 1, -1, -1], [0, -1, -1, -1], [-1, -1, -1, -1]]],
            dtype="int64",
        )
        # The kernel's LSE always has the DSA-fixed 64 heads, not the layer's.
        lse = paddle.full([1, self.S, 64], 1.5, dtype="bfloat16")
        return query, kv, idx, lse

    def _run(self, target_value=2.0, h_q=None):
        seen = {}

        def fake(q, kv, lse, idx, scale):
            seen["types"] = [type(t) for t in (q, kv, lse, idx)]
            # The real wrapper keys its kernel cache on this tuple; lists would
            # raise TypeError here, which is the whole point of
            # ``_HashableTensor``.
            seen["key"] = hash(
                tuple((t.dtype, t.shape, t.stride()) for t in (q, kv, lse, idx))
            )
            seen["q_shape"] = list(q.shape)
            seen["q"] = q.cast("float32").numpy().copy()
            seen["lse_shape"] = list(lse.shape)
            seen["lse"] = lse.numpy().copy()
            seen["lse_dtype"] = lse.dtype
            seen["idx_dtype"] = idx.dtype
            seen["scale"] = scale
            return {
                "target": paddle.full(idx.shape, target_value, dtype="float32")
            }

        query, kv, idx, lse = self._inputs(h_q)
        with _fake_score_recompute(fake):
            target = MQALatentAttention._attn_target_cudnn(
                SimpleNamespace(softmax_scale=self.SCALE), query, kv, idx, lse
            )
        return seen, target

    def test_kernel_arguments(self):
        seen, _ = self._run()
        self.assertEqual(seen["types"], [_HashableTensor] * 4)
        self.assertEqual(seen["lse_dtype"], paddle.float32)
        self.assertEqual(seen["idx_dtype"], paddle.int32)
        self.assertEqual(seen["scale"], self.SCALE)

    def test_narrow_head_count_is_padded_with_an_infinite_lse(self):
        """``h < 16`` is padded up, and the pad heads contribute nothing.

        The kernel's MMA ``M`` tile is the query-head count and it silently
        returns an all-zero target below 16 heads, so the wrapper pads. The pad
        heads must not join the head sum, which an infinite LSE guarantees
        exactly: ``exp(finite - inf) == 0``.
        """
        seen, _ = self._run()
        self.assertEqual(seen["q_shape"], [1, self.S, 16, self.DK_])
        self.assertEqual(seen["lse_shape"], [1, self.S, 16])
        # Real heads keep the layer's LSE, sliced out of the kernel's 64-wide
        # one; the pad heads are +inf.
        np.testing.assert_array_equal(seen["lse"][:, :, : self.H_Q], 1.5)
        self.assertTrue(bool(np.isposinf(seen["lse"][:, :, self.H_Q :]).all()))
        np.testing.assert_array_equal(seen["q"][:, :, self.H_Q :], 0.0)

    def test_supported_head_count_is_passed_through(self):
        """A power-of-two ``h >= 16`` (production is 64) is not padded."""
        seen, _ = self._run(h_q=64)
        self.assertEqual(seen["q_shape"], [1, self.S, 64, self.DK_])
        self.assertEqual(seen["lse_shape"], [1, self.S, 64])
        self.assertTrue(bool(np.isfinite(seen["lse"]).all()))

    def test_empty_slots_are_zeroed_and_rows_renormalised(self):
        """Whatever the kernel writes at ``-1`` slots is discarded.

        A uniform kernel output makes the expectation exact: each row becomes
        uniform over its valid columns, and the all-empty row stays all zeros
        rather than turning into ``0/0`` -- the KL reduction divides by the
        valid-row count, so a padded row must contribute nothing.
        """
        _, target = self._run()
        np.testing.assert_allclose(
            target.numpy(),
            np.array([[[0.5, 0.5, 0.0, 0.0], [1.0, 0, 0, 0], [0, 0, 0, 0]]]),
            atol=1e-6,
        )
        self.assertEqual(target.dtype, paddle.float32)


class TestMQASparseAttnLseSideChannelMocked(unittest.TestCase):
    """``mqa_sparse_attn``'s ``lse_indexer`` side channel, kernel mocked out.

    The LSE cannot be a PyLayer output (every returned tensor would demand a
    matching backward gradient), so it travels on a class attribute that the
    wrapper pops. That popping is pure Python and is what this checks, together
    with the ``indexer_topk`` pass-through and the two return signatures.
    """

    S, H_Q, DK_, DV_ = 4, 8, 576, 512
    SCALE = 0.1

    def _inputs(self):
        query = paddle.zeros([1, self.S, self.H_Q, self.DK_], dtype="bfloat16")
        kv = paddle.zeros([1, self.S, self.DK_], dtype="bfloat16")
        idx = paddle.arange(self.S, dtype="int32").reshape([1, 1, self.S])
        idx = paddle.where(
            idx <= paddle.arange(self.S, dtype="int32").reshape([1, self.S, 1]),
            idx.tile([1, self.S, 1]),
            paddle.full([1, self.S, self.S], -1, dtype="int32"),
        )
        return query, kv, idx.contiguous()

    def _patched_kernel(self, calls):
        import paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn as fwd_mod

        def fake(q_pad, kv, sink, token_indices, **kwargs):
            calls.append(kwargs)
            b, s = int(q_pad.shape[0]), int(q_pad.shape[1])
            out = paddle.zeros([b, s, 64, kwargs["d_v"]], dtype=q_pad.dtype)
            lse = paddle.zeros([b, s, 64], dtype="float32")
            # The kernel returns ``None`` for a 0 budget; a non-None value for
            # a real one, so a leak would be visible on the next call.
            lse_indexer = (
                None
                if kwargs["indexer_topk"] == 0
                else paddle.full([b, s, 64], 1.5, dtype="float32")
            )
            return out, lse, lse_indexer

        return mock.patch.object(fwd_mod, "flash_mla_sparse_attn", fake)

    def _call(self, indexer_topk, calls):
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        query, kv, idx = self._inputs()
        with self._patched_kernel(calls):
            return mqa_sparse_attn(
                query,
                kv,
                idx,
                self.SCALE,
                self.DV_,
                attn_sink=None,
                indexer_topk=indexer_topk,
            )

    def tearDown(self):
        from paddlefleet.fusions.mqa_sparse_attn import _MQASparseAttention

        _MQASparseAttention._lse_indexer = None

    def test_zero_budget_keeps_the_single_output_signature(self):
        calls = []
        out = self._call(0, calls)
        self.assertIsInstance(out, paddle.Tensor)
        self.assertEqual(list(out.shape), [1, self.S, self.H_Q * self.DV_])
        self.assertEqual(calls[0]["indexer_topk"], 0)

    def test_positive_budget_returns_and_forwards_the_lse(self):
        calls = []
        out, lse_indexer = self._call(512, calls)
        self.assertEqual(calls[0]["indexer_topk"], 512)
        self.assertEqual(list(out.shape), [1, self.S, self.H_Q * self.DV_])
        self.assertEqual(list(lse_indexer.shape), [1, self.S, 64])
        self.assertEqual(lse_indexer.dtype, paddle.float32)

    def test_the_side_channel_never_outlives_one_call(self):
        """Popped on every call, so a stale LSE cannot reach the next one.

        Without the reset a 0-budget call following a 512 one would still find
        the previous tensor on the class -- harmless today (the return value is
        gated on ``indexer_topk``) but it would pin one LSE per layer alive for
        the whole step.
        """
        from paddlefleet.fusions.mqa_sparse_attn import _MQASparseAttention

        calls = []
        self._call(512, calls)
        self.assertIsNone(_MQASparseAttention._lse_indexer)
        self._call(0, calls)
        self.assertIsNone(_MQASparseAttention._lse_indexer)


class TestSparseAttnPlumbingMocked(unittest.TestCase):
    """``_sparse_attn`` forwards the sink and the budget, and nothing else.

    Mocking ``mqa_sparse_attn`` keeps this off the SM100 kernels; the method's
    only job is to hand over ``self.softmax_offset`` and ``indexer_topk``.
    """

    def setUp(self):
        _CAPTURED.clear()
        self.module = _build_module(_create_mqa_config("mqa_dsa"), bf16=True)

    def _patched(self, calls):
        import paddlefleet.fusions.mqa_sparse_attn as fusion

        def fake(query, kv, token_indices, sm_scale, d_v, **kwargs):
            calls.append((float(sm_scale), int(d_v), kwargs))
            return "core_out"

        return mock.patch.object(fusion, "mqa_sparse_attn", fake)

    def test_budget_and_sink_are_forwarded(self):
        calls = []
        with self._patched(calls):
            got = self.module._sparse_attn(
                paddle.zeros([1, 2, H, DK], dtype="bfloat16"),
                paddle.zeros([1, 2, DK], dtype="bfloat16"),
                paddle.zeros([1, 2, 4], dtype="int32"),
                self.module.softmax_scale,
                DV,
                indexer_topk=512,
            )
        self.assertEqual(got, "core_out")
        scale, d_v, kwargs = calls[0]
        self.assertEqual((scale, d_v), (self.module.softmax_scale, DV))
        self.assertEqual(kwargs["indexer_topk"], 512)
        # No sink configured in this fixture: the backend reads ``None`` as
        # "sinkless softmax", it is not an omitted argument.
        self.assertIsNone(kwargs["attn_sink"])
        self.assertIn("attn_sink", kwargs)


class TestForwardDsaFusedDispatchMocked(unittest.TestCase):
    """Which target path ``_forward_sparse`` selects, with both kernels mocked.

    ``TestMQADSACudnnTarget`` covers the same decision on real kernels; this
    reaches it without any, so the branch stays checked on machines below
    SM100. Mocked: the indexer top-k kernel, the sparse-attention call and the
    target itself -- everything between them (window table, concat order, KL,
    loss logging) is the real code.
    """

    S = 256
    WIDE_TOPK = 384  # a legal index_topk the kernel does not implement

    def _build(self, topk):
        config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
        config.dsa_index_topk = topk
        module = _build_module(config, bf16=True)
        module.train()
        return module

    def _topk_table(self, topk):
        """``[1, S, topk]`` causal-ish table, right-padded with ``-1``."""
        cols = paddle.arange(topk, dtype="int32").reshape([1, 1, topk])
        rows = paddle.arange(self.S, dtype="int32").reshape([1, self.S, 1])
        return paddle.where(
            cols <= rows,
            cols.tile([1, self.S, 1]),
            paddle.full([1, self.S, topk], -1, dtype="int32"),
        ).contiguous()

    def _run(self, topk):
        module = self._build(topk)
        DSAIndexerLossLoggingHelper.tracker.clear()
        seen = {}
        table = self._topk_table(topk)

        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod

        def fake_topk(q, k, w, **kwargs):
            width = int(kwargs["topk_effective"])
            scores = paddle.rand([1, self.S, width], dtype="float32")
            return self._topk_table(width), None, scores

        def fake_indexer(x, qr, position_offset, cp_group):
            return (
                paddle.zeros(
                    [1, self.S, INDEX_HEADS, INDEX_HEAD_DIM], dtype="bfloat16"
                ),
                paddle.zeros([1, self.S, INDEX_HEAD_DIM], dtype="bfloat16"),
                paddle.zeros([1, self.S, INDEX_HEADS], dtype="bfloat16"),
            )

        def fake_sparse_attn(
            query, kv, token_indices, sm_scale, d_v, indexer_topk=0
        ):
            seen["indexer_topk"] = int(indexer_topk)
            seen["token_indices"] = token_indices.numpy().copy()
            core_out = query.reshape([1, self.S, H * DK])[:, :, : H * DV]
            if indexer_topk == 0:
                return core_out
            return core_out, paddle.full([1, self.S, 64], 1.5, dtype="float32")

        def fake_target(query, kv, topk_indices, lse_indexer=None):
            seen["lse_indexer"] = lse_indexer
            width = int(topk_indices.shape[-1])
            return paddle.full([1, self.S, width], 1.0 / width, dtype="float32")

        query, key, w_v, x, qr = _make_inputs(self.S, seed=0, with_hidden=True)
        tensors = [t.clone() for t in (query, key, x, qr)]
        for tensor in tensors:
            tensor.stop_gradient = False
        module.indexer.forward_before_topk = fake_indexer
        module._sparse_attn = fake_sparse_attn
        module._attn_target = fake_target
        with mock.patch.object(fwd_mod, "cudnn_indexer_topk_fwd", fake_topk):
            out = module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([self.S], self.S),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
            )
        seen["output"] = out
        seen["table"] = table.numpy()
        return seen

    def test_supported_budget_requests_the_lse_and_reuses_it(self):
        topk = _LSE_INDEXER_TOPKS[0]
        seen = self._run(topk)
        self.assertEqual(seen["indexer_topk"], topk)
        # The indexer columns must come first: the kernel's LSE covers the
        # leading ``indexer_topk`` columns of the table, so the window has to
        # sit in the tail.
        table = seen["token_indices"]
        self.assertEqual(table.shape[-1], topk + WINDOW)
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            _row_end([self.S], self.S), self.S
        )
        window = (
            _build_window_topk_idxs_from_doc_bounds(
                1, self.S, WINDOW, doc_start, is_valid
            )
            .cast("int32")
            .numpy()
        )
        np.testing.assert_array_equal(table[:, :, topk:], window)
        # ``_forward_sparse`` blanks the rows whose candidate range is empty (the
        # first ``window_size`` tokens of a document), so compare where the
        # prefix survived.
        prefix, selected = table[:, :, :topk], seen["table"]
        kept = prefix != -1
        self.assertGreater(int(kept.sum()), 0)
        np.testing.assert_array_equal(prefix[kept], selected[kept])
        self.assertIsNotNone(seen["lse_indexer"])
        self.assertEqual(list(seen["lse_indexer"].shape), [1, self.S, 64])
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))

    def test_unsupported_budget_asks_for_no_lse(self):
        self.assertNotIn(self.WIDE_TOPK, _LSE_INDEXER_TOPKS)
        seen = self._run(self.WIDE_TOPK)
        self.assertEqual(seen["indexer_topk"], 0)
        self.assertIsNone(seen["lse_indexer"])
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
