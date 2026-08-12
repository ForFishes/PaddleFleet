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

"""Recompute / MTP / RoPE regression for the phase-2 (warmup) shape of
``hybrid_mla_attention="mqa_dsa"``.

The warmup shape is selected by ``dsa_indexer_use_sparse_loss=False``, and since
that phase has no top-k on *either* side it runs phase 1's **dense MHA**
(``MHADSAWarmupAttention``, a ``DotProductAttention`` subclass) with the indexer
bolted on; the absorbed latent MQA of ``MQALatentAttention`` is phase 3/4 only
(``hybrid_mla_indexer.latent_mqa_enabled``). So there is no ``[b, s, s]`` index
table and no block-sparse call anywhere below -- ``_CAPTURED`` staying empty is
itself an assertion -- and the object the backward consumes is the KL target
(``_WARMUP_TARGETS``) instead. This module covers the three axes that compose
*around* the single forward:

1. **Recompute** (``TestWarmupRecompute``) -- the real production wrapping,
   ``paddle.distributed.fleet.utils.recompute`` around
   ``MHADSAWarmupAttention.forward``, which is how ``full_recompute`` wraps
   ``_forward_impl``. ON must equal OFF, the KL target must be re-derived
   bit-identically, and the indexer loss must be attached exactly once, on the
   grad-enabled pass: ``_needs_indexer_loss`` gates on
   ``paddle.is_grad_enabled()`` (``hybrid_mla_indexer.py:110-121``), so the
   ``no_grad`` pass skips the indexer entirely -- which phase 3, whose attention
   consumes the ranking, cannot do. That contrast is the discriminator of
   ``test_no_grad_pass_skips_the_indexer_only_in_warmup``.
2. **MTP** (``TestWarmupMTP``) -- ``MultiTokenPredictionLayer`` builds its
   ``transformer_layer`` without passing ``pg_collection``, so the MTP ``-2``
   layer is the same core-attention class reading the same config; the warmup
   shape must therefore be live there too, and bit-identical to the phase-1
   dense MTP layer. Plus the tracker denominator: ``track_indexer_metrics``
   takes the enum string and must count the MTP ``-2`` entry of
   ``csa_compress_ratios``.
3. **RoPE** (``TestWarmupRope``) -- the switch must not touch RoPE. The main
   attention's rotary application happens in ``MLASelfAttention`` before the
   ``mqa_latent`` branch, and the DSA indexer keeps its own plain-RoPE
   (``dsa_indexer_rotary_interleaved``), evaluated in warmup even though
   attention does not consume its ranking -- both phases reach it through the
   shared ``HybridMLAIndexerMixin._indexer_projections``. Also covers the
   construction-time ``apply_rope_fusion`` x latent-MQA behaviour: the latent
   layer downgrades to eager RoPE and warns while the non-latent layers --
   phase 1, ``mha``, and now phase 2 -- keep the global fusion, plus the
   ``mqa_latent_rope_fusion=True`` opt-in that fuses it instead.

Every RoPE assertion is against the independent fp64 reference of
``test_hybrid_mla_rope_audit``, never against the implementation itself.

``TestRecomputeInnerForwardBitIdentical`` closes the one gap axis 1 leaves open:
recompute-ON vs recompute-OFF says nothing about whether the *two* forwards of a
recomputed step agree with each other, since paddle discards the first one's
output. That class keeps both and requires ``maxabs == 0.0``, on all three
``hybrid_mla_attention`` shapes -- the two latent ones on their own call shape,
untouched -- and on a ``seqlen`` where the sparse budget does not already cover
the whole causal range.
"""

import types
import unittest
from unittest import mock

import numpy as np
import paddle

from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.mha_dsa_warmup_attention import (
    MHADSAWarmupAttention,
)
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)

from .hybrid_mla_utils import (
    _CAPTURED,
    _GPU,
    _WARMUP_TARGETS,
    HIDDEN,
    INDEX_TOPK,
    Q_LORA,
    BiasedLinear,
    LayerNormStub,
    _build_module,
    _build_phase1_dense_module,
    _create_mqa_config,
    _make_dense_inputs,
    _make_inputs,
    _rel,
    _row_end,
)
from .test_hybrid_mla_rope_audit import (
    ROPE_THETA,
    ref_angles,
    ref_inv_freq,
    ref_rope_halfsplit,
)

SEQLEN = 256
# Two documents, the second longer than the forced window (so the indexer's
# candidate range is non-empty), plus the single full-length document that the
# phase-3 recompute test cannot use because its top-k emission order drifts.
TWO_DOCS = [40, 216]
ONE_DOC = [SEQLEN]

# Production 44-slot ratios: -2 at [8, 17, 26, 34, 42, 43]; 43 is the MTP layer.
_PROD_MINUS2 = [8, 17, 26, 34, 42, 43]


def _prod_csa_ratios():
    ratios = [128] * 8 + [-2] + [128] * 8 + [-2] + [128] * 8 + [-2]
    ratios += [128] * 7 + [-2] + [128] * 7 + [-2, -2]
    assert len(ratios) == 44
    assert [i for i, v in enumerate(ratios) if v == -2] == _PROD_MINUS2
    return ratios


def _warmup_config(loss_coeff=0.01, **overrides):
    """A ``"mqa_dsa"`` config with the phase-2 switch off from construction.

    ``sparse_loss=False`` is what makes ``_build_module`` pick the dense
    ``MHADSAWarmupAttention`` backend, through the production predicate
    ``hybrid_mla_indexer.latent_mqa_enabled``, so it has to be set *at*
    construction rather than assigned afterwards.
    """
    return _create_mqa_config(
        "mqa_dsa", loss_coeff=loss_coeff, sparse_loss=False, **overrides
    )


def _fp32(tensor):
    """bf16 -> fp32 numpy; the widening is exact, so bit equality survives."""
    return tensor.cast("float32").numpy()


def _leaf(tensor):
    out = tensor.clone().detach()
    out.stop_gradient = False
    return out


def _dense_call(module, query, key, value, row_end, x, qr, input_ids=None):
    """The phase-2 forward call shape.

    Per-head q/k/v (``_make_dense_inputs``), no ``v_b_proj_weight`` (nothing is
    absorbed), and an **explicit** ``attn_mask_type``: the inherited
    ``DotProductAttention.forward`` resolves ``is_causal`` from this arg and
    leaves it False when it is omitted, so dropping it would silently compare a
    bidirectional attention against a causal reference.
    """
    return module(
        query,
        key,
        value,
        None,
        row_end,
        attn_mask_type=AttnMaskType.causal,
        x=x,
        qr=qr,
        input_ids=input_ids,
    )


class _ForwardSpy:
    """Count and keep the output of every ``module.forward`` call.

    ``_CAPTURED`` used to double as the forward counter (one entry per
    block-sparse call), but the phase-2 backend never reaches that kernel, so
    "did recompute really re-forward the layer?" has to be answered by the layer
    itself now. Patching the bound method rather than the class keeps the two
    concurrent module instances of ``_indexer_call_count`` independent.
    """

    def __init__(self, module):
        self.module = module
        self.outputs = []

    def __enter__(self):
        real = type(self.module).forward

        def spy(zelf, *args, **kwargs):
            result = real(zelf, *args, **kwargs)
            tensor = result[0] if isinstance(result, tuple) else result
            self.outputs.append(_fp32(tensor.detach()).copy())
            return result

        self.module.forward = types.MethodType(spy, self.module)
        return self

    def __exit__(self, *exc_info):
        del self.module.forward
        return False

    def __len__(self):
        return len(self.outputs)


def _grad_rel(g_on, g_off):
    if g_on is None and g_off is None:
        return 0.0
    assert (g_on is None) == (g_off is None), "one grad present, the other None"
    return _rel(g_on, g_off)


def _tracker_value(slot):
    values = DSAIndexerLossLoggingHelper.tracker.get("values")
    return None if values is None else float(values.numpy()[slot])


@_GPU
class TestWarmupRecompute(unittest.TestCase):
    """``recompute(MHADSAWarmupAttention.forward)`` in the warmup phase.

    This is the production wrapping: ``full_recompute`` hands
    ``TransformerLayer._forward_impl`` to ``paddle.distributed.fleet.utils.
    recompute``, whose reentrant implementation runs the wrapped callable once
    under ``no_grad`` (to produce the output) and once more with grad enabled
    during backward. Both forwards must agree, or the backward differentiates
    something the forward never produced.
    """

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
        self.module = _build_module(_warmup_config(), bf16=True)
        # ``indexer_use_sparse_loss`` is a latent-MQA attribute; the phase-2
        # backend has no such field, the config is the only place the phase
        # lives.
        self.assertIsInstance(self.module, MHADSAWarmupAttention)
        self.assertIsNotNone(self.module.indexer)
        self.assertFalse(self.module.config.dsa_indexer_use_sparse_loss)

    def _run(self, module, row_end, use_recompute, seed=7):
        """One train step, with or without recompute.

        Returns ``(output, grads, n_forwards)``.
        """
        from paddle.distributed.fleet.utils import recompute

        query, key, value, x, qr = _make_dense_inputs(SEQLEN, seed=seed)
        module.train()
        module.clear_gradients()
        q = _leaf(query)

        def fn(qin):
            return _dense_call(module, qin, key, value, row_end, x, qr)

        with _ForwardSpy(module) as spy:
            out = recompute(fn, q) if use_recompute else fn(q)
            out.cast("float32").sum().backward()
            n_forwards = len(spy)
        grads = {
            name: (None if p.grad is None else p.grad.detach().cast("float32"))
            for name, p in module.named_parameters()
        }
        grads["__query__"] = (
            None if q.grad is None else q.grad.detach().cast("float32")
        )
        return out.detach().cast("float32"), grads, n_forwards

    def _check_target_rows(self, target):
        """The KL target is a per-row distribution over the candidate set.

        The phase-2 replacement for ``_check_index_invariants``: there is no
        column table left to audit, so audit what the backward actually consumes
        instead. ``mha_dsa_warmup_attention.py:414-417`` documents the shape and
        the contract -- ``[1, s_local, s_global]``, rows summing to 1, empty
        rows all-zero -- and ``:445-450`` is where the explicit zeroing of an
        all-masked row happens (a uniform softmax there would poison a KL
        reduction that divides by the valid-row count, not by the row sum).
        """
        self.assertEqual(list(target.shape), [1, SEQLEN, SEQLEN])
        self.assertTrue(bool((target >= 0.0).all()), "negative target mass")
        sums = target[0].sum(axis=-1)
        for row in range(SEQLEN):
            # ``_row_end`` makes every row valid, so every row is a distr.
            self.assertAlmostEqual(
                float(sums[row]),
                1.0,
                delta=1e-4,
                msg=f"target row {row} does not sum to 1",
            )

    def _equivalence(self, layout):
        row_end = _row_end(layout, SEQLEN)

        _CAPTURED.clear()
        _WARMUP_TARGETS.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        out_off, g_off, n_fwd_off = self._run(
            self.module, row_end, use_recompute=False
        )
        target_off = _WARMUP_TARGETS[-1]
        loss_off = _tracker_value(self.module.layer_number - 1)

        _CAPTURED.clear()
        _WARMUP_TARGETS.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        out_on, g_on, n_fwd_on = self._run(
            self.module, row_end, use_recompute=True
        )
        loss_on = _tracker_value(self.module.layer_number - 1)

        # The recompute forward really ran (otherwise the rest proves nothing).
        self.assertEqual(n_fwd_off, 1)
        self.assertGreaterEqual(
            n_fwd_on, 2, "recompute did not re-forward the layer"
        )
        # Phase 2 is dense: no ``[b, s, s]`` table is built and the block-sparse
        # kernel is never entered, on either pass.
        self.assertEqual(
            len(_CAPTURED), 0, "phase 2 reached the block-sparse kernel"
        )
        # The KL target is built on the grad-enabled pass only, so two forwards
        # produce exactly one -- and it is bit-identical to the single-forward
        # run's. No tolerance: the recomputed forward re-executes the same
        # kernels on the same saved inputs.
        self.assertEqual(len(_WARMUP_TARGETS), 1)
        np.testing.assert_array_equal(_WARMUP_TARGETS[-1], target_off)
        self._check_target_rows(target_off)

        out_rel = _rel(out_on, out_off)
        self.assertEqual(set(g_on), set(g_off))
        grad_rels = {name: _grad_rel(g_on[name], g_off[name]) for name in g_off}
        worst = max(grad_rels, key=grad_rels.get)
        print(
            f"[warmup recompute {layout}] out_rel={out_rel:.3e} "
            f"worst_grad={worst}:{grad_rels[worst]:.3e} "
            f"loss_off={loss_off!r} loss_on={loss_on!r} "
            f"all_grads={ {k: f'{v:.2e}' for k, v in grad_rels.items()} }"
        )
        # Same tolerances as the phase-3 equivalence test
        # (test_hybrid_mla_recompute_mtp_ckpt.TestRecomputeEquivalence).
        self.assertLess(out_rel, 1e-5, f"{layout} output rel={out_rel}")
        for name, rel in grad_rels.items():
            self.assertLess(rel, 5e-3, f"{layout} grad[{name}] rel={rel}")
        # The indexer loss is attached on the grad-enabled pass only, so it is
        # counted once -- not twice, and not lost.
        self.assertIsNotNone(loss_off)
        self.assertIsNotNone(loss_on)
        self.assertGreater(loss_off, 0.0)
        self.assertAlmostEqual(
            loss_on / loss_off,
            1.0,
            delta=1e-3,
            msg=f"indexer loss counted {loss_on / loss_off:.3f}x under recompute",
        )

    def test_recompute_equivalence_two_documents(self):
        self._equivalence(TWO_DOCS)

    def test_recompute_equivalence_single_document(self):
        """The layout phase 3 cannot assert on.

        Phase 3's top-k emission order drifts between two forwards of the same
        single full-length document, so it can only compare the selected *set*.
        Phase 2 selects nothing, so its target is reproduced exactly.
        """
        self._equivalence(ONE_DOC)

    def _run_latent(self, module, row_end, seed=7):
        """One recomputed train step on the **phase-3** latent module.

        Deliberately kept on the latent call shape (``_make_inputs`` plus
        ``v_b_proj_weight``): phase 3 is unchanged by the dense-warmup rework
        and appears here only as the contrast of
        ``test_no_grad_pass_skips_the_indexer_only_in_warmup``.
        """
        from paddle.distributed.fleet.utils import recompute

        query, key, w_v, x, qr = _make_inputs(
            SEQLEN, seed=seed, with_hidden=True
        )
        module.train()
        module.clear_gradients()
        q = _leaf(query)

        def fn(qin):
            return module(
                qin, key, None, None, row_end, v_b_proj_weight=w_v, x=x, qr=qr
            )

        with _ForwardSpy(module) as spy:
            out = recompute(fn, q)
            out.cast("float32").sum().backward()
            return len(spy)

    def _indexer_call_count(self, use_sparse_loss):
        """Indexer selector calls across both passes, per backend.

        Two selectors have to be counted separately: phase 3 selects with the
        **cuDNN** top-k kernel, phase 2 with the **tilelang** one at
        ``topk_effective = s_global`` (its full-candidate mode,
        ``mha_dsa_warmup_attention.py:292-300``). Counting only cuDNN would
        report "zero top-k calls" for warmup and hide the one call it does make.

        Returns ``(n_before_topk, cudnn_topks, tilelang_topks, n_forwards,
        n_sparse_kernel_calls)``.
        """
        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod
        import paddlefleet.tilelang_ops as tl_mod

        config = _create_mqa_config(
            "mqa_dsa", loss_coeff=0.01, sparse_loss=use_sparse_loss
        )
        module = _build_module(config, bf16=True)
        # The phase switch picks the backend, so the fixture cannot silently
        # test the same class twice.
        self.assertEqual(
            isinstance(module, MHADSAWarmupAttention), not use_sparse_loss
        )

        cudnn_calls = []
        tl_calls = []
        before_calls = []
        inner_topk = fwd_mod.cudnn_indexer_topk_fwd
        inner_tl = tl_mod.csa_indexer_topk_fwd
        inner_before = module.indexer.forward_before_topk

        def rec_topk(*args, **kwargs):
            cudnn_calls.append(int(kwargs["topk_effective"]))
            return inner_topk(*args, **kwargs)

        def rec_tl(*args, **kwargs):
            tl_calls.append(int(kwargs["topk_effective"]))
            return inner_tl(*args, **kwargs)

        def rec_before(*args, **kwargs):
            before_calls.append(1)
            return inner_before(*args, **kwargs)

        fwd_mod.cudnn_indexer_topk_fwd = rec_topk
        tl_mod.csa_indexer_topk_fwd = rec_tl
        module.indexer.forward_before_topk = rec_before
        _CAPTURED.clear()
        _WARMUP_TARGETS.clear()
        row_end = _row_end(TWO_DOCS, SEQLEN)
        try:
            if use_sparse_loss:
                n_forwards = self._run_latent(module, row_end)
            else:
                n_forwards = self._run(module, row_end, use_recompute=True)[2]
        finally:
            fwd_mod.cudnn_indexer_topk_fwd = inner_topk
            tl_mod.csa_indexer_topk_fwd = inner_tl
            module.indexer.forward_before_topk = inner_before
        return (
            len(before_calls),
            cudnn_calls,
            tl_calls,
            n_forwards,
            len(_CAPTURED),
        )

    def test_no_grad_pass_skips_the_indexer_only_in_warmup(self):
        """The warmup early exit, observed through the recompute double pass.

        Under recompute the layer is forwarded twice: once under ``no_grad`` (no
        loss needed) and once grad-enabled. ``_needs_indexer_loss`` is False
        on the first one (``hybrid_mla_indexer.py:117-121``), so in warmup the
        indexer projections run exactly *once* even though attention was built
        twice, and the **cuDNN** top-k kernel -- phase 3's selector -- runs
        **zero** times: warmup reads no ``index_topk``. What it does run is one
        **tilelang** call at ``topk_effective == SEQLEN``, its full-candidate
        mode, and exactly one, on the grad-enabled pass only. Phase 3 has no
        such exit: attention consumes the ranking, so the projections and the
        cuDNN kernel both run on both passes. The contrast is the discriminator.

        The sparse-kernel counts are the other half of the contrast: phase 3
        enters it once per forward, phase 2 never.
        """
        n_before_w, cudnn_w, tl_w, n_fwd_w, n_sparse_w = (
            self._indexer_call_count(False)
        )
        n_before_s, cudnn_s, tl_s, n_fwd_s, n_sparse_s = (
            self._indexer_call_count(True)
        )
        print(
            f"[warmup indexer calls] warmup before_topk={n_before_w} "
            f"cudnn={cudnn_w} tilelang={tl_w} forwards={n_fwd_w} "
            f"sparse_kernel={n_sparse_w} || "
            f"phase3 before_topk={n_before_s} cudnn={cudnn_s} "
            f"tilelang={tl_s} forwards={n_fwd_s} sparse_kernel={n_sparse_s}"
        )
        self.assertGreaterEqual(n_fwd_w, 2, "recompute did not re-forward")
        self.assertGreaterEqual(n_fwd_s, 2, "recompute did not re-forward")
        self.assertEqual(n_before_w, 1)
        self.assertEqual(cudnn_w, [], "warmup called the cuDNN top-k kernel")
        # One tilelang call, on the grad-enabled pass only, over every column.
        self.assertEqual(tl_w, [SEQLEN])
        self.assertEqual(n_sparse_w, 0, "warmup ran the block-sparse kernel")
        # Same wrapping, sparse phase: no early exit, both passes pay for it.
        self.assertEqual(n_before_s, 2)
        self.assertEqual(len(cudnn_s), 2)
        self.assertEqual(cudnn_s, [INDEX_TOPK, INDEX_TOPK])
        self.assertEqual(tl_s, [], "phase 3 selected with the tilelang kernel")
        self.assertGreaterEqual(n_sparse_s, 2)


def _mtp_config(use_sparse_loss, loss_coeff=0.01):
    """Production MTP shape: 43 backbone layers + 1 next-n predict layer."""
    config = _create_mqa_config(
        "mqa_dsa",
        loss_coeff=loss_coeff,
        num_hidden_layers=43,
        sparse_loss=use_sparse_loss,
    )
    config.num_nextn_predict_layers = 1
    config.pad_token_id = 0
    return config


@_GPU
class TestWarmupMTP(unittest.TestCase):
    """The warmup shape inside the MTP layer.

    ``MultiTokenPredictionLayer.__init__`` builds its ``transformer_layer``
    without passing ``pg_collection`` (``multi_token_prediction.py:419-423``), so
    the MTP ``-2`` layer reads the same config as the backbone ones and its core
    attention comes from the same ``latent_mqa_enabled`` dispatch -- there is no
    MTP-specific branch that could keep it on the phase-3 latent shape. These
    tests pin that: same construction as
    ``test_hybrid_mla_mtp_layer43_w6._build_mtp_module`` but with the phase-2
    switch off.
    """

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
        DSAIndexerLossLoggingHelper.num_layers = None

    @staticmethod
    def _build(use_sparse_loss=False, loss_coeff=0.01):
        module = _build_module(
            _mtp_config(use_sparse_loss, loss_coeff),
            layer_number=0,
            bf16=True,
            is_mtp=True,
        )
        assert module.layer_number == 0
        return module

    @staticmethod
    def _call(module, row_end, training=True, input_ids=None):
        query, key, value, x, qr = _make_dense_inputs(SEQLEN, seed=5)
        module.train() if training else module.eval()
        return _dense_call(module, query, key, value, row_end, x, qr, input_ids)

    @staticmethod
    def _call_phase1(module, row_end):
        """Same inputs, on a bare ``DotProductAttention``.

        It does not accept ``input_ids`` (``accepts_input_ids`` is a capability
        of the two indexer-owning core attentions only), so the phase-2 call
        helper cannot be reused verbatim.
        """
        query, key, value, x, qr = _make_dense_inputs(SEQLEN, seed=5)
        module.eval()
        return module(
            query,
            key,
            value,
            None,
            row_end,
            attn_mask_type=AttnMaskType.causal,
            x=x,
            qr=qr,
        )

    def test_mtp_warmup_attention_is_the_phase1_dense_layer(self):
        """The MTP ``-2`` layer's attention is bit-identical to phase 1's.

        The inverse of the assertion this test used to make (that the MTP layer
        built the same full-causal ``[b, s, s]`` table as an indexer-less
        ``mqa_full_causal`` MTP layer): phase 2 delegates its whole attention
        half to ``DotProductAttention.forward``
        (``mha_dsa_warmup_attention.py:179-198``) and the indexer only rides on
        the output's gradient, so the forward must equal the plain dense layer
        exactly -- and no block-sparse call may appear.
        """
        row_end = _row_end(TWO_DOCS, SEQLEN)
        config = _mtp_config(use_sparse_loss=False)
        module = _build_module(config, layer_number=0, bf16=True, is_mtp=True)
        self.assertIsInstance(module, MHADSAWarmupAttention)
        reference = _build_phase1_dense_module(
            config, layer_number=0, bf16=True, is_mtp=True
        )
        self.assertNotIsInstance(reference, MHADSAWarmupAttention)
        # Phase 2 adds indexer parameters and nothing else, which is why no
        # state_dict has to be copied for the comparison below to be meaningful:
        # the attention half is parameter-free in this fixture (no sink
        # configured, so ``build_softmax_offset`` returns None).
        self.assertEqual(
            {k for k in module.state_dict() if not k.startswith("indexer.")},
            set(reference.state_dict()),
        )

        _CAPTURED.clear()
        _WARMUP_TARGETS.clear()
        out_warm = _fp32(self._call(module, row_end, training=True))
        out_ref = _fp32(self._call_phase1(reference, row_end))

        self.assertEqual(
            len(_CAPTURED), 0, "the MTP warmup layer built an index table"
        )
        self.assertEqual(
            len(_WARMUP_TARGETS), 1, "the MTP warmup layer skipped the indexer"
        )
        maxabs = float(np.max(np.abs(out_warm - out_ref)))
        print(f"[warmup mtp] out maxabs vs the phase-1 dense MTP = {maxabs!r}")
        np.testing.assert_array_equal(out_warm, out_ref)

    def test_mtp_indexer_loss_denominator_counts_the_mtp_minus2_layer(self):
        """``get_total_num_layers`` is 44 and the tracker row is the MTP one.

        The MTP layer keeps ``layer_number=0``, so ``save_loss_to_tracker``
        writes ``values[-1]``: a pre-existing logging blemish already pinned by
        ``test_hybrid_mla_recompute_mtp_ckpt.TestMTPTrackerSlot``. Asserted here
        as-is, deliberately not "fixed" -- what this test adds is that the
        warmup phase still reaches the tracker at all.
        """
        module = self._build()
        self.assertEqual(
            DSAIndexerLossLoggingHelper.get_total_num_layers(module.config), 44
        )
        out = self._call(module, _row_end(TWO_DOCS, SEQLEN), training=True)
        out.cast("float32").sum().backward()
        values = DSAIndexerLossLoggingHelper.tracker["values"]
        self.assertEqual(list(values.shape), [44])
        nonzero = [i for i, v in enumerate(values.numpy()) if v != 0.0]
        loss = float(values.numpy()[-1])
        print(f"[warmup mtp tracker] nonzero_slots={nonzero} values[-1]={loss}")
        self.assertEqual(nonzero, [43])
        self.assertGreater(loss, 0.0)

    def test_track_indexer_metrics_denominator_over_the_enum(self):
        """The cross-repo API now takes the enum string, and only ``mqa_dsa``
        adds the ``-2`` layers to the denominator.

        ``_prod_csa_ratios`` has no CSA layer (``1 < ratio < 128``) at all, so
        the ``-2`` contribution is the whole denominator: ``mqa_dsa`` gives 6
        (five backbone + the MTP layer), the other two modes give 0, which is
        the "no indexer anywhere" path that clears the tracker and reports
        nothing.
        """
        ratios = _prod_csa_ratios()
        total = 0.0
        for mode, expect in (
            ("mqa_dsa", 6),
            ("mqa_full_causal", None),
            ("mha", None),
        ):
            with self.subTest(mode=mode):
                DSAIndexerLossLoggingHelper.tracker.clear()
                values = paddle.zeros([44])
                for slot in _PROD_MINUS2:
                    values[slot] = 1.0
                DSAIndexerLossLoggingHelper.tracker["values"] = values
                total = float(values.sum())
                sink = {}
                DSAIndexerLossLoggingHelper.track_indexer_metrics(
                    loss_scale=1.0,
                    iteration=0,
                    total_loss_dict=sink,
                    num_layers=44,
                    csa_compress_ratios=ratios,
                    hybrid_mla_attention=mode,
                )
                got = sink.get("indexer loss")
                got = None if got is None else float(got)
                print(f"[track_indexer_metrics {mode}] sum={total} avg={got}")
                if expect is None:
                    self.assertIsNone(got)
                else:
                    self.assertEqual(len(_PROD_MINUS2), expect)
                    self.assertAlmostEqual(got, total / expect, places=6)
        self.assertEqual(total, 6.0)


def ref_rope_interleaved(x, pos, base):
    """GPT-J / interleaved layout: pair channel 2j with 2j+1 in place."""
    d = x.shape[-1]
    ang = ref_angles(np.array([pos]), d, base)[0]
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=np.float64)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = b * cos + a * sin
    return out


class _WeightExposingLinear(BiasedLinear):
    """``BiasedLinear`` plus the ``.weight`` the absorption reads off
    ``kv_b_proj``."""

    @property
    def weight(self):
        return self.linear.weight


_MLA_SPEC = MLASelfAttentionSublayersSpec(
    q_proj=BiasedLinear,
    q_a_proj=BiasedLinear,
    q_b_proj=BiasedLinear,
    kv_a_proj_with_mqa=BiasedLinear,
    kv_b_proj=_WeightExposingLinear,
    core_attention=DotProductAttention,
    o_proj=BiasedLinear,
    q_a_layernorm=LayerNormStub,
    kv_a_layernorm=LayerNormStub,
)


@_GPU
class TestWarmupRope(unittest.TestCase):
    """The phase-2 switch must not reach RoPE, anywhere.

    Two independent RoPE users live on a ``-2`` layer: ``MLASelfAttention``
    rotates q/k *before* dispatching on ``mqa_latent``, and the DSA indexer
    keeps its own plain RoPE. The switch changes neither; every assertion below
    is against the fp64 reference of ``test_hybrid_mla_rope_audit``, never
    against
    the implementation.
    """

    def _mla(self, mode, use_sparse_loss=True, rope_fusion=False):
        config = _create_mqa_config(mode, sparse_loss=use_sparse_loss)
        # The shared fixture pins fusion off (``hybrid_mla_utils.py:214``);
        # the production phase-1/phase-2 yamls turn it on, so one test needs
        # to opt back in.
        config.apply_rope_fusion = rope_fusion
        paddle.seed(123)
        return MLASelfAttention(
            config=config, sublayers_spec=_MLA_SPEC, layer_number=1
        )

    def test_main_attention_rope_is_untouched_by_the_phase_switch(self):
        """warmup q/k == the ``mha`` q/k exactly, and == phase 3's rope block.

        ``get_query_key_value_tensors`` is where the rotation happens. Sharing
        one ``state_dict`` across the three modes (the key sets are identical --
        that is the whole point of activation-level absorption) makes the
        comparison meaningful, and the result must be bit-equality, since the
        switch is not read on this code path at all.

        Phase 2 is now *non*-latent (``latent_mqa_enabled`` is False for
        ``mqa_dsa`` + ``sparse_loss=False``), which strengthens the first half
        of this test from "the rope sub-block matches" to "the whole q and k
        match phase 1's"; the cross-layout rope-sub-block comparison moves to
        warmup-vs-phase-3, where the shapes genuinely differ.
        """
        mha = self._mla("mha")
        warm = self._mla("mqa_dsa", use_sparse_loss=False)
        ph3 = self._mla("mqa_dsa", use_sparse_loss=True)
        self.assertEqual(set(mha.state_dict()), set(warm.state_dict()))
        self.assertEqual(set(mha.state_dict()), set(ph3.state_dict()))
        state = mha.state_dict()
        warm.set_state_dict(state)
        ph3.set_state_dict(state)
        self.assertFalse(mha.mqa_latent)
        self.assertFalse(warm.mqa_latent)
        self.assertTrue(ph3.mqa_latent)

        paddle.seed(7)
        hidden = paddle.randn([1, 64, HIDDEN]) * 0.5
        out = {}
        for name, module in (("mha", mha), ("warm", warm), ("ph3", ph3)):
            module.eval()
            query, key = module.get_query_key_value_tensors(hidden)[:2]
            out[name] = (query, key)

        rope_dim = mha.config.hybrid_mla_qk_rope_head_dim
        pairs = {
            # Phase 2 runs phase 1's dense path, so this is full-tensor
            # equality, not a sub-block one.
            "mha_vs_warm_q": (out["mha"][0], out["warm"][0]),
            "mha_vs_warm_k": (out["mha"][1], out["warm"][1]),
            # The latent q/k carry the rope block in their trailing dims; the
            # nope halves differ in shape between dense and latent, the rope
            # sub-block does not. The dense path keeps K per head, the latent
            # path keeps the single shared head, so take head 0 on both.
            "warm_vs_ph3_q_pe": (
                out["warm"][0][..., -rope_dim:],
                out["ph3"][0][..., -rope_dim:],
            ),
            "warm_vs_ph3_k_pe": (
                out["warm"][1][:, :, :1, -rope_dim:],
                out["ph3"][1][:, :, :1, -rope_dim:],
            ),
        }
        measured = {}
        for name, (a, b) in pairs.items():
            measured[name] = float((a - b).abs().max())
        print(f"[warmup rope main] rope_dim={rope_dim} maxabs={measured}")
        for name, (a, b) in pairs.items():
            np.testing.assert_array_equal(
                a.numpy(), b.numpy(), err_msg=f"{name} moved"
            )

    def test_warmup_matches_phase1_under_fused_rope(self):
        """The same q/k equality, but with ``apply_rope_fusion`` actually on.

        The fixture family runs with fusion off
        (``hybrid_mla_utils.py:214``), so without this the "phase 2 == phase 1"
        claim was only ever measured on the eager kernel -- while the production
        phase-2 yaml sets ``apply_rope_fusion: true`` to match the baseline. The
        two are mathematically equal and *not* bit-identical, so a fixture that
        silently kept eager could not have caught a production config where one
        phase fuses and the other does not (which is exactly what the phase-2
        yaml carried until this rework: ``false`` against the baseline's
        ``true``).

        Phase 3 is deliberately absent here: it is latent, so it downgrades
        itself to eager (``test_apply_rope_fusion_downgrades_on_latent_mqa``)
        and a bitwise comparison against a fused tensor would be meaningless.
        """
        mha = self._mla("mha", rope_fusion=True)
        warm = self._mla("mqa_dsa", use_sparse_loss=False, rope_fusion=True)
        self.assertFalse(mha.mqa_latent)
        self.assertFalse(warm.mqa_latent)
        self.assertTrue(mha.config.apply_rope_fusion)
        self.assertTrue(warm.config.apply_rope_fusion)
        warm.set_state_dict(mha.state_dict())

        paddle.seed(7)
        hidden = paddle.randn([1, 64, HIDDEN]) * 0.5
        # Training mode is mandatory here: the fused MLA RoPE path raises
        # ``NotImplementedError: apply_rope_fusion does not support dynamic
        # inference yet`` under ``eval()``
        # (``multi_latent_attention.py:1808-1812``). Training is the only mode
        # the three phases are ever run in, so that gap is out of scope.
        self.assertTrue(mha.training and warm.training)
        q_ref, k_ref = mha.get_query_key_value_tensors(hidden)[:2]
        q_got, k_got = warm.get_query_key_value_tensors(hidden)[:2]
        print(
            "[warmup rope fused] q maxabs="
            f"{float((q_got - q_ref).abs().max()):.3e} k maxabs="
            f"{float((k_got - k_ref).abs().max()):.3e}"
        )
        np.testing.assert_array_equal(
            q_got.numpy(), q_ref.numpy(), err_msg="fused q moved"
        )
        np.testing.assert_array_equal(
            k_got.numpy(), k_ref.numpy(), err_msg="fused k moved"
        )

    def test_indexer_rope_is_plain_and_correct_in_warmup(self):
        """The indexer's own RoPE, in warmup, against the fp64 reference.

        Reached off the dense phase-2 backend now (``_build_module`` returns an
        ``MHADSAWarmupAttention``), but it is the very same ``DSAIndexer``
        instance: both phases build it from ``_indexer_layer_spec`` and call it
        through ``HybridMLAIndexerMixin._indexer_projections``.

        Three things at once: the frequency table is plain RoPE (base 10000, not
        the compressed layers' YaRN / ``csa_compress_rotary_base``), the
        ``dsa_indexer_rotary_interleaved`` layout switch is live (each setting
        matches its own reference and *not* the other one -- the cross error is
        the self-calibration), and the ``nope`` tail is left bit-untouched.
        """
        seqlen = 16
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                config = _warmup_config()
                config.dsa_indexer_rotary_interleaved = interleaved
                paddle.seed(99)
                indexer = _build_module(config, bf16=False).indexer
                self.assertIsNotNone(indexer)

                inv = np.asarray(
                    indexer.rotary_pos_emb.inv_freq.astype("float64").numpy()
                )
                inv_err = float(
                    np.max(
                        np.abs(
                            inv
                            - ref_inv_freq(indexer.rope_head_dim, ROPE_THETA)
                        )
                    )
                )

                rng = np.random.default_rng(3)
                # fp32 round-tripped up front, so "untouched" can be asserted
                # bit-exactly rather than against an fp64 draw the kernel never
                # saw.
                x = (
                    rng.standard_normal((1, seqlen, 1, indexer.head_dim))
                    .astype("float32")
                    .astype("float64")
                )
                freqs = indexer.rotary_pos_emb(seqlen, packed_seq=False)
                got = np.asarray(
                    indexer._apply_rope(
                        paddle.to_tensor(x.astype("float32")), freqs, 1.0
                    ).numpy(),
                    dtype=np.float64,
                )
                rope_dim = indexer.rope_head_dim
                nope_err = float(
                    np.max(np.abs(got[..., rope_dim:] - x[..., rope_dim:]))
                )
                pe = x[..., :rope_dim]
                own = (
                    ref_rope_interleaved if interleaved else ref_rope_halfsplit
                )
                other = (
                    ref_rope_halfsplit if interleaved else ref_rope_interleaved
                )
                own_err = max(
                    float(
                        np.max(
                            np.abs(
                                got[0, p, 0, :rope_dim]
                                - own(pe[0, p, 0], p, ROPE_THETA)
                            )
                        )
                    )
                    for p in range(seqlen)
                )
                other_err = max(
                    float(
                        np.max(
                            np.abs(
                                got[0, p, 0, :rope_dim]
                                - other(pe[0, p, 0], p, ROPE_THETA)
                            )
                        )
                    )
                    for p in range(seqlen)
                )
                print(
                    f"[warmup rope indexer interleaved={interleaved}] "
                    f"inv_freq_err={inv_err:.3e} nope_untouched={nope_err!r} "
                    f"own_layout_err={own_err:.3e} "
                    f"other_layout_err={other_err:.3e}"
                )
                self.assertLess(inv_err, 1e-6, "not plain rope base 10000")
                self.assertEqual(nope_err, 0.0, "nope half was rotated")
                self.assertLess(own_err, 1e-5)
                self.assertGreater(other_err, 1e-2, "layout switch is not live")

    def test_indexer_rope_output_matches_phase_three_bitwise(self):
        """``forward_before_topk`` q/k are identical across the two phases.

        The indexer still runs in warmup; only the *consumption* of its ranking
        changes -- and, since the rework, the class that owns it: phase 2's
        ``MHADSAWarmupAttention`` and phase 3's ``MQALatentAttention`` build the
        same ``DSAIndexer`` under the same ``indexer.*`` names, which is what
        makes copying one ``state_dict`` across legal. The pre-top-k activations
        -- which is where all the RoPE is -- must then be bit-equal.
        """
        seqlen = 64
        warm = _build_module(_warmup_config(), bf16=True)
        ph3 = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01), bf16=True
        )
        self.assertIsInstance(warm, MHADSAWarmupAttention)
        self.assertTrue(ph3.indexer_use_sparse_loss)
        self.assertEqual(set(warm.state_dict()), set(ph3.state_dict()))
        ph3.set_state_dict(warm.state_dict())

        paddle.seed(11)
        x = (paddle.randn([1, seqlen, HIDDEN]) * 0.5).cast("bfloat16")
        qr = (paddle.randn([1, seqlen, Q_LORA]) * 0.5).cast("bfloat16")
        outs = []
        for module in (warm, ph3):
            module.eval()
            with paddle.no_grad():
                outs.append(module.indexer.forward_before_topk(x, qr))
        measured = []
        for a, b in zip(outs[0], outs[1]):
            if not isinstance(a, paddle.Tensor):
                self.assertEqual(a, b)
                continue
            measured.append(float((a - b).cast("float32").abs().max()))
            np.testing.assert_array_equal(
                a.cast("float32").numpy(), b.cast("float32").numpy()
            )
        print(f"[warmup rope indexer vs phase3] maxabs={measured}")
        self.assertTrue(measured)

    def test_apply_rope_fusion_downgrades_on_latent_mqa(self):
        """Latent MQA cannot use the fused MLA kernel: it needs the per-head K/V
        that absorption skips. Instead of erroring at construction, the layer now
        downgrades *itself* to eager RoPE and warns, so the non-latent HCA/CSA
        layers of the same model keep the global fusion. This checks:

          - both latent modes (``mqa_full_causal`` and the *sparse* pairing of
            ``mqa_dsa``) construct, stay ``mqa_latent=True`` and resolve the
            per-layer decision (``apply_rope_fusion and not mqa_latent``) to
            False (eager), emitting the downgrade warning;
          - the non-latent controls -- ``mha`` and, since the rework, phase 2
            (``mqa_dsa`` + ``sparse_loss=False``) -- keep fusion on and emit
            no warning, proving the downgrade is scoped to *absorption*, not to
            "``mqa_dsa`` breaks fusion". Phase 2 running on the dense path is
            exactly why it keeps it;
          - ``mqa_latent_rope_fusion=True`` is the opt-in alternate path: the
            latent layer constructs without the downgrade warning because it
            takes the fused rotate_half branch instead.
        """
        import paddlefleet.transformer.multi_latent_attention as _mla

        def _effective(module, config):
            # Mirrors the per-use-site decision in get_query_key_value_tensors.
            return bool(config.apply_rope_fusion) and not module.mqa_latent

        _DOWNGRADE = "has no effect on the RoPE"

        def _build(mode, sparse):
            config = _create_mqa_config(mode, sparse_loss=sparse)
            config.apply_rope_fusion = True
            with mock.patch.object(_mla.logger, "warning") as warn:
                module = MLASelfAttention(
                    config=config,
                    sublayers_spec=_MLA_SPEC,
                    layer_number=1,
                )
            warned = any(
                _DOWNGRADE in str(c.args[0]) for c in warn.call_args_list
            )
            return config, module, warned

        # ---- latent modes downgrade to eager and warn ----
        for mode, sparse in (("mqa", True), ("mqa_dsa", True)):
            with self.subTest(
                mode=mode, use_sparse_loss=sparse, path="downgrade"
            ):
                config, module, warned = _build(mode, sparse)
                self.assertTrue(module.mqa_latent)
                self.assertFalse(_effective(module, config))
                self.assertTrue(
                    warned, f"{mode}: expected the eager-downgrade warning"
                )

        # ---- non-latent controls keep the global fusion, silently ----
        for mode, sparse in (("mha", True), ("mqa_dsa", False)):
            with self.subTest(
                mode=mode, use_sparse_loss=sparse, path="enabled"
            ):
                config, module, warned = _build(mode, sparse)
                self.assertFalse(module.mqa_latent)
                self.assertTrue(_effective(module, config))
                self.assertFalse(
                    warned,
                    f"{mode}: a non-latent layer must not be downgraded",
                )

        # ---- opt-in fused rotate_half path: no downgrade warning ----
        for mode in ("mqa", "mqa_dsa"):
            with self.subTest(mode=mode, path="mqa_latent_rope_fusion"):
                config = _create_mqa_config(mode, sparse_loss=True)
                config.apply_rope_fusion = True
                config.mqa_latent_rope_fusion = True
                with mock.patch.object(_mla.logger, "warning") as warn:
                    module = MLASelfAttention(
                        config=config,
                        sublayers_spec=_MLA_SPEC,
                        layer_number=1,
                    )
                self.assertTrue(module.mqa_latent)
                self.assertFalse(
                    any(
                        _DOWNGRADE in str(c.args[0])
                        for c in warn.call_args_list
                    ),
                    f"{mode}: mqa_latent_rope_fusion must suppress the "
                    "eager-downgrade warning",
                )


@_GPU
class TestRecomputeInnerForwardBitIdentical(unittest.TestCase):
    """The forward *inside* backward must equal the forward outside it, bitwise.

    ``TestWarmupRecompute`` above compares recompute-on against recompute-off and
    compares the two index tables, which pins the sparsity pattern. It does not
    compare the two forwards' *outputs*, because paddle discards the first one --
    so a divergence in R2's activations that happened to leave the column set
    alone would go unnoticed and be differentiated silently.

    Here both returns are kept and compared. ``maxabs == 0.0`` is the expected
    result, not a tolerance: the recomputed forward re-executes the same kernels
    on the same saved inputs, and the index table is an integer function of the
    document bounds. The captured-call count is asserted first, otherwise a
    single-forward implementation would make the comparison vacuous.

    The output comparison covers all three ``hybrid_mla_attention`` shapes; the
    *index table* comparison only applies to the two latent ones, because the
    phase-2 backend has no index table. Its analogue is the KL target, which
    exists once per recomputed step (the no-grad forward attaches no loss), so
    for phase 2 the invariant is a count, not a diff.
    """

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    # ``mqa`` -> "mqa_full_causal"; ``("mqa_dsa", True)`` is phase 3 (narrow
    # loss, top-k attention). Both are latent MQA and keep the latent call
    # shape. ``("mqa_dsa", False)`` -- phase 2 -- is dense MHA and is listed
    # apart because only the output half of this class applies to it.
    _LATENT_MODES = (("mqa_dsa", True), ("mqa", None))
    _WARMUP_MODE = ("mqa_dsa", False)
    _MODES = (_WARMUP_MODE, *_LATENT_MODES)
    # 256 saturates window+topk, 512 does not -- see the module docstring of
    # ``test_hybrid_mla_warmup_doc_mask_loss``.
    _SHAPES = ((SEQLEN, TWO_DOCS), (512, [200, 312]))

    def _module(self, mode, sparse_loss):
        """``sparse_loss`` has to reach ``_create_mqa_config`` as a kwarg.

        It selects the attention backend through ``latent_mqa_enabled``, so
        assigning ``config.dsa_indexer_use_sparse_loss`` after construction
        would build the wrong class.
        """
        kwargs = {} if sparse_loss is None else {"sparse_loss": sparse_loss}
        config = _create_mqa_config(mode, loss_coeff=0.01, **kwargs)
        config.pad_token_id = 0
        return _build_module(config, bf16=True)

    def _capture_two_forwards(self, module, seqlen, layout):
        """Run one recomputed train step; return both forwards' outputs.

        The call shape follows the backend the production dispatch picked
        (``latent_mqa_enabled`` inside ``_build_module``): the latent rows keep
        the absorbed signature, phase 2 gets the per-head dense one.
        """
        from paddle.distributed.fleet.utils import recompute

        dense = isinstance(module, MHADSAWarmupAttention)
        input_ids = paddle.ones([1, seqlen], dtype="int64")
        row_end = _row_end(layout, seqlen)
        if dense:
            query, key, value, x, qr = _make_dense_inputs(seqlen, seed=7)

            def _call(qin):
                return _dense_call(
                    module, qin, key, value, row_end, x, qr, input_ids
                )

        else:
            query, key, w_v, x, qr = _make_inputs(
                seqlen, seed=7, with_hidden=True
            )

            def _call(qin):
                return module(
                    qin,
                    key,
                    None,
                    None,
                    row_end,
                    v_b_proj_weight=w_v,
                    x=x,
                    qr=qr,
                    input_ids=input_ids,
                )

        module.train()
        module.clear_gradients()
        q = _leaf(query)

        with _ForwardSpy(module) as spy:
            out = recompute(_call, q)
            out.cast("float32").sum().backward()
        return spy.outputs

    def test_inner_forward_output_is_bit_identical(self):
        for mode, sparse_loss in self._MODES:
            for seqlen, layout in self._SHAPES:
                with self.subTest(mode=mode, sparse=sparse_loss, s=seqlen):
                    _CAPTURED.clear()
                    _WARMUP_TARGETS.clear()
                    DSAIndexerLossLoggingHelper.tracker.clear()
                    module = self._module(mode, sparse_loss)
                    outs = self._capture_two_forwards(module, seqlen, layout)
                    self.assertGreaterEqual(
                        len(outs),
                        2,
                        "recompute did not run a second forward, so this "
                        "comparison would be vacuous",
                    )
                    self.assertEqual(
                        float(np.abs(outs[0] - outs[1]).max()),
                        0.0,
                        "the forward inside backward differs from the outer "
                        "forward",
                    )

    def test_inner_forward_index_table_is_bit_identical(self):
        """Latent MQA only -- phase 2 has no index table to compare."""
        for mode, sparse_loss in self._LATENT_MODES:
            for seqlen, layout in self._SHAPES:
                with self.subTest(mode=mode, sparse=sparse_loss, s=seqlen):
                    _CAPTURED.clear()
                    DSAIndexerLossLoggingHelper.tracker.clear()
                    module = self._module(mode, sparse_loss)
                    self._capture_two_forwards(module, seqlen, layout)
                    self.assertGreaterEqual(
                        len(_CAPTURED), 2, "only one sparse-kernel call"
                    )
                    first, second = _CAPTURED[0], _CAPTURED[-1]
                    self.assertEqual(
                        int((first != second).sum()),
                        0,
                        "the recomputed forward selected different columns",
                    )

    def test_warmup_recompute_has_no_table_and_one_kl_target(self):
        """The phase-2 counterpart of the index-table comparison above.

        Two forwards happen, yet the block-sparse kernel is never reached and
        exactly one KL target is built: the no-grad forward is gated out by
        ``_needs_indexer_loss`` (``hybrid_mla_indexer.py:110-121``), so the
        recomputed step pays for the indexer once rather than twice.
        """
        mode, sparse_loss = self._WARMUP_MODE
        for seqlen, layout in self._SHAPES:
            with self.subTest(s=seqlen):
                _CAPTURED.clear()
                _WARMUP_TARGETS.clear()
                DSAIndexerLossLoggingHelper.tracker.clear()
                module = self._module(mode, sparse_loss)
                self.assertIsInstance(module, MHADSAWarmupAttention)
                outs = self._capture_two_forwards(module, seqlen, layout)
                self.assertGreaterEqual(len(outs), 2, "no second forward")
                self.assertEqual(
                    len(_CAPTURED),
                    0,
                    "the warmup phase reached the block-sparse kernel",
                )
                self.assertEqual(
                    len(_WARMUP_TARGETS),
                    1,
                    "the KL target was built on the no-grad forward too",
                )
                self.assertEqual(
                    tuple(_WARMUP_TARGETS[-1].shape), (1, seqlen, seqlen)
                )


if __name__ == "__main__":
    unittest.main()
