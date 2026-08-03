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

``un_absorbed_mqa=True`` turns the hybrid MLA (``csa_compress_ratios == -2``)
layers of a ``dsv4_hybrid`` model into :class:`MQALatentAttention`. The module
picks its path from the sublayers spec, not from any config string:

* ``MQALatentAttentionSublayersSpec(indexer=None)`` -- per-document full-causal
  dense attention on the latent, mathematically equal to MHA. In production the
  indexer is always built (``gpt_layer_specs`` when ``un_absorbed_mqa`` is set),
  so this indexer-less path exists only for the absorption-equivalence tests
  here, driven by constructing the layer directly with ``indexer=None``.
* an indexer spec -- forced local window + Lightning-indexer top-k, i.e. DSA on
  the KV latent.

Coverage:
  1. Guards -- unsupported configurations fail loudly (no GPU needed).
  2. Index construction over adversarial multi-document layouts: the forced
     128-window and the indexer candidate range are disjoint yet jointly equal
     the per-document causal set (no duplicate column, no lost window column).
  3. The indexer-less dense path equals a dense fp32 reference, because the
     activation-level absorption is exactly score preserving.
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
"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.transformer.csa_attention import (
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
)
from paddlefleet.transformer.dsa_attention import (
    DSAIndexer,
    DSAIndexerLossLoggingHelper,
    DSAIndexerSublayersSpec,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.mqa_latent_attention import (
    MQALatentAttention,
    MQALatentAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


# ---------------------------------------------------------------------------
# Stub layers (same pattern as test_dsa_attention.py)
# ---------------------------------------------------------------------------
class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        if x.dtype != self.linear.weight.dtype:
            x = x.cast(self.linear.weight.dtype)
        return self.linear(x), self.linear.bias


class LayerNormStub(paddle.nn.Layer):
    """LayerNorm stub accepting either ``hidden_size``/``eps`` naming."""

    def __init__(
        self,
        hidden_size=None,
        eps=None,
        normalized_shape=None,
        epsilon=None,
        **kwargs,
    ):
        super().__init__()
        size = hidden_size if hidden_size is not None else normalized_shape
        self.eps = (
            eps
            if eps is not None
            else (epsilon if epsilon is not None else 1e-5)
        )
        self.weight = paddle.nn.Parameter(paddle.ones([size]))
        self.bias = paddle.nn.Parameter(paddle.zeros([size]))

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = x.var(axis=-1, keepdim=True, unbiased=False)
        x = (x - mean) / paddle.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# Geometry. DK/DV are hard requirements of the FlashMLA sparse kernel
# (d_qk in {512, 576}, d_v == 512). K_CHANNELS is the *MHA* q_head_dim
# (qk_nope 192 + qk_rope 64): absorption preserves scores exactly, so the MHA
# softmax scale must be kept instead of the 576-wide latent one.
# INDEX_TOPK must stay a multiple of 128 (``indexer_backward_sm100`` asserts
# ``topk % block_I == 0``) and INDEX_HEADS >= 64 (``assert heads >= 64``).
# ---------------------------------------------------------------------------
H = 8
DK = 576
DV = 512
V_HEAD_DIM = 64
K_CHANNELS = 256
WINDOW = 128
INDEX_TOPK = 128
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
HIDDEN = 256
Q_LORA = 128

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


def _dsa_kernels_available():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        from paddlefleet.cudnn_ops.block_sparse_mqa_dsa import is_dsa_available

        return bool(is_dsa_available())
    except Exception:
        return False


_GPU = unittest.skipUnless(
    _dsa_kernels_available(),
    "requires SM100+ FlashMLA sparse fwd + cuDNN DSA bwd kernels",
)


def _create_mqa_config(mode="mqa", loss_coeff=0.0, num_hidden_layers=2):
    """dsv4_hybrid config for a ``csa_compress_ratios == -2`` layer.

    ``mode`` is a test-only convenience: both ``"mqa"`` (dense, indexer-less)
    and ``"mqa_dsa"`` (DSA) set ``un_absorbed_mqa=True`` on the config; the
    dense/sparse distinction is expressed by whether ``_build_module`` attaches
    an indexer to the sublayers spec, mirroring the production source which
    reads the layer path from the spec, not from a config string.

    Attributes are assigned after construction so that ``__post_init__``
    validation (exercised by the production model config, not by this unit) is
    bypassed -- same convention as ``test_dsa_attention.py``.
    """
    config = TransformerConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=HIDDEN,
        num_attention_heads=H,
    )
    config.num_key_value_heads = H
    config.head_dim = K_CHANNELS
    config.experimental_attention_variant = "dsv4_hybrid"
    config.un_absorbed_mqa = True
    # Test-only marker read by ``_build_module``: production always builds the
    # indexer when ``un_absorbed_mqa`` is set, so the indexer-less dense path is
    # reachable only by constructing the layer directly with
    # ``MQALatentAttentionSublayersSpec(indexer=None)``.
    config._build_dsa_indexer = mode == "mqa_dsa"
    config.hybrid_mla_q_lora_rank = Q_LORA
    config.hybrid_mla_kv_lora_rank = DV
    config.hybrid_mla_qk_nope_head_dim = 192
    config.hybrid_mla_qk_rope_head_dim = 64
    config.hybrid_mla_v_head_dim = V_HEAD_DIM
    config.hybrid_mla_num_attention_heads = H
    config.hybrid_mla_num_key_value_heads = H
    # The indexer dims are model-wide (HF json aliases index_n_heads /
    # index_head_dim / index_topk), shared with the CSA layers.
    config.dsa_index_n_heads = INDEX_HEADS
    config.dsa_index_head_dim = INDEX_HEAD_DIM
    config.dsa_index_topk = INDEX_TOPK
    config.csa_window_size = WINDOW
    config.dsa_indexer_loss_coeff = loss_coeff
    config.dsa_indexer_use_sparse_loss = True
    config.dsa_indexer_rotary_interleaved = False
    # The -2 layers are uncompressed, hence plain RoPE (base 10000); YaRN only
    # applies to the compressed HCA layers.
    config.rope_type = "rope"
    config.rope_theta = 10000.0
    config.rotary_interleaved = False
    config.rotary_percent = 1.0
    config.apply_rope_fusion = False
    config.num_nextn_predict_layers = 0
    config.mtp_num_layers = 0
    config.init_method = init_method_normal(0.02)
    config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
    config.rms_norm_eps = 1e-5
    config.context_parallel_size = 1
    config.sequence_parallel = False
    return config


_CAPTURED = []


class RecordingMQA(MQALatentAttention):
    """Captures the ``token_indices`` handed to the sparse kernel."""

    def _sparse_attn(self, query, kv, token_indices, sm_scale, d_v):
        _CAPTURED.append(token_indices.numpy().copy())
        return super()._sparse_attn(query, kv, token_indices, sm_scale, d_v)


def _build_module(config, layer_number=1, bf16=False, sink=None):
    indexer = None
    if getattr(config, "_build_dsa_indexer", False):
        indexer = LayerSpec(
            layer=DSAIndexer,
            sublayers_spec=DSAIndexerSublayersSpec(
                linear_wq_b=BiasedLinear,
                linear_wk=BiasedLinear,
                k_norm=LayerNormStub,
                linear_weights_proj=BiasedLinear,
            ),
            extra_kwargs={"is_hybrid_mla_indexer": True},
        )
    module = RecordingMQA(
        config=config,
        sublayers_spec=MQALatentAttentionSublayersSpec(indexer=indexer),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        k_channels=K_CHANNELS,
    )
    if bf16:
        # ``rotate_activation`` asserts bf16 inputs, so the indexer projections
        # must hold bf16 weights.
        module.to(dtype="bfloat16")
    if sink is not None:
        # In production ``MQALatentAttention.__init__`` builds this parameter
        # via the shared ``build_softmax_offset`` helper (name
        # ``core_attention.softmax_offset``, identical to the dense
        # ``DotProductAttention`` phase, so an MHA checkpoint stays loadable).
        # This unit uses a default config with no sink configured, so the
        # helper returns ``None``; inject the sink here instead. Created *after*
        # ``to(dtype=...)`` and in the module dtype: production uses
        # ``params_dtype`` (bf16), which is what the FA4 cute kernel of the
        # dense path requires, and the DSA path returns the sink gradient in the
        # parameter's own dtype.
        module.softmax_offset = module.create_parameter(
            shape=[H],
            dtype="bfloat16" if bf16 else "float32",
            default_initializer=paddle.nn.initializer.Assign(
                np.asarray(sink, dtype="float32")
            ),
        )
    return module


def _row_end(doc_lens, seqlen):
    """``[1, 1, s, 1]`` int32 exclusive per-token document end row."""
    out = np.empty([seqlen], dtype="int32")
    pos = 0
    for length in doc_lens:
        end = pos + length
        out[pos : min(end, seqlen)] = end
        pos = end
        if pos >= seqlen:
            break
    if pos < seqlen:
        out[pos:] = seqlen
    return paddle.to_tensor(out).reshape([1, 1, seqlen, 1])


def _doc_meta(row_end, seqlen):
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
    return doc_start.numpy(), is_valid.numpy()


def _make_inputs(seqlen, seed=0):
    paddle.seed(seed)
    query = (paddle.randn([1, seqlen, H, DK]) * 0.5).cast("bfloat16")
    key = (paddle.randn([1, seqlen, 1, DK]) * 0.5).cast("bfloat16")
    w_v = (paddle.randn([DV, H, V_HEAD_DIM]) * 0.05).cast("bfloat16")
    return query, key, w_v


def _rel(actual, expected):
    a = actual.cast("float32")
    e = expected.cast("float32")
    return float((a - e).norm() / e.norm().clip(min=1e-12))


def _dense_reference(query, key, w_v, row_end, scale, sink=None):
    """Per-document full-causal attention on the latent, computed in fp32.

    ``sink`` is a ``[H]`` per-head logit appended as one extra softmax column
    that carries no value vector, i.e. it only drains probability mass -- the
    exact semantics of ``attn_sink`` in the block-sparse kernel.
    """
    seqlen = int(query.shape[1])
    doc_start, is_valid = _doc_meta(row_end, seqlen)
    pos = np.arange(seqlen)
    allowed = (
        (pos[None, :] <= pos[:, None])
        & (pos[None, :] >= doc_start[:, None])
        & is_valid[:, None]
    )
    q = query[0].cast("float32")
    k = key.squeeze(2)[0].cast("float32")
    scores = paddle.einsum("shd,td->sht", q, k) * scale
    keep = paddle.to_tensor(allowed).unsqueeze(1)
    scores = paddle.where(keep, scores, paddle.full_like(scores, -1e30))
    if sink is None:
        probs = F.softmax(scores, axis=-1)
    else:
        sink_col = paddle.to_tensor(np.asarray(sink, dtype="float32")).reshape(
            [1, H, 1]
        )
        sink_col = paddle.expand(sink_col, [seqlen, H, 1])
        probs = F.softmax(paddle.concat([scores, sink_col], axis=-1), axis=-1)
        probs = probs[:, :, :seqlen]
    ctx = paddle.einsum("sht,tl->shl", probs, k[:, :DV])
    out = paddle.einsum("shl,lhv->shv", ctx, w_v.cast("float32"))
    row_ok = paddle.to_tensor(is_valid).cast("float32").reshape([seqlen, 1, 1])
    return (out * row_ok).reshape([1, seqlen, H * V_HEAD_DIM])


def _check_index_invariants(test, indices, row_end, seqlen, expect_full=False):
    """Assert the per-row column set is sound.

    Invariants: no duplicate column (a duplicate would double-count in the
    softmax), every column causal and inside the query's own document, the
    forced ``WINDOW`` columns always present, and pad rows select nothing.
    """
    doc_start, is_valid = _doc_meta(row_end, seqlen)
    for q in range(seqlen):
        cols = indices[0, q]
        cols = cols[cols >= 0].tolist()
        test.assertEqual(
            len(cols), len(set(cols)), f"row {q}: duplicate column"
        )
        if not is_valid[q]:
            test.assertEqual(cols, [], f"pad row {q} must select nothing")
            continue
        start = int(doc_start[q])
        test.assertTrue(
            all(start <= c <= q for c in cols),
            f"row {q}: non-causal or cross-document column",
        )
        window = set(range(max(start, q - WINDOW + 1), q + 1))
        test.assertEqual(
            window - set(cols), set(), f"row {q}: lost forced-window columns"
        )
        if expect_full:
            test.assertEqual(
                set(cols),
                set(range(start, q + 1)),
                f"row {q}: not the full causal set",
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

    def test_softmax_scale_is_the_mha_scale(self):
        # Absorption is exactly score preserving, so the scale must stay the MHA
        # q_head_dim one (256**-0.5), never the 576-wide latent one.
        self.assertAlmostEqual(
            self.module.softmax_scale, K_CHANNELS**-0.5, places=12
        )
        self.assertAlmostEqual(self.module.softmax_scale, 0.0625, places=12)


class TestMQAIndexRanges(unittest.TestCase):
    """The forced window and the indexer candidate range partition the causal
    set: no overlap (would double-count) and no gap (would waste budget)."""

    def setUp(self):
        self.module = _build_module(_create_mqa_config("mqa"))

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
                    seqlen, doc_start, doc_len, is_valid
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
    """The hybrid MLA config surface after the ``un_absorbed_mqa`` refactor.

    The old 3-state ``hybrid_mla_attn_mode`` and the ``hybrid_mla_attn_sink``
    switch (with its mutual-exclusion ValueError against the model-wide sinks)
    are gone. There is now a single boolean ``un_absorbed_mqa`` and a single
    sink switch (``add_full_attention_sink_bias`` / ``softmax_type``), so the
    two can no longer conflict. When ``un_absorbed_mqa=True`` the -2 layers run
    a cuDNN DSA indexer, so the config validates the model-wide ``dsa_index_*``
    fields (index_n_heads / index_head_dim / index_topk).
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
    def _un_absorbed_kwargs(cls, **overrides):
        # ``un_absorbed_mqa=True`` triggers the DSA-indexer validation, so a
        # valid baseline must carry the model-wide index dims.
        base = {
            "un_absorbed_mqa": True,
            "dsa_index_n_heads": INDEX_HEADS,
            "dsa_index_head_dim": INDEX_HEAD_DIM,
            "dsa_index_topk": INDEX_TOPK,
        }
        base.update(overrides)
        return cls._kwargs(**base)

    def test_un_absorbed_mqa_defaults_off(self):
        config = TransformerConfig(**self._kwargs())
        self.assertFalse(config.un_absorbed_mqa)

    def test_un_absorbed_mqa_true_accepted_with_valid_index_dims(self):
        config = TransformerConfig(**self._un_absorbed_kwargs())
        self.assertTrue(config.un_absorbed_mqa)
        self.assertEqual(config.dsa_index_head_dim, 128)

    def test_sink_coexists_with_un_absorbed_mqa(self):
        # The old mutual-exclusion ValueError is gone: one sink switch only, so
        # enabling a model-wide sink alongside un_absorbed_mqa must be accepted.
        for sink in (
            {"add_full_attention_sink_bias": True},
            {"softmax_type": "learnable"},
        ):
            with self.subTest(sink=sink):
                config = TransformerConfig(**self._un_absorbed_kwargs(**sink))
                self.assertTrue(config.un_absorbed_mqa)

    def test_index_head_dim_must_be_128(self):
        with self.assertRaisesRegex(ValueError, "index_head_dim"):
            TransformerConfig(**self._un_absorbed_kwargs(dsa_index_head_dim=64))

    def test_index_topk_must_be_multiple_of_128(self):
        with self.assertRaisesRegex(ValueError, "index_topk"):
            TransformerConfig(**self._un_absorbed_kwargs(dsa_index_topk=100))

    def test_index_topk_at_most_2048(self):
        with self.assertRaisesRegex(ValueError, "index_topk"):
            TransformerConfig(
                **self._un_absorbed_kwargs(dsa_index_topk=2048 + 128)
            )

    def test_index_dims_must_be_positive_ints(self):
        with self.assertRaisesRegex(ValueError, "index_n_heads"):
            TransformerConfig(
                **self._un_absorbed_kwargs(dsa_index_n_heads=None)
            )

    def test_index_dims_unvalidated_when_un_absorbed_mqa_off(self):
        # With the flag off the -2 layers are dense MHA; the indexer fields are
        # unused and left at their defaults without triggering the validation.
        config = TransformerConfig(**self._kwargs(un_absorbed_mqa=False))
        self.assertFalse(config.un_absorbed_mqa)


@_GPU
class TestMQAEquivalence(unittest.TestCase):
    """The indexer-less dense path is mathematically identical to MHA."""

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
        self.assertLess(_rel(out, ref), 5e-3)

    def test_multi_document_layouts_match_dense(self):
        seqlen = 256
        for layout in _LAYOUTS:
            with self.subTest(layout=layout):
                _CAPTURED.clear()
                out, ref = self._run(seqlen, layout)
                self.assertLess(_rel(out, ref), 5e-3)
                _check_index_invariants(
                    self,
                    _CAPTURED[-1],
                    _row_end(layout, seqlen),
                    seqlen,
                    expect_full=True,
                )

    def test_packed_equals_independent_per_document_runs(self):
        """The core correctness requirement: packing must be a no-op."""
        seqlen, layout = 256, [40, 88, 128]
        query, key, w_v = _make_inputs(seqlen)
        packed = (
            self.module(
                query,
                key,
                None,
                None,
                _row_end(layout, seqlen),
                v_b_proj_weight=w_v,
            )
            .cast("float32")
            .numpy()
        )
        pos = 0
        for length in layout:
            piece = (
                self.module(
                    query[:, pos : pos + length].contiguous(),
                    key[:, pos : pos + length].contiguous(),
                    None,
                    None,
                    _row_end([length], length),
                    v_b_proj_weight=w_v,
                )
                .cast("float32")
                .numpy()
            )
            diff = float(np.abs(packed[:, pos : pos + length] - piece).max())
            self.assertEqual(diff, 0.0, f"document at {pos} differs by {diff}")
            pos += length

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
        self.assertLess(_rel(out, ref), 5e-3)
        # ... and the sink genuinely changed the result: a positive logit drains
        # probability mass, so the sinkless reference is far away.
        sinkless = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale
        )
        self.assertGreater(_rel(sinkless, ref), 5e-2)


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
        query, key, w_v = _make_inputs(seqlen, seed=seed)
        x = (paddle.randn([1, seqlen, HIDDEN]) * 0.5).cast("bfloat16")
        qr = (paddle.randn([1, seqlen, Q_LORA]) * 0.5).cast("bfloat16")
        return query, key, w_v, x, qr

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
                self.assertLess(_rel(out, ref), 5e-3)
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
        self.assertLess(_rel(out, ref), 5e-3)
        sinkless = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale
        )
        self.assertGreater(_rel(sinkless, ref), 5e-2)

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


if __name__ == "__main__":
    unittest.main()
