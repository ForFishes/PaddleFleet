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

"""Document masking and indexer-loss arithmetic of the hybrid-MLA WARMUP phase.

The warmup phase is ``hybrid_mla_attention="mqa_dsa"`` with
``dsa_indexer_use_sparse_loss=False``. It no longer runs the absorbed latent
MQA: with no top-k on either side the block-sparse kernel was being handed a
zero-sparsity ``[b, s, s]`` int32 index table and made to walk all ``s``
columns anyway, so the phase now runs *exactly* phase 1's dense attention --
``MHADSAWarmupAttention(HybridMLAIndexerMixin, DotProductAttention)``, which
delegates its whole attention half to ``super().forward`` -- with the indexer
bolted on: one ``paddlefleet.tilelang_ops.csa_indexer_topk_fwd`` call in
full-candidate mode (``ratio=1``, ``topk_effective=s_global``) and a KL against
the head-summed dense attention distribution
(``MHADSAWarmupAttention._dense_attn_target``).

That inverts the two central claims of the previous revision of this file.
Where it asserted "the attention index table is exactly the full per-document
causal set", the assertion is now "no index table is built at all and the
block-sparse kernel is never reached"; where it asserted "warmup output ==
``mqa_full_causal`` output bitwise", it is now "warmup output == phase-1 dense
output bitwise". Everything else -- mask semantics, the KL column set, the
reduction denominator, gradient health -- survives unchanged in intent, on the
dense per-head call shape.

What is proven here, and nowhere else:

* ``TestWarmupCandidateRange`` -- ``_indexer_valid_range(window=0)`` is exactly
  the per-document causal span ``[doc_start[i], i]``, on every layout below
  plus layouts with genuine pad rows and with documents shorter than the sparse
  phase's forced window. Pure integer arithmetic, no kernel, hence not GPU
  gated.
* ``TestWarmupIsPhase1Dense`` -- the output is **bit-identical** to a plain
  ``DotProductAttention`` built from the same config, ``_CAPTURED`` stays empty
  (no block-sparse call), and exactly one KL target of shape ``[1, s, s]`` is
  built per grad-enabled forward.
* ``TestWarmupCrossDocumentIsolation`` -- zero cross-document leakage, measured
  in the only form that stays exact on a dense kernel: replace every *other*
  document's K/V with noise and this document's output does not move one bit.
* ``TestWarmupPadRows`` -- rows with ``is_valid == False`` contribute nothing
  and receive nothing: output, ``dq`` and their KL rows are exactly zero.
* ``TestWarmupIndexerLossPrecision`` -- the KL column set (the whole causal
  set, with no cuDNN top-k call anywhere), the row mask coming from
  ``input_ids`` (not from the document metadata), the reduction denominator,
  and the claim that the target is the head-summed dense attention
  distribution.
* ``TestDenseAttnTarget`` -- ``_dense_attn_target`` against a naive
  plain-paddle/numpy reference, including an all-padding row and a non-identity
  column permutation. This is the one piece of phase-2 maths with no upstream
  counterpart, so it gets a direct reference check rather than only the
  end-to-end one above.
* ``TestWarmupGradHealth`` -- the indexer parameters, the detached indexer
  inputs, and the attention-side ``dq`` / ``dk`` / ``dv``.
* ``TestShortDocumentLayouts`` -- the layouts that starve the *sparse* phase's
  indexer cannot starve this one, and phase 2 still equals phase 1 on them.

Shared fixtures come from ``hybrid_mla_utils``.

Run::

    R=<erniebot checkout>
    PYTHONPATH=$R/third_party/PaddleFleet/src:$R/third_party/PaddleFormers \\
        CUDA_VISIBLE_DEVICES=5 FLAGS_selected_gpus=0 \\
        python -m pytest <this file> -q -p no:randomly
"""

import contextlib
import inspect
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlefleet.transformer.mha_dsa_warmup_attention as warmup_mod
from paddlefleet.transformer.csa_attention import _derive_csa_doc_boundaries
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.hybrid_mla_indexer import (
    HybridMLAIndexerMixin,
    latent_mqa_enabled,
)
from paddlefleet.transformer.mha_dsa_warmup_attention import (
    MHADSAWarmupAttention,
)

from .hybrid_mla_utils import (
    _CAPTURED,
    _GPU,
    _WARMUP_TARGETS,
    V_HEAD_DIM,
    WINDOW,
    H,
    _build_module,
    _build_phase1_dense_module,
    _create_mqa_config,
    _dense_mha_reference,
    _doc_meta,
    _flash_attn_version,
    _make_dense_inputs,
    _production_fa_version,
    _rel,
    _row_end,
)

_EPS = 1e-10  # mha_dsa_warmup_attention._EPS, the KL/renormalisation epsilon

# The required layouts, as ``(doc_lens, seqlen)``. ``_row_end`` turns any
# trailing gap into one more *valid* document, so "with a trailing gap" is still
# a well-formed multi-document layout here; genuine pad rows need
# ``_pad_row_end`` and live in ``_PAD_LAYOUTS``.
#
# The two ``seqlen=512`` entries are kept, with new justifications. They used to
# be the only shapes where an exact-column-set assertion could tell the warmup
# index table apart from the sparse phase's ``WINDOW + INDEX_TOPK == 128 + 128
# == 256`` budget. Phase 2 builds no index table at all now, so that
# discrimination is gone -- but 512 is still the only shape here that reaches
# two properties:
#
# * a row's causal length exceeds the sparse phase's forced ``WINDOW``, so
#   ``_indexer_valid_range(window=0)`` and ``window=WINDOW`` disagree on most
#   rows instead of only on the tail -- what ``TestWarmupCandidateRange``
#   measures;
# * ``_dense_attn_target`` runs **more than one row chunk**: ``chunk =
#   _TARGET_ROW_SLOTS // s_global == 131072 // 512 == 256 < 512``
#   (``mha_dsa_warmup_attention.py:399-400``), whereas at ``s = 256`` the chunk
#   is 512 and the loop body runs once. The ``paddle.concat`` of the parts and
#   the per-chunk row offsets are therefore only exercised at 512.
_LAYOUTS = [
    ([256], 256),  # one document spanning the whole buffer
    ([40, 216], 256),  # two documents, tiles the buffer
    ([100, 50, 106], 256),  # three documents, none a multiple of the window
    ([127, 65], 256),  # trailing gap -> a third (valid) document of 64
    ([512], 512),  # two target row chunks; causal length > WINDOW
    ([200, 312], 512),  # two target row chunks, multi-document
]

# Layouts whose trailing gap stays *outside* every document, i.e. real pad rows.
_PAD_LAYOUTS = [
    ([200], 256),  # 56 pad rows after a single document
    ([40, 88], 256),  # 128 pad rows after two documents
    ([100, 50, 60], 256),  # 46 pad rows after three documents
]

# Documents no longer than the sparse phase's forced window. ``cand_i =
# max(causal_len_i - WINDOW, 0)``, so with ``window=WINDOW`` every row of such a
# document has an *empty* candidate range -- the starvation that made the old
# warmup log a KL of exactly 0.0. Phase 2 passes ``window=0``, so the same
# layouts keep their full causal span; and since its attention half is phase 1's,
# they are no longer special for the attention either. Both claims are what
# ``TestShortDocumentLayouts`` keeps regressing.
_SHORT_DOC_LAYOUTS = [
    ([64, 64, 64, 64], 256, "every doc half the window: 256/256 starved"),
    ([128, 128], 256, "every doc exactly the window: 256/256 starved"),
    ([1] * 8 + [120, 128], 256, "single-token docs mixed with window-sized"),
    ([2, 3, 5, 7, 11, 100], 128, "prime tiny docs, all far below the window"),
    ([129, 127], 256, "one row past the window: 255/256 starved"),
    ([1] * 8 + [248], 256, "single-token docs then one long document"),
    ([255, 1], 256, "long document plus a single-token tail"),
    ([WINDOW], WINDOW, "s == csa_window_size: the whole buffer is starved"),
]


def _pad_row_end(doc_lens, seqlen):
    """``[1, 1, s, 1]`` int32 ``row_end`` that produces real pad rows.

    ``hybrid_mla_utils._row_end`` fills the trailing gap with ``seqlen``, which
    ``_derive_csa_doc_boundaries`` reads as one more document, so every row comes
    back ``is_valid``. Repeating the *last document's* end instead leaves
    ``doc_len_per_pos`` short of ``pos_in_doc`` for the tail, which is the
    ``is_valid == False`` state a packed batch's padding actually takes.
    """
    out = np.empty([seqlen], dtype="int32")
    pos, end = 0, 0
    for length in doc_lens:
        end = pos + length
        out[pos : min(end, seqlen)] = end
        pos = end
        if pos >= seqlen:
            break
    if pos < seqlen:
        out[pos:] = end
    return paddle.to_tensor(out).reshape([1, 1, seqlen, 1])


def _segments(row_end, seqlen):
    """``[(start, length), ...]`` for every document, from the production
    deriver, so a per-document perturbation hits exactly the span the kernel is
    supposed to keep isolated."""
    _, _, _, doc_lens, doc_starts = _derive_csa_doc_boundaries(row_end, seqlen)
    return list(zip(doc_starts.numpy().tolist(), doc_lens.numpy().tolist()))


def _valid_range(row_end, seqlen, window=0, position_offset=0, s_local=None):
    """``(valid_range, row_empty)`` as numpy, from the production mixin.

    ``HybridMLAIndexerMixin._indexer_valid_range`` touches no instance state --
    it is pure integer arithmetic on the document bounds -- so it is called
    unbound with ``None`` for ``self``. That keeps every candidate-range
    assertion kernel-free, which is what the old
    ``_build_full_causal_indices``-based assertions were: the phase-2 candidate
    *range* is the successor of the phase-2 index *table*, the table being what
    this change removed.
    """
    doc_start, doc_len, is_valid, _, _ = _derive_csa_doc_boundaries(
        row_end, seqlen
    )
    valid_range, row_empty = HybridMLAIndexerMixin._indexer_valid_range(
        None,
        seqlen,
        doc_start,
        doc_len,
        is_valid,
        window,
        position_offset,
        s_local,
    )
    return valid_range.numpy(), row_empty.numpy()


def _warmup_config(loss_coeff=0.01):
    """The phase-2 config: ``"mqa_dsa"`` with the phase switch off.

    ``sparse_loss=False`` is not just the loss width any more -- it is what
    ``hybrid_mla_indexer.latent_mqa_enabled`` reads to pick the *backend*
    (``hybrid_mla_indexer.py:56-59``), so this one kwarg is what makes the module
    below a dense ``MHADSAWarmupAttention`` rather than a latent
    ``MQALatentAttention``.
    """
    config = _create_mqa_config(
        "mqa_dsa", loss_coeff=loss_coeff, sparse_loss=False
    )
    config.pad_token_id = 0
    return config


def _warmup_module(loss_coeff=0.01, sink=None):
    """A phase-2 module, asserted to *be* the dense warmup backend.

    The old form of this assertion was ``module.indexer_use_sparse_loss is
    False``. That attribute belonged to ``MQALatentAttention``'s phase dispatch
    and ``MHADSAWarmupAttention`` has none -- the phase is now expressed by the
    class itself, so assert the class. ``latent_mqa_enabled`` is asserted too,
    because it is the production predicate that both this fixture's builder and
    ``gpt_layer_specs`` dispatch on (``gpt_layer_specs.py``,
    ``hybrid_mla_utils.py:305``).
    """
    config = _warmup_config(loss_coeff)
    assert not latent_mqa_enabled(config)
    module = _build_module(config, bf16=True, sink=sink)
    assert module.indexer is not None
    assert isinstance(module, MHADSAWarmupAttention), type(module)
    return module


# Positional signature of ``TileLangCSAIndexerLossAutoScaler.forward`` (minus
# ``ctx``) -- the PyLayer phase 2 attaches its loss through, imported into
# ``mha_dsa_warmup_attention`` from ``csa_attention``
# (``csa_attention.py:1204-1218``). The spy below binds by position, so a
# reordered signature would silently hand it the wrong tensors instead of
# failing. Assert the order rather than trust it.
_LOSS_ARGS = [
    "output",
    "target",
    "index_q",
    "weights",
    "index_k_comp",
    "topk_indices",
    "topk_probs",
    "loss_coeff",
    "indexer_backend",
    "num_rows_override",
    "loss_mask",
]


def _positional(columns, values=None):
    """Scatter a column-layout table back into position space.

    The tilelang indexer returns ``[b, s, width]`` tables in **column layout**:
    slot ``j`` of row ``i`` refers to token ``columns[i, j]``, ordered by
    descending score, with ``-1`` in the unused slots. That is not position
    order, so anything compared against a positional reference has to be
    scattered first. ``values is None`` returns the boolean "row ``i`` scored
    column ``c``" table.
    """
    b, s, width = columns.shape
    dtype = bool if values is None else values.dtype
    out = np.zeros([b, s, width], dtype=dtype)
    for batch in range(b):
        for row in range(s):
            cols = columns[batch, row]
            keep = cols >= 0
            live = cols[keep]
            assert live.size == 0 or int(live.max()) < width, (
                f"row {row}: column id {int(live.max())} outside the "
                f"{width}-wide position space"
            )
            out[batch, row, live] = (
                True if values is None else values[batch, row, keep]
            )
    return out


@contextlib.contextmanager
def _capture_loss_args():
    """Capture what the warmup KL is actually reduced over.

    One spy, on the ``TileLangCSAIndexerLossAutoScaler`` boundary phase 2
    attaches its loss at, because every observable is an argument of that single
    call: ``P`` (``topk_probs``, already softmaxed by the kernel), ``Q``
    (``target``, from ``_dense_attn_target``), the column ids, the row mask, the
    denominator and the coefficient. These are literally the tensors the backward
    (upstream's tilelang ``csa_indexer_bwd``) differentiates, so anything
    asserted here is what the gradient sees.

    Patched in ``mha_dsa_warmup_attention``'s namespace: phase 3 imports the
    *same* PyLayer into ``mqa_latent_attention``, so patching the shared
    definition would spy on both phases at once. Rebinding this module's name
    makes "the spy fired" mean "phase 2's code path ran", with no need for the
    old ``indexer_backend`` discriminator -- which is instead asserted, since
    ``"tilelang"`` is the only backend this phase may select
    (``mha_dsa_warmup_attention.py:350``).

    ``cap["probs"]`` / ``cap["target"]`` / ``cap["columns"]`` are in the kernel's
    column layout; ``cap["live"]`` and ``cap["dense_target"]`` are the
    position-space scatters. A sum over the last axis -- which is all the KL does
    -- is permutation invariant and may be taken on the column layout directly.
    """
    real = warmup_mod.TileLangCSAIndexerLossAutoScaler
    actual = [
        name
        for name in inspect.signature(real.forward).parameters
        if name != "ctx"
    ]
    assert actual == _LOSS_ARGS, (
        "TileLangCSAIndexerLossAutoScaler.forward was reordered: "
        f"{actual} != {_LOSS_ARGS}; the positional spy below would "
        "capture the wrong tensors"
    )
    cap = {}

    class _Spy:
        @staticmethod
        def apply(
            output,
            target,
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            topk_probs,
            loss_coeff=1.0,
            indexer_backend="tilelang",
            num_rows_override=None,
            loss_mask=None,
        ):
            cap["backend"] = indexer_backend
            cap["columns"] = topk_indices.numpy().copy()
            cap["probs"] = topk_probs.astype("float32").numpy().copy()
            cap["target"] = target.astype("float32").numpy().copy()
            cap["width"] = int(topk_indices.shape[-1])
            # ``loss_mask`` / ``num_rows_override`` are ``None`` when no
            # ``input_ids`` reached the layer -- the same unmasked branch
            # ``csa_attention`` takes -- so record that rather than assuming a
            # synthesised all-ones mask.
            cap["mask"] = (
                None
                if loss_mask is None
                else loss_mask.astype("float32").numpy().copy()
            )
            cap["num_rows"] = (
                None if num_rows_override is None else float(num_rows_override)
            )
            cap["coeff"] = float(loss_coeff)
            return real.apply(
                output,
                target,
                index_q,
                weights,
                index_k_comp,
                topk_indices,
                topk_probs,
                loss_coeff,
                indexer_backend,
                num_rows_override,
                loss_mask,
            )

    warmup_mod.TileLangCSAIndexerLossAutoScaler = _Spy
    try:
        yield cap
    finally:
        warmup_mod.TileLangCSAIndexerLossAutoScaler = real
        if cap:
            cap["live"] = _positional(cap["columns"])
            cap["dense_target"] = _positional(cap["columns"], cap["target"])


def _forward(module, tensors, row_end, training, input_ids=None):
    """One forward in the **dense per-head** call shape.

    ``tensors`` is ``(query, key, value, x, qr)`` from ``_make_dense_inputs``.
    Three things changed with the backend and all three are load-bearing:

    * a real per-head ``value`` replaces ``v_b_proj_weight``, which the dense
      path has no use for -- ``kv_b_proj`` ran before the core attention;
    * ``attn_mask_type`` must be passed explicitly, because
      ``DotProductAttention`` derives ``is_causal`` from it
      (``dot_product_attention.py:562``) and defaults it to ``None``, i.e. *not*
      causal. Omitting it silently tests a bidirectional attention;
    * ``input_ids`` is accepted only because ``MHADSAWarmupAttention`` declares
      ``accepts_input_ids`` and strips it before ``super().forward``
      (``mha_dsa_warmup_attention.py:175-210``).
    """
    module.train() if training else module.eval()
    query, key, value, x, qr = tensors
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


def _phase1_forward(module, tensors, row_end, training):
    """The same forward on a plain ``DotProductAttention`` (phase 1)."""
    module.train() if training else module.eval()
    query, key, value, _, _ = tensors
    return module(
        query,
        key,
        value,
        None,
        row_end,
        attn_mask_type=AttnMaskType.causal,
    )


def _leaves(seqlen, seed=1):
    """``[query, key, value, x, qr]`` as differentiable leaves."""
    tensors = list(_make_dense_inputs(seqlen, seed=seed))
    for tensor in tensors:
        tensor.stop_gradient = False
    return tensors


def _fp32(tensor):
    """bf16 -> fp32 numpy; the widening is exact, so bit equality survives."""
    return tensor.cast("float32").numpy()


class _TargetHost:
    """Minimal host for an unbound ``_dense_attn_target`` call.

    The method reads exactly one attribute of its host, ``self.softmax_scale``
    (``mha_dsa_warmup_attention.py:427``), and calls no other method, so it can
    be evaluated without building an attention -- which keeps
    ``TestDenseAttnTarget`` kernel-free and lets it use fp32 inputs instead of
    the bf16 the flashmask path requires.
    """

    def __init__(self, softmax_scale):
        self.softmax_scale = softmax_scale


def _dense_target(query, key, columns, doc_start, is_valid, scale, offset=0):
    """``_dense_attn_target`` under test, with the shapes it gets in production."""
    s_local, s_global = int(query.shape[1]), int(key.shape[1])
    return MHADSAWarmupAttention._dense_attn_target(
        _TargetHost(scale),
        query,
        key,
        columns,
        doc_start,
        is_valid,
        offset,
        s_local,
        s_global,
    )


def _dyadic(shape, seed):
    """fp32 values ``k / 8``, ``|k| <= 8`` -- exact in fp32 *and* in TF32.

    Both need 4 mantissa bits, their products 8, and a 256-term dot product of
    them stays a multiple of ``1/64`` below 256, i.e. under 15 bits. So the
    matmul inside ``_dense_attn_target`` is exact whatever the matmul precision
    flag is, and the only residual against a float64 reference is the fp32
    softmax. Random bf16 inputs would instead bury the comparison under a 1e-3
    rounding floor.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(-8, 9, size=shape).astype("float32") / 8.0


def _fake_columns(
    doc_start, is_valid, s_local, s_global, offset=0, reverse=False
):
    """A stand-in for the indexer's ``[1, s_local, s_global]`` column table.

    Same contract as the kernel's output: the live slots of a row are its causal
    candidate set, left-packed, the rest ``-1``, and an invalid row is all
    ``-1``. ``reverse=True`` reverses the live run, which is the only way to
    catch a ``_dense_attn_target`` that forgot the ``take_along_axis``
    permutation -- with the identity order, "permuted" and "not permuted" are the
    same array.
    """
    cols = np.full([s_local, s_global], -1, dtype="int32")
    for local_row in range(s_local):
        row = offset + local_row
        if not bool(is_valid[row]):
            continue
        live = list(range(int(doc_start[row]), row + 1))
        if reverse:
            live.reverse()
        cols[local_row, : len(live)] = live
    return paddle.to_tensor(cols).unsqueeze(0)


def _naive_target(query, key, columns, doc_start, is_valid, scale, offset=0):
    """float64 loop reference for ``_dense_attn_target``.

    Deliberately the slowest possible spelling of the docstring's claim: for each
    query row, score its causal columns one head at a time, softmax that head in
    float64, sum the heads, L1-normalise, then place the result at the slot
    ``columns`` names. No chunking, no masking-by-``-inf``, no vectorisation, so
    it shares no structure with the implementation.
    """
    q = query.numpy()[0].astype("float64")
    k = key.numpy()[0].astype("float64")
    cols_np = columns.numpy()[0]
    s_local, heads = q.shape[0], q.shape[1]
    s_global = k.shape[0]
    out = np.zeros([s_local, s_global], dtype="float64")
    for local_row in range(s_local):
        row = offset + local_row
        if not bool(is_valid[row]):
            continue
        live = list(range(int(doc_start[row]), row + 1))
        acc = np.zeros([s_global], dtype="float64")
        for head in range(heads):
            logits = np.array(
                [
                    float(q[local_row, head] @ k[col, head]) * scale
                    for col in live
                ]
            )
            probs = np.exp(logits - logits.max())
            acc[live] += probs / probs.sum()
        out[local_row] = acc / max(acc.sum(), _EPS)
    permuted = np.zeros_like(out)
    for local_row in range(s_local):
        slots = cols_np[local_row]
        keep = slots >= 0
        permuted[local_row, keep] = out[local_row, slots[keep]]
    return permuted


class TestWarmupCandidateRange(unittest.TestCase):
    """The warmup candidate range is exactly the per-document causal span.

    This is the successor of ``TestWarmupFullCausalTable``. That class asserted
    that row ``i`` of the phase-2 ``[b, s, s]`` *index table* held exactly
    ``[doc_start[i], i]``; there is no index table any more, and the surviving
    carrier of the same claim is the ``valid_range`` phase 2 hands the indexer
    kernel -- ``_indexer_valid_range(..., window=0)``, whose two columns are the
    inclusive-exclusive bounds of that identical set
    (``hybrid_mla_indexer.py:174-183``). Still pure integer arithmetic on the
    document bounds, no kernel and no float, hence not GPU gated.
    """

    def _assert_range(self, row_end, seqlen):
        doc_start, is_valid = _doc_meta(row_end, seqlen)
        valid_range, row_empty = _valid_range(row_end, seqlen, window=0)
        self.assertEqual(list(valid_range.shape), [1, seqlen, 2])
        self.assertEqual(list(row_empty.shape), [1, seqlen, 1])
        valid = is_valid.astype(bool)
        for row in range(seqlen):
            low, high = valid_range[0, row].tolist()
            if valid[row]:
                self.assertEqual(
                    [low, high],
                    [int(doc_start[row]), row + 1],
                    f"row {row}: candidate range is not [doc_start, i]",
                )
            else:
                self.assertEqual(
                    high - low, 0, f"row {row}: a pad row has candidates"
                )
        # ``row_empty`` is what zeroes a row's columns and probs
        # (``mha_dsa_warmup_attention.py:301-306``), so it must be exactly the
        # pad-row predicate: with ``window=0`` every valid row keeps at least its
        # own diagonal, so no valid row may be reported empty.
        np.testing.assert_array_equal(
            row_empty.reshape(-1).astype(bool), ~valid
        )

    def test_candidate_range_all_layouts(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                self._assert_range(_row_end(layout, seqlen), seqlen)

    def test_candidate_range_with_pad_rows(self):
        for layout, seqlen in _PAD_LAYOUTS:
            with self.subTest(layout=layout, pad=True):
                row_end = _pad_row_end(layout, seqlen)
                _, is_valid = _doc_meta(row_end, seqlen)
                self.assertEqual(
                    int((~is_valid.astype(bool)).sum()),
                    seqlen - sum(layout),
                    "``_pad_row_end`` did not produce the pad rows",
                )
                self._assert_range(row_end, seqlen)

    def test_window_zero_is_what_saves_the_short_documents(self):
        """``window=0`` vs ``window=WINDOW`` on documents shorter than the window.

        The kernel-free half of the old
        ``test_window_length_sequence_still_trains_the_indexer_in_warmup``: the
        sparse phase clamps the candidate end a full ``csa_window_size`` before
        the diagonal, so on these layouts *every* row's range is empty and the KL
        is identically 0.0 -- a floor in the very phase where the indexer does all
        of its learning. Phase 2 passes ``0``
        (``mha_dsa_warmup_attention.py:283-291``) and keeps every row. Asserting
        both sides of the comparison here, rather than building a phase-3 module,
        keeps this file free of latent-MQA construction: ``MQALatentAttention``
        with an indexer and ``sparse_loss=False`` is now a hard error state
        (``mqa_latent_attention.py:253-288``), so a phase-3 control would only be
        exercising another suite's backend.
        """
        for layout, seqlen, note in _SHORT_DOC_LAYOUTS:
            with self.subTest(layout=layout, note=note):
                row_end = _row_end(layout, seqlen)
                _, empty_sparse = _valid_range(row_end, seqlen, window=WINDOW)
                _, empty_warmup = _valid_range(row_end, seqlen, window=0)
                self.assertGreater(
                    int(empty_sparse.sum()),
                    0,
                    f"{note}: the sparse window starves no row here",
                )
                self.assertEqual(
                    int(empty_warmup.sum()),
                    0,
                    f"{note}: warmup starved a row although window == 0",
                )
                self._assert_range(row_end, seqlen)

    def test_range_row_slice_matches_the_global_rows(self):
        """The ``s_local`` branch is a pure row slice of the global range.

        ``_indexer_valid_range`` builds over ``s_global`` then slices
        ``[position_offset : position_offset + s_local]``
        (``hybrid_mla_indexer.py:184-190``), which is what makes the returned
        bounds *global* token ids that the kernel's ``seq_offset`` causal bound
        can consume. Cheap to pin here and integer-exact.
        """
        seqlen, layout = 256, [40, 216]
        row_end = _row_end(layout, seqlen)
        full, full_empty = _valid_range(row_end, seqlen, window=0)
        half = seqlen // 2
        for offset in (0, half):
            with self.subTest(offset=offset):
                got, got_empty = _valid_range(
                    row_end, seqlen, 0, position_offset=offset, s_local=half
                )
                self.assertEqual(list(got.shape), [1, half, 2])
                np.testing.assert_array_equal(
                    got[0], full[0, offset : offset + half]
                )
                np.testing.assert_array_equal(
                    got_empty[0], full_empty[0, offset : offset + half]
                )


@_GPU
class TestWarmupIsPhase1Dense(unittest.TestCase):
    """Phase 2's attention half **is** phase 1's, and nothing else runs.

    The inversion of the old ``warmup == mqa_full_causal`` bit-identity claim.
    ``MHADSAWarmupAttention.forward`` forwards every attention-side argument to
    ``DotProductAttention.forward`` untouched and returns its output through a
    PyLayer that passes it through unchanged (``mha_dsa_warmup_attention.py:
    175-210``, ``csa_attention.py:1230-1239``), so anything but **bit equality**
    against a plain ``DotProductAttention`` means the warmup phase is no longer
    "phase 1 plus an indexer loss".

    Three things are asserted together, because each alone is satisfiable for the
    wrong reason:

    * bit equality with ``_build_phase1_dense_module``;
    * ``_CAPTURED`` empty -- no index table was built and the block-sparse kernel
      was never entered. This is the direct successor of the old
      ``[b, s, s]``-table assertions: there is nothing left to inspect, and the
      absence *is* the property (``hybrid_mla_utils.py:240-253``);
    * agreement with the fp32 per-document causal reference, which is what rules
      out "both modules are wrong in the same way" -- bit equality against a
      broken phase 1 would hold happily.
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

    def _compare(self, layout, seqlen, training):
        config = _warmup_config()
        warmup = _build_module(config, bf16=True)
        self.assertIsInstance(warmup, MHADSAWarmupAttention)
        phase1 = _build_phase1_dense_module(config, bf16=True)
        # Both are parameter-free on the attention side (q/k/v arrive already
        # projected and no sink is configured), so bit equality needs no weight
        # copy -- only the same ``softmax_scale``, which both derive from the same
        # config (``dot_product_attention.py:211-215``).
        self.assertEqual(warmup.softmax_scale, phase1.softmax_scale)

        tensors = _leaves(seqlen)
        row_end = _row_end(layout, seqlen)
        _CAPTURED.clear()
        _WARMUP_TARGETS.clear()
        got = _forward(
            warmup,
            tensors,
            row_end,
            training=training,
            input_ids=paddle.ones([1, seqlen], dtype="int64"),
        )
        want = _phase1_forward(phase1, tensors, row_end, training=training)
        self.assertEqual(list(got.shape), [1, seqlen, H * V_HEAD_DIM])
        self.assertEqual(
            float(np.abs(_fp32(got) - _fp32(want)).max()),
            0.0,
            f"{layout}: warmup output is not bit-identical to phase 1",
        )
        self.assertEqual(
            _CAPTURED, [], f"{layout}: the block-sparse kernel was reached"
        )
        return got, tensors, row_end

    def test_bit_identical_to_phase1_eval(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                self._compare(layout, seqlen, training=False)
                self.assertEqual(
                    _WARMUP_TARGETS,
                    [],
                    "eval built a KL target: the loss is not grad-gated",
                )

    def test_bit_identical_to_phase1_train(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                got, _, _ = self._compare(layout, seqlen, training=True)
                got.cast("float32").sum().backward()
                # Exactly one KL target per grad-enabled forward, spanning every
                # column: the loss is attached once, not per chunk and not twice
                # under the two forwards recompute would run.
                self.assertEqual(len(_WARMUP_TARGETS), 1)
                self.assertEqual(_WARMUP_TARGETS[0].shape, (1, seqlen, seqlen))

    def test_output_matches_the_fp32_per_document_causal_reference(self):
        """Both modules compute per-document causal MHA, not just the same thing.

        The residual is the bf16 flashmask kernel against an fp32 einsum
        reference; measured 1.969e-3 relative at ``s=256``, docs ``[100, 156]``.
        A leaked cross-document column or a lost row mask moves this by orders of
        magnitude, which is what makes a loose bound sufficient here -- the tight
        statement is the bit equality above.
        """
        seqlen, layout = 256, [100, 156]
        warmup = _warmup_module()
        tensors = _leaves(seqlen)
        row_end = _row_end(layout, seqlen)
        got = _forward(warmup, tensors, row_end, training=False)
        want = _dense_mha_reference(
            tensors[0], tensors[1], tensors[2], row_end, warmup.softmax_scale
        )
        rel = _rel(got, want)
        print(f"\n[warmup] dense output vs fp32 reference: rel={rel:.3e}")
        self.assertLess(rel, 1e-2)


@_GPU
class TestWarmupCrossDocumentIsolation(unittest.TestCase):
    """Zero cross-document leakage, in the only exact form a dense kernel allows.

    The old mechanism -- run each document alone and demand the packed output
    match bitwise -- was sound on the block-sparse kernel, where each query row
    reduces over its own listed columns and nothing else. It cannot survive here:
    a single-document rerun changes the sequence length, so flashmask picks
    different tiles and a different accumulation order, and bf16 reassociation
    makes the comparison approximate for reasons that have nothing to do with
    leakage.

    The intent survives at full strength with the shapes held fixed instead:
    replace every *other* document's K/V with noise and require this document's
    rows not to move one bit. Same claim ("this document's output is a function of
    this document only"), same exactness, and it is now the *masking* that is
    under test rather than the kernel's tiling.
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
        self.module = _warmup_module()

    def _worst_leak(self, layout, seqlen, training):
        """Worst per-document deviation, or ``None`` for a one-document layout."""
        row_end = _row_end(layout, seqlen)
        segments = _segments(row_end, seqlen)
        if len(segments) < 2:
            return None  # nothing to leak *from*
        tensors = _leaves(seqlen)
        DSAIndexerLossLoggingHelper.tracker.clear()
        base = _fp32(_forward(self.module, tensors, row_end, training=training))
        worst = 0.0
        for start, length in segments:
            sl = slice(start, start + length)
            noisy = list(tensors)
            paddle.seed(23)
            for idx in (1, 2):  # key, value
                other = paddle.randn(tensors[idx].shape).cast("bfloat16") * 4.0
                other[:, sl] = tensors[idx][:, sl]
                noisy[idx] = other
            DSAIndexerLossLoggingHelper.tracker.clear()
            perturbed = _fp32(
                _forward(self.module, noisy, row_end, training=training)
            )
            worst = max(
                worst, float(np.abs(base[:, sl] - perturbed[:, sl]).max())
            )
        return worst

    def _check_all(self, training):
        measured = 0
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                worst = self._worst_leak(layout, seqlen, training)
                if worst is None:
                    continue
                measured += 1
                self.assertEqual(
                    worst, 0.0, f"{layout}: cross-document leakage"
                )
        # The single-document layouts of ``_LAYOUTS`` have no other document to
        # leak from; the rest must all have been measured.
        self.assertEqual(measured, 4)

    def test_other_documents_cannot_move_this_one_eval(self):
        self._check_all(training=False)

    def test_other_documents_cannot_move_this_one_train(self):
        self._check_all(training=True)


@_GPU
class TestWarmupPadRows(unittest.TestCase):
    """Rows outside every document produce nothing and receive nothing.

    Same intent as before, on the new backend and with one more observable. The
    old carrier of "a pad row selected no column" was the index table's all-``-1``
    row; there is no table, so the KL's own column table takes over -- phase 2
    forces ``columns`` to ``-1`` and ``probs`` to 0 wherever ``row_empty``
    (``mha_dsa_warmup_attention.py:301-306``), and ``_dense_attn_target`` leaves
    those rows all-zero, so the pad rows contribute exactly 0 to the KL as well as
    to the attention output.

    Output and ``dq`` must be exactly zero: a non-zero output would inject
    padding into the residual stream and a non-zero ``dq`` would train on it.
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

    def _check(self, layout, seqlen, sink):
        module = _warmup_module(sink=sink)
        row_end = _pad_row_end(layout, seqlen)
        _, is_valid = _doc_meta(row_end, seqlen)
        pad = ~is_valid.astype(bool)
        self.assertTrue(pad.any(), f"{layout}: no pad row was produced")

        tensors = _leaves(seqlen)
        with _capture_loss_args() as cap:
            out = _forward(
                module,
                tensors,
                row_end,
                training=True,
                input_ids=paddle.ones([1, seqlen], dtype="int64"),
            )
        paddle.seed(7)
        upstream = paddle.randn([1, seqlen, H * V_HEAD_DIM]).cast("float32")
        (out.cast("float32") * upstream).sum().backward()

        self.assertTrue(
            bool((cap["columns"][0][pad] == -1).all()),
            f"{layout}: a pad row kept an indexer candidate",
        )
        self.assertEqual(
            float(np.abs(cap["target"][0][pad]).max()),
            0.0,
            f"{layout}: a pad row carries KL target mass",
        )
        self.assertEqual(
            float(np.abs(cap["probs"][0][pad]).max()),
            0.0,
            f"{layout}: a pad row carries indexer probability mass",
        )
        out_np = _fp32(out)[0]
        self.assertEqual(
            float(np.abs(out_np[pad]).max()),
            0.0,
            f"{layout}: pad row output is not exactly zero",
        )
        dq = _fp32(tensors[0].grad)[0]
        self.assertEqual(
            float(np.abs(dq[pad]).max()),
            0.0,
            f"{layout}: pad row dq is not exactly zero",
        )
        # The real rows must still be alive -- an all-zero output would satisfy
        # the assertions above for the wrong reason.
        self.assertGreater(float(np.abs(out_np[~pad]).max()), 0.0)
        self.assertGreater(float(np.abs(dq[~pad]).max()), 0.0)
        self.assertGreater(float(np.abs(cap["target"][0][~pad]).max()), 0.0)
        module.clear_gradients()

    def test_pad_rows_are_inert_sinkless(self):
        for layout, seqlen in _PAD_LAYOUTS:
            with self.subTest(layout=layout):
                self._check(layout, seqlen, sink=None)

    def test_pad_rows_are_inert_with_sink(self):
        """The sink is the one column masking cannot remove, so it keeps its own
        case -- under the production ``FLAGS_flash_attn_version``, and skipped
        where the installed flashmask cannot take a learnable sink.

        Phase 2 inherits phase 1's sink constraint deliberately ("phase 2 runs
        wherever phase 1 runs"): dense MLA with ``add_full_attention_sink_bias``
        requires ``FLAGS_flash_attn_version in (3, 4)``
        (``multi_latent_attention.py:584-614``), which a bare pytest process does
        not set, hence the ``_flash_attn_version`` pin. Beyond that, this
        fixture's head dims (``K_CHANNELS=256`` / ``V_HEAD_DIM=64``) are outside
        FA4's supported set, so ``get_fa_version`` downgrades to FA2
        (``flash_mask_facade.py:39-101``) where ``learnable_sink`` is only
        available if the installed kernel takes the kwarg
        (``flash_mask_facade.py:104-199``). That is a property of the environment
        rather than of this change, so it skips instead of failing.
        """
        sink = np.linspace(1.0, 3.0, H)
        with _flash_attn_version(_production_fa_version()):
            for layout, seqlen in _PAD_LAYOUTS:
                with self.subTest(layout=layout):
                    try:
                        self._check(layout, seqlen, sink=sink)
                    except NotImplementedError as exc:
                        self.skipTest(f"no learnable_sink support here: {exc}")


@_GPU
class TestWarmupIndexerLossPrecision(unittest.TestCase):
    """The warmup KL: column set, row mask, denominator, target.

    Every number below is read at the ``TileLangCSAIndexerLossAutoScaler``
    boundary, i.e. exactly what the backward differentiates, and cross-checked
    against an independent fp32 recomputation of the logged scalar. Unchanged in
    intent from the previous revision; what changed is that there is no attention
    index table to compare the KL's column set against, so the reference is the
    per-document causal predicate itself.
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

    def _step(self, module, seqlen, layout, input_ids=None, seed=1):
        """One training step; returns ``(logged_kl, captured)``."""
        tensors = _leaves(seqlen, seed=seed)
        _CAPTURED.clear()
        _WARMUP_TARGETS.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        with _capture_loss_args() as cap:
            out = _forward(
                module,
                tensors,
                _row_end(layout, seqlen),
                training=True,
                input_ids=input_ids,
            )
        out.cast("float32").sum().backward()
        logged = float(
            DSAIndexerLossLoggingHelper.tracker["values"]
            .astype("float32")
            .sum()
        )
        module.clear_gradients()
        # No index table exists in this phase, on any step.
        self.assertEqual(_CAPTURED, [])
        return logged, cap

    @staticmethod
    def _kl_per_row(cap):
        """fp32 recomputation of ``kl.sum(axis=-1)`` from the captured pair."""
        target, probs = cap["target"], cap["probs"]
        return (target * (np.log(target + _EPS) - np.log(probs + _EPS))).sum(
            axis=-1
        )

    def test_kl_spans_the_whole_causal_set_and_no_topk_runs(self):
        """The KL's live columns are the per-document causal set, from one
        full-candidate tilelang call and no cuDNN top-k.

        The previous revision asserted an equality between the KL's live columns
        and the *attention* index table, both being the full causal set. Phase 2
        has no attention table, so the reference is now the causal predicate
        itself -- which is the stronger of the two statements anyway: it does not
        depend on a second table being right.

        Two call counts pin *which* selector produced those columns: the cuDNN
        top-k kernel (phase 3's) is called zero times, and the tilelang indexer
        exactly once per step at ``topk_effective == s``, its documented
        full-candidate mode (``mha_dsa_warmup_attention.py:292-300``).
        """
        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod
        import paddlefleet.tilelang_ops as tl_mod

        module = _warmup_module()
        inner = fwd_mod.cudnn_indexer_topk_fwd
        inner_tl = tl_mod.csa_indexer_topk_fwd
        topk_calls = []
        tl_widths = []

        def recording(*args, **kwargs):
            topk_calls.append(1)
            return inner(*args, **kwargs)

        def recording_tl(*args, **kwargs):
            tl_widths.append(int(kwargs["topk_effective"]))
            return inner_tl(*args, **kwargs)

        fwd_mod.cudnn_indexer_topk_fwd = recording
        tl_mod.csa_indexer_topk_fwd = recording_tl
        try:
            for seqlen in (16, 128, 256, 300, 384, 512):
                with self.subTest(seqlen=seqlen):
                    before = len(tl_widths)
                    _, cap = self._step(module, seqlen, [seqlen])
                    # The KL table is exactly the causal span -- no rounding. The
                    # tilelang wrapper pads ``topk_effective`` up to its block
                    # internally and crops the result back
                    # (``csa_indexer_fwd.py:430-462``), so short and
                    # non-power-of-two lengths are served too; ``seqlen=16`` is
                    # below the block size on purpose.
                    self.assertEqual(cap["backend"], "tilelang")
                    self.assertEqual(cap["target"].shape[-1], seqlen)
                    self.assertEqual(cap["probs"].shape[-1], seqlen)
                    self.assertEqual(cap["width"], seqlen)
                    self.assertEqual(tl_widths[before:], [seqlen])
                    # One document over the whole buffer, so the causal set of
                    # row ``i`` is ``[0, i]`` -- a lower-triangular live table.
                    live = cap["live"][0]
                    expected = np.tril(np.ones([seqlen, seqlen], dtype=bool))
                    np.testing.assert_array_equal(live, expected)
        finally:
            fwd_mod.cudnn_indexer_topk_fwd = inner
            tl_mod.csa_indexer_topk_fwd = inner_tl
        self.assertEqual(topk_calls, [], "warmup called the cuDNN top-k kernel")

    def test_pad_tail_excluded_from_indexer_loss_warmup(self):
        """Warmup counterpart of
        ``test_hybrid_mla_doc_equivalence.TestIndexerLossPadMaskRequestW``.

        One document spans the whole buffer, so ``is_valid`` is all ``True`` and
        the document metadata cannot express the padding tail. Only
        ``input_ids != pad_token_id`` can, and it must drive both the KL sum and
        its denominator -- the ``num_rows_override`` the backward divides by
        (``TileLangCSAIndexerLossAutoScaler.forward``'s ``ctx.num_rows``).
        """
        seqlen, real_tokens = 256, 200
        module = _warmup_module()
        ids = np.zeros([1, seqlen], dtype="int64")
        ids[0, :real_tokens] = np.arange(1, real_tokens + 1)
        logged, cap = self._step(
            module, seqlen, [seqlen], input_ids=paddle.to_tensor(ids)
        )

        _, is_valid = _doc_meta(_row_end([seqlen], seqlen), seqlen)
        self.assertEqual(int(is_valid.astype("int32").sum()), seqlen)
        self.assertEqual(float(cap["mask"].sum()), float(real_tokens))
        self.assertEqual(cap["num_rows"], float(real_tokens))
        mask = cap["mask"].reshape(-1)
        self.assertTrue(bool((mask[:real_tokens] == 1).all()))
        self.assertTrue(bool((mask[real_tokens:] == 0).all()))

        kl_per_row = self._kl_per_row(cap)
        ref = (kl_per_row * cap["mask"]).sum() / real_tokens * cap["coeff"]
        self.assertLess(abs(logged - float(ref)) / abs(float(ref)), 1e-5)
        # Two discriminators, so this is not a tautology.
        # (1) the denominator: had ``B*Sq`` driven it, the scalar would be
        # 256/200 = 1.28x smaller. The full-candidate KL is the same order of
        # magnitude on every row, so comparing against the plain mean of the same
        # rows would separate nothing -- the denominator has to be pinned
        # directly.
        wrong_denominator = float(
            (kl_per_row * cap["mask"]).sum() / seqlen * cap["coeff"]
        )
        self.assertAlmostEqual(
            float(ref) / wrong_denominator, seqlen / real_tokens, delta=1e-6
        )
        self.assertGreater(abs(wrong_denominator - logged) / abs(logged), 0.2)
        # (2) the mask is not a no-op: the rows it drops carry real KL mass.
        dropped = float((kl_per_row * (1.0 - cap["mask"])).sum())
        self.assertGreater(dropped, 0.0)
        # Single card: cp_size == 1, so the coefficient reaches the backward
        # unscaled. The ``/cp_size`` branch is a multi-card concern.
        self.assertEqual(cap["coeff"], module.indexer_loss_coeff)

    def test_unset_pad_token_id_falls_back_to_zero(self):
        """``pad_token_id=None`` masks the same rows as ``0``; it is not fatal.

        ``TransformerConfig.from_config`` can copy a ``None`` straight out of an
        external/HF config, and every other consumer of the field in this
        repository folds it to ``0`` (``gpt_embedding.py:214-216``,
        ``mtp_embedding_layer.py:105-107``, ``moe_router.py:605-607`` and four
        more sites). The earlier revision of ``_indexer_loss_mask`` asserted
        ``is not None`` instead, which aborted such a run at its *first* indexer
        loss -- and under ``python -O``, where asserts are stripped, would have
        compared the ids against ``None`` and masked nothing. Flagged in the
        upstream review of PR #1721; this pins the fallback.
        """
        seqlen, real_tokens = 256, 200
        ids = np.zeros([1, seqlen], dtype="int64")
        ids[0, :real_tokens] = np.arange(1, real_tokens + 1)
        ids_t = paddle.to_tensor(ids)

        module = _warmup_module()
        module.config.pad_token_id = None
        mask_none, rows_none = module._indexer_loss_mask(ids_t, 1, seqlen)

        reference = _warmup_module()
        self.assertEqual(reference.config.pad_token_id, 0)
        mask_zero, rows_zero = reference._indexer_loss_mask(ids_t, 1, seqlen)

        self.assertEqual(rows_none, float(real_tokens))
        self.assertEqual(rows_none, rows_zero)
        np.testing.assert_array_equal(mask_none.numpy(), mask_zero.numpy())

        # And the whole loss path, not just the helper: the fallback has to reach
        # both the KL sum and the denominator the backward divides by.
        logged, cap = self._step(module, seqlen, [seqlen], input_ids=ids_t)
        self.assertEqual(float(cap["mask"].sum()), float(real_tokens))
        self.assertEqual(cap["num_rows"], float(real_tokens))
        kl_per_row = self._kl_per_row(cap)
        ref = float(
            (kl_per_row * cap["mask"]).sum() / real_tokens * cap["coeff"]
        )
        self.assertLess(abs(logged - ref) / abs(ref), 1e-5)

    def test_no_input_ids_uses_the_plain_row_mean(self):
        """Without ``input_ids`` the reduction is ``kl.mean() * coeff``.

        ``_indexer_loss_mask`` returns ``(None, None)``
        (``hybrid_mla_indexer.py:212-213``) and ``_attach_indexer_loss`` passes
        that straight down, which is the same unmasked branch
        ``csa_attention._compute_fused_indexer_target`` takes: the backward then
        falls back to the kernel's own ``1/(B*Sq)``, and only the *logged* scalar
        carries the ``/cp_size`` CP correction. So what to assert here is that
        both reach the backward as ``None`` -- a synthesised all-ones mask would
        silently change which denominator the gradient uses.
        """
        seqlen = 256
        module = _warmup_module()
        logged, cap = self._step(module, seqlen, [seqlen], input_ids=None)
        self.assertIsNone(cap["mask"])
        self.assertIsNone(cap["num_rows"])
        ref = float(self._kl_per_row(cap).mean() * cap["coeff"])
        self.assertLess(abs(logged - ref) / abs(ref), 1e-5)
        self.assertEqual(cap["coeff"], module.indexer_loss_coeff)
        self.assertEqual(
            module._indexer_loss_mask(None, 1, seqlen), (None, None)
        )

    def test_kl_target_is_normalised_and_zero_off_the_causal_set(self):
        """The KL target is L1-normalised per row and exactly zero on columns the
        per-document causal mask excludes.

        A row shorter than the ``s``-wide table comes back ``-1``-padded, and
        those dead slots are where the "excluded column" assertion lands.
        """
        seqlen, layout = 256, [40, 216]
        module = _warmup_module()
        _, cap = self._step(module, seqlen, layout)
        target = cap["target"][0]
        masked = cap["columns"][0] < 0
        self.assertTrue(masked.any(), "layout masks nothing")
        self.assertFalse(masked.all(), "layout masks everything")
        self.assertEqual(
            float(np.abs(target[masked]).max()),
            0.0,
            "a masked column carries probability mass",
        )
        self.assertGreaterEqual(float(target.min()), 0.0)
        self.assertLess(
            float(np.abs(target.sum(axis=-1) - 1.0).max()),
            1e-5,
            "target rows are not L1-normalised",
        )

    def test_kl_target_is_the_full_causal_attention_distribution(self):
        """The intended semantics, measured end to end.

        Both sides of the warmup KL span the whole per-document causal set, so the
        target is the head-summed *dense* attention distribution over all causal
        columns, L1-normalised. The reference below is an independent fp32
        recomputation on the per-head layout -- ``"shd,thd->sht"`` now that K is
        per-head rather than the shared latent -- so agreement is evidence about
        the semantics rather than a restatement of the code. The residual is the
        bf16 rounding of the inputs.
        """
        seqlen, layout = 256, [40, 216]
        module = _warmup_module()
        tensors = _leaves(seqlen)
        query, key = tensors[0], tensors[1]
        row_end = _row_end(layout, seqlen)
        with _capture_loss_args() as cap:
            out = _forward(
                module, tensors, row_end, training=True, input_ids=None
            )
        out.cast("float32").sum().backward()
        module.clear_gradients()

        doc_start, is_valid = _doc_meta(row_end, seqlen)
        positions = np.arange(seqlen)
        allowed = (
            (positions[None, :] <= positions[:, None])
            & (positions[None, :] >= doc_start[:, None])
            & is_valid.astype(bool)[:, None]
        )
        scores = (
            paddle.einsum(
                "shd,thd->sht",
                query.detach()[0].cast("float32"),
                key.detach()[0].cast("float32"),
            )
            * module.softmax_scale
        )
        scores = paddle.where(
            paddle.to_tensor(allowed).unsqueeze(1),
            scores,
            paddle.full_like(scores, -1e30),
        )
        head_sum = F.softmax(scores, axis=-1).sum(axis=1).numpy()
        reference = head_sum / np.maximum(
            head_sum.sum(axis=-1, keepdims=True), _EPS
        )
        # The mask the implementation used must be the same predicate. The kernel
        # emits columns score-descending, so the comparison is on the
        # position-space scatter, not on the raw column order.
        np.testing.assert_array_equal(cap["live"][0], allowed)

        got = cap["dense_target"][0]
        max_abs = float(np.abs(got - reference).max())
        norm_rel = float(
            np.linalg.norm(got - reference) / np.linalg.norm(reference)
        )
        print(
            f"\n[warmup] KL target vs full-causal head sum: "
            f"max_abs={max_abs:.3e} norm_rel={norm_rel:.3e}"
        )
        self.assertLess(norm_rel, 3e-2)
        self.assertLess(max_abs, 3e-2)

    def test_window_length_sequence_still_trains_the_indexer(self):
        """At ``s == csa_window_size`` the warmup KL is strictly positive.

        The kernel half of the old
        ``test_window_length_sequence_still_trains_the_indexer_in_warmup``: with
        ``window=0`` every row keeps its full causal span, so the KL is nonzero
        (measured 9.10e-05) at the very shape where the sparse phase's clamp
        empties every candidate range and logs exactly 0.0. The phase-3 control
        that used to sit here is now the kernel-free comparison in
        ``TestWarmupCandidateRange
        .test_window_zero_is_what_saves_the_short_documents``, because building an
        ``MQALatentAttention`` with ``sparse_loss=False`` is a hard error now
        (``mqa_latent_attention.py:253-288``).
        """
        module = _warmup_module()
        logged, cap = self._step(module, WINDOW, [WINDOW])
        self.assertGreater(logged, 0.0)
        self.assertEqual(cap["target"].shape[-1], WINDOW)
        self.assertGreater(float(np.abs(cap["target"]).max()), 0.0)
        # ... and it keeps scaling with the sequence, as before.
        logged_2w, cap_2w = self._step(module, 2 * WINDOW, [2 * WINDOW])
        self.assertGreater(logged_2w, 0.0)
        self.assertGreater(float(np.abs(cap_2w["target"]).max()), 0.0)


class TestDenseAttnTarget(unittest.TestCase):
    """``_dense_attn_target`` against a naive reference. NEW in this revision.

    The KL target builder is the one piece of phase-2 arithmetic with no upstream
    counterpart: phase 3 gathers the selected keys and scores them in the kernel's
    own column order, which the per-head layout makes impossible here (``[chunk,
    width, h, dk]`` is 3.2GB at production shapes), so this phase scores in
    *natural* column order and permutes afterwards
    (``mha_dsa_warmup_attention.py:380-460``). Everywhere else in this file the
    target is only checked end to end, where a wrong permutation and a wrong mask
    are hard to tell apart.

    The reference is a float64 triple loop -- one row, one head, one column at a
    time -- sharing no structure with the implementation: no chunking, no
    ``-inf`` masking, no vectorisation.

    Inputs are ``_dyadic`` fp32 rather than the production bf16, so the matmul is
    exact and the only residual is the fp32 softmax; a bf16 fixture would bury the
    comparison under a ~1e-3 rounding floor. Kernel-free, hence not GPU gated.
    """

    SEQ, HEADS, DK, SCALE = 24, 4, 8, 0.125
    DOCS, PAD_DOCS = [10, 14], [10, 8]

    def _case(self, row_end, seed=5, offset=0, s_local=None, reverse=False):
        """``(got, want, is_valid)`` for one configuration."""
        seqlen = self.SEQ
        rows = seqlen if s_local is None else s_local
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            row_end, seqlen
        )
        starts, valid = doc_start.numpy(), is_valid.numpy()
        query = paddle.to_tensor(_dyadic([1, rows, self.HEADS, self.DK], seed))
        key = paddle.to_tensor(
            _dyadic([1, seqlen, self.HEADS, self.DK], seed + 1)
        )
        columns = _fake_columns(
            starts, valid, rows, seqlen, offset=offset, reverse=reverse
        )
        got = _dense_target(
            query, key, columns, doc_start, is_valid, self.SCALE, offset
        )
        want = _naive_target(
            query, key, columns, starts, valid, self.SCALE, offset
        )
        self.assertEqual(list(got.shape), [1, rows, seqlen])
        return got.numpy()[0].astype("float64"), want, valid

    def _assert_close(self, got, want, note):
        max_abs = float(np.abs(got - want).max())
        self.assertLess(max_abs, 5e-6, f"{note}: max_abs={max_abs:.3e}")

    def test_matches_the_naive_reference(self):
        for label, row_end in (
            ("all rows valid", _row_end(self.DOCS, self.SEQ)),
            ("with pad rows", _pad_row_end(self.PAD_DOCS, self.SEQ)),
        ):
            for reverse in (False, True):
                with self.subTest(layout=label, reverse=reverse):
                    got, want, _ = self._case(row_end, reverse=reverse)
                    self._assert_close(got, want, f"{label}/{reverse}")

    def test_the_permutation_is_actually_applied(self):
        """A reversed column table must produce a reversed target row.

        Without this the identity-ordered comparison above would pass on an
        implementation that dropped the ``take_along_axis``: the two orders are
        the same array. The kernel really does emit score-descending order, so the
        permutation is the normal case, not an edge one.
        """
        row_end = _row_end(self.DOCS, self.SEQ)
        natural, _, _ = self._case(row_end, reverse=False)
        reversed_, _, _ = self._case(row_end, reverse=True)
        moved = 0
        for row in range(self.SEQ):
            live = int(np.count_nonzero(natural[row]))
            if live < 2:
                continue
            moved += 1
            np.testing.assert_allclose(
                reversed_[row, :live], natural[row, :live][::-1], atol=5e-6
            )
        self.assertGreater(moved, 0, "no row had two live columns")

    def test_empty_rows_stay_zero_and_the_rest_sum_to_one(self):
        """The all-padding row, and the row-sum contract around it.

        An empty row is the only place the implementation cannot rely on the
        softmax: every column is ``-inf``-masked, so ``F.softmax`` returns a
        *uniform* row, which is why the mask is re-applied after it and the
        normalisation clips its denominator
        (``mha_dsa_warmup_attention.py:440-449``). A pad row leaking ``1/s`` per
        head would still look normalised, so it has to be pinned at exactly zero:
        the KL reduction divides by the valid-row count, not by the row sum.
        """
        row_end = _pad_row_end(self.PAD_DOCS, self.SEQ)
        got, _, valid = self._case(row_end)
        pad = ~valid.astype(bool)
        self.assertTrue(pad.any(), "the fixture produced no pad row")
        self.assertEqual(
            float(np.abs(got[pad]).max()), 0.0, "a pad row is not exactly zero"
        )
        sums = got[~pad].sum(axis=-1)
        self.assertLess(float(np.abs(sums - 1.0).max()), 5e-6)

    def test_chunking_does_not_change_the_result(self):
        """Row chunking is a memory device, not part of the maths.

        ``chunk = max(1, _TARGET_ROW_SLOTS // s_global)``
        (``mha_dsa_warmup_attention.py:399-400``) is 512 rows at the production
        ``s = 256`` and 1 row at ``s = 131072``, so in most unit fixtures the loop
        body runs exactly once and its row offsets are never exercised. Shrinking
        the budget to 2, 5 and 1 rows per chunk is the cheap way to cover them;
        with ``_dyadic`` inputs the matmul is exact, so every run must agree
        **bitwise** with the single-chunk one, and a mis-indexed chunk shows up as
        a shifted row rather than a rounding difference.
        """
        row_end = _row_end(self.DOCS, self.SEQ)
        whole, _, _ = self._case(row_end)
        original = warmup_mod._TARGET_ROW_SLOTS
        try:
            # 24 columns, so these give chunk = 2, 5 and 1 rows.
            for slots in (48, 120, 24):
                with self.subTest(row_slots=slots):
                    warmup_mod._TARGET_ROW_SLOTS = slots
                    chunked, want, _ = self._case(row_end)
                    self.assertEqual(
                        float(np.abs(chunked - whole).max()),
                        0.0,
                        "chunking changed the target",
                    )
                    self._assert_close(chunked, want, f"slots={slots}")
        finally:
            warmup_mod._TARGET_ROW_SLOTS = original

    def test_context_parallel_row_slice(self):
        """``s_local < s_global``: this rank's rows, global columns.

        The CP shape, which the single-card end-to-end tests never reach: rows are
        ``[position_offset, position_offset + s_local)`` while the columns and the
        document metadata stay global. Cheap to cover here because the method is
        pure paddle; the actual CP wiring is asserted in the multi-card suite.
        """
        row_end = _row_end(self.DOCS, self.SEQ)
        half = self.SEQ // 2
        for offset in (0, half):
            with self.subTest(offset=offset):
                got, want, _ = self._case(row_end, offset=offset, s_local=half)
                self._assert_close(got, want, f"offset={offset}")
                self.assertEqual(got.shape, (half, self.SEQ))


@_GPU
class TestWarmupGradHealth(unittest.TestCase):
    """Warmup is where the indexer does all of its learning, so a silently
    gradient-free indexer parameter would waste the whole phase.

    Same intent as before, with the attention side moved to the dense layout:
    ``dq`` / ``dk`` / ``dv`` instead of ``dq`` / ``dkv`` / ``dw_v``, since
    ``v_b_proj_weight`` never reaches this backend -- ``kv_b_proj`` has already
    materialised per-head K/V. They are only checked finite and non-zero: bf16
    flashmask reduction order gives them a small run-to-run jitter shared with
    every other dense layer, so their exact values are not a warmup property.
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

    def _assert_live(self, name, grad):
        self.assertIsNotNone(grad, f"{name} has no gradient")
        values = grad.cast("float32")
        self.assertTrue(
            bool(paddle.isfinite(values).all()),
            f"{name} gradient is not finite",
        )
        self.assertGreater(
            float(values.abs().max()), 0.0, f"{name} gradient is all zero"
        )

    def _check(self, sink):
        seqlen, layout = 256, [40, 216]
        module = _warmup_module(sink=sink)
        tensors = _leaves(seqlen)
        out = _forward(
            module,
            tensors,
            _row_end(layout, seqlen),
            training=True,
            input_ids=paddle.ones([1, seqlen], dtype="int64"),
        )
        paddle.seed(11)
        upstream = paddle.randn([1, seqlen, H * V_HEAD_DIM]).cast("float32")
        (out.cast("float32") * upstream).sum().backward()

        indexer = module.indexer
        for name, param in (
            ("indexer.wq_b.weight", indexer.wq_b.linear.weight),
            ("indexer.wk.weight", indexer.wk.linear.weight),
            ("indexer.k_norm.weight", indexer.k_norm.weight),
            ("indexer.k_norm.bias", indexer.k_norm.bias),
            ("indexer.weights_proj.weight", indexer.weights_proj.linear.weight),
        ):
            self._assert_live(name, param.grad)
        query, key, value, x, qr = tensors
        self._assert_live("dq", query.grad)
        self._assert_live("dk", key.grad)
        self._assert_live("dv", value.grad)
        if sink is not None:
            self._assert_live("d_sink", module.softmax_offset.grad)
            self.assertEqual(
                module.softmax_offset.grad.dtype, module.softmax_offset.dtype
            )
        # The indexer learns from its own KL only: its inputs stay detached
        # (``hybrid_mla_indexer.py:140-141``), so no indexer gradient may leak
        # into the backbone through ``x`` / ``qr`` -- and since the dense
        # attention half ignores both, ``None`` is the only correct answer.
        self.assertIsNone(
            x.grad, "x.grad is not None: indexer input not detached"
        )
        self.assertIsNone(
            qr.grad, "qr.grad is not None: indexer input not detached"
        )
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)
        module.clear_gradients()

    def test_grad_health_sinkless(self):
        self._check(sink=None)

    def test_grad_health_with_sink(self):
        with _flash_attn_version(_production_fa_version()):
            try:
                self._check(sink=np.linspace(1.0, 3.0, H))
            except NotImplementedError as exc:
                self.skipTest(f"no learnable_sink support here: {exc}")


@_GPU
class TestShortDocumentLayouts(unittest.TestCase):
    """Packed documents no longer than the sparse phase's forced window.

    Successor of ``TestStarvedIndexerCandidates``. Real packing is dominated by
    short documents, so this is not a synthetic edge; what changed is that phase 2
    cannot be starved by them at all. The three old claims map as follows:

    * "the phase-3 attention table still contains the whole causal set" -> there
      is no table; the candidate *range* claim is asserted kernel-free in
      ``TestWarmupCandidateRange
      .test_window_zero_is_what_saves_the_short_documents``;
    * "all three ``hybrid_mla_attention`` shapes agree bitwise" -> the comparison
      that matters now is phase 2 against **phase 1**, and it must hold on every
      layout rather than only on the fully starved ones, because the two run the
      same kernel;
    * "an empty candidate row is the classic NaN source, so check finiteness" ->
      survives unchanged. It is now the *indexer* rows that can be empty (pad
      rows) rather than whole documents, and the KL still divides by a clipped
      row sum, so the check is still worth its cost.
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

    def test_short_documents_still_equal_phase1_bitwise(self):
        for layout, seqlen, note in _SHORT_DOC_LAYOUTS:
            with self.subTest(note=note):
                config = _warmup_config()
                warmup = _build_module(config, bf16=True)
                phase1 = _build_phase1_dense_module(config, bf16=True)
                tensors = _leaves(seqlen)
                row_end = _row_end(layout, seqlen)
                _CAPTURED.clear()
                got = _forward(
                    warmup,
                    tensors,
                    row_end,
                    training=True,
                    input_ids=paddle.ones([1, seqlen], dtype="int64"),
                )
                want = _phase1_forward(phase1, tensors, row_end, training=True)
                self.assertEqual(
                    float(np.abs(_fp32(got) - _fp32(want)).max()),
                    0.0,
                    f"{note}: warmup != phase 1",
                )
                self.assertEqual(_CAPTURED, [])

    def test_short_documents_stay_finite(self):
        for layout, seqlen, note in _SHORT_DOC_LAYOUTS:
            with self.subTest(note=note):
                module = _warmup_module()
                tensors = _leaves(seqlen)
                out = _forward(
                    module,
                    tensors,
                    _row_end(layout, seqlen),
                    training=True,
                    input_ids=paddle.ones([1, seqlen], dtype="int64"),
                )
                out.cast("float32").sum().backward()
                self.assertTrue(
                    np.isfinite(_fp32(out)).all(),
                    f"{note}: non-finite output",
                )
                grads = [
                    (name, tensor.grad)
                    for name, tensor in zip(
                        ("query", "key", "value", "x", "qr"), tensors
                    )
                ]
                grads += [
                    (name, param.grad)
                    for name, param in module.named_parameters()
                ]
                for name, grad in grads:
                    if grad is None:
                        continue
                    self.assertTrue(
                        np.isfinite(grad.cast("float32").numpy()).all(),
                        f"{note}: gradient {name} is not finite",
                    )
                module.clear_gradients()


if __name__ == "__main__":
    unittest.main()
