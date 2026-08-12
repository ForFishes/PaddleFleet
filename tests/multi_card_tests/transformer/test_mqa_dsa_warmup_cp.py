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

"""Context parallel for the DSA **warmup** phase (phase 2), and for padded
layouts.

Phase 2 is ``hybrid_mla_attention="mqa_dsa"`` with
``dsa_indexer_use_sparse_loss=False``: no top-k on either side. It is no longer
latent MQA -- ``hybrid_mla_indexer.latent_mqa_enabled`` returns False for it
(``hybrid_mla_indexer.py:59-60``), so the spec builds
:class:`MHADSAWarmupAttention`, which delegates its whole attention half to
``DotProductAttention.forward`` (``mha_dsa_warmup_attention.py:103``,
``:179-198``) and only adds the full-candidate indexer KL. There is no
``[b, s, s]`` index table and no block-sparse kernel call in this phase at all.

That moves, but does not remove, the CP evidence this file owes:

1. Two independent CP states have to be right at once. The attention half reads
   the *process-global* ``paddlefleet.parallel_state`` group, at construction
   (``dot_product_attention.py:155``) and at dispatch (``:397``, ``:565-570``);
   the indexer half reads ``pg_collection.cp``
   (``hybrid_mla_indexer.py:88-108``). ``fleet.init`` sets only the former's
   fleet counterpart, so the runner below toggles ``parallel_state`` explicitly
   and the CP=1 reference is built *and* run with it off.
2. "The full-causal ``[b, s, s]`` table is the global build, row-sliced"
   inverts into two claims: the per-rank output is **bit-identical** to the
   phase-1 dense attention fed the same local slice
   (``hybrid_mla_utils._build_phase1_dense_module``), and the *indexer's*
   candidate set -- ``csa_indexer_topk_fwd``'s ``columns``, which are global
   token ids -- is the whole per-document causal set built over the global
   sequence and row-sliced (``hybrid_mla_indexer._indexer_valid_range``).
3. The full-candidate KL still has to normalise across the CP group: the masked
   branch divides by the **global** valid-row count
   (``hybrid_mla_indexer.py:222-227``) and the unmasked one folds ``/cp_size``
   into the coefficient handed to the *backward*
   (``mha_dsa_warmup_attention.py:322-326``).
4. A layout with genuine row-validity pad rows. ``_STRADDLE`` sums to exactly
   ``S_GLOBAL`` and ``U._row_end`` folds any trailing gap into one final
   document, so ``is_valid`` is all-``True`` in every other CP test.
   ``[475] @ s=512`` puts 37 pad rows on the last rank only, which is also the
   pad imbalance the loss denominator has to survive. The phase-3 (latent,
   block-sparse) pad-row cases stay here as well, on ``test_mqa_dsa_cp``'s
   harness, so a pad-row failure can be attributed to a phase.

Phase 3/4's own loss normalisation is swept in
``test_mqa_dsa_cp.py::TestMQADSACP::test_7``; this file owns phase 2.

No ``if rank == X`` short-circuit exists in this file: every collective is
issued on all ranks and only the assertions are rank-conditional.

Run (2 or 4 GPUs)::

    PYTHONPATH=./third_party/PaddleFleet/src:./third_party/PaddleFormers \
    python -m paddle.distributed.launch --devices 0,1 --nnodes 1 \
        --master 127.0.0.1:<port> \
        third_party/PaddleFleet/tests/multi_card_tests/transformer/\
test_mqa_dsa_warmup_cp.py
"""

import contextlib
import types
import unittest

import numpy as np
import paddle
import paddle.distributed as dist

# ``paddle.distributed.launch <thisfile>`` puts this directory on ``sys.path``,
# so the sibling harness imports as a top-level module (same as
# ``test_mla_cp_recompute`` importing ``test_mla_cp_contiguous_allgather``).
import test_mqa_dsa_cp as H

from paddlefleet import parallel_state as ps
from paddlefleet.transformer import mha_dsa_warmup_attention as warmup_mod
from paddlefleet.transformer.csa_attention import _derive_csa_doc_boundaries
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.hybrid_mla_indexer import latent_mqa_enabled
from paddlefleet.transformer.mha_dsa_warmup_attention import (
    MHADSAWarmupAttention,
)

U = H.U

S_GLOBAL = H.S_GLOBAL
_STRADDLE = H._STRADDLE

# One document covering [0, 475) of a 512-long batch: rows 475..511 are pad
# rows. 37 is deliberately not a multiple of 512/cp_size, so the pad rows land
# on the last rank only for both CP=2 (256) and CP=4 (128). It also matches
# ``H._input_ids``' own ``n_pad=37``, so the loss row mask and the document
# validity agree.
_PAD_DOC_LEN = 475
_N_PAD = S_GLOBAL - _PAD_DOC_LEN

FWD_RTOL = H.FWD_RTOL
GRAD_RTOL = H.GRAD_RTOL

# Floor for the warmup-vs-phase-1 ``dq`` comparison, used only when the phase-1
# module's own two runs happen to agree exactly on this batch: one bf16 ulp at
# the scale of these gradients (2**-8 relative, |dq| ~ 1e-2 in this fixture).
# The measured spread between two runs of the same module is 3.052e-05 at
# s=512, so this floor is not what carries the assertion -- see
# ``_assert_phase1_identical``.
_DQ_ATOMIC_FLOOR = 1e-4

# The logged indexer loss is a bf16-fed fp32 KL reduction, so the per-rank sum
# lands a few 1e-4 off the single-rank value. A wrong denominator is off by
# ``cp_size`` (100% at CP=2), so this bound is three orders of magnitude away
# from the failure it has to catch.
LOSS_RTOL = 5e-3

# Positional arguments of ``TileLangCSAIndexerLossAutoScaler.apply`` as phase 2
# calls it (``mha_dsa_warmup_attention.py:342-354``).
_ARG_LOSS_COEFF = 7
_ARG_LOSS_MASK = 10


def setUpModule():
    H.setUpModule()
    # The attention half is ``DotProductAttention``'s and takes its CP state
    # from the *process-global* ``parallel_state`` group, at construction
    # (``dot_product_attention.py:155``) and at dispatch (``:397``, ``:565``),
    # not from ``pg_collection.cp`` the way the indexer half does
    # (``hybrid_mla_indexer.py:88-108``). ``fleet.init`` does not set it, so
    # keep it off by default -- the CP=1 reference must build *and* run non-CP
    # -- and turn it on only around the CP modules.
    ps._CONTEXT_PARALLEL_GROUP = None


@contextlib.contextmanager
def _cp_enabled():
    """Make ``paddlefleet.parallel_state`` report this test's CP group."""
    ps._CONTEXT_PARALLEL_GROUP = H.CP_GROUP
    try:
        yield
    finally:
        ps._CONTEXT_PARALLEL_GROUP = None


@contextlib.contextmanager
def _capture_loss_args():
    """Record every ``TileLangCSAIndexerLossAutoScaler.apply`` argument list.

    Phase 2 attaches its loss through ``mha_dsa_warmup_attention``'s own symbol
    (``mha_dsa_warmup_attention.py:342``), so that is the one to patch;
    ``mqa_latent_attention``'s is phase 3's.
    """
    real = warmup_mod.TileLangCSAIndexerLossAutoScaler
    calls = []

    class _Spy:
        @staticmethod
        def apply(*args, **kwargs):
            calls.append(args)
            return real.apply(*args, **kwargs)

    warmup_mod.TileLangCSAIndexerLossAutoScaler = _Spy
    try:
        yield calls
    finally:
        warmup_mod.TileLangCSAIndexerLossAutoScaler = real


def _pad_row_end(doc_len, s_global):
    """``[1, 1, s_global, 1]`` int32 mask whose tail rows are genuine pad rows.

    ``U._row_end`` cannot express this: it closes the trailing gap with one more
    document, which makes every row valid. Pointing the tail back at the
    previous document's end instead leaves ``pos_in_doc >= doc_len_per_pos``,
    which is exactly ``_derive_csa_doc_boundaries``' ``is_valid`` test
    (``csa_attention.py:136-138``).
    """
    return paddle.full([1, 1, s_global, 1], doc_len, dtype="int32")


def _doc_bounds(row_end, s_global):
    """``(doc_start, is_valid)`` as int lists, over the global sequence."""
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, s_global)
    return doc_start.tolist(), [bool(v) for v in is_valid.tolist()]


def _maxabs(a, b):
    return float((a.cast("float32") - b.cast("float32")).abs().max())


def _bit_equal(a, b):
    """Bit equality of two bf16 tensors, tolerating matching ``NaN``s.

    A fully masked (pad) row is left to the flashmask kernel, which may return
    ``NaN`` there; ``_maxabs`` would then read ``nan`` and compare false against
    every bound, including ``== 0.0``. What this file asserts is that phase 2
    and phase 1 produce *the same bits*, ``NaN`` included.
    """
    x = a.cast("float32").numpy()
    y = b.cast("float32").numpy()
    return bool(np.array_equal(x, y, equal_nan=True))


def _local_slice():
    """``(row_offset, rows)`` of this CP rank's query rows."""
    rows = S_GLOBAL // H.CP_SIZE
    return H.CP_RANK * rows, rows


def _warmup_cfg(cp_size, loss_coeff=0.0):
    """Phase-2 config: ``mqa_dsa`` with the sparse-loss switch off."""
    cfg = U._create_mqa_config(
        mode="mqa_dsa", loss_coeff=loss_coeff, sparse_loss=False
    )
    cfg.cp_balance_mode = "contiguous_allgather"
    # Production EB dataflow hands every rank the *global* ``input_ids``, which
    # is the branch ``_indexer_loss_mask`` takes when this flag is set
    # (``hybrid_mla_indexer.py:217-221``).
    cfg.experimental_dataflow = True
    cfg.pad_token_id = 0
    cfg.context_parallel_size = cp_size
    assert not latent_mqa_enabled(cfg), (
        "mqa_dsa + dsa_indexer_use_sparse_loss=False must not select latent "
        "MQA (hybrid_mla_indexer.py:59-60); this suite would silently be "
        "testing phase 3 instead of the warmup phase"
    )
    return cfg


def _build_warmup(cp_group, loss_coeff=0.0, seed=7):
    """CP=1 reference (``cp_group is None``) or CP layer, identical weights.

    Must be called under :func:`_cp_enabled` for the CP layer and outside it for
    the reference: ``pg_collection.cp`` drives the indexer half only.
    """
    cfg = _warmup_cfg(1 if cp_group is None else cp_group.nranks, loss_coeff)
    paddle.seed(seed)
    module = U._build_module(
        cfg,
        bf16=True,
        pg_collection=types.SimpleNamespace(tp=None, cp=cp_group),
    )
    assert isinstance(module, MHADSAWarmupAttention), (
        f"the phase-2 fixture built {type(module).__name__}, not the dense "
        "MHADSAWarmupAttention"
    )
    return module


def _capture_columns(module, store):
    """Record ``_dense_attn_target``'s ``columns`` (global token ids).

    ``RecordingWarmupMHA`` keeps the KL *target*, which is already permuted into
    the kernel's column order, so the candidate ids themselves have to be taken
    from the call's third positional argument
    (``mha_dsa_warmup_attention.py:305-314``).
    """
    inner = module._dense_attn_target

    def wrapper(*args, **kwargs):
        store.append(args[2].numpy().copy())
        return inner(*args, **kwargs)

    module._dense_attn_target = wrapper


def _forward(module, q, k, v, row_end, x=None, qr=None, ids=None):
    """One dense forward, per-document causal, ``row_end`` at global length.

    ``attn_mask_type`` has to be passed explicitly: on a ``None`` argument
    ``DotProductAttention`` derives ``is_causal = attn_mask_type ==
    AttnMaskType.causal`` (``dot_product_attention.py:562``), i.e. non-causal.
    The mask is global on both sides because
    ``expand_attn_mask_startend_row_indices_for_cp`` builds its second column
    from ``arange(s_local * cp_size)`` and expands it onto the mask's own rows
    (``:319-342``), so a local-length table would not even broadcast; the CP
    kernel then slices it per rank (``context_parallel_utils.py:1979-1992``).

    ``x`` left ``None`` selects the phase-1 module's narrower signature (it
    takes no ``input_ids`` -- ``accepts_input_ids`` is the phase-2 mixin's,
    ``hybrid_mla_indexer.py:76``).
    """
    extra = {} if x is None else {"x": x, "qr": qr, "input_ids": ids}
    return module(
        q,
        k,
        v,
        None,
        attn_mask_startend_row_indices=row_end.clone(),
        attn_mask_type=AttnMaskType.causal,
        **extra,
    )


def run_warmup_cp(
    doc_lens,
    loss_coeff=0.0,
    with_input_ids=False,
    row_end=None,
    s_global=S_GLOBAL,
):
    """CP=1 reference, the CP layer, and the phase-1 dense module on one batch.

    Three modules rather than the harness' two: the phase-2 attention half is
    ``DotProductAttention.forward`` verbatim
    (``mha_dsa_warmup_attention.py:179-198``), so the phase-1 module fed this
    rank's slice is the reference for *bit* equality, while the CP=1 phase-2
    module is the reference for CP equivalence.
    """
    sl = s_global // H.CP_SIZE
    off = H.CP_RANK * sl
    if row_end is None:
        row_end = U._row_end(doc_lens, s_global)
    q, k, v, x, qr = U._make_dense_inputs(s_global, seed=3)
    ids = H._input_ids(s_global) if with_input_ids else None

    ref = _build_warmup(None, loss_coeff)
    with _cp_enabled():
        cpl = _build_warmup(H.CP_GROUP, loss_coeff)
        phase1 = U._build_phase1_dense_module(_warmup_cfg(H.CP_SIZE), bf16=True)
    cpl.set_state_dict(ref.state_dict())

    U._CAPTURED.clear()
    U._WARMUP_TARGETS.clear()
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    cols_ref = []
    _capture_columns(ref, cols_ref)
    ra = [H._leaf(t) for t in (q, k, v, x, qr)]
    out_ref = _forward(ref, ra[0], ra[1], ra[2], row_end, ra[3], ra[4], ids)
    out_ref.sum().backward()
    logged_ref = H._logged_indexer_loss()
    targets_ref = list(U._WARMUP_TARGETS)

    U._CAPTURED.clear()
    U._WARMUP_TARGETS.clear()
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    cols_cp = []
    _capture_columns(cpl, cols_cp)
    cb = [H._leaf(t[:, off : off + sl]) for t in (q, k, v, x, qr)]
    p1 = [H._leaf(t[:, off : off + sl]) for t in (q, k, v)]
    # Second phase-1 run on fresh leaves, same module, same inputs: the
    # kernel's own run-to-run spread, used as the bound for the warmup-vs-phase-1
    # ``dq`` comparison. Measured: the forward, ``dk`` and ``dv`` are bitwise
    # reproducible, ``dq`` is not (3.052e-05 at s=512 between two runs of the
    # *same* module), because the flashmask backward accumulates ``dq`` across
    # column blocks with atomics.
    p1c = [H._leaf(t[:, off : off + sl]) for t in (q, k, v)]
    with _cp_enabled(), _capture_loss_args() as loss_args:
        out = _forward(cpl, cb[0], cb[1], cb[2], row_end, cb[3], cb[4], ids)
        out.sum().backward()
        out_p1 = _forward(phase1, p1[0], p1[1], p1[2], row_end)
        out_p1.sum().backward()
        out_p1c = _forward(phase1, p1c[0], p1c[1], p1c[2], row_end)
        out_p1c.sum().backward()
    logged_cp = H._logged_indexer_loss()

    # Parameter grads: this rank only saw its own query rows, so the CP group's
    # SUM is the reference. The attention half owns no parameter in this fixture
    # (no sink is configured, so ``build_softmax_offset`` returns ``None``),
    # which is also why the phase-1 module needs no ``set_state_dict``.
    ref_named = dict(ref.named_parameters())
    param_err = {}
    for name, p in cpl.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.contiguous()
        dist.all_reduce(g, group=H.CP_GROUP)
        rg = ref_named[name].grad
        param_err[name] = None if rg is None else H._rel(g, rg)

    return {
        "fwd": H._rel(out, out_ref[:, off : off + sl]),
        "dq": H._rel(cb[0].grad, ra[0].grad[:, off : off + sl]),
        "dk": H._rel(cb[1].grad, ra[1].grad[:, off : off + sl]),
        "dv": H._rel(cb[2].grad, ra[2].grad[:, off : off + sl]),
        "param_err": param_err,
        "out": out.detach(),
        "ref_out": out_ref.detach(),
        "dq_local": cb[0].grad.detach(),
        # The phase-1 dense module on this rank's slice: the reference for the
        # output (bitwise) and for ``dq`` (within the control spread below).
        "phase1_out": out_p1.detach(),
        "phase1_dq": p1[0].grad.detach(),
        "phase1_out_control": out_p1c.detach(),
        "phase1_dq_control": p1c[0].grad.detach(),
        # ``columns`` are global token ids on both sides, so the reference's
        # rows compare to this rank's directly, with no offset arithmetic.
        "cols_ref_slice": cols_ref[-1][:, off : off + sl] if cols_ref else None,
        "cols_cp": cols_cp[-1] if cols_cp else None,
        "targets_ref": targets_ref,
        "targets_cp": list(U._WARMUP_TARGETS),
        # ``RecordingMQA``'s hook: any entry means a block-sparse kernel call
        # happened, which phase 2 must never make.
        "captured": len(U._CAPTURED),
        "loss_args": list(loss_args),
        "logged_ref": logged_ref,
        "logged_cp": logged_cp,
        "row_end": row_end,
        "off": off,
        "rows": sl,
    }


class _CPChecks(unittest.TestCase):
    """Shared assertions.

    ``_check`` / ``_check_index_sets`` are the harness' own implementations,
    used by the latent (phase 3/4) pad-row cases; everything below them is the
    dense phase-2 contract.
    """

    def _check(self, res, tag):
        H.TestMQADSACP._check(self, res, tag)

    def _check_index_sets(self, res, tag):
        H.TestMQADSACP._check_index_sets(self, res, tag)

    def _check_dense(self, res, tag):
        """CP=N == CP=1 on this rank's slice: output and every gradient."""
        for key, bound in (
            ("fwd", FWD_RTOL),
            ("dq", GRAD_RTOL),
            ("dk", GRAD_RTOL),
            ("dv", GRAD_RTOL),
        ):
            self.assertLess(res[key], bound, f"{tag}: {key} {res[key]:.3e}")
        for name, err in res["param_err"].items():
            self.assertIsNotNone(
                err, f"{tag}: reference has no grad for {name}"
            )
            self.assertLess(err, GRAD_RTOL, f"{tag}: param {name} {err:.3e}")
        worst = max(res["param_err"].values(), default=0.0)
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"fwd={res['fwd']:.2e} dq={res['dq']:.2e} dk={res['dk']:.2e} "
            f"dv={res['dv']:.2e} param_max={worst:.2e}",
            flush=True,
        )

    def _assert_no_block_sparse(self, res, tag):
        """No ``[b, s, s]`` table, no block-sparse call -- the inverted claim.

        The old suite asserted phase 2 built the *same* full-causal index table
        as ``mqa_full_causal``. Phase 2 no longer builds one at all, so the
        evidence is that ``RecordingMQA``'s ``_sparse_attn`` hook never fired.
        """
        self.assertEqual(
            res["captured"],
            0,
            f"{tag}: {res['captured']} block-sparse kernel call(s) were made "
            "in the warmup phase, which must run dense flashmask only",
        )

    def _assert_phase1_identical(self, res, tag):
        """Per-rank output == the phase-1 dense module's bitwise; ``dq`` == it to
        within the kernel's own run-to-run spread.

        This is the other half of the inversion: "phase 2 is phase 1 plus an
        indexer loss" is only true if the attention half is untouched, and the
        indexer loss reaches the output solely through
        ``TileLangCSAIndexerLossAutoScaler``'s *backward*
        (``mha_dsa_warmup_attention.py:342-354``), so even a live loss must not
        move the forward. Asserted with the CP row-slicing in the path.

        ``dq`` cannot be asserted bitwise, and the bound is measured rather than
        chosen: two backward passes of the *same* phase-1 module on the same
        inputs already disagree (3.052e-05 at s=512), because the flashmask
        backward accumulates ``dq`` over column blocks with atomics. The
        forward, ``dk`` and ``dv`` are bitwise reproducible, so those stay exact.
        The control is computed in this same run
        (``run_warmup_cp``: ``phase1_dq_control``), so a real divergence -- which
        would be orders of magnitude larger, as the un-sliced-mask and
        wrong-offset controls in this suite show -- still fails.
        """
        same = _bit_equal(res["out"], res["phase1_out"])
        delta = _maxabs(res["out"], res["phase1_out"])
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: output vs "
            f"phase-1 dense maxabs={delta:.3e} bit_equal={same}",
            flush=True,
        )
        self.assertTrue(
            same,
            f"{tag}: the warmup output is not bit-identical to the phase-1 "
            f"dense module's (maxabs={delta:.3e}), so phase 2 is no longer "
            "'phase 1 plus an indexer loss'",
        )
        dq_delta = _maxabs(res["dq_local"], res["phase1_dq"])
        control = _maxabs(res["phase1_dq"], res["phase1_dq_control"])
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: dq vs phase-1 "
            f"dense maxabs={dq_delta:.3e} (phase-1 self-control "
            f"{control:.3e})",
            flush=True,
        )
        self.assertLessEqual(
            dq_delta,
            max(control, _DQ_ATOMIC_FLOOR),
            f"{tag}: the warmup dq differs from the phase-1 dense module's by "
            f"{dq_delta:.3e}, more than that module's own run-to-run spread "
            f"({control:.3e}), so phase 2's attention backward is not phase "
            "1's",
        )

    def _assert_full_causal_columns(self, res, tag):
        """Every local row's candidate set == its whole per-document causal set.

        ``csa_indexer_topk_fwd`` runs in full-candidate mode here (``ratio=1``,
        ``topk_effective=s_global``), fed a ``valid_range`` built over the
        global sequence and row-sliced with ``window=0``
        (``mha_dsa_warmup_attention.py:283-300``,
        ``hybrid_mla_indexer.py:151-191``). Under phase 3's window + top-k a row
        longer than ``window + index_topk`` would select a strict subset, so
        this assertion separates the two phases without needing phase 3's
        columns for comparison.
        """
        off, rows = _local_slice()
        doc_start, is_valid = _doc_bounds(res["row_end"], S_GLOBAL)
        cols = res["cols_cp"]
        self.assertIsNotNone(cols, f"{tag}: the indexer produced no columns")
        self.assertEqual(int(cols.shape[1]), rows, f"{tag}: row count")
        for r in range(rows):
            q = off + r
            got = {int(c) for c in cols[0][r] if c >= 0}
            want = set(range(doc_start[q], q + 1)) if is_valid[q] else set()
            self.assertEqual(
                got, want, f"{tag}: row {r} (global {q}) is not full-causal"
            )

    def _assert_loss_cp_sum(self, res, tag):
        """Sum the per-rank logged loss and compare to the CP=1 value."""
        total = paddle.to_tensor([res["logged_cp"]], dtype="float64")
        dist.all_reduce(total, group=H.CP_GROUP)
        got, want = float(total[0]), res["logged_ref"]
        self.assertGreater(abs(want), 0.0, f"{tag}: reference logged no loss")
        rel = abs(got - want) / abs(want)
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"this_rank={res['logged_cp']:.6e} sum(loss)={got:.6e} "
            f"ref={want:.6e} rel={rel:.3e}",
            flush=True,
        )
        self.assertLess(
            rel,
            LOSS_RTOL,
            f"{tag}: CP loss sum {got:.6e} != CP=1 {want:.6e} "
            f"(rel {rel:.3e}); a per-rank denominator would be off by "
            f"~{H.CP_SIZE}x",
        )
        return want

    def _assert_loss_coeff(self, res, masked, loss_coeff, tag):
        """The unmasked ``/cp_size`` must reach the *backward*, not just the log.

        ``csa_attention`` folds it into the logged scalar only; phase 2 puts it
        in the coefficient it hands the autoscaler
        (``mha_dsa_warmup_attention.py:322-326``), which is the value the
        tilelang backward multiplies ``(P - Q)`` by. The logged loss cannot see
        the difference -- both placements produce the same scalar -- so read the
        argument itself.
        """
        self.assertEqual(len(res["loss_args"]), 1, f"{tag}: one loss per layer")
        args = res["loss_args"][0]
        got = float(args[_ARG_LOSS_COEFF])
        want = loss_coeff if masked else loss_coeff / H.CP_SIZE
        self.assertAlmostEqual(
            got,
            want,
            places=9,
            msg=f"{tag}: the backward got loss_coeff={got:.6e}, expected "
            f"{want:.6e} (masked={masked}, cp_size={H.CP_SIZE})",
        )
        mask = args[_ARG_LOSS_MASK]
        if masked:
            self.assertIsNotNone(mask, f"{tag}: input_ids built no row mask")
            self.assertEqual(
                list(mask.shape), [1, res["rows"]], f"{tag}: mask is not local"
            )
        else:
            self.assertIsNone(mask, f"{tag}: a mask appeared without input_ids")

    def _assert_columns_match_reference(self, res, tag):
        """The CP layer's candidate set == the CP=1 layer's, for these rows.

        Set equality per row rather than sequence equality: the kernel's slot
        order is its own business, and ``columns`` is what the KL target is
        permuted into (``mha_dsa_warmup_attention.py:454-459``).
        """
        want, got = res["cols_ref_slice"], res["cols_cp"]
        self.assertIsNotNone(got, f"{tag}: the CP layer produced no columns")
        self.assertIsNotNone(want, f"{tag}: the reference produced no columns")
        self.assertEqual(list(got.shape), list(want.shape), f"{tag}: shape")
        for r in range(int(got.shape[1])):
            a = {int(c) for c in want[0][r] if c >= 0}
            b = {int(c) for c in got[0][r] if c >= 0}
            self.assertEqual(
                b,
                a,
                f"{tag}: row {r} (global {res['off'] + r}) differs: "
                f"missing={sorted(a - b)[:8]} extra={sorted(b - a)[:8]}",
            )

    def _rel_on_valid_rows(self, res):
        """Relative error against the CP=1 reference, valid rows only.

        A fully masked pad row is the flashmask kernel's business and may come
        back as ``NaN``, which would poison a whole-slice norm. Bit equality to
        the phase-1 module (``_assert_phase1_identical``) is what covers those
        rows here.
        """
        off, rows = _local_slice()
        _, is_valid = _doc_bounds(res["row_end"], S_GLOBAL)
        keep = [r for r in range(rows) if is_valid[off + r]]
        got = res["out"].cast("float32").numpy()[0][keep]
        want = res["ref_out"].cast("float32").numpy()[0][off : off + rows][keep]
        denom = max(float(np.linalg.norm(want)), 1e-12)
        return float(np.linalg.norm(got - want)) / denom


class TestWarmupCP(_CPChecks):
    """``mqa_dsa`` + ``dsa_indexer_use_sparse_loss=False`` under CP."""

    @H.U._GPU
    def test_1_warmup_forward_equivalence(self):
        """CP=N == CP=1 on the warmup path, with and without a live loss.

        ``dsa_indexer_loss_coeff == 0`` returns straight after the dense
        attention and never touches the indexer projections
        (``mha_dsa_warmup_attention.py:199-200``); ``> 0`` additionally runs the
        full-candidate indexer KL. Neither may perturb the attention output, so
        both are checked against the same reference *and* against the phase-1
        dense module.
        """
        for coeff in (0.0, 0.1):
            with self.subTest(loss_coeff=coeff):
                tag = f"warmup/coeff={coeff}"
                res = run_warmup_cp(
                    _STRADDLE, loss_coeff=coeff, with_input_ids=coeff > 0
                )
                self._check_dense(res, tag)
                self._assert_no_block_sparse(res, tag)
                self._assert_phase1_identical(res, tag)
                if coeff > 0:
                    self._assert_full_causal_columns(res, tag)

    @H.U._GPU
    def test_2_candidate_columns_are_the_global_set_row_sliced(self):
        """The indexer's candidates == the global build, sliced.

        ``columns`` are global token ids, so this rank's rows must equal the
        CP=1 reference's same rows with no rebasing. The control at the end
        constructs the set a CP-unaware build would produce -- ``valid_range``
        derived from *local* coordinates, i.e. rows clipped at
        ``q - position_offset`` with the prefix owned by lower ranks dropped --
        and asserts it differs, so the comparison cannot be vacuous on rank > 0.
        """
        tag = "columns"
        res = run_warmup_cp(_STRADDLE, loss_coeff=0.1, with_input_ids=True)
        self._assert_columns_match_reference(res, tag)
        self._assert_full_causal_columns(res, tag)

        off, rows = _local_slice()
        doc_start, is_valid = _doc_bounds(res["row_end"], S_GLOBAL)
        widths = [
            len({int(c) for c in res["cols_cp"][0][r] if c >= 0})
            for r in range(rows)
        ]
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] candidate widths: "
            f"min={min(widths)} max={max(widths)}",
            flush=True,
        )
        if H.CP_RANK == 0:
            return
        local = [
            len(range(max(doc_start[off + r] - off, 0), r + 1))
            if is_valid[off + r]
            else 0
            for r in range(rows)
        ]
        self.assertNotEqual(
            widths,
            local,
            "a local-coordinate candidate build was indistinguishable from "
            "the global one, so this test is vacuous",
        )

    @H.U._GPU
    def test_3_warmup_output_is_real_per_document_causal_attention(self):
        """The non-vacuity control for the phase-1 bit-identity claim.

        ``_assert_phase1_identical`` compares two runs of the same code path, so
        on its own it would also pass if both returned garbage (or zeros). Pin
        this rank's rows to an independent fp32 per-document causal MHA
        (``hybrid_mla_utils._dense_mha_reference``) over the *global* batch, at
        the module's own scale: ``softmax_scale = 1/sqrt(k_channels)``
        (``dot_product_attention.py:210-215``, with ``k_channels=K_CHANNELS``
        from ``_build_phase1_dense_module``), which is what
        ``_dense_attn_target`` uses for the KL target too
        (``mha_dsa_warmup_attention.py:440``).
        """
        off, rows = _local_slice()
        res = run_warmup_cp(_STRADDLE, loss_coeff=0.1, with_input_ids=True)
        q, k, v, _, _ = U._make_dense_inputs(S_GLOBAL, seed=3)
        want = U._dense_mha_reference(
            q, k, v, res["row_end"], U.K_CHANNELS**-0.5
        )
        err = H._rel(res["out"], want[:, off : off + rows])
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] vs fp32 dense MHA: "
            f"rel={err:.3e}",
            flush=True,
        )
        self.assertLess(
            err,
            FWD_RTOL,
            f"the warmup output is not per-document causal MHA (rel {err:.3e})",
        )

    @H.U._GPU
    def test_4_indexer_loss_cp_normalisation(self):
        """The logged loss must sum to the CP=1 value, on both mask branches.

        Read straight out of ``DSAIndexerLossLoggingHelper``, so it observes the
        denominator rather than its shadow in the gradients. Masked divides by
        the **global** valid-row count (``hybrid_mla_indexer.py:222-227``);
        unmasked takes the plain local mean with ``/cp_size`` folded into the
        coefficient, which ``_assert_loss_coeff`` checks reaches the backward.
        """
        for masked in (True, False):
            with self.subTest(masked=masked):
                tag = f"loss/masked={masked}"
                res = run_warmup_cp(
                    _STRADDLE, loss_coeff=0.1, with_input_ids=masked
                )
                self.assertTrue(
                    any(n.startswith("indexer.") for n in res["param_err"]),
                    f"{tag}: the indexer received no gradient",
                )
                self._check_dense(res, tag)
                self._assert_no_block_sparse(res, tag)
                self._assert_loss_cp_sum(res, tag)
                self._assert_loss_coeff(res, masked, 0.1, tag)

    @H.U._GPU
    def test_5_full_candidate_kl_is_wider_than_a_windowed_one(self):
        """One 512-long document: the KL support is the whole causal row.

        The width itself is the phase difference, so assert it directly instead
        of comparing against a phase-3 run: the widest local candidate row must
        exceed ``window + index_topk``, which is all phase 3 can ever supervise,
        and the target the KL is taken over must be ``s_global`` wide
        (``mha_dsa_warmup_attention.py:391-417``). Taken as a MAX over the CP
        group: on rank 0 at CP=2 the widest row is exactly ``s_local``, so a
        per-rank assertion would be a false failure there.
        """
        tag = "wide"
        res = run_warmup_cp([S_GLOBAL], loss_coeff=0.1, with_input_ids=True)
        self._check_dense(res, tag)
        self._assert_loss_cp_sum(res, tag)
        self._assert_full_causal_columns(res, tag)
        self.assertEqual(
            list(res["targets_cp"][-1].shape),
            [1, res["rows"], S_GLOBAL],
            f"{tag}: the KL target is not the full candidate width",
        )
        widest = max(
            len({int(c) for c in res["cols_cp"][0][r] if c >= 0})
            for r in range(res["rows"])
        )
        group_max = paddle.to_tensor([widest], dtype="int64")
        dist.all_reduce(group_max, group=H.CP_GROUP, op=dist.ReduceOp.MAX)
        narrow = U.WINDOW + U.INDEX_TOPK
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: widest local "
            f"row={widest} group_max={int(group_max[0])} narrow={narrow}",
            flush=True,
        )
        self.assertGreater(
            int(group_max[0]),
            narrow,
            f"{tag}: the widest candidate row across the CP group is "
            f"{int(group_max[0])} <= window+index_topk ({narrow}), so this "
            "layout cannot tell the full-candidate KL from a windowed one",
        )


class TestPadRowsCP(_CPChecks):
    """``[475] @ s=512``: 37 real pad rows, all on the last CP rank.

    Every other CP suite uses layouts whose documents tile the sequence, so
    ``is_valid`` is all-``True`` and the pad-row path is never reached under CP.
    The rows also sit entirely on the last rank, which is the pad imbalance a
    per-rank loss denominator cannot survive. Both phases are covered so a
    pad-row failure can be attributed to one: the latent (phase 3/4) cases run
    on the harness' own runner, the dense warmup case on this file's.
    """

    def _run_latent(self, mode, sparse_loss):
        return H.run_core_cp(
            mode,
            None,
            loss_coeff=0.1,
            with_input_ids=True,
            sparse_loss=sparse_loss,
            row_end=_pad_row_end(_PAD_DOC_LEN, S_GLOBAL),
        )

    def _local_pad_rows(self, row_end, tag):
        """This rank's pad rows, after asserting the group really has ``_N_PAD``.

        Only the last rank owns them, so the count is checked on the group.
        """
        off, rows = _local_slice()
        _, is_valid = _doc_bounds(row_end, S_GLOBAL)
        local_pad = [r for r in range(rows) if not is_valid[off + r]]
        n_pad = paddle.to_tensor([len(local_pad)], dtype="int64")
        dist.all_reduce(n_pad, group=H.CP_GROUP)
        self.assertEqual(
            int(n_pad[0]),
            _N_PAD,
            f"{tag}: the layout produced {int(n_pad[0])} pad rows, expected "
            f"{_N_PAD} -- the fixture no longer tests what it claims",
        )
        return off, local_pad

    def _check_pad_latent(self, res, tag):
        """Phase 3/4: an all-``-1`` index row must give a zero output and dq."""
        off, local_pad = self._local_pad_rows(res["row_end"], tag)
        out = res["out"].cast("float32")
        dq = res["dq_local"].cast("float32")
        for r in local_pad:
            self.assertEqual(
                float(out[0, r].abs().max()),
                0.0,
                f"{tag}: pad row {r} (global {off + r}) has a non-zero output",
            )
            self.assertEqual(
                float(dq[0, r].abs().max()),
                0.0,
                f"{tag}: pad row {r} (global {off + r}) has a non-zero dq",
            )
            cols = res["idx_cp"][0][r]
            self.assertEqual(
                int((cols >= 0).sum()),
                0,
                f"{tag}: pad row {r} (global {off + r}) selected columns",
            )
        print(
            f"[padrows-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"local_pad_rows={len(local_pad)} fwd={res['fwd']:.2e}",
            flush=True,
        )

    @H.U._GPU
    def test_1_pad_rows_mqa_full_causal(self):
        res = self._run_latent("mqa", True)
        self._check(res, "pad/mqa_full_causal")
        self._check_index_sets(res, "pad/mqa_full_causal")
        self._check_pad_latent(res, "pad/mqa_full_causal")

    @H.U._GPU
    def test_2_pad_rows_warmup_dense(self):
        """Phase 2's pad rows: whatever phase 1 does, plus an empty KL row.

        The output of a fully masked row is the dense flashmask kernel's own
        behaviour, not this phase's, so it is pinned by bit equality to the
        phase-1 module rather than by a value; the ``NaN``-tolerant comparison
        exists for exactly that reason. What phase 2 *does* own is the indexer
        half: a pad row must carry no candidate and no KL mass
        (``mha_dsa_warmup_attention.py:301-304``, ``:450``), and the loss
        denominator must still be the global valid-row count with 37 of the pad
        rows on one rank.
        """
        tag = "pad/warmup"
        row_end = _pad_row_end(_PAD_DOC_LEN, S_GLOBAL)
        res = run_warmup_cp(
            None, loss_coeff=0.1, with_input_ids=True, row_end=row_end
        )
        off, local_pad = self._local_pad_rows(row_end, tag)
        self._assert_no_block_sparse(res, tag)
        self._assert_phase1_identical(res, tag)
        self._assert_full_causal_columns(res, tag)

        target = res["targets_cp"][-1]
        for r in local_pad:
            self.assertEqual(
                int((res["cols_cp"][0][r] >= 0).sum()),
                0,
                f"{tag}: pad row {r} (global {off + r}) has candidates",
            )
            self.assertEqual(
                float(np.abs(target[0][r]).max()),
                0.0,
                f"{tag}: pad row {r} (global {off + r}) carries KL mass",
            )

        err = self._rel_on_valid_rows(res)
        self.assertLess(err, FWD_RTOL, f"{tag}: valid-row forward {err:.3e}")
        for name, perr in res["param_err"].items():
            self.assertIsNotNone(perr, f"{tag}: no reference grad for {name}")
            self.assertLess(perr, GRAD_RTOL, f"{tag}: param {name} {perr:.3e}")
        self._assert_loss_cp_sum(res, tag)
        self._assert_loss_coeff(res, True, 0.1, tag)
        print(
            f"[padrows-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"local_pad_rows={len(local_pad)} valid_row_fwd={err:.2e}",
            flush=True,
        )

    @H.U._GPU
    def test_3_pad_rows_sparse(self):
        """Same layout on the phase-3 (``window + top-k``) path."""
        res = self._run_latent("mqa_dsa", True)
        self._check(res, "pad/sparse")
        self._check_index_sets(res, "pad/sparse")
        self._check_pad_latent(res, "pad/sparse")


if __name__ == "__main__":
    unittest.main(verbosity=2)
