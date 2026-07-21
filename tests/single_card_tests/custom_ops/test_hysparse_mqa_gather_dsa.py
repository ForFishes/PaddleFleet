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

"""Correctness tests for the HySparse block-sparse *DSA* backend
(:func:`paddlefleet.cudnn_ops.block_sparse_mqa_attention_dsa`), which routes the
block selection through DeepSeek-v4's FlashMLA sparse forward + cuDNN DSA
backward.

We validate the absorbed-MQA layout (Dk=576 query/key, Dv=512 value == leading
slice of the shared latent) against a differentiable dense masked-softmax
reference over the exact selected-block column set (forward output and dq/dkv
grads). The comparison requires cosine > 0.99. The suite skips gracefully when
FlashMLA / cuDNN-frontend / a Blackwell (SM100) GPU is unavailable.
"""

import math
import os
import sys
import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

# Reusable precision metrics helper at the single_card_tests root.
_TESTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
from _hysparse_metrics import assert_close

from paddlefleet.tilelang_ops.hysparse.reference_sparse import (
    build_random_block_indices,
    make_causal_valid_range,
    ref_block_sparse_mqa,
)


def _dsa_unavailable_reason():
    if not paddle.device.is_compiled_with_cuda():
        return "CUDA build of Paddle required"
    if paddle.device.cuda.device_count() == 0:
        return "no CUDA device available"
    try:
        from paddlefleet.cudnn_ops import is_dsa_available

        if not is_dsa_available():
            return "FlashMLA sparse fwd + cuDNN DSA bwd not available"
    except (ImportError, RuntimeError):
        return "hysparse DSA import failed"
    cc = paddle.device.cuda.get_device_capability()
    if cc[0] < 10:
        return f"DSA sparse fwd requires SM100+, got {cc}"
    return None


_SKIP_REASON = None


def _skip_if_no_dsa(tc):
    global _SKIP_REASON
    if _SKIP_REASON is None:
        _SKIP_REASON = _dsa_unavailable_reason() or ""
    if _SKIP_REASON:
        tc.skipTest(_SKIP_REASON)


def _allow_mask(indices, valid_range, s_kv, block_B):
    """Bool [B, S, S_kv]: col allowed iff in a selected block ∩ [bos, eos)."""
    import numpy as np

    idx = indices.numpy()
    vr = valid_range.numpy()
    b, s, _ = idx.shape
    allow = np.zeros([b, s, s_kv], dtype=bool)
    for bi in range(b):
        for i in range(s):
            bos, eos = int(vr[bi, i, 0]), int(vr[bi, i, 1])
            for blk in idx[bi, i]:
                if blk < 0:
                    continue
                c0 = bos + int(blk) * block_B
                for col in range(c0, min(c0 + block_B, eos)):
                    if 0 <= col < s_kv:
                        allow[bi, i, col] = True
    return paddle.to_tensor(allow)


def _ref_masked_attn(q, k, v, allow, sm_scale, attn_sink=None):
    """Differentiable dense masked MQA attention reference (fp32).

    ``attn_sink`` [H] fp32 (optional) adds a virtual sink column to the softmax
    denominator (attention sink / off-by-one).
    """
    neg_inf = float("-inf")
    logits = paddle.einsum("bshd,bkd->bshk", q, k) * sm_scale
    neg = paddle.full_like(logits, neg_inf)
    mask = allow.unsqueeze(2)  # [B,S,1,Skv]
    logits = paddle.where(mask, logits, neg)
    row_has = allow.any(axis=-1)
    m = logits.max(axis=-1, keepdim=True)
    if attn_sink is not None:
        m = paddle.maximum(
            m, attn_sink.reshape([1, 1, -1, 1]).astype("float32")
        )
    m = paddle.where(paddle.isfinite(m), m, paddle.zeros_like(m))
    p = paddle.exp(logits - m)
    denom = p.sum(axis=-1, keepdim=True)
    if attn_sink is not None:
        denom = denom + paddle.exp(
            attn_sink.reshape([1, 1, -1, 1]).astype("float32") - m
        )
    denom = paddle.where(denom > 0, denom, paddle.ones_like(denom))
    p = p / denom
    out = paddle.einsum("bshk,bkc->bshc", p, v)
    out = out * row_has.astype("float32").unsqueeze(-1).unsqueeze(-1)
    return out


def _causal_valid_range(b, s):
    eos = (
        paddle.arange(1, s + 1, dtype="int32")
        .reshape([1, s, 1])
        .expand([b, s, 1])
    )
    bos = paddle.zeros([b, s, 1], dtype="int32")
    return paddle.concat([bos, eos], axis=-1).contiguous()


def _cos(a, b):
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float(np.dot(af, bf) / denom)


def _select_blocks(b, s, block_B, topk):
    """Per-token relative block ids: block 0 plus the running block (pos//BB),
    padded with -1 to width ``topk``."""
    pos = paddle.arange(s)
    b0 = paddle.zeros([b, s, 1], dtype="int32")
    b1 = (pos // block_B).cast("int32").reshape([1, s, 1]).expand([b, s, 1])
    idx = paddle.concat([b0, b1], axis=-1)  # [b, s, 2]
    if topk > 2:
        pad = paddle.full([b, s, topk - 2], -1, dtype="int32")
        idx = paddle.concat([idx, pad], axis=-1)
    return idx.contiguous()


class TestBlockSparseDSA(unittest.TestCase):
    BLOCK_B = 64
    Dk = 576
    Dv = 512

    def _make(self, b, s, h, topk, seed=7):
        paddle.seed(seed)
        q = paddle.randn([b, s, h, self.Dk]).cast("bfloat16")
        kf = paddle.randn([b, s, self.Dk]).cast("bfloat16")
        vr = _causal_valid_range(b, s)
        idx = _select_blocks(b, s, self.BLOCK_B, topk)
        sm_scale = 1.0 / math.sqrt(self.Dk)
        return q, kf, vr, idx, sm_scale

    def _run_dsa(self, q, kf, idx, vr, sm_scale):
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        qd = q.detach().clone()
        qd.stop_gradient = False
        kd = kf.detach().clone()
        kd.stop_gradient = False
        out, _ = block_sparse_mqa_attention_dsa(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
        )
        out.sum().backward()
        return out, qd.grad, kd.grad

    def _run_ref(self, q, kf, idx, vr, sm_scale):
        b, s, h = q.shape[0], q.shape[1], q.shape[2]
        s_kv = kf.shape[1]
        qr = q.detach().cast("float32")
        qr.stop_gradient = False
        kr = kf.detach().cast("float32")
        kr.stop_gradient = False
        vr_ = kr[:, :, : self.Dv]
        allow = _allow_mask(idx, vr, s_kv, self.BLOCK_B)
        out = _ref_masked_attn(qr, kr, vr_, allow, sm_scale)
        out = out.reshape([b, s, h * self.Dv])
        out.sum().backward()
        return out, qr.grad, kr.grad

    def test_dsa_vs_dense_reference_h4(self):
        _skip_if_no_dsa(self)
        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk)
        out_dsa, dq_dsa, dkv_dsa = self._run_dsa(q, kf, idx, vr, sm)
        out_ref, dq_ref, dkv_ref = self._run_ref(q, kf, idx, vr, sm)
        self.assertEqual(list(out_dsa.shape), [b, s, h * self.Dv])
        self.assertGreater(_cos(out_dsa, out_ref), 0.99)
        self.assertGreater(_cos(dq_dsa, dq_ref), 0.99)
        self.assertGreater(_cos(dkv_dsa, dkv_ref), 0.99)

    def test_dsa_vs_dense_reference_h64(self):
        _skip_if_no_dsa(self)
        b, s, h, topk = 1, 256, 64, 4
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=13)
        out_dsa, dq_dsa, dkv_dsa = self._run_dsa(q, kf, idx, vr, sm)
        out_ref, dq_ref, dkv_ref = self._run_ref(q, kf, idx, vr, sm)
        self.assertGreater(_cos(out_dsa, out_ref), 0.99)
        self.assertGreater(_cos(dq_dsa, dq_ref), 0.99)
        self.assertGreater(_cos(dkv_dsa, dkv_ref), 0.99)

    def test_learnable_sink_matches_reference(self):
        # A finite learnable per-head sink: DSA forward + dq/dkv/d_sink grads
        # must match the dense reference that folds the sink into the softmax.
        _skip_if_no_dsa(self)
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=21)
        paddle.seed(29)
        # Moderate sink magnitude: a large random sink shrinks the effective
        # softmax mass and amplifies bf16 rounding in the dq GEMM, which would
        # make the comparison test bf16 noise rather than the sink math.
        sink0 = paddle.randn([h], dtype="float32") * 0.5

        qd = q.detach().clone()
        kd = kf.detach().clone()
        sink_d = sink0.detach().clone()
        for t in (qd, kd, sink_d):
            t.stop_gradient = False
        out_dsa, _ = block_sparse_mqa_attention_dsa(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_d,
        )
        out_dsa.sum().backward()

        s_kv = kf.shape[1]
        qr = q.detach().cast("float32")
        kr = kf.detach().cast("float32")
        sink_r = sink0.detach().clone()
        for t in (qr, kr, sink_r):
            t.stop_gradient = False
        vr_ = kr[:, :, : self.Dv]
        allow = _allow_mask(idx, vr, s_kv, self.BLOCK_B)
        out_ref = _ref_masked_attn(qr, kr, vr_, allow, sm, attn_sink=sink_r)
        out_ref = out_ref.reshape([b, s, h * self.Dv])
        out_ref.sum().backward()

        self.assertGreater(_cos(out_dsa, out_ref), 0.99)
        self.assertIsNotNone(sink_d.grad)
        self.assertGreater(_cos(qd.grad, qr.grad), 0.99)
        self.assertGreater(_cos(sink_d.grad, sink_r.grad), 0.99)

    def test_neg_sink_matches_sinkless(self):
        # Sinkless (attn_sink=None) must match a very-negative explicit sink:
        # exp(sink - m) -> 0, so the softmax denominator is unchanged.
        _skip_if_no_dsa(self)
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=33)
        neg_sink = paddle.full([h], -1e30, dtype="float32")
        out_none, _ = block_sparse_mqa_attention_dsa(
            q,
            kf,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
        )
        out_neg, _ = block_sparse_mqa_attention_dsa(
            q,
            kf,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=neg_sink,
        )
        self.assertGreater(_cos(out_none, out_neg), 0.999999)


class TestBlockSparseDSAPackedMultiDoc(unittest.TestCase):
    """Packed multi-document [40, 88, 133, 27] backward coverage for the DSA
    gather backend, plus a packed-vs-solo query-gradient equivalence check.

    Document-relative block ids + per-query ``[bos, eos)`` mean the DSA op
    natively supports packed sequences (nonzero ``bos``); the reference is the
    shared naive ``ref_block_sparse_mqa`` (散算子) over the same coordinates.
    """

    BLOCK_B = 64
    Dk = 576
    Dv = 512
    DOC_LENS = [40, 88, 133, 27]  # sum = 288, all unaligned to BLOCK_B

    def _make(self, h, topk, doc_lens, seed=7):
        s = sum(doc_lens)
        paddle.seed(seed)
        q = paddle.randn([1, s, h, self.Dk]).cast("bfloat16")
        kf = paddle.randn([1, s, self.Dk]).cast("bfloat16")
        vr = make_causal_valid_range(s, batch=1, doc_lengths=doc_lens)
        idx = build_random_block_indices(
            vr, topk, self.BLOCK_B, s, seed=seed + 1
        )
        sm_scale = 1.0 / math.sqrt(self.Dk)
        return q, kf, vr, idx, sm_scale

    def _run_dsa(self, q, kf, idx, vr, sm, sink=None):
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        qd = q.detach().clone()
        kd = kf.detach().clone()
        qd.stop_gradient = False
        kd.stop_gradient = False
        sink_d = None
        if sink is not None:
            sink_d = sink.detach().clone()
            sink_d.stop_gradient = False
        out, _ = block_sparse_mqa_attention_dsa(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_d,
        )
        out.sum().backward()
        return (
            out,
            qd.grad,
            kd.grad,
            (sink_d.grad if sink_d is not None else None),
        )

    def _run_ref(self, q, kf, idx, vr, sm, sink=None):
        qr = q.detach().cast("float32")
        kr = kf.detach().cast("float32")
        qr.stop_gradient = False
        kr.stop_gradient = False
        sink_r = None
        if sink is not None:
            sink_r = sink.detach().clone()
            sink_r.stop_gradient = False
        out = ref_block_sparse_mqa(
            qr,
            kr,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_r,
        )
        out.sum().backward()
        return (
            out,
            qr.grad,
            kr.grad,
            (sink_r.grad if sink_r is not None else None),
        )

    def test_packed_backward_sinkless(self):
        _skip_if_no_dsa(self)
        h, topk = 4, 4
        q, kf, vr, idx, sm = self._make(h, topk, self.DOC_LENS, seed=7)
        self.assertGreater(int(vr[0, :, 0].max()), 0)  # nonzero bos present
        out_d, dq_d, dkv_d, _ = self._run_dsa(q, kf, idx, vr, sm)
        out_r, dq_r, dkv_r, _ = self._run_ref(q, kf, idx, vr, sm)
        assert_close(self, "dsa_packed_out", out_d, out_r, min_cos=0.99)
        assert_close(self, "dsa_packed_dq", dq_d, dq_r, min_cos=0.99)
        assert_close(self, "dsa_packed_dkv", dkv_d, dkv_r, min_cos=0.99)

    def test_packed_backward_finite_sink(self):
        # REGRESSION for the DSA finite-sink dQ fix. Before the fix, a finite
        # per-head learnable sink on a PACKED multi-doc sequence made the DSA
        # backward dQ diverge from the dense reference (cos 0.976, max_abs 2.844,
        # RMSE 0.06364) while forward / sinkless grads / dKV / d_sink all matched.
        # Root cause: the cuDNN DSA backward's Dk!=Dv (absorbed-MQA 576/512)
        # branch consumes the passed LSE verbatim, and the forward returns a
        # KV-only LSE, so every p_k was overestimated for a finite sink. The
        # wrapper now feeds a sink-inclusive LSE (logaddexp(lse_kv, sink)) and
        # neutralizes the passed sink on that branch; the analytic d_sink still
        # uses the original KV-only LSE. This assertion is STRICT (min_cos=0.99)
        # and must now PASS. The TileLang oracle equivalent already passed,
        # isolating the original defect to the DSA backend.
        _skip_if_no_dsa(self)
        h, topk = 4, 4
        q, kf, vr, idx, sm = self._make(h, topk, self.DOC_LENS, seed=21)
        paddle.seed(29)
        # Moderate sink magnitude: a large sink shrinks the softmax mass and
        # turns the dq comparison into bf16 noise rather than the sink math.
        sink = paddle.randn([h], dtype="float32") * 0.5
        out_d, dq_d, dkv_d, ds_d = self._run_dsa(q, kf, idx, vr, sm, sink=sink)
        out_r, dq_r, dkv_r, ds_r = self._run_ref(q, kf, idx, vr, sm, sink=sink)
        assert_close(self, "dsa_packed_sink_out", out_d, out_r, min_cos=0.99)
        assert_close(self, "dsa_packed_sink_dq", dq_d, dq_r, min_cos=0.99)
        assert_close(self, "dsa_packed_sink_dkv", dkv_d, dkv_r, min_cos=0.99)
        self.assertIsNotNone(ds_d)
        assert_close(self, "dsa_packed_sink_dsink", ds_d, ds_r, min_cos=0.99)

    def test_packed_vs_solo_dq_equivalence(self):
        # A document run alone (bos=0) and the SAME document packed behind an
        # arbitrary-length prefix must produce identical query gradients for the
        # document's rows: block ids are document-relative and eos is
        # per-document causal, so the gather sees the exact same key columns
        # (shifted by the prefix). This is the backward analogue of the block
        # scorer's pack-equivalence property.
        _skip_if_no_dsa(self)
        h, topk = 4, 4
        prefix_len, doc_len = 91, 133  # both unaligned to BLOCK_B
        sm = 1.0 / math.sqrt(self.Dk)

        paddle.seed(101)
        q_doc = paddle.randn([1, doc_len, h, self.Dk]).cast("bfloat16")
        k_doc = paddle.randn([1, doc_len, self.Dk]).cast("bfloat16")
        q_pre = paddle.randn([1, prefix_len, h, self.Dk]).cast("bfloat16")
        k_pre = paddle.randn([1, prefix_len, self.Dk]).cast("bfloat16")

        # Document-relative block ids for the document's own rows (identical in
        # both runs because ids are relative to each row's bos).
        vr_solo = make_causal_valid_range(doc_len, batch=1)
        idx_doc = build_random_block_indices(
            vr_solo, topk, self.BLOCK_B, doc_len, seed=202
        )

        # ---- SOLO: the document alone (bos = 0). ----
        _, dq_solo, _, _ = self._run_dsa(q_doc, k_doc, idx_doc, vr_solo, sm)

        # ---- PACKED: [prefix, document]; the document's rows get bos>0. ----
        total = prefix_len + doc_len
        q_pack = paddle.concat([q_pre, q_doc], axis=1)
        k_pack = paddle.concat([k_pre, k_doc], axis=1)
        vr_pack = make_causal_valid_range(
            total, batch=1, doc_lengths=[prefix_len, doc_len]
        )
        vr_pre = make_causal_valid_range(prefix_len, batch=1)
        idx_pre = build_random_block_indices(
            vr_pre, topk, self.BLOCK_B, prefix_len, seed=303
        )
        idx_pack = paddle.concat([idx_pre, idx_doc], axis=1).contiguous()
        _, dq_pack, _, _ = self._run_dsa(q_pack, k_pack, idx_pack, vr_pack, sm)

        dq_pack_doc = dq_pack[:, prefix_len:, :, :]
        assert_close(
            self, "dsa_pack_vs_solo_dq", dq_pack_doc, dq_solo, min_cos=0.99
        )


def _rel_l2(got, ref):
    """Relative L2 error ||got-ref||_2 / (||ref||_2 + eps), fp32, finite-only."""
    import numpy as np

    g = got.astype("float32").numpy().reshape(-1)
    r = ref.astype("float32").numpy().reshape(-1)
    both = np.isfinite(g) & np.isfinite(r)
    g, r = g[both], r[both]
    return float(np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-12))


class TestBlockSparseDSAOnlineDimsRandomCotangent(unittest.TestCase):
    """Online ``ernielite_layer43`` DSA dims (Dk=512, Dv=448, H=64) driven by a
    RANDOM upstream cotangent instead of the all-ones ``out.sum()`` backward.

    Two things this class covers that the Dk=576/Dv=512 suites above do not:

    * **Dv=448 (< 512)** exercises the value zero-pad *re-lay* path in
      ``block_sparse_mqa_attention_dsa`` (``block_sparse_mqa_dsa.py`` pad_v=64:
      latent ``[val(448)|rope(64)]`` -> ``[val(448)|zeros(64)|rope(64)]``, run
      internally at Dk=576/Dv=512 then sliced back). This is the true online
      ernielite config (``model_config_separated/.../ernielite_layer43_0713_
      8k_bzz3``: kv_lora_rank=448, qk_rope_head_dim=64, num_attention_heads=64).
    * **Random cotangent.** ``out.sum().backward()`` feeds a constant all-ones
      upstream grad, which correlates the per-row error and can mask a defect in
      the sink-corrected dQ path. A per-element ``randn`` cotangent (identical
      values fed to both DSA and the fp32 reference) decorrelates it, so the
      dQ/dKV/dSink comparison reflects the kernel, not the probe.

    Dk is the *absorbed* query/key width, confirmed from the production tensor
    construction (multi_latent_attention.py:1928-1955), NOT from the raw config
    field: ``q_nope_absorbed = einsum("bshd,lhd->bshl", q_no_pe, W)`` projects
    the nope part (qk_nope_head_dim) up to ``kv_lora_rank`` via the reshaped
    kv_b_proj weight, then ``query = concat([q_nope_absorbed, q_pos_emb])`` and
    ``key = concat([kv_compressed, k_pos_emb])``. So Dk = kv_lora_rank +
    qk_rope_head_dim = 448 + 64 = 512 here (the pre-absorption q_head_dim
    qk_nope(192)+rope(64)=256 is a different, non-absorbed quantity and is NOT
    what the DSA op sees). dsv4's kv_lora_rank=512 gives the 576/512 case above.

    POST-FIX artifact: these all run with the wrapper's sink-inclusive-LSE dQ
    fix applied, and are meant as post-fix confirmation at online dims + random
    dO. They are NOT pre-fix evidence; the pre-fix finite-sink dQ defect was the
    separately-reproduced cos 0.976095 / max_abs 2.844 / RMSE 0.06364.

    Reference = fp32 naive ``ref_block_sparse_mqa`` (散算子) over identical
    coordinates. Every quantity is reported with max_abs / max_rel / RMSE /
    relative-L2 / cosine / dtype-aware allclose; assertions use a cosine floor
    (bf16 vs fp32 makes elementwise allclose too strict, but it is still logged).
    """

    BLOCK_B = 64
    Dk = 512  # absorbed q/k width = kv_lora_rank(448) + qk_rope_head_dim(64)
    Dv = 448  # ernielite_layer43 kv_lora_rank -> exercises the zero-pad re-lay
    H = 64
    DOC_LENS = [40, 88, 133, 27]  # packed ragged, all unaligned to BLOCK_B

    def _make(self, topk, seed):
        s = sum(self.DOC_LENS)
        paddle.seed(seed)
        q = paddle.randn([1, s, self.H, self.Dk]).cast("bfloat16")
        kf = paddle.randn([1, s, self.Dk]).cast("bfloat16")
        vr = make_causal_valid_range(s, batch=1, doc_lengths=self.DOC_LENS)
        idx = build_random_block_indices(
            vr, topk, self.BLOCK_B, s, seed=seed + 1
        )
        sm_scale = 1.0 / math.sqrt(self.Dk)
        return q, kf, vr, idx, sm_scale

    def _cotangent(self, s, seed):
        # fp32 master cotangent; cast per-branch to the output dtype so DSA
        # (bf16 out) and the reference (fp32 out) see the same numeric values.
        paddle.seed(seed)
        return paddle.randn([1, s, self.H * self.Dv], dtype="float32")

    def _run_dsa(self, q, kf, idx, vr, sm, cot, sink=None):
        from paddlefleet.cudnn_ops import block_sparse_mqa_attention_dsa

        qd, kd = q.detach().clone(), kf.detach().clone()
        qd.stop_gradient = False
        kd.stop_gradient = False
        sink_d = None
        if sink is not None:
            sink_d = sink.detach().clone()
            sink_d.stop_gradient = False
        out, _ = block_sparse_mqa_attention_dsa(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_d,
        )
        out.backward(cot.astype(out.dtype))
        return (
            out,
            qd.grad,
            kd.grad,
            (sink_d.grad if sink_d is not None else None),
        )

    def _run_ref(self, q, kf, idx, vr, sm, cot, sink=None):
        qr, kr = q.detach().cast("float32"), kf.detach().cast("float32")
        qr.stop_gradient = False
        kr.stop_gradient = False
        sink_r = None
        if sink is not None:
            sink_r = sink.detach().clone()
            sink_r.stop_gradient = False
        out = ref_block_sparse_mqa(
            qr,
            kr,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.Dv,
            attn_sink=sink_r,
        )
        out.backward(cot.astype(out.dtype))
        return (
            out,
            qr.grad,
            kr.grad,
            (sink_r.grad if sink_r is not None else None),
        )

    def _report(self, name, got, ref, min_cos=0.99, rel_tol=2e-2):
        m = assert_close(self, name, got, ref, min_cos=min_cos)
        rel = _rel_l2(got, ref)
        print(f"[{name}] rel_l2={rel:.3e}")
        self.assertLessEqual(
            rel, rel_tol, f"{name}: rel-L2 {rel:.3e} > {rel_tol:g}"
        )
        return m

    def test_online_dims_random_cotangent_sinkless(self):
        _skip_if_no_dsa(self)
        s = sum(self.DOC_LENS)
        q, kf, vr, idx, sm = self._make(topk=4, seed=41)
        self.assertGreater(int(vr[0, :, 0].max()), 0)  # nonzero bos present
        cot = self._cotangent(s, seed=141)
        out_d, dq_d, dkv_d, _ = self._run_dsa(q, kf, idx, vr, sm, cot)
        out_r, dq_r, dkv_r, _ = self._run_ref(q, kf, idx, vr, sm, cot)
        self.assertEqual(list(out_d.shape), [1, s, self.H * self.Dv])
        self._report("dsa448_rc_sinkless_out", out_d, out_r)
        self._report("dsa448_rc_sinkless_dq", dq_d, dq_r)
        self._report("dsa448_rc_sinkless_dkv", dkv_d, dkv_r)

    def test_online_dims_random_cotangent_finite_sink(self):
        # The finite learnable sink + Dk!=Dv (absorbed-MQA) path is the one the
        # wrapper's sink-inclusive-LSE backward fix targets. At online Dv=448
        # (zero-pad re-lay) with a random cotangent, dQ must still track the
        # dense reference; a regression here re-opens the packed finite-sink dQ
        # defect (previously cos ~0.976).
        _skip_if_no_dsa(self)
        s = sum(self.DOC_LENS)
        q, kf, vr, idx, sm = self._make(topk=4, seed=53)
        paddle.seed(59)
        sink = paddle.randn([self.H], dtype="float32") * 0.5
        cot = self._cotangent(s, seed=153)
        out_d, dq_d, dkv_d, ds_d = self._run_dsa(
            q, kf, idx, vr, sm, cot, sink=sink
        )
        out_r, dq_r, dkv_r, ds_r = self._run_ref(
            q, kf, idx, vr, sm, cot, sink=sink
        )
        self._report("dsa448_rc_sink_out", out_d, out_r)
        self._report("dsa448_rc_sink_dq", dq_d, dq_r)
        self._report("dsa448_rc_sink_dkv", dkv_d, dkv_r)
        self.assertIsNotNone(ds_d)
        self._report("dsa448_rc_sink_dsink", ds_d, ds_r)

    def test_online_dims_random_cotangent_negative_sink(self):
        # A very-negative sink must be indistinguishable from sinkless in every
        # differentiated quantity (exp(sink - m) -> 0), and both must match the
        # sinkless dense reference. Validates the negative-sink branch under the
        # zero-pad re-lay + random cotangent.
        _skip_if_no_dsa(self)
        s = sum(self.DOC_LENS)
        q, kf, vr, idx, sm = self._make(topk=4, seed=67)
        cot = self._cotangent(s, seed=167)
        neg_sink = paddle.full([self.H], -1e30, dtype="float32")
        out_none, dq_none, dkv_none, _ = self._run_dsa(q, kf, idx, vr, sm, cot)
        out_neg, dq_neg, dkv_neg, _ = self._run_dsa(
            q, kf, idx, vr, sm, cot, sink=neg_sink
        )
        # DSA(neg sink) vs DSA(sinkless): must be near bit-identical.
        self._report(
            "dsa448_rc_neg_vs_none_out", out_neg, out_none, min_cos=0.999999
        )
        self._report(
            "dsa448_rc_neg_vs_none_dq", dq_neg, dq_none, min_cos=0.999999
        )
        self._report(
            "dsa448_rc_neg_vs_none_dkv", dkv_neg, dkv_none, min_cos=0.999999
        )
        # DSA(neg sink) vs the sinkless dense reference.
        _, dq_r, dkv_r, _ = self._run_ref(q, kf, idx, vr, sm, cot)
        self._report("dsa448_rc_neg_vs_ref_dq", dq_neg, dq_r)
        self._report("dsa448_rc_neg_vs_ref_dkv", dkv_neg, dkv_r)


if __name__ == "__main__":
    unittest.main()
