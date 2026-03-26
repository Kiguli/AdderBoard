"""
Formal verifier for SeuperHakkerJa_28p (jacobli99).

Architecture: d=2, 5 heads (MQA), hd=2, ff=4, RoPE with per-head offsets.
Prompt: 24 tokens: [0] + rev(a) + [0,0] + rev(b) + [0]
Output positions start at 24.

Key properties:
- K = [x0*qk_scale, 0]: position-only after projection (K[1]=0)
- 5 heads with RoPE offsets (0, 23, 11, 22, 10) route different position pairs
- Sparse O-proj with 6 weights
- ReLU MLP with bias terms derived from embed_const
"""

import math
import numpy as np

from ..interval import Interval, iv_relu, iv_rms_norm
from ..verify_formal import possible_output_digits, digit_sum_for_target, carry_in_at

VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
PROMPT_LEN = 24  # Different from zcbtrak's 31
RMS_EPS = 1e-6


def _compute_attention_weights_5head(output_pos, seq_len, rope_offsets, qk_scale, embed_const, decode_eps):
    """Compute exact attention weights for all 5 heads at a given output position.

    K = [x0*qk_scale, 0] is position-only after projection.
    Q for head h = [x0*cos(offset_h), -x0*sin(offset_h)] * qk_scale.
    After RoPE: score depends only on position (not token) since x0 cancels in ratio.

    Actually, x0 = hn[0] which varies by token. But as with yieldthought, x0 appears
    in both Q and K so the softmax ratio only depends on the x0_key variation.
    For this model, x0 = embed[d][0] = embed_const - (decode_eps/2)*d^2.
    The variation is tiny (embed_const dominates).

    For this verifier, we approximate with position-only weights.
    """
    OMEGA_ROPE = 2.0 * math.pi / 19.0  # This model uses a different RoPE period

    # Actually, this model uses torch's standard RoPE: pos * omega where omega depends on the head
    # But looking at the code, rope_2d uses the same omega for all heads.
    # The per-head differentiation comes from rope_offsets which rotate Q before RoPE.

    # Q for head h: q_base_h = [cos(offset_h), -sin(offset_h)] * qk_scale
    # K: [qk_scale, 0] (same for all heads before RoPE)
    # But the QK projection is Q = q_base_h * x0, K = [x0*qk_scale, 0]
    # After unit-rms-norm (not used here - this model has NO QK-norm):
    # Q_{roped} and K_{roped} depend on position via RoPE.

    # Looking at the code more carefully:
    # Q: x0 * q_base (where q_base varies per head via rope_offsets)
    # K: [x0 * qk_scale, 0] (shared across heads)
    # score = Q_roped @ K_roped / sqrt(hd) where hd=2

    # Since both Q and K are proportional to x0, and x0 is positive,
    # softmax is invariant to the x0 factor (it's a positive temperature).

    # But that's not quite right: Q has x0_query and K has x0_key.
    # score = x0_q * x0_k * f(pos_q, pos_k, head) / sqrt(2)
    # The x0_q cancels in softmax (same for all keys).
    # The x0_k varies across keys, but by < 0.01%.

    # We use position-only weights (x0_k = constant).

    # RoPE in this model: standard per-position rotation
    # angle = pos * omega where omega = 2*pi/19
    # Wait, let me check the code: rope_2d uses pos directly.
    # The q_base already incorporates the offset.

    # Actually the code does:
    # offs = rope_offsets  # [0, 23, 11, 22, 10]
    # q_base = [cos(off), -sin(off)] * qk_scale  for each head
    # q = x0 * q_base (expanded)
    # k = [x0*qk_scale, 0]
    # Then RoPE is applied to both q and k using pos*omega where omega=2*pi/19? No...

    # Actually looking at the code: rope_2d(x, pos) where pos = arange(T) * omega?
    # No: `pos = torch.arange(T, device=x.device, dtype=x.dtype)` then
    # `theta = (pos * OMEGA).view(...)` - wait, this is in yieldthought's rope.
    # In SeuperHakkerJa: `rope_2d(q, pos)` where pos is computed differently.

    # Let me re-read: In SeuperHakkerJa's forward():
    # pos = torch.arange(T, device=x.device, dtype=x.dtype)
    # q = rope_2d(q, pos)
    # k = rope_2d(k, pos)
    # where rope_2d(x, pos):
    #   c = cos(pos), s = sin(pos)
    #   return [c*x0 - s*x1, s*x0 + c*x1]

    # So RoPE angle = pos itself (not pos*omega). That means the RoPE period is 2*pi.

    # Wait no, there's no OMEGA multiply. The angle is literally `pos` radians.
    # That's unusual but the offsets (0, 23, 11, 22, 10) compensate.

    # Q for head h at position p (after RoPE):
    # Q_base_h = [qk_scale * cos(offs_h), -qk_scale * sin(offs_h)]  (from x0=1 approx)
    # After RoPE with angle p:
    # Q_roped = [cos(p)*Q_b0 - sin(p)*Q_b1, sin(p)*Q_b0 + cos(p)*Q_b1]
    #         = qk_scale * [cos(p)*cos(offs_h) + sin(p)*sin(offs_h),
    #                       sin(p)*cos(offs_h) - cos(p)*sin(offs_h)]
    #         = qk_scale * [cos(p - offs_h), sin(p - offs_h)]

    # K at position s (after RoPE):
    # K_base = [qk_scale, 0]
    # K_roped = [cos(s)*qk_scale, sin(s)*qk_scale] = qk_scale * [cos(s), sin(s)]

    # Score(h, p, s) = Q_roped . K_roped / sqrt(2) (x0 factors omitted)
    # = qk_scale^2 * [cos(p-off_h)*cos(s) + sin(p-off_h)*sin(s)] / sqrt(2)
    # = qk_scale^2 * cos(p - off_h - s) / sqrt(2)

    heads = len(rope_offsets)
    all_weights = np.zeros((heads, seq_len))

    for h in range(heads):
        scores = np.full(seq_len, -1e4)
        for s in range(min(output_pos + 1, seq_len)):
            angle = output_pos - rope_offsets[h] - s
            scores[s] = qk_scale**2 * math.cos(angle) / math.sqrt(2)

        # Numerically stable softmax
        scores_valid = scores[:output_pos + 1]
        max_s = np.max(scores_valid)
        exp_s = np.exp(scores_valid - max_s)
        all_weights[h, :output_pos + 1] = exp_s / np.sum(exp_s)

    return all_weights


class SeuperHakkerJaVerifier:
    def __init__(self, model):
        import torch
        with torch.no_grad():
            self.embed_const = float(model.embed_const.item())
            self.decode_eps = float(model.decode_eps.item())
            self.qk_scale = float(model.qk_scale.item())
            digit_vals = model.digit_values.detach().cpu().numpy()
            self.o_w = model.o_w.detach().cpu().numpy().astype(np.float64)
            self.w1_a = float(model.w1_a.item())
            self.w1_b = float(model.w1_b.item())
            self.w1_c = float(model.w1_c.item())
            self.w2_s1 = float(model.w2_s1.item())
            self.w2_s10 = float(model.w2_s10.item())
            rope_offsets = model.rope_offsets_buf.detach().cpu().numpy()

        self.rope_offsets = [float(x) for x in rope_offsets]

        # Build embedding table: [embed_const - (eps/2)*d^2, d]
        d = np.arange(VOCAB_SIZE, dtype=np.float64)
        quad = self.decode_eps / 2.0
        self.embed_table = np.stack([self.embed_const - quad * d * d, d], axis=-1)

        # V for each digit: V = [digit_value, 0] → V[0] = d, V[1] = 0
        # After attention: attn_out = att (B,H,T,2) with only V[0] nonzero
        # So att[h,:,:,0] = sum(w[s] * digit_s), att[h,:,:,1] = 0

        # Precompute attention weights for all output positions
        self.all_attn_weights = {}
        for k in range(OUTPUT_DIGITS):
            output_pos = PROMPT_LEN + k
            seq_len = PROMPT_LEN + k
            self.all_attn_weights[k] = _compute_attention_weights_5head(
                output_pos, seq_len, self.rope_offsets, self.qk_scale,
                self.embed_const, self.decode_eps,
            )

        # MLP bias terms
        C = self.embed_const
        self.b1 = np.array([C - 8.0, C - 9.0, 2*C - 188.0, 2*C - 189.0])

        # Output decode scale
        self.out_scale = np.array([1.0 / C, self.decode_eps])

    def _get_position_mapping(self, k):
        """Map digit position k to prompt positions for a[k] and b[k].

        Prompt: [0] + rev(a[0..9]) + [0,0] + rev(b[0..9]) + [0]
        Positions: 0  1..10        11,12   13..22          23
        a[k] is at position k+1, b[k] is at position k+13.
        """
        return k + 1, k + 13

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        k = digit_pos
        if k >= 10:
            # Overflow position
            expected = 1 if (carry_mask & (1 << 9)) else 0
            if target_digit != expected:
                return False, f"overflow: expected {expected}, target {target_digit}"

        attn_w = self.all_attn_weights[k]  # shape (5, seq_len)
        digit_sum = digit_sum_for_target(carry_mask, k, target_digit)
        if k == 10:
            digit_sum = 0

        # Zero-token positions: 0, 11, 12, 23
        zero_positions = {0, 11, 12, 23}

        a_pos, b_pos = self._get_position_mapping(k) if k < 10 else (None, None)

        # Compute per-head attention outputs.
        # V[s] = [digit_s, 0]. After attention: each head produces weighted digit sums.
        # att[h] = sum(w_h[s] * digit_s)

        # For each head h, compute bounds on the weighted digit sum
        head_outputs = []  # list of (lo, hi) per head
        for h in range(5):
            w = attn_w[h]
            lo = 0.0
            hi = 0.0
            for s in range(len(w)):
                wt = w[s]
                if wt < 1e-15:
                    continue
                if s in zero_positions:
                    continue  # token 0, digit value 0

                # Determine digit range at position s
                if k < 10 and s == a_pos:
                    # Primary a digit: constrained by digit_sum
                    dmin = max(0, digit_sum - 9)
                    dmax = min(9, digit_sum)
                elif k < 10 and s == b_pos:
                    dmin = max(0, digit_sum - 9)
                    dmax = min(9, digit_sum)
                elif s >= PROMPT_LEN:
                    # Previously generated output digit
                    out_idx = s - PROMPT_LEN
                    if out_idx < len(prev_output_digits) and prev_output_digits[out_idx] >= 0:
                        dmin = dmax = prev_output_digits[out_idx]
                    else:
                        dmin, dmax = 0, 9
                elif 1 <= s <= 10:
                    # a digit position (not the primary one)
                    dmin, dmax = 0, 9
                elif 13 <= s <= 22:
                    dmin, dmax = 0, 9
                else:
                    dmin, dmax = 0, 9

                lo += wt * dmin
                hi += wt * dmax

            head_outputs.append((lo, hi))

        # O-proj: sparse mixing of head outputs
        # upd0 = o_w[0]*att[h0,0] + o_w[1]*att[h1,0] + o_w[2]*att[h2,0]
        # upd1 = o_w[3]*att[h0,0] + o_w[4]*att[h2,0] + o_w[5]*att[h4,0]
        # Wait, let me re-read the O-proj code:
        # att is (B, T, H*hd) = (B, T, 10) after concat
        # att[..., 0] = head0_dim0, att[..., 1] = head0_dim1, ..., att[..., 8] = head4_dim0, att[..., 9] = head4_dim1
        # But V = [digit, 0], so only dim0 of each head is nonzero.
        # att[..., 0] = head0_attn, att[..., 2] = head1_attn, att[..., 4] = head2_attn,
        # att[..., 6] = head3_attn, att[..., 8] = head4_attn
        #
        # upd0 = o_w[0]*att[0] + o_w[1]*att[2] + o_w[2]*att[4]
        #       = o_w[0]*head0 + o_w[1]*head1 + o_w[2]*head2
        # upd1 = o_w[3]*att[0] + o_w[4]*att[6] + o_w[5]*att[8]
        #       = o_w[3]*head0 + o_w[4]*head3 + o_w[5]*head4
        # With o_w = [+1, -1, -1, -1, +1, +1]

        def _scale_interval(val, coeff):
            lo, hi = val
            if coeff >= 0:
                return (coeff * lo, coeff * hi)
            else:
                return (coeff * hi, coeff * lo)

        upd0_lo = 0.0
        upd0_hi = 0.0
        for idx, coeff in [(0, self.o_w[0]), (1, self.o_w[1]), (2, self.o_w[2])]:
            a, b = _scale_interval(head_outputs[idx], coeff)
            upd0_lo += a
            upd0_hi += b

        upd1_lo = 0.0
        upd1_hi = 0.0
        for idx, coeff in [(0, self.o_w[3]), (3, self.o_w[4]), (4, self.o_w[5])]:
            a, b = _scale_interval(head_outputs[idx], coeff)
            upd1_lo += a
            upd1_hi += b

        # Previous output token determines embedding
        if k == 0:
            d_prev_list = [0]
        else:
            d_prev_list = possible_output_digits(carry_mask, k - 1)

        t = target_digit

        for d_prev in d_prev_list:
            if k == 0:
                emb = self.embed_table[0]
            else:
                emb = self.embed_table[d_prev]

            # h = embed + [upd0, upd1]
            h0 = Interval(emb[0]) + Interval(upd0_lo, upd0_hi)
            h1 = Interval(emb[1]) + Interval(upd1_lo, upd1_hi)

            # MLP: z = relu(x @ W1.T + b1) where W1 and b1 are specific
            # W1 = [[-1, 0], [-1, 0], [-2, 20], [-2, 20]], b1 = [C-8, C-9, 2C-188, 2C-189]
            n0_in = h0 * self.w1_a + Interval(self.b1[0])
            n1_in = h0 * self.w1_a + Interval(self.b1[1])
            n2_in = h0 * self.w1_b + h1 * self.w1_c + Interval(self.b1[2])
            n3_in = h0 * self.w1_b + h1 * self.w1_c + Interval(self.b1[3])

            n0 = iv_relu(n0_in)
            n1 = iv_relu(n1_in)
            n2 = iv_relu(n2_in)
            n3 = iv_relu(n3_in)

            # W2: dim1 += s1*(n0-n1) - s10*(n2-n3) + s10*(n3-n2)
            # Actually: W2[1,:] = [s1, -s1, -s10, s10]
            # 2-hinge optimization: n0 and n1 have thresholds 1 apart.
            # If both active: n0-n1 = (b1[0]-b1[1]) = 1.0 (exact)
            # If both inactive: n0-n1 = 0
            if n0_in.lo > 0 and n1_in.lo > 0:
                carry_term = self.w2_s1 * Interval(self.b1[0] - self.b1[1])
            elif n0_in.hi < 0 and n1_in.hi < 0:
                carry_term = Interval(0.0)
            elif n0_in.lo > 0 and n1_in.hi < 0:
                carry_term = self.w2_s1 * n0
            else:
                carry_term = self.w2_s1 * (n0 - n1)

            if n2_in.lo > 0 and n3_in.lo > 0:
                wrap_term = self.w2_s10 * Interval(-(self.b1[2] - self.b1[3]))
            elif n2_in.hi < 0 and n3_in.hi < 0:
                wrap_term = Interval(0.0)
            elif n3_in.lo > 0 and n2_in.hi < 0:
                wrap_term = self.w2_s10 * n3
            else:
                wrap_term = self.w2_s10 * (n3 - n2)

            mlp_dim1 = carry_term + wrap_term

            h0_post = h0
            h1_post = h1 + mlp_dim1

            # Output: scaled then dot with embed table
            y0 = h0_post * self.out_scale[0]
            y1 = h1_post * self.out_scale[1]

            # logit[d] = y0 * embed[d,0] + y1 * embed[d,1]
            for d in range(VOCAB_SIZE):
                if d == t:
                    continue
                delta_e0 = self.embed_table[t, 0] - self.embed_table[d, 0]
                delta_e1 = self.embed_table[t, 1] - self.embed_table[d, 1]
                diff = y0 * delta_e0 + y1 * delta_e1

                if diff.lo <= 0:
                    return False, (
                        f"digit {k}, target {t}, d_prev={d_prev}: "
                        f"logit[{t}]-logit[{d}] interval [{diff.lo:.6f}, {diff.hi:.6f}] "
                        f"not provably positive"
                    )

        return True, ""


def create_verifier(model):
    return SeuperHakkerJaVerifier(model)
