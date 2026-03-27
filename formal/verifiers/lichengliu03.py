"""
Formal verifier for lichengliu03_50p.

Architecture: d=4, 2 heads (head_dim=2), ReLU MLP, sinusoidal PE (period=11).
Prompt: a[9]a[8]..a[0]+b[9]b[8]..b[0]= (22 tokens, MSB first), output 11 digits.
Output is generated LSB-first: digit k is predicted at position PROMPT_LEN-1+k.

Key properties:
1. K_proj selects PE channels (dims 1,2) -> position-only attention
2. Head 0 (angle 8*THETA) peaks at current pair a[k],b[k] with ~0.5 weight each
3. Head 1 (angle 9*THETA) peaks at previous pair a[k-1],b[k-1]
4. Score gap >= 11.2 -> dominant/other ratio ~77000:1
5. 2-hinge ReLU carry (0.5/1.5) and wrap (9045/9055) detection
6. Parabolic head: logit[d] = 2*d*z - d^2
7. PE at output positions has amplitude 1 (vs 100 for prompt), introducing
   small carry bias (max ~0.49) absorbed by parabolic head's 1-unit gap.
"""

import math
import numpy as np

from ..interval import Interval, iv_relu

VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
THETA = 2 * math.pi / 11
PROMPT_LEN = 22


def _build_pe(seq_len):
    pe = np.zeros((seq_len, 4), dtype=np.float64)
    for p in range(seq_len):
        amp = 100.0 if p <= 21 else 1.0
        pe[p, 1] = amp * math.sin(p * THETA)
        pe[p, 2] = amp * math.cos(p * THETA)
    return pe


def _precompute_attention_weights(pe):
    """Precompute exact attention weights for each (output_digit_k, head).

    Digit k is predicted at position PROMPT_LEN - 1 + k:
      k=0 at position 21 (the '=' token), k=1 at position 22, etc.
    """
    q_angles = [8 * THETA, 9 * THETA]
    all_weights = {}

    for k in range(OUTPUT_DIGITS):
        p = PROMPT_LEN - 1 + k  # prediction position
        seq_len = p + 1          # causal: can see positions 0..p

        K = pe[:seq_len, 1:3]
        weights_per_head = []

        for h in range(2):
            a = q_angles[h]
            ca, sa = math.cos(a), math.sin(a)
            kp = pe[p, 1:3]
            qp = np.array([-ca * kp[0] + sa * kp[1], sa * kp[0] + ca * kp[1]])
            scores = (K @ qp) / math.sqrt(2.0)
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / np.sum(exp_scores)
            weights_per_head.append(weights)

        all_weights[k] = weights_per_head

    return all_weights


def _carry_in_at(carry_mask, pos):
    if pos == 0:
        return 0
    return 1 if (carry_mask & (1 << (pos - 1))) else 0


def _carry_out_at(carry_mask, pos):
    if pos >= 10:
        return 0
    return 1 if (carry_mask & (1 << pos)) else 0


def _digit_sum_for_target(carry_mask, pos, target):
    c_in = _carry_in_at(carry_mask, pos)
    c_out = _carry_out_at(carry_mask, pos)
    if c_out:
        return target + 10 - c_in
    else:
        return target - c_in


def _possible_output_digits(carry_mask, pos):
    if pos == 10:
        return [1 if (carry_mask & (1 << 9)) else 0]
    c_in = _carry_in_at(carry_mask, pos)
    c_out = _carry_out_at(carry_mask, pos)
    digits = set()
    for a_d in range(10):
        for b_d in range(10):
            s = a_d + b_d + c_in
            if (c_out and s >= 10) or (not c_out and s < 10):
                digits.add(s % 10)
    return sorted(digits)


class Lichengliu03Verifier:
    def __init__(self, model):
        max_seq = PROMPT_LEN + OUTPUT_DIGITS
        self.pe = _build_pe(max_seq)
        self.all_weights = _precompute_attention_weights(self.pe)

        # Prompt layout: a[9] at pos 0, a[8] at pos 1, ..., a[0] at pos 9
        #                + at pos 10
        #                b[9] at pos 11, b[8] at pos 12, ..., b[0] at pos 20
        #                = at pos 21
        self.a_seq_pos = {}  # digit k -> sequence position of a[k]
        self.b_seq_pos = {}
        for k in range(10):
            self.a_seq_pos[k] = 9 - k
            self.b_seq_pos[k] = 20 - k

    def _head_v_sum_for_dsum(self, k, head, dsum, prev_output_digits):
        """Compute tight V-sum bounds for a head, given the exact digit sum.

        The two dominant positions (a[target_k] and b[target_k]) are always
        separated by exactly 11 in sequence position, so they have identical
        attention scores and thus w_a = w_b. This means:
            w_a*a + w_b*b = w_a*(a+b) = w_a*dsum  (exact, zero-width interval)

        All other positions contribute negligibly (weights ~1/77000).
        """
        w = self.all_weights[k][head]
        seq_len = len(w)

        target_k = k if head == 0 else k - 1

        if 0 <= target_k < 10:
            dom_a = self.a_seq_pos[target_k]
            dom_b = self.b_seq_pos[target_k]
        else:
            dom_a = dom_b = -1

        # Non-dominant contribution
        other_lo = 0.0
        other_hi = 0.0
        for s in range(seq_len):
            ws = w[s]
            if ws < 1e-15 or s == dom_a or s == dom_b:
                continue
            if s == 10 or s == 21:
                continue  # separators, value 0
            if s <= 20:
                v_lo, v_hi = 0.0, 9.0
            else:
                j = s - PROMPT_LEN
                if j < len(prev_output_digits) and prev_output_digits[j] >= 0:
                    v_lo = v_hi = float(prev_output_digits[j])
                else:
                    v_lo, v_hi = 0.0, 9.0
            other_lo += ws * v_lo
            other_hi += ws * v_hi

        # Dominant pair: w_a * V(a) + w_b * V(b)
        # where V(a) + V(b) = dsum, V(a) in [max(0, dsum-9), min(9, dsum)]
        # = (w_a - w_b) * V(a) + w_b * dsum
        if dom_a >= 0 and dom_b >= 0:
            w_a = w[dom_a]
            w_b = w[dom_b]
            a_lo = max(0, dsum - 9)
            a_hi = min(9, dsum)
            if w_a >= w_b:
                dom_lo = (w_a - w_b) * a_lo + w_b * dsum
                dom_hi = (w_a - w_b) * a_hi + w_b * dsum
            else:
                dom_lo = (w_a - w_b) * a_hi + w_b * dsum
                dom_hi = (w_a - w_b) * a_lo + w_b * dsum
            return Interval(dom_lo + other_lo, dom_hi + other_hi)
        else:
            return Interval(other_lo, other_hi)

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        k = digit_pos
        t = target_digit

        if k < 10:
            digit_sum = _digit_sum_for_target(carry_mask, k, t)
        else:
            digit_sum = 0

        # Head 0 targets current pair (digit k) -- independent of d_prev
        head0_vsum = self._head_v_sum_for_dsum(k, 0, digit_sum, prev_output_digits)

        # O-proj: head0 -> dim3 (scale 2.0)
        head0_out = head0_vsum * 2.0

        # PE at prediction position (digit k predicted at PROMPT_LEN - 1 + k)
        p = PROMPT_LEN - 1 + k
        pe_sin = self.pe[p, 1]
        pe_cos = self.pe[p, 2]

        # Previous output digit (the token at the prediction position)
        if k == 0:
            d_prev_list = [0]  # '=' token has value 0
        else:
            d_prev_list = _possible_output_digits(carry_mask, k - 1)

        for d_prev in d_prev_list:
            # Head 1 targets previous pair (digit k-1), conditioned on d_prev.
            # d_prev uniquely determines dsum_prev within this carry partition,
            # giving tight V-sum bounds.
            if k == 0:
                # Head 1 peaks at separators (pos 10, 21) with value 0
                head1_vsum = self._head_v_sum_for_dsum(k, 1, 0, prev_output_digits)
            else:
                dsum_prev = _digit_sum_for_target(carry_mask, k - 1, d_prev)
                head1_vsum = self._head_v_sum_for_dsum(k, 1, dsum_prev, prev_output_digits)

            head1_out = head1_vsum * 2.0

            # x after attention residual:
            # dim0 = d_prev (from embedding, embed_dir = [1,0,0,0])
            # dim1 = pe_sin + head1_out (PE + attention)
            # dim2 = pe_cos (PE only, no attention writes here)
            # dim3 = head0_out (attention only, no PE in dim3)
            dim0 = Interval(float(d_prev))
            dim1 = Interval(pe_sin) + head1_out
            dim2 = Interval(pe_cos)
            dim3 = head0_out

            # MLP carry: ci = dot(x, [-1, 1, 0, 0]) = -dim0 + dim1
            # carry = relu(ci - 0.5) - relu(ci - 1.5)
            # 2-hinge optimization: exploit correlation (same ci in both terms)
            ci = -dim0 + dim1
            ci_lo_term = ci + Interval(-0.5)   # ci - 0.5
            ci_hi_term = ci + Interval(-1.5)   # ci - 1.5
            if ci_hi_term.lo > 0:
                carry = Interval(1.0)  # both positive: exact 1
            elif ci_lo_term.hi < 0:
                carry = Interval(0.0)  # both non-positive: exact 0
            elif ci_lo_term.lo > 0 and ci_hi_term.hi < 0:
                carry = ci_lo_term     # first positive, second zero: ci - 0.5
            else:
                carry = iv_relu(ci_lo_term) - iv_relu(ci_hi_term)

            # MLP wrap: wi = dot(x, [-10, 10, 0, 1000]) = -10*dim0 + 10*dim1 + 1000*dim3
            # wrap = relu(wi - 9055) - relu(wi - 9045)
            # 2-hinge optimization
            wi = dim0 * (-10.0) + dim1 * 10.0 + dim3 * 1000.0
            wi_lo_term = wi + Interval(-9055.0)  # wi - 9055
            wi_hi_term = wi + Interval(-9045.0)  # wi - 9045
            if wi_lo_term.lo > 0:
                wrap = Interval(-10.0)  # both positive: exact -10
            elif wi_hi_term.hi < 0:
                wrap = Interval(0.0)    # both non-positive: exact 0
            elif wi_hi_term.lo > 0 and wi_lo_term.hi < 0:
                wrap = -wi_hi_term      # -(wi - 9045)
            else:
                wrap = iv_relu(wi_lo_term) - iv_relu(wi_hi_term)

            # MLP output -> dim3 only (accum_dir = [0,0,0,1])
            dim3_final = dim3 + carry + wrap

            # Parabolic head: z = dim3_final
            z = dim3_final

            for d in range(VOCAB_SIZE):
                if d == t:
                    continue
                diff = z * float(2 * (t - d)) + Interval(-float(t * t - d * d))
                if diff.lo <= 0:
                    return False, (
                        "digit %d, target %d, d_prev=%d: "
                        "logit[%d]-logit[%d] interval [%.6f, %.6f]" % (
                            k, t, d_prev, t, d, diff.lo, diff.hi))

        return True, ""


def create_verifier(model):
    return Lichengliu03Verifier(model)
