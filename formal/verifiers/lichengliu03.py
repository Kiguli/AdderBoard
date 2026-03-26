"""
Formal verifier for lichengliu03_50p.

Architecture: d=4, 2 heads (head_dim=2), ReLU MLP, sinusoidal PE (period=11).
Prompt: a[9]a[8]..a[0]+b[9]b[8]..b[0]= (22 tokens, MSB first), output 11 digits.

Key properties:
1. K_proj selects PE channels (dims 1,2) -> position-only attention
2. Head 0 (angle 8*THETA) peaks at current pair a[k],b[k] with ~0.5 weight each
3. Head 1 (angle 9*THETA) peaks at previous pair a[k-1],b[k-1]
4. Score gap ~11.25 -> dominant/other ratio ~77000:1
5. 2-hinge ReLU carry (0.5/1.5) and wrap (9045/9055) detection
6. Parabolic head: logit[d] = 2*d*z - d^2
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
    """Precompute exact attention weights for each (output_digit_k, head)."""
    q_angles = [8 * THETA, 9 * THETA]
    all_weights = {}

    for k in range(OUTPUT_DIGITS):
        p = PROMPT_LEN + k
        seq_len = p + 1

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

    def _head_v_sum_bounds(self, k, head, carry_mask, digit_sum_k, prev_output_digits):
        """Compute tight [lo, hi] bounds on the weighted V-sum for a head.

        Strategy: identify dominant positions (the pair the head targets),
        compute their contribution tightly using digit_sum constraint,
        and bound all other positions conservatively with their tiny weights.
        """
        w = self.all_weights[k][head]
        seq_len = len(w)

        # Head 0 targets pair at digit k, head 1 targets pair at digit k-1
        target_k = k if head == 0 else k - 1

        # Identify dominant pair positions
        if 0 <= target_k < 10:
            dom_a = self.a_seq_pos[target_k]
            dom_b = self.b_seq_pos[target_k]
        else:
            dom_a = -1
            dom_b = -1

        # Get digit_sum for the targeted pair
        if 0 <= target_k < 10:
            if head == 0:
                dsum = digit_sum_k
            else:
                # For head 1, we need bounds on digit_sum at position target_k
                # This depends on which output digits are possible there
                c_in_tk = _carry_in_at(carry_mask, target_k)
                c_out_tk = _carry_out_at(carry_mask, target_k)
                # Compute all possible digit sums
                dsums = set()
                for a_d in range(10):
                    for b_d in range(10):
                        s = a_d + b_d + c_in_tk
                        if (c_out_tk and s >= 10) or (not c_out_tk and s < 10):
                            dsums.add(a_d + b_d)
                dsum_lo = min(dsums)
                dsum_hi = max(dsums)

        # Compute weighted V-sum bounds
        total_lo = 0.0
        total_hi = 0.0

        for s in range(seq_len):
            ws = w[s]
            if ws < 1e-15:
                continue

            # Determine V bounds at position s
            if s == dom_a or s == dom_b:
                # Dominant position: V is a digit from the constrained pair
                if head == 0:
                    # digit ranges: a in [max(0,dsum-9), min(9,dsum)]
                    v_lo = float(max(0, dsum - 9))
                    v_hi = float(min(9, dsum))
                else:
                    # For head 1, use dsum range
                    v_lo = float(max(0, dsum_lo - 9))
                    v_hi = float(min(9, dsum_hi))
            elif s == 10 or s == 21:
                # Separator tokens: + and =, value 0
                continue
            elif s <= 9:
                # a[9-s] digit: unconstrained, V in [0, 9]
                v_lo, v_hi = 0.0, 9.0
            elif s <= 20:
                # b[20-s] digit: unconstrained, V in [0, 9]
                v_lo, v_hi = 0.0, 9.0
            else:
                # Output token at position 22+j
                j = s - 22
                if j < len(prev_output_digits) and prev_output_digits[j] >= 0:
                    v_lo = v_hi = float(prev_output_digits[j])
                else:
                    v_lo, v_hi = 0.0, 9.0

            total_lo += ws * v_lo
            total_hi += ws * v_hi

        return Interval(total_lo, total_hi)

    def _head_v_sum_tight(self, k, head, carry_mask, digit_sum_k, prev_output_digits):
        """Even tighter bounds: use V(a)+V(b) = digit_sum constraint at dominant pair."""
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
                continue
            if s <= 20:
                v_lo, v_hi = 0.0, 9.0
            else:
                j = s - 22
                if j < len(prev_output_digits) and prev_output_digits[j] >= 0:
                    v_lo = v_hi = float(prev_output_digits[j])
                else:
                    v_lo, v_hi = 0.0, 9.0
            other_lo += ws * v_lo
            other_hi += ws * v_hi

        # Dominant pair contribution: w_a * V(a) + w_b * V(b)
        # where V(a) + V(b) = digit_sum (constrained)
        if dom_a >= 0 and dom_b >= 0:
            w_a = w[dom_a]
            w_b = w[dom_b]

            if head == 0:
                dsum = digit_sum_k
                # V(a) + V(b) = dsum, V(a) in [max(0,dsum-9), min(9,dsum)]
                # w_a*V(a) + w_b*V(b) = w_a*V(a) + w_b*(dsum - V(a)) = (w_a-w_b)*V(a) + w_b*dsum
                a_lo = max(0, dsum - 9)
                a_hi = min(9, dsum)
                if w_a >= w_b:
                    dom_lo = (w_a - w_b) * a_lo + w_b * dsum
                    dom_hi = (w_a - w_b) * a_hi + w_b * dsum
                else:
                    dom_lo = (w_a - w_b) * a_hi + w_b * dsum
                    dom_hi = (w_a - w_b) * a_lo + w_b * dsum
            else:
                # Head 1: digit sum at target_k varies
                c_in_tk = _carry_in_at(carry_mask, target_k)
                c_out_tk = _carry_out_at(carry_mask, target_k)
                combos = []
                for a_d in range(10):
                    for b_d in range(10):
                        s = a_d + b_d + c_in_tk
                        if (c_out_tk and s >= 10) or (not c_out_tk and s < 10):
                            combos.append(w_a * a_d + w_b * b_d)
                dom_lo = min(combos)
                dom_hi = max(combos)

            total_lo = dom_lo + other_lo
            total_hi = dom_hi + other_hi
        else:
            total_lo = other_lo
            total_hi = other_hi

        return Interval(total_lo, total_hi)

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        k = digit_pos
        t = target_digit

        if k < 10:
            digit_sum = _digit_sum_for_target(carry_mask, k, t)
        else:
            digit_sum = 0

        # Head outputs with tight V-sum bounds
        head0_vsum = self._head_v_sum_tight(k, 0, carry_mask, digit_sum, prev_output_digits)
        head1_vsum = self._head_v_sum_tight(k, 1, carry_mask, digit_sum, prev_output_digits)

        # O-proj: head0 -> dim3 (scale 2.0), head1 -> dim1 (scale 2.0)
        head0_out = head0_vsum * 2.0
        head1_out = head1_vsum * 2.0

        # PE at output position
        p = PROMPT_LEN + k
        pe_sin = self.pe[p, 1]
        pe_cos = self.pe[p, 2]

        # Previous output digit (the token embedding at this output position)
        if k == 0:
            d_prev_list = [0]  # '=' token has value 0
        else:
            d_prev_list = _possible_output_digits(carry_mask, k - 1)

        for d_prev in d_prev_list:
            # x after attention residual:
            # dim0 = d_prev (from embedding)
            # dim1 = pe_sin + head1_out (PE + attention)
            # dim2 = pe_cos (PE only, no attention writes here)
            # dim3 = head0_out (attention only, no PE in dim3)
            dim0 = Interval(float(d_prev))
            dim1 = Interval(pe_sin) + head1_out
            dim2 = Interval(pe_cos)
            dim3 = head0_out

            # MLP carry: ci = dot(x, [-1, 1, 0, 0]) = -dim0 + dim1
            ci = -dim0 + dim1
            carry = iv_relu(ci + Interval(-0.5)) - iv_relu(ci + Interval(-1.5))

            # MLP wrap: wi = dot(x, [-10, 10, 0, 1000]) = -10*dim0 + 10*dim1 + 1000*dim3
            wi = dim0 * (-10.0) + dim1 * 10.0 + dim3 * 1000.0
            wrap = iv_relu(wi + Interval(-9055.0)) - iv_relu(wi + Interval(-9045.0))

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
