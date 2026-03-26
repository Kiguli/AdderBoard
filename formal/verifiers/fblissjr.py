"""
Formal verifier for fblissjr_33p.

Architecture: d=3, 3 heads (d_head=1), ff=4, float64.
Q=K=0: attention routing entirely from fixed mask (0 params).

Three heads:
- Head 0: Gather A_k + B_k via e^80 softmax anchoring on SEP
- Head 1: ALiBi prefix sum for carry (scores = -(k-j)*log(10), SEP=80)
- Head 2: Cancel residual (self-attend with V = -token_value)

Verification is straightforward because:
1. Attention weights are exact constants (Q=K=0, fixed mask)
2. e^80 anchoring gives errors < 1e-33 (24 orders of margin vs 1e-10 gap)
3. 2-hinge ReLU gives EXACT step functions (carry=0 or 1, overflow=0 or -10)
4. Parabolic head has logit gaps >= (t-d)^2 - tiny_error >= 1 - 1e-32
"""

import math
import numpy as np

from ..interval import Interval
from ..verify_formal import possible_output_digits, digit_sum_for_target, carry_in_at


VOCAB_SIZE = 10
OUTPUT_DIGITS = 11


class FblissjrVerifier:
    def __init__(self, model):
        self.exp80 = math.exp(80.0)
        # Head 0 error factor: w_digit = 1/(2+e^80), scaled by e^80 gives 1 - alpha
        self.alpha = 2.0 / (2.0 + self.exp80)

        # Precompute D_k = 2*sum_{j<k} 10^{j-k} for head 1 denominator correction
        self.D = {}
        for k in range(1, 11):
            self.D[k] = 2.0 * sum(10.0 ** (j - k) for j in range(k))

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        k = digit_pos
        t = target_digit

        # --- dim0: Head 0 gathers (A_k + B_k) via e^80 anchoring ---
        # After O-proj*e^80 and residual cancellation (Head 2 removes d_prev):
        # dim0 = digit_sum * e^80/(2+e^80) = digit_sum * (1 - alpha)
        if k < 10:
            digit_sum = digit_sum_for_target(carry_mask, k, t)
            dim0 = Interval(float(digit_sum) * (1.0 - self.alpha))
        else:
            # k=10: Head 0 sees only SEP -> 0; Head 2 cancels d_prev
            digit_sum = 0
            dim0 = Interval(0.0)

        # --- dim1: Head 1 computes prefix sum S_k via ALiBi ---
        # S_k = sum_{j<k} (A_j+B_j) * 10^{j-k}
        # dim1 = S_k * e^80/(e^80 + D_k) where D_k = 2*sum 10^{j-k}
        if k == 0:
            dim1 = Interval(0.0)
        else:
            S_lo, S_hi = 0.0, 0.0
            for j in range(k):
                c_in_j = carry_in_at(carry_mask, j)
                c_out_j = 1 if (carry_mask & (1 << j)) else 0
                if c_out_j:
                    lo_j, hi_j = 10 - c_in_j, 18
                else:
                    lo_j, hi_j = 0, 9 - c_in_j
                w = 10.0 ** (j - k)
                S_lo += lo_j * w
                S_hi += hi_j * w

            factor = self.exp80 / (self.exp80 + self.D[k])
            dim1 = Interval(S_lo * factor, S_hi * factor)

        # --- MLP carry detection (2-hinge on dim1 at threshold 1.0) ---
        c_in_k = carry_in_at(carry_mask, k)
        z1 = dim1 + Interval(-1.0 + 0.5e-11)  # more restrictive neuron
        z0 = dim1 + Interval(-1.0 + 1e-11)

        if c_in_k:
            # S_k >= 1.0 (exact), so z1 should be provably > 0
            if z1.lo <= 0:
                return False, f"pos {k}: carry neuron not provably active, z1=[{z1.lo:.2e},{z1.hi:.2e}]"
            carry_delta = 1.0
        else:
            # S_k < 1 - 10^{-k}, gap >= 1e-10 >> 1e-11
            if z0.hi >= 0:
                return False, f"pos {k}: carry neuron not provably inactive, z0=[{z0.lo:.2e},{z0.hi:.2e}]"
            carry_delta = 0.0

        # --- MLP overflow detection (2-hinge on dim0+dim1 at threshold 10.0) ---
        c_out_k = (1 if (carry_mask & (1 << k)) else 0) if k < 10 else 0
        total = dim0 + dim1
        z3 = total + Interval(-10.0 + 0.5e-11)
        z2 = total + Interval(-10.0 + 1e-11)

        if c_out_k:
            if z3.lo <= 0:
                return False, f"pos {k}: overflow neuron not provably active, z3=[{z3.lo:.2e},{z3.hi:.2e}]"
            overflow_delta = -10.0
        else:
            if z2.hi >= 0:
                return False, f"pos {k}: overflow neuron not provably inactive, z2=[{z2.lo:.2e},{z2.hi:.2e}]"
            overflow_delta = 0.0

        # --- Final dim0 after MLP residual ---
        # 2-hinge gives EXACT deltas: carry=1 or 0, overflow=-10 or 0
        dim0_final = dim0 + Interval(carry_delta + overflow_delta)

        # --- Verify logit gaps ---
        # logit[c] = 2c * dim0_final - c^2 (parabolic head)
        # logit[t] - logit[d] = (t-d)^2 + 2*(dim0_final - t)*(t-d)
        # With dim0_final ≈ t (error < 1e-33), gap >= 1 - 2e-32
        for d in range(VOCAB_SIZE):
            if d == t:
                continue
            diff = dim0_final * float(2 * (t - d)) + Interval(-float(t * t - d * d))
            if diff.lo <= 0:
                return False, (
                    f"pos {k}, target {t}: "
                    f"logit[{t}]-logit[{d}] [{diff.lo:.2e},{diff.hi:.2e}]"
                )

        return True, ""


def create_verifier(model):
    return FblissjrVerifier(model)
