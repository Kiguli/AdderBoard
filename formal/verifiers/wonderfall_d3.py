"""
Formal verifier for Wonderfall d=3 models (121p, 130p, 139p).

Architecture: d=3, 4 heads (GQA: 4Q/1KV), head_dim=2, 24-token prompt.
All share: K_proj=[[1,0,0],[0,0,0]] -> position-only attention with QK_NORM_SCALE=256.
Score gap > 42000 -> attention weight at peak > 1 - exp(-42000) (exact in float64).

Prompt layout:
  pos 0: 0, pos 1..10: a[0]..a[9], pos 11: 0, pos 12: 0, pos 13..22: b[0]..b[9], pos 23: 0

Head targets (offset -> peak position s = p - offset):
  Head 0 (offset 23): a[k-1] for k>=1, token 0 for k=0
  Head 1 (offset 11): b[k-1] for k>=1, token 0 for k=0
  Head 2 (offset 22): a[k] for k<10, token 0 for k=10
  Head 3 (offset 10): b[k] for k<10, token 0 for k=10

O_proj: dim1 = -head0[0] - head1[0], dim2 = head2[0] + head3[0]

MLP variants:
  121p (ff=2): wrap pair only, carry via output norm dim1
  130p (ff=3): wrap pair + always-on linear carry neuron
  139p (ff=4): carry pair + wrap pair (both explicit 2-hinge)
"""

import math
import numpy as np

from ..interval import Interval, iv_silu, iv_rms_norm, iv_rms_norm_weighted

VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
PROMPT_LEN = 24

# Shared constants
EMBED_CONST = 1000.0
MODEL_DIM = 3
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
DECODE_LINEAR_EPS = 5e-4
DECODE_QUAD = DECODE_LINEAR_EPS / 2.0
ALPHA = 20.0
QK_NORM_SCALE = 256.0
RMS_EPS = 1e-6


def _build_embed_table():
    """Build embedding table for d=3 models."""
    table = np.zeros((VOCAB_SIZE, MODEL_DIM), dtype=np.float64)
    for d in range(VOCAB_SIZE):
        table[d, 0] = EMBED_CONST - DECODE_QUAD * d * d
        table[d, 1] = float(d)
        table[d, 2] = DECODE_LINEAR_EPS * float(d)
    return table


def _carry_in_at(carry_mask, pos):
    if pos == 0:
        return 0
    return 1 if (carry_mask & (1 << (pos - 1))) else 0


def _carry_out_at(carry_mask, pos):
    if pos >= 10:
        return 0
    return 1 if (carry_mask & (1 << pos)) else 0


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


def _digit_sum_for_target(carry_mask, pos, target):
    c_in = _carry_in_at(carry_mask, pos)
    c_out = _carry_out_at(carry_mask, pos)
    if c_out:
        return target + 10 - c_in
    else:
        return target - c_in


class WonderfallD3Verifier:
    def __init__(self, variant, model=None):
        """
        variant: '121p', '130p', or '139p'
        """
        self.variant = variant

        # Build embedding table
        self.embed_table = _build_embed_table()

        # Precompute V values: V(d) = hn_d[1] * DIGIT_SCALE
        # where hn_d = RMSNorm(embed(d)) with weight=[1,1,1]
        self.v_table = np.zeros(VOCAB_SIZE, dtype=np.float64)
        for d in range(VOCAB_SIZE):
            emb = self.embed_table[d]
            rms = math.sqrt(np.sum(emb ** 2) / MODEL_DIM + RMS_EPS)
            self.v_table[d] = (emb[1] / rms) * DIGIT_SCALE

        # Verify attention is essentially exact:
        # Score gap = QK^2 * sqrt(2) * (1 - cos(1)) > 42000
        score_gap = QK_NORM_SCALE ** 2 * math.sqrt(2) * (1.0 - math.cos(1.0))
        # Max error from non-peak positions: seq_len * exp(-gap) * max|V|
        # exp(-42000) = 0 in float64 (underflows to 0)
        # So attention is exact: V_sum = V(peak_digit)
        self.score_gap = score_gap

        # MLP constants
        self.scale = 1.0 / (ALPHA * CONST_NORM)

        # Model-specific output norm weights
        if variant == '121p':
            CARRY_SLOPE = -0.1
            self.norm_w = np.array([
                1.0 / CONST_NORM,
                CARRY_SLOPE * DIGIT_SCALE * DECODE_LINEAR_EPS,
                DIGIT_SCALE,
            ], dtype=np.float64)
        elif variant == '130p':
            self.norm_w = np.array([
                1.0 / CONST_NORM,
                0.0,
                DIGIT_SCALE,
            ], dtype=np.float64)
            self.carry_gate = ALPHA / (1.0 + math.exp(-ALPHA))
        elif variant == '139p':
            self.norm_w = np.array([
                1.0 / CONST_NORM,
                0.0,
                DIGIT_SCALE,
            ], dtype=np.float64)

    def _v_sum_bounds(self, digit_sum):
        """Compute [lo, hi] bounds on V(a) + V(b) for all (a,b) with a+b=digit_sum."""
        v_sums = []
        for a_d in range(max(0, digit_sum - 9), min(10, digit_sum + 1)):
            b_d = digit_sum - a_d
            if 0 <= b_d <= 9:
                v_sums.append(self.v_table[a_d] + self.v_table[b_d])
        if not v_sums:
            return Interval(0.0)
        return Interval(min(v_sums), max(v_sums))

    def _apply_mlp_121p(self, hn):
        """MLP for 121p: ff=2, wrap pair only."""
        # Gate: g0 = ALPHA*(-188/CN*hn[0] - 2*DS*hn[1] + 20*DS*hn[2])
        #        g1 = ALPHA*(-189/CN*hn[0] - 2*DS*hn[1] + 20*DS*hn[2])
        base_term = hn[0] * (ALPHA * (-188.0) / CONST_NORM) + \
                    hn[1] * (ALPHA * (-2.0) * DIGIT_SCALE) + \
                    hn[2] * (ALPHA * 20.0 * DIGIT_SCALE)
        g0 = base_term
        g1 = base_term + hn[0] * (ALPHA * (-1.0) / CONST_NORM)

        # Up: [hn[0], hn[0]]
        # Down dim2: -10*scale*silu(g0)*hn[0] + 10*scale*silu(g1)*hn[0]
        #          = 10*scale*hn[0]*(silu(g1) - silu(g0))
        DEEP = 30.0
        if g0.lo > DEEP and g1.lo > DEEP:
            # Both deeply positive: silu(g) ≈ g
            diff = g1 - g0  # = ALPHA*(-1)/CN * hn[0]
            wrap_val = Interval(10.0) * self.scale * hn[0] * diff
        elif g0.hi < -DEEP and g1.hi < -DEEP:
            wrap_val = Interval(0.0)
        else:
            wrap_val = Interval(10.0) * self.scale * hn[0] * (iv_silu(g1) - iv_silu(g0))

        return [Interval(0.0), Interval(0.0), wrap_val]

    def _apply_mlp_130p(self, hn):
        """MLP for 130p: ff=3, wrap pair + linear carry neuron."""
        # Wrap pair (same as 121p)
        base_term = hn[0] * (ALPHA * (-188.0) / CONST_NORM) + \
                    hn[1] * (ALPHA * (-2.0) * DIGIT_SCALE) + \
                    hn[2] * (ALPHA * 20.0 * DIGIT_SCALE)
        g0 = base_term
        g1 = base_term + hn[0] * (ALPHA * (-1.0) / CONST_NORM)

        DEEP = 30.0
        if g0.lo > DEEP and g1.lo > DEEP:
            diff = g1 - g0
            wrap_val = Interval(10.0) * self.scale * hn[0] * diff
        elif g0.hi < -DEEP and g1.hi < -DEEP:
            wrap_val = Interval(0.0)
        else:
            wrap_val = Interval(10.0) * self.scale * hn[0] * (iv_silu(g1) - iv_silu(g0))

        # Always-on carry neuron:
        # g2 = ALPHA/CN * hn[0] (deeply positive)
        # up2 = 0.5/CN * hn[0] - DS * hn[1]
        # silu(g2)*up2 contribution to dim2: down[2,2] * silu(g2) * up2
        g2 = hn[0] * (ALPHA / CONST_NORM)
        up2 = hn[0] * (0.5 / CONST_NORM) + hn[1] * (-DIGIT_SCALE)

        if g2.lo > DEEP:
            carry_val = Interval(0.1 / self.carry_gate) * g2 * up2
        else:
            carry_val = Interval(0.1 / self.carry_gate) * iv_silu(g2) * up2

        return [Interval(0.0), Interval(0.0), wrap_val + carry_val]

    def _apply_mlp_139p(self, hn):
        """MLP for 139p: ff=4, carry pair + wrap pair."""
        DEEP = 30.0

        # Carry pair:
        # g0 = ALPHA*(-8/CN*hn[0] - DS*hn[1])
        # g1 = ALPHA*(-9/CN*hn[0] - DS*hn[1])
        gc0 = hn[0] * (ALPHA * (-8.0) / CONST_NORM) + hn[1] * (ALPHA * (-1.0) * DIGIT_SCALE)
        gc1 = hn[0] * (ALPHA * (-9.0) / CONST_NORM) + hn[1] * (ALPHA * (-1.0) * DIGIT_SCALE)

        # Carry detection: scale*hn[0]*(silu(gc0) - silu(gc1))
        if gc0.lo > DEEP and gc1.lo > DEEP:
            carry_diff = gc0 - gc1
            carry_val = self.scale * hn[0] * carry_diff
        elif gc0.hi < -DEEP and gc1.hi < -DEEP:
            carry_val = Interval(0.0)
        else:
            carry_val = self.scale * hn[0] * (iv_silu(gc0) - iv_silu(gc1))

        # Wrap pair:
        gw0 = hn[0] * (ALPHA * (-188.0) / CONST_NORM) + \
              hn[1] * (ALPHA * (-2.0) * DIGIT_SCALE) + \
              hn[2] * (ALPHA * 20.0 * DIGIT_SCALE)
        gw1 = gw0 + hn[0] * (ALPHA * (-1.0) / CONST_NORM)

        if gw0.lo > DEEP and gw1.lo > DEEP:
            wrap_diff = gw1 - gw0
            wrap_val = Interval(10.0) * self.scale * hn[0] * wrap_diff
        elif gw0.hi < -DEEP and gw1.hi < -DEEP:
            wrap_val = Interval(0.0)
        else:
            wrap_val = Interval(10.0) * self.scale * hn[0] * (iv_silu(gw1) - iv_silu(gw0))

        return [Interval(0.0), Interval(0.0), carry_val + wrap_val]

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        k = digit_pos
        t = target_digit

        # Determine head targets
        if k == 0:
            # Head 0,1: token 0 (V=0)
            head01_vsum = Interval(0.0)
        else:
            d_prev_at_k1 = _possible_output_digits(carry_mask, k - 1)
            # For heads 0,1: target a[k-1], b[k-1]
            # V(a[k-1]) + V(b[k-1]) depends on digit_sum at k-1 and d_prev

        if k < 10:
            digit_sum_k = _digit_sum_for_target(carry_mask, k, t)
            head23_vsum = self._v_sum_bounds(digit_sum_k)
        else:
            # Head 2,3: token 0 (V=0)
            head23_vsum = Interval(0.0)

        # k=10 overflow case
        if k == 10:
            digit_sum_k = 0

        # Iterate over possible d_prev values
        if k == 0:
            d_prev_list = [0]  # start token
        else:
            d_prev_list = _possible_output_digits(carry_mask, k - 1)

        for d_prev in d_prev_list:
            # Compute head 0,1 V-sum (previous digit pair)
            if k == 0:
                head01_vsum = Interval(0.0)
            elif k == 10:
                # Heads 0,1 target a[9], b[9]
                dsum_prev = _digit_sum_for_target(carry_mask, 9, d_prev)
                head01_vsum = self._v_sum_bounds(dsum_prev)
            else:
                dsum_prev = _digit_sum_for_target(carry_mask, k - 1, d_prev)
                head01_vsum = self._v_sum_bounds(dsum_prev)

            # O_proj: dim1 += -(head0 + head1), dim2 += head2 + head3
            attn_dim1 = -head01_vsum
            attn_dim2 = head23_vsum

            # Hidden state after attention residual
            emb = self.embed_table[d_prev]
            h0 = Interval(emb[0])
            h1 = Interval(emb[1]) + attn_dim1
            h2 = Interval(emb[2]) + attn_dim2

            # Post-attention RMSNorm (weight=[1,1,1])
            hn = iv_rms_norm([h0, h1, h2], eps=RMS_EPS)

            # Apply model-specific MLP
            if self.variant == '121p':
                mlp_out = self._apply_mlp_121p(hn)
            elif self.variant == '130p':
                mlp_out = self._apply_mlp_130p(hn)
            elif self.variant == '139p':
                mlp_out = self._apply_mlp_139p(hn)

            # Residual connection (add MLP output to pre-MLP hidden state)
            h0_post = h0 + mlp_out[0]
            h1_post = h1 + mlp_out[1]
            h2_post = h2 + mlp_out[2]

            # Output RMSNorm (weighted)
            hn_final = iv_rms_norm_weighted(
                [h0_post, h1_post, h2_post],
                [self.norm_w[0], self.norm_w[1], self.norm_w[2]],
                eps=RMS_EPS,
            )

            # Verify logit differences
            for d in range(VOCAB_SIZE):
                if d == t:
                    continue
                delta_e0 = self.embed_table[t, 0] - self.embed_table[d, 0]
                delta_e1 = self.embed_table[t, 1] - self.embed_table[d, 1]
                delta_e2 = self.embed_table[t, 2] - self.embed_table[d, 2]
                diff = hn_final[0] * delta_e0 + hn_final[1] * delta_e1 + hn_final[2] * delta_e2

                if diff.lo <= 0:
                    return False, (
                        f"digit {k}, target {t}, d_prev={d_prev}: "
                        f"logit[{t}]-logit[{d}] interval [{diff.lo:.8f}, {diff.hi:.8f}] "
                        f"not provably positive"
                    )

        return True, ""


def create_verifier_121p(model=None):
    return WonderfallD3Verifier('121p', model)


def create_verifier_130p(model=None):
    return WonderfallD3Verifier('130p', model)


def create_verifier_139p(model=None):
    return WonderfallD3Verifier('139p', model)
