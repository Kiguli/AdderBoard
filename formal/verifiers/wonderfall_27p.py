"""
Formal verifier for Wonderfall_27p.

Same d=2, 1h, RoPE-19 structure as yieldthought_20p but with:
1. Cross-tied w_vo matrix (attention output and MLP W2 share weights)
2. ReLU MLP (piecewise-linear, exact within activation patterns)
3. No QK-norm: attention scores have tiny token dependence (hn[0] varies by ~5.7e-5)

Since it requires mlx (Apple only), we hardcode all parameters derived
from the source code.

Strategy: Same as yieldthought — use position-only attention weights with
a rigorous error bound on the V-sum, then verify the ReLU MLP with
interval arithmetic.
"""

import math
import numpy as np

from ..interval import Interval, iv_relu, iv_rms_norm, iv_rms_norm_weighted
from ..verify_smt import (
    _precompute_all_attention_weights,
    _compute_v_sum_bounds, _compute_v_at_token,
    VOCAB_SIZE, OUTPUT_DIGITS, PROMPT_LEN, RMS_EPS,
    _carry_in_at, _possible_output_digits, _digit_sum_for_target,
)


# Architecture constants (same as zcbtrak/yieldthought family)
EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(2)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
DECODE_QUAD = 1e-3
DECODE_CURVATURE = 0.1
CARRY_ALPHA = 256.0 / CONST_NORM

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))

# Hardcoded parameters from Wonderfall_27p.py hand_set_weights()
GATE_A = CARRY_ALPHA * (-94.0) / CONST_NORM          # w1_row0_col0 = -12032
W1_COL1_SCALE = CARRY_ALPHA / CONST_NORM              # = 128
GATE_C = W1_COL1_SCALE * EMBED_CONST                  # = 128000
VO_C = 100.0 / CARRY_ALPHA                            # w_vo entry magnitude
V_SCALE_OLD = -22.0 * DIGIT_SCALE                     # original v_proj_w
V_SCALE = V_SCALE_OLD / (2.0 * VO_C * VO_C)           # v_scale param
EFF_V_PROJ_W = 2.0 * VO_C * VO_C * V_SCALE            # = V_SCALE_OLD

# Output norm weights
NORM_W0 = (DECODE_CURVATURE / DECODE_QUAD) / CONST_NORM  # = 50*sqrt(2)
NORM_W1 = -(DIGIT_SCALE / 50.0)                          # = -10*sqrt(2)


def _build_embed_table():
    """Build embedding table: [const0 + quad0*d^2, token_values]."""
    d = np.arange(VOCAB_SIZE, dtype=np.float64)
    col0 = EMBED_CONST + (-DECODE_QUAD) * d * d
    col1 = -d  # token_values = [-0, -1, ..., -9]
    return np.stack([col0, col1], axis=-1)


class Wonderfall27pVerifier:
    def __init__(self, model=None):
        # Build embedding table
        self.embed_table = _build_embed_table()

        # Precompute V values per token: V[d] = hn_d[1] * eff_v_proj_w
        self.v_table = np.zeros(VOCAB_SIZE)
        for dd in range(VOCAB_SIZE):
            emb = self.embed_table[dd]
            rms = math.sqrt((emb[0]**2 + emb[1]**2) / 2 + RMS_EPS)
            self.v_table[dd] = (emb[1] / rms) * EFF_V_PROJ_W

        # Compute hn[0] range for bounding attention weight errors
        self.hn0_table = np.zeros(VOCAB_SIZE)
        for dd in range(VOCAB_SIZE):
            emb = self.embed_table[dd]
            rms = math.sqrt((emb[0]**2 + emb[1]**2) / 2 + RMS_EPS)
            self.hn0_table[dd] = emb[0] / rms

        self.hn0_min = float(np.min(self.hn0_table))
        self.hn0_max = float(np.max(self.hn0_table))

        # Precompute position-only attention weights (same as zcbtrak)
        self.all_attn_weights = _precompute_all_attention_weights()

        # Folded output table for logit computation
        self.folded = np.zeros((VOCAB_SIZE, 2))
        for dd in range(VOCAB_SIZE):
            self.folded[dd, 0] = self.embed_table[dd, 0] * NORM_W0
            self.folded[dd, 1] = self.embed_table[dd, 1] * NORM_W1

        # Per-position attention weight error from hn[0] variation
        delta_hn0 = self.hn0_max - self.hn0_min  # ~5.7e-5
        self.max_weight_perturbation = (QK_SCALE**2 / math.sqrt(2)) * self.hn0_max * delta_hn0

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        k = digit_pos
        attn_w = self.all_attn_weights[k]
        digit_sum = _digit_sum_for_target(carry_mask, k, target_digit)
        if k == 10:
            digit_sum = 0

        # V-sum bounds (same approach as zcbtrak/yieldthought)
        v_min_all = float(np.min(self.v_table))
        v_max_all = float(np.max(self.v_table))

        primary_positions = set()
        if k < 10:
            primary_positions = {k + 1, k + 20}
        secondary_positions = {k, k + 19}
        zero_v_positions = {0} | set(range(11, 20)) | {30}

        # "Other" contribution (negligible weights)
        other_lo = 0.0
        other_hi = 0.0
        for s in range(len(attn_w)):
            w = attn_w[s]
            if w < 1e-15:
                continue
            if s in primary_positions or s in secondary_positions or s in zero_v_positions:
                continue
            other_lo += w * min(v_min_all, 0)
            other_hi += w * max(v_max_all, 0)

        # Primary contribution
        primary_lo = 0.0
        primary_hi = 0.0
        if k < 10:
            v_sum_iv = _compute_v_sum_bounds(self.embed_table, EFF_V_PROJ_W, digit_sum)
            w_pa = attn_w[k + 1]
            w_pb = attn_w[k + 20]
            if abs(w_pa - w_pb) < 1e-12:
                primary_lo = w_pa * v_sum_iv.lo
                primary_hi = w_pa * v_sum_iv.hi
            else:
                combos = []
                for a_d in range(max(0, digit_sum - 9), min(10, digit_sum + 1)):
                    b_d = digit_sum - a_d
                    if 0 <= b_d <= 9:
                        combos.append(w_pa * self.v_table[a_d] + w_pb * self.v_table[b_d])
                primary_lo = min(combos)
                primary_hi = max(combos)

        # Secondary + previous digit
        if k == 0:
            d_prev_list = [0]
            secondary_contribs = {0: (0.0, 0.0)}
        else:
            d_prev_list = _possible_output_digits(carry_mask, k - 1)
            w_sa = attn_w[k]
            w_sb = attn_w[k + 19]
            secondary_contribs = {}
            for d_prev in d_prev_list:
                dsum_prev = _digit_sum_for_target(carry_mask, k - 1, d_prev)
                v_sum_prev = _compute_v_sum_bounds(self.embed_table, EFF_V_PROJ_W, dsum_prev)
                if abs(w_sa - w_sb) < 1e-12:
                    secondary_contribs[d_prev] = (w_sa * v_sum_prev.lo, w_sa * v_sum_prev.hi)
                else:
                    combos = []
                    for a_d in range(max(0, dsum_prev - 9), min(10, dsum_prev + 1)):
                        b_d = dsum_prev - a_d
                        if 0 <= b_d <= 9:
                            combos.append(w_sa * self.v_table[a_d] + w_sb * self.v_table[b_d])
                    secondary_contribs[d_prev] = (min(combos), max(combos))

        # Attention weight error from hn[0] variation
        v_max_abs = float(np.max(np.abs(self.v_table)))
        err = 0.0
        for s in range(len(attn_w)):
            w = attn_w[s]
            if w < 1e-15:
                continue
            if s in zero_v_positions:
                continue
            delta_w = self.max_weight_perturbation * w * (1.0 - w)
            err += delta_w * v_max_abs

        # Verify for each d_prev
        t = target_digit

        for d_prev in d_prev_list:
            if k == 0:
                emb_0 = self.embed_table[0][0]
                emb_1 = self.embed_table[0][1]
            else:
                emb_0 = self.embed_table[d_prev][0]
                emb_1 = self.embed_table[d_prev][1]

            sec_lo, sec_hi = secondary_contribs[d_prev]
            attn_lo = primary_lo + sec_lo + other_lo - err
            attn_hi = primary_hi + sec_hi + other_hi + err

            # h after attention residual
            h0 = Interval(emb_0)
            h1 = Interval(emb_1) + Interval(attn_lo, attn_hi)

            # Post-attention RMSNorm (parameterless)
            hn_mlp = iv_rms_norm([h0, h1], eps=RMS_EPS)

            # ReLU MLP
            # g0 = hn[0]*gate_a + hn[1]*gate_c
            # g1 = hn[0]*(gate_a - w1_col1_scale) + hn[1]*gate_c
            g0 = hn_mlp[0] * GATE_A + hn_mlp[1] * GATE_C
            g1 = hn_mlp[0] * (GATE_A - W1_COL1_SCALE) + hn_mlp[1] * GATE_C

            # Exploit correlation: when both positive, relu(g1)-relu(g0) = g1-g0
            if g0.lo > 0 and g1.lo > 0:
                mlp_diff = hn_mlp[0] * (-W1_COL1_SCALE)
            elif g0.hi < 0 and g1.hi < 0:
                mlp_diff = Interval(0.0)
            elif g0.lo > 0 and g1.hi < 0:
                mlp_diff = -g0
            else:
                mlp_diff = iv_relu(g1) - iv_relu(g0)

            # MLP output scaled by vo_c (the w_vo matrix entry)
            mlp_scalar = VO_C * mlp_diff

            # h after MLP residual
            h0_post = h0
            h1_post = h1 + mlp_scalar

            # Output norm (weighted RMSNorm)
            hn_final = iv_rms_norm_weighted(
                [h0_post, h1_post],
                [NORM_W0, NORM_W1],
                eps=RMS_EPS,
            )

            # Verify logit differences
            for d in range(VOCAB_SIZE):
                if d == t:
                    continue
                delta_e0 = self.embed_table[t, 0] - self.embed_table[d, 0]
                delta_e1 = self.embed_table[t, 1] - self.embed_table[d, 1]
                diff = hn_final[0] * delta_e0 + hn_final[1] * delta_e1

                if diff.lo <= 0:
                    return False, (
                        f"digit {k}, target {t}, d_prev={d_prev}: "
                        f"logit[{t}]-logit[{d}] interval [{diff.lo:.6f}, {diff.hi:.6f}] "
                        f"not provably positive"
                    )

        return True, ""


def create_verifier(model=None):
    return Wonderfall27pVerifier(model)
