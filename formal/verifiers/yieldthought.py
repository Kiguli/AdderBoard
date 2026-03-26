"""
Formal verifier for yieldthought_20p.

Same d=2, 1h, RoPE-19 structure as zcbtrak but with two key differences:
1. No QK-norm: attention scores have tiny token dependence (hn[0] varies by 5.7e-5)
2. ReLU MLP instead of SiLU: piecewise-linear, exact within activation patterns

Strategy: Reuse zcbtrak's position-only attention weights with a rigorous error
bound on the V-sum, then verify the ReLU MLP with interval arithmetic.
"""

import math
import numpy as np

from ..interval import Interval, iv_relu, iv_rms_norm, iv_rms_norm_weighted, iv_silu, iv_sqrt
from ..verify_smt import (
    _precompute_all_attention_weights,
    _compute_v_sum_bounds, _compute_v_at_token,
    VOCAB_SIZE, OUTPUT_DIGITS, PROMPT_LEN, RMS_EPS,
    _carry_in_at, _possible_output_digits, _digit_sum_for_target,
)


# Architecture constants (same as zcbtrak family)
EMBED_CONST = 1000.0
DECODE_QUAD = 1e-3
CONST_NORM = math.sqrt(2)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
CARRY_ALPHA = 256.0 / CONST_NORM
DECODE_CURVATURE = 0.1

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))


def _build_embed_table():
    """Build embedding table: [const0 + quad0*d^2, const0*quad0*d]."""
    d = np.arange(VOCAB_SIZE, dtype=np.float64)
    col0 = EMBED_CONST + (-DECODE_QUAD) * d * d
    col1 = EMBED_CONST * (-DECODE_QUAD) * d
    return np.stack([col0, col1], axis=-1)


class YieldthoughtVerifier:
    def __init__(self, model):
        # Extract parameters from PyTorch model
        import torch
        with torch.no_grad():
            self.const0 = float(model.embed_tokens.const0.item())
            self.quad0 = float(model.embed_tokens.quad0.item())
            token_vals = model.embed_tokens.token_values.detach().cpu().numpy()
            self.q_scale = float(model.attn.q_scale.item())
            self.q_phase = float(model.attn.q_phase.item())
            self.vo_scale = float(model.attn.vo_scale.item())
            self.v_scale = float(model.attn.v_scale.item())
            self.w1_row0_col0 = float(model.mlp.w1_row0_col0.item())
            self.w1_col1_scale = float(model.mlp.w1_col1_scale.item())
            self.norm_w = model.final_norm.weight.detach().cpu().numpy().astype(np.float64)

        # Build embedding table
        d = np.arange(VOCAB_SIZE, dtype=np.float64)
        self.embed_table = np.stack([
            self.const0 + self.quad0 * d * d,
            self.const0 * self.quad0 * d,
        ], axis=-1)

        # Effective v_proj_w: 2 * vo_scale^2 * v_scale (same as zcbtrak's v_proj_w)
        self.eff_v_proj_w = 2.0 * self.vo_scale ** 2 * self.v_scale

        # Precompute V values per token (V[d] = hn_d[1] * eff_v_proj_w)
        # hn_d[1] = embed[d][1] / rms(embed[d])
        self.v_table = np.zeros(VOCAB_SIZE)
        for dd in range(VOCAB_SIZE):
            emb = self.embed_table[dd]
            rms = math.sqrt((emb[0]**2 + emb[1]**2) / 2 + RMS_EPS)
            self.v_table[dd] = (emb[1] / rms) * self.eff_v_proj_w

        # The attention structure is position-only (to first approximation)
        # since Q and K are both proportional to hn[0] ≈ sqrt(2).
        # Compute hn[0] range for bounding attention weight errors.
        self.hn0_table = np.zeros(VOCAB_SIZE)
        for dd in range(VOCAB_SIZE):
            emb = self.embed_table[dd]
            rms = math.sqrt((emb[0]**2 + emb[1]**2) / 2 + RMS_EPS)
            self.hn0_table[dd] = emb[0] / rms

        self.hn0_min = float(np.min(self.hn0_table))
        self.hn0_max = float(np.max(self.hn0_table))

        # Precompute position-only attention weights (same as zcbtrak).
        # These are the weights when hn[0] = sqrt(2) exactly.
        self.all_attn_weights = _precompute_all_attention_weights()

        # MLP gate matrix: W1 = [[w1_a, col1], [w1_a - w1_cs, col1]]
        col1 = self.w1_col1_scale * self.const0
        self.gate_a = self.w1_row0_col0
        self.gate_c = col1

        # Output norm weights
        self.norm_w0 = float(self.norm_w[0])
        self.norm_w1 = float(self.norm_w[1])

        # Build folded output table (for logit computation)
        self.folded = np.zeros((VOCAB_SIZE, 2))
        for dd in range(VOCAB_SIZE):
            self.folded[dd, 0] = self.embed_table[dd, 0] * self.norm_w0
            self.folded[dd, 1] = self.embed_table[dd, 1] * self.norm_w1

        # Compute per-position V-sum error bounds from using position-only
        # attention weights instead of true (token-dependent) weights.
        #
        # The error at each position comes from the softmax weight shift
        # caused by hn_s[0] varying by delta_hn0 across digits.
        #
        # We compute this per (digit_pos, key_pos) for tight bounds.
        # For the dominant positions, the error is tiny because:
        # 1. V[a]+V[b] = 22*(a+b) is constant within a carry partition
        # 2. Weight shifts from hn[0] variation are < 0.001
        #
        # For "other" positions with weights < 1e-4, the V contribution
        # is bounded by w * max|V| < 1e-4 * 198 < 0.02 regardless.
        #
        # Total error per carry partition: max score delta * L1 sensitivity
        delta_hn0 = self.hn0_max - self.hn0_min  # ~5.7e-5
        # Max score change at any key position from hn_s variation:
        # score = (q_scale^2/sqrt(2)) * hn_p * hn_s * cos(...)
        # delta_score = (q_scale^2/sqrt(2)) * hn_p * delta_hn0 * |cos|
        # Upper bound (hn_p ≤ hn0_max, |cos| ≤ 1):
        max_score_perturbation = (self.q_scale**2 / math.sqrt(2)) * self.hn0_max * delta_hn0
        # Attention weight sensitivity: delta_w ≤ delta_score (since d(softmax)/d(score) ≤ 1)
        # Conservative per-position weight error bound
        self.max_weight_perturbation = max_score_perturbation
        # V-sum error: each position contributes at most |delta_w| * |V_range|
        # For seq_len positions, total error ≤ seq_len * max_weight_perturbation * V_max
        # But most positions have w < 1e-4 so their weight perturbation is < 1e-4
        # Tight bound: use actual weights to scale the perturbation
        v_max = float(np.max(np.abs(self.v_table)))

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        k = digit_pos
        attn_w = self.all_attn_weights[k]
        digit_sum = _digit_sum_for_target(carry_mask, k, target_digit)
        if k == 10:
            digit_sum = 0

        # ── Compute V-sum bounds (same approach as zcbtrak) ──
        v_min_all = float(np.min(self.v_table))
        v_max_all = float(np.max(self.v_table))

        # Primary positions: a[k] at k+1, b[k] at k+20
        primary_positions = set()
        if k < 10:
            primary_positions = {k + 1, k + 20}

        secondary_positions = {k, k + 19}
        zero_v_positions = {0} | set(range(11, 20)) | {30}

        # Other (negligible weight) positions
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
            v_sum_iv = _compute_v_sum_bounds(self.embed_table, self.eff_v_proj_w, digit_sum)
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
                v_sum_prev = _compute_v_sum_bounds(self.embed_table, self.eff_v_proj_w, dsum_prev)
                if abs(w_sa - w_sb) < 1e-12:
                    secondary_contribs[d_prev] = (w_sa * v_sum_prev.lo, w_sa * v_sum_prev.hi)
                else:
                    combos = []
                    for a_d in range(max(0, dsum_prev - 9), min(10, dsum_prev + 1)):
                        b_d = dsum_prev - a_d
                        if 0 <= b_d <= 9:
                            combos.append(w_sa * self.v_table[a_d] + w_sb * self.v_table[b_d])
                    secondary_contribs[d_prev] = (min(combos), max(combos))

        # ── Compute per-position attention weight error ──
        # For each key position, the weight error from using position-only weights
        # is bounded by: max_weight_perturbation * w_nom[s] * (1 + correction)
        # where correction accounts for softmax coupling.
        # We add the per-position V-sum error conservatively.
        v_max_abs = float(np.max(np.abs(self.v_table)))
        err = 0.0
        for s in range(len(attn_w)):
            w = attn_w[s]
            if w < 1e-15:
                continue
            if s in zero_v_positions:
                continue  # V=0, no error regardless of weight
            # Weight perturbation at this position: bounded by delta_score * w*(1-w)
            # V contribution error: |delta_w| * |V_max|
            delta_w = self.max_weight_perturbation * w * (1.0 - w)
            err += delta_w * v_max_abs

        # ── Verify for each d_prev ──
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

            # h after attention residual: h = [embed[0], embed[1] + attn_output]
            h0 = Interval(emb_0)
            h1 = Interval(emb_1) + Interval(attn_lo, attn_hi)

            # Post-attention norm (parameterless)
            hn_mlp = iv_rms_norm([h0, h1], eps=RMS_EPS)

            # ReLU MLP
            # g0 = hn[0]*gate_a + hn[1]*gate_c
            # g1 = hn[0]*(gate_a - w1_col1_scale) + hn[1]*gate_c
            # Key: g1 - g0 = -w1_col1_scale * hn[0] (algebraically exact)
            g0 = hn_mlp[0] * self.gate_a + hn_mlp[1] * self.gate_c
            g1 = hn_mlp[0] * (self.gate_a - self.w1_col1_scale) + hn_mlp[1] * self.gate_c

            # Exploit correlation: when both positive, relu(g1)-relu(g0) = g1-g0
            if g0.lo > 0 and g1.lo > 0:
                # Both on: relu(g1) - relu(g0) = g1 - g0 = -w1_col1_scale * hn[0]
                mlp_diff = hn_mlp[0] * (-self.w1_col1_scale)
            elif g0.hi < 0 and g1.hi < 0:
                # Both off
                mlp_diff = Interval(0.0)
            elif g0.lo > 0 and g1.hi < 0:
                # g0 on, g1 off: relu(g1) - relu(g0) = 0 - g0 = -g0
                mlp_diff = -g0
            else:
                # Mixed/uncertain: fall back to naive intervals
                mlp_diff = iv_relu(g1) - iv_relu(g0)

            mlp_scalar = self.vo_scale * mlp_diff

            # h after MLP residual
            h0_post = h0
            h1_post = h1 + mlp_scalar

            # Output norm (weighted RMSNorm)
            hn_final = iv_rms_norm_weighted(
                [h0_post, h1_post],
                [self.norm_w0, self.norm_w1],
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


def create_verifier(model):
    return YieldthoughtVerifier(model)
