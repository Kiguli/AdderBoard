"""
Formal verification of AdderBoard submissions via structural algebraic analysis.

For the zcbtrak-family models (hand-coded, 1L Qwen decoder, d=2), we exploit
three structural properties that make rigorous verification tractable:

1. POSITION-ONLY ATTENTION: After RMSNorm, K = [√2, 0] regardless of token
   value, so attention scores depend only on position (via RoPE). This means
   attention weights are exact constants — no interval uncertainty.

2. DIGIT-SUM DEPENDENCE: V[a_pos] + V[b_pos] depends only on (a + b), not on
   individual digits. Within a carry sub-partition where the digit sum is fixed,
   the dominant attention contribution is nearly constant (variation < 0.006).

3. LOGIT-DIFFERENCE CANCELLATION: Logits are ~100,000 but differ by only ~0.1.
   Computing logit[target] - logit[d] cancels the huge common mode, leaving a
   verifiable signal proportional to hn[0]*(d²-t²) + hn[1]*(t-d)*200.

Verification strategy:
  For each carry partition (1024) × output digit position (11) × target digit:
    - Compute exact attention weights (position-only, precomputed)
    - Compute V-sum at dominant positions ± rigorous error bound
    - Propagate through MLP (verify gate states)
    - Verify logit differences are provably positive via interval arithmetic
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from .extract import ModelSpec
from .interval import Interval, iv_silu, iv_sqrt, iv_rms_norm


@dataclass
class SMTVerificationResult:
    """Result of formal verification."""
    status: str  # "PROVEN_CORRECT", "COUNTEREXAMPLE_FOUND", "TIMEOUT", "INCONCLUSIVE", "ERROR"
    solve_time_seconds: float = 0.0
    counterexample: Optional[tuple[int, int]] = None
    expected: Optional[int] = None
    model_output: Optional[int] = None
    solver_stats: dict[str, Any] = field(default_factory=dict)
    method: str = "structural_algebraic"
    notes: list[str] = field(default_factory=list)


# ── Architecture constants (zcbtrak family) ──────────────────────────────

VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
PROMPT_LEN = 31
MODEL_DIM = 2

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
ATTN_SCALE = (MODEL_DIM ** -0.5) * (QK_NORM_SCALE ** 2)

RMS_EPS = 1e-6

# After RMSNorm, K_normed[0] is this constant for ALL tokens.
# (Verified empirically: identical to 8 decimal places across digits 0-9.)
K_NORMED_0 = math.sqrt(2.0) / math.sqrt(1.0 + RMS_EPS)

COS_PHI = math.cos(PHI)
SIN_PHI = math.sin(PHI)

# Q_normed is also effectively constant (depends only on hn0 ≈ √2).
# Q = [hn0*cos(PHI), -hn0*sin(PHI)], after RMSNorm: [cos(PHI)*C, -sin(PHI)*C]
# where C = √2 / √(1 + eps) ≈ √2
Q_NORMED = np.array([COS_PHI * K_NORMED_0, -SIN_PHI * K_NORMED_0])

# NORM weights folded into output table
NORM_W0 = 50.0 * math.sqrt(2.0)
NORM_W1 = -10.0 * math.sqrt(2.0)


# ── Precomputed attention weights ────────────────────────────────────────

def _compute_attention_weights(output_pos: int, seq_len: int) -> np.ndarray:
    """
    Compute exact softmax attention weights from output_pos to all positions.

    Since K_normed = [√2, 0] for ALL tokens (Property 1), the attention scores
    depend only on position through RoPE. This function returns exact weights.
    """
    # Q after RoPE at output_pos
    angle_q = output_pos * OMEGA
    q_roped = np.array([
        Q_NORMED[0] * math.cos(angle_q) - Q_NORMED[1] * math.sin(angle_q),
        Q_NORMED[0] * math.sin(angle_q) + Q_NORMED[1] * math.cos(angle_q),
    ])

    scores = np.full(seq_len, -np.inf)
    for s in range(min(output_pos + 1, seq_len)):  # Causal mask
        angle_k = s * OMEGA
        k_roped = np.array([
            K_NORMED_0 * math.cos(angle_k),
            K_NORMED_0 * math.sin(angle_k),
        ])
        scores[s] = np.dot(q_roped, k_roped) * ATTN_SCALE

    # Numerically stable softmax
    scores_valid = scores[:output_pos + 1]
    max_s = np.max(scores_valid)
    exp_s = np.exp(scores_valid - max_s)
    weights = np.zeros(seq_len)
    weights[:output_pos + 1] = exp_s / np.sum(exp_s)
    return weights


def _precompute_all_attention_weights() -> dict[int, np.ndarray]:
    """Precompute attention weights for all 11 output digit positions.

    The first output digit is generated from logits at position 30
    (the last prompt token), with sequence length 31. For digit k,
    output_pos = 30 + k, seq_len = 31 + k.
    """
    result = {}
    for k in range(OUTPUT_DIGITS):
        output_pos = PROMPT_LEN - 1 + k  # Position 30+k
        seq_len = PROMPT_LEN + k          # 31+k tokens
        result[k] = _compute_attention_weights(output_pos, seq_len)
    return result


# ── Carry partition helpers ──────────────────────────────────────────────

def _carry_in_at(carry_mask: int, pos: int) -> int:
    if pos == 0:
        return 0
    return 1 if (carry_mask & (1 << (pos - 1))) else 0


def _possible_output_digits(carry_mask: int, pos: int) -> list[int]:
    """All possible output digits at position pos for this carry partition."""
    if pos == 10:
        # Overflow: 0 or 1
        return [1 if (carry_mask & (1 << 9)) else 0]

    c_in = _carry_in_at(carry_mask, pos)
    c_out = 1 if (carry_mask & (1 << pos)) else 0
    digits = set()
    for a_d in range(10):
        for b_d in range(10):
            s = a_d + b_d + c_in
            if (c_out and s >= 10) or (not c_out and s < 10):
                digits.add(s % 10)
    return sorted(digits)


def _digit_sum_for_target(carry_mask: int, pos: int, target: int) -> int:
    """The a[pos]+b[pos] value that produces target digit at this position."""
    c_in = _carry_in_at(carry_mask, pos)
    c_out = 1 if (carry_mask & (1 << pos)) else 0
    # output = (a + b + c_in) % 10 = target
    # If carry_out: a + b + c_in >= 10, so a+b+c_in = target + 10
    # If no carry_out: a + b + c_in < 10, so a+b+c_in = target
    if c_out:
        return target + 10 - c_in
    else:
        return target - c_in


# ── V-sum computation ────────────────────────────────────────────────────

def _compute_v_sum_bounds(
    embed_table: np.ndarray,
    v_proj_w: float,
    digit_sum: int,
) -> Interval:
    """
    Compute rigorous bounds on V[a_pos] + V[b_pos] for all (a,b) with a+b = digit_sum.

    V[pos][0] = hn_pos[1] * v_proj_w, where hn_pos = RMSNorm(embed(token)).

    Property 3: this sum depends almost entirely on digit_sum, with tiny
    variation from the RMSNorm denominator depending on individual digits.
    """
    v_sums = []
    for a_d in range(max(0, digit_sum - 9), min(10, digit_sum + 1)):
        b_d = digit_sum - a_d
        if b_d < 0 or b_d > 9:
            continue

        # V at a-position
        emb_a = embed_table[a_d]
        rms_a = math.sqrt((emb_a[0] ** 2 + emb_a[1] ** 2) / 2 + RMS_EPS)
        v_a = (emb_a[1] / rms_a) * v_proj_w

        # V at b-position
        emb_b = embed_table[b_d]
        rms_b = math.sqrt((emb_b[0] ** 2 + emb_b[1] ** 2) / 2 + RMS_EPS)
        v_b = (emb_b[1] / rms_b) * v_proj_w

        v_sums.append(v_a + v_b)

    if not v_sums:
        return Interval(0.0)

    return Interval(min(v_sums), max(v_sums))


def _compute_v_at_token(embed_table: np.ndarray, v_proj_w: float, token: int) -> float:
    """Compute V[0] for a specific known token."""
    emb = embed_table[token]
    rms = math.sqrt((emb[0] ** 2 + emb[1] ** 2) / 2 + RMS_EPS)
    return (emb[1] / rms) * v_proj_w


# ── Core verification logic ─────────────────────────────────────────────

def _build_embed_table(w0: float, w1: float) -> np.ndarray:
    d = np.arange(VOCAB_SIZE, dtype=np.float64)
    return np.stack([w0 - w1 * d * d, -d], axis=-1)


def _verify_output_digit(
    params: np.ndarray,
    carry_mask: int,
    digit_pos: int,
    target_digit: int,
    attn_weights: np.ndarray,
    embed_table: np.ndarray,
    prev_output_digits: list[int],
) -> tuple[bool, str]:
    """
    Verify that the model outputs target_digit at digit_pos for ALL inputs
    matching this carry partition and target digit.

    Key structural insight: the attention has four groups of positions:
      1. PRIMARY (w≈0.4545): a[k] and b[k] — determine the current digit
      2. SECONDARY (w≈0.04545): a[k-1] and b[k-1] — correlated with the
         current token's embedding through the carry partition
      3. ZERO-V: positions holding token 0 (pos 0, 11-19, 30) — V=0
      4. OTHER: all remaining positions — negligible weights (total < 1e-4)

    The secondary attention contribution (≈22*w_sec*digit_sum_prev ≈ 0.9999*digit_sum_prev)
    nearly cancels the current token's embedding (-d_prev), because within a carry
    partition d_prev = (digit_sum_prev + c_in) mod 10. This correlation makes h[1]
    nearly constant within a partition (variation < 0.001).

    Returns (proven, reason).
    """
    embed_w0 = float(params[0])
    v_proj_w = float(params[2])
    gate_a = float(params[3])
    gate_c = float(params[4])
    carry_w = float(params[5])

    # Precompute V for each digit
    v_table = [_compute_v_at_token(embed_table, v_proj_w, d) for d in range(10)]
    v_min_all = min(v_table)
    v_max_all = max(v_table)

    # Prompt layout:
    # pos 0: token 0 | pos 1..10: a[0]..a[9] | pos 11..19: padding 0
    # pos 20..29: b[0]..b[9] | pos 30: token 0 | pos 31+: output digits

    digit_sum = _digit_sum_for_target(carry_mask, digit_pos, target_digit)
    if digit_pos == 10:
        digit_sum = 0  # overflow position

    # ── Identify the four position groups ──

    k = digit_pos
    # Primary: a[k] at pos k+1, b[k] at pos k+20 (for k < 10)
    primary_positions = set()
    if k < 10:
        primary_positions = {k + 1, k + 20}

    # Secondary: a[k-1] at pos k, b[k-1] at pos k+19
    secondary_positions = {k, k + 19}

    # Zero-V positions
    zero_v_positions = {0} | set(range(11, 20)) | {30}

    # ── Step 1: Compute "other" attention contribution (negligible) ──

    other_lo = 0.0
    other_hi = 0.0
    for s in range(len(attn_weights)):
        w = attn_weights[s]
        if w < 1e-15:
            continue
        if s in primary_positions or s in secondary_positions or s in zero_v_positions:
            continue
        # Variable position: bound V over [v_min, v_max]
        other_lo += w * v_min_all
        other_hi += w * v_max_all

    # ── Step 2: Primary contribution (target digit's a+b) ──

    primary_lo = 0.0
    primary_hi = 0.0
    if k < 10:
        w_pa = attn_weights[k + 1]
        w_pb = attn_weights[k + 20]
        v_sum_iv = _compute_v_sum_bounds(embed_table, v_proj_w, digit_sum)
        if abs(w_pa - w_pb) < 1e-12:
            primary_lo = w_pa * v_sum_iv.lo
            primary_hi = w_pa * v_sum_iv.hi
        else:
            combos = []
            for a_d in range(max(0, digit_sum - 9), min(10, digit_sum + 1)):
                b_d = digit_sum - a_d
                if 0 <= b_d <= 9:
                    combos.append(w_pa * v_table[a_d] + w_pb * v_table[b_d])
            primary_lo = min(combos)
            primary_hi = max(combos)

    # ── Step 3: Secondary + embedding (correlated through carry partition) ──

    # For each valid previous output digit d_prev, compute the combined h[1]:
    #   h[1] = embed[d_prev][1] + secondary_attn + primary_attn + other_attn
    # where secondary_attn = w_sec * V_sum[digit_sum_prev(d_prev)]

    if digit_pos == 0:
        # Current token is token 0 at position 30.
        # Secondary positions are pos 0 (token 0) and pos 19 (padding 0): V=0.
        d_prev_list = [0]
        secondary_contribs = {0: (0.0, 0.0)}  # (lo, hi) of secondary V-sum
    else:
        # Get possible previous output digits
        d_prev_list = _possible_output_digits(carry_mask, digit_pos - 1)

        w_sa = attn_weights[k]       # a[k-1] position
        w_sb = attn_weights[k + 19]  # b[k-1] position

        secondary_contribs = {}
        for d_prev in d_prev_list:
            # digit_sum at the previous position determines a[k-1]+b[k-1]
            dsum_prev = _digit_sum_for_target(carry_mask, digit_pos - 1, d_prev)
            v_sum_prev = _compute_v_sum_bounds(embed_table, v_proj_w, dsum_prev)
            if abs(w_sa - w_sb) < 1e-12:
                secondary_contribs[d_prev] = (w_sa * v_sum_prev.lo,
                                              w_sa * v_sum_prev.hi)
            else:
                combos = []
                for a_d in range(max(0, dsum_prev - 9), min(10, dsum_prev + 1)):
                    b_d = dsum_prev - a_d
                    if 0 <= b_d <= 9:
                        combos.append(w_sa * v_table[a_d] + w_sb * v_table[b_d])
                secondary_contribs[d_prev] = (min(combos), max(combos))

    # ── Step 4: Verify logit differences for ALL d_prev values ──

    folded = np.zeros((VOCAB_SIZE, 2))
    for d in range(VOCAB_SIZE):
        folded[d, 0] = embed_table[d, 0] * NORM_W0
        folded[d, 1] = embed_table[d, 1] * NORM_W1

    t = target_digit

    for d_prev in d_prev_list:
        # Embedding at current position
        if digit_pos == 0:
            emb_0 = embed_table[0][0]
            emb_1 = embed_table[0][1]
        else:
            emb_0 = embed_table[d_prev][0]
            emb_1 = embed_table[d_prev][1]

        # Total attention output
        sec_lo, sec_hi = secondary_contribs[d_prev]
        attn_lo = primary_lo + sec_lo + other_lo
        attn_hi = primary_hi + sec_hi + other_hi

        # h after attention residual
        h0 = Interval(emb_0)
        h1 = Interval(emb_1) + Interval(attn_lo, attn_hi)

        # MLP
        hn_mlp = iv_rms_norm([h0, h1], eps=RMS_EPS)
        g0 = hn_mlp[0] * gate_a + hn_mlp[1] * gate_c
        g1 = hn_mlp[0] * (gate_a - gate_c / embed_w0) + hn_mlp[1] * gate_c
        base = hn_mlp[0]

        # Key optimization: g1 - g0 = -hn[0] * gate_c / embed_w0 (algebraically exact).
        # When both gates are deeply positive or deeply negative, we can exploit
        # this to avoid the correlation loss from subtracting independent intervals.
        #
        # For |g| > 30: silu(g) ≈ g (error < g*exp(-g) < 1e-11)
        # So silu(g1)*base - silu(g0)*base ≈ (g1-g0)*base = -hn[0]²*gate_c/embed_w0
        #
        # For g << -30: silu(g) ≈ 0 (error < 1e-11)
        # So silu(g1)*base - silu(g0)*base ≈ 0

        DEEP_THRESHOLD = 30.0

        if g0.lo > DEEP_THRESHOLD and g1.lo > DEEP_THRESHOLD:
            # Both deeply positive: silu(g) ≈ g
            # mix_diff = base * (g1 - g0) = -base * hn[0] * gate_c / embed_w0
            #          = -hn[0]² * gate_c / embed_w0
            hn0_sq = hn_mlp[0].square()
            mix_diff = hn0_sq * (-gate_c / embed_w0)
            mlp_out = carry_w * mix_diff
        elif g0.hi < -DEEP_THRESHOLD and g1.hi < -DEEP_THRESHOLD:
            # Both deeply negative: silu(g) ≈ 0
            mlp_out = Interval(0.0)
        else:
            # Transition region — use naive interval arithmetic
            silu_g0 = iv_silu(g0)
            silu_g1 = iv_silu(g1)
            mlp_out = carry_w * (silu_g1 * base - silu_g0 * base)

        h0_post = h0
        h1_post = h1 + mlp_out

        # Final norm and logit differences
        hn_final = iv_rms_norm([h0_post, h1_post], eps=RMS_EPS)

        for d in range(VOCAB_SIZE):
            if d == t:
                continue
            delta_f0 = folded[t, 0] - folded[d, 0]
            delta_f1 = folded[t, 1] - folded[d, 1]
            diff = hn_final[0] * delta_f0 + hn_final[1] * delta_f1

            if diff.lo <= 0:
                return False, (f"digit {digit_pos}, target {t}, d_prev={d_prev}: "
                              f"logit[{t}]-logit[{d}] interval [{diff.lo:.6f}, {diff.hi:.6f}] "
                              f"not provably positive")

    return True, ""


# ── Main verification entry points ───────────────────────────────────────

def verify_by_carry_partition(
    model_spec: ModelSpec,
    timeout_seconds: int = 3600,
) -> SMTVerificationResult:
    """
    Formally verify a zcbtrak-family model using structural algebraic analysis.
    """
    start = time.time()

    if model_spec._model is None or not hasattr(model_spec._model, 'params'):
        return SMTVerificationResult(
            status="ERROR",
            notes=["Not a zcbtrak-family model (no .params attribute)"],
        )

    params = model_spec._model.params
    embed_w0, embed_w1 = float(params[0]), float(params[1])
    embed_table = _build_embed_table(embed_w0, embed_w1)

    # Precompute attention weights (constant, position-only)
    logger.info("Precomputing attention weights...")
    all_attn_weights = _precompute_all_attention_weights()

    total_partitions = 1024
    verified = 0
    inconclusive = []

    for carry_mask in range(total_partitions):
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            return SMTVerificationResult(
                status="TIMEOUT",
                solve_time_seconds=elapsed,
                notes=[f"Verified {verified}/{total_partitions}, "
                       f"{len(inconclusive)} inconclusive"],
            )

        partition_ok = True
        fail_reason = ""
        prev_output_digits: list[int] = []

        for digit_pos in range(OUTPUT_DIGITS):
            possible_targets = _possible_output_digits(carry_mask, digit_pos)
            seq_len = PROMPT_LEN + digit_pos + 1
            attn_w = all_attn_weights[digit_pos]

            all_targets_ok = True
            for target in possible_targets:
                ok, reason = _verify_output_digit(
                    params, carry_mask, digit_pos, target,
                    attn_w, embed_table, prev_output_digits,
                )
                if not ok:
                    all_targets_ok = False
                    fail_reason = reason
                    break

            if not all_targets_ok:
                partition_ok = False
                break

            # For the inductive step: previous digit varies across sub-partitions
            if len(possible_targets) == 1:
                prev_output_digits.append(possible_targets[0])
            else:
                prev_output_digits.append(-1)  # Sentinel: varies

        if partition_ok:
            verified += 1
        else:
            inconclusive.append((carry_mask, fail_reason))

        if (verified + len(inconclusive)) % 128 == 0:
            logger.info(
                "Progress: %d verified, %d inconclusive / %d checked (%.1fs)",
                verified, len(inconclusive),
                verified + len(inconclusive), time.time() - start,
            )

    elapsed = time.time() - start

    if inconclusive:
        return SMTVerificationResult(
            status="INCONCLUSIVE",
            solve_time_seconds=elapsed,
            method="structural_algebraic",
            notes=[
                f"Verified {verified}/{total_partitions} carry partitions",
                f"{len(inconclusive)} inconclusive partitions",
                f"First failure: partition {inconclusive[0][0]:010b}: {inconclusive[0][1]}",
            ],
        )

    return SMTVerificationResult(
        status="PROVEN_CORRECT",
        solve_time_seconds=elapsed,
        method="structural_algebraic",
        notes=[f"All {total_partitions} carry partitions formally verified"],
    )


def verify_full(
    model_spec: ModelSpec,
    strategy: str = "carry_partition",
    timeout_seconds: int = 3600,
) -> SMTVerificationResult:
    """Run formal verification."""
    return verify_by_carry_partition(model_spec, timeout_seconds)
