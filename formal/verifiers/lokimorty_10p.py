"""
Formal verifier for Lokimorty_10p.

This model is algebraically identical to the zcbtrak 6-param family.
Since it requires mlx (Apple only), we hardcode the parameters derived
from the source code rather than extracting from a live model.

Parameter derivation from Lokimorty_10p.py:
  embed_w0 = EMBED_CONST = 1000
  embed_w1 = DECODE_QUAD = 0.001
  v_proj_w = -22 * DIGIT_SCALE = -22 * 1000/sqrt(2)
  gate_w0  = CARRY_ALPHA*(-94)/CONST_NORM = 256*(-94)/2 = -12032
  gate_w1  = CARRY_ALPHA*DIGIT_SCALE = 256*1000/2 = 128000
  carry_w  = (100/CARRY_ALPHA)*(1/CONST_NORM) = 100/256
"""

import math
import numpy as np

from ..verify_smt import (
    _build_embed_table, _precompute_all_attention_weights,
    _verify_output_digit,
    VOCAB_SIZE, OUTPUT_DIGITS,
)

EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(2)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
CARRY_ALPHA = 256.0 / CONST_NORM
DECODE_QUAD = 1e-3


class Lokimorty10pVerifier:
    """Verifier for Lokimorty_10p using zcbtrak's structural algebraic proof."""

    def __init__(self, model=None):
        # Hardcoded parameters (mlx model not needed)
        self.params = np.array([
            EMBED_CONST,                    # embed_w0
            DECODE_QUAD,                    # embed_w1
            -22.0 * DIGIT_SCALE,            # v_proj_w
            CARRY_ALPHA * (-94.0) / CONST_NORM,  # gate_w0
            CARRY_ALPHA * DIGIT_SCALE,      # gate_w1
            100.0 / 256.0,                  # carry_w
        ], dtype=np.float64)

        self.embed_table = _build_embed_table(self.params[0], self.params[1])
        self.all_attn_weights = _precompute_all_attention_weights()

    def verify_digit(self, carry_mask, digit_pos, target_digit, prev_output_digits):
        attn_w = self.all_attn_weights[digit_pos]
        return _verify_output_digit(
            self.params, carry_mask, digit_pos, target_digit,
            attn_w, self.embed_table, prev_output_digits,
        )


def create_verifier(model=None):
    return Lokimorty10pVerifier(model)
