"""
Formal verifier for kswain98_8p.

This model is algebraically identical to zcbtrak_6p when c=1000:
  embed_w0 = c, embed_w1 = 1/c, v_proj_w = -22*c/sqrt(2),
  gate_w0 = g, gate_w1 = 128*c, carry_w = 100/256.

We extract c and g from the PyTorch model and reuse verify_smt's
interval arithmetic proof directly.
"""

import math
import numpy as np

from ..verify_smt import (
    _build_embed_table, _precompute_all_attention_weights,
    _verify_output_digit, _possible_output_digits,
    VOCAB_SIZE, OUTPUT_DIGITS,
)
from ..verify_formal import ModelVerifier, possible_output_digits, digit_sum_for_target


V_FACTOR = -22.0 / math.sqrt(2)
S_CONST = 100.0 / 256.0


class Kswain98Verifier:
    """Verifier for kswain98_8p using zcbtrak's structural algebraic proof."""

    def __init__(self, model):
        # Extract parameters from PyTorch model
        c = float(model.c.item())
        g = float(model.g.item())

        # Map to zcbtrak's 6-param format
        self.params = np.array([
            c,             # embed_w0
            1.0 / c,       # embed_w1
            V_FACTOR * c,  # v_proj_w
            g,             # gate_w0
            128.0 * c,     # gate_w1
            S_CONST,       # carry_w
        ], dtype=np.float64)

        self.embed_table = _build_embed_table(self.params[0], self.params[1])
        self.all_attn_weights = _precompute_all_attention_weights()

    def verify_digit(
        self,
        carry_mask: int,
        digit_pos: int,
        target_digit: int,
        prev_output_digits: list[int],
    ) -> tuple[bool, str]:
        attn_w = self.all_attn_weights[digit_pos]
        return _verify_output_digit(
            self.params, carry_mask, digit_pos, target_digit,
            attn_w, self.embed_table, prev_output_digits,
        )


def create_verifier(model) -> Kswain98Verifier:
    return Kswain98Verifier(model)
