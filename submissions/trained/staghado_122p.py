"""
Submission for Nano Transformer Adder leaderboard.

This submits the 122-param model (1h/1kv, ff=3), smallest to clear 99%.
The repo also contains checkpoints for:
  - 137-param (2h/1kv, ff=2): 99.73% accuracy
  - 146-param (2h/1kv, ff=3): 99.98% accuracy
  - 155-param (ff=4): 99.92% accuracy
  - 173-param (ff=6): 99.93% accuracy
  - 200-param (ff=9): 99.99% accuracy
  - 228-param (d=4, ff=6): 100% accuracy
Change INTERMEDIATE_SIZE (and MODEL_DIM for 228) to switch.
"""

import os
import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs

MODEL_DIM = 3
ATTENTION_HEADS = 1
KEY_VALUE_HEADS = 1
HEAD_DIM = 4
INTERMEDIATE_SIZE = 3
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
ROPE_THETA = 3


def _build_model_args():
    return ModelArgs(
        model_type="qwen3",
        hidden_size=MODEL_DIM,
        num_hidden_layers=1,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=ATTENTION_HEADS,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB_SIZE,
        tie_word_embeddings=True,
        num_key_value_heads=KEY_VALUE_HEADS,
        max_position_embeddings=64,
        rope_theta=ROPE_THETA,
        head_dim=HEAD_DIM,
    )


def _encode(a: int, b: int) -> list[int]:
    pa, pb = f"{a:010d}", f"{b:010d}"
    return ([0] + [int(c) for c in reversed(pa)] + [0] +
            [0] + [int(c) for c in reversed(pb)] + [0])


def _count_params(model):
    from mlx.utils import tree_flatten
    return sum(x.size for _, x in tree_flatten(model.parameters()))


def build_model():
    model = Model(_build_model_args())
    n_params = _count_params(model)
    checkpoint = os.path.join(os.path.dirname(__file__), "checkpoint", f"best_{n_params}.npz")
    weights = list(mx.load(checkpoint).items())
    model.load_weights(weights)
    model.eval()
    mx.eval(model.parameters())

    metadata = {
        "name": f"{n_params}-param Qwen3 Adder",
        "author": "staghado",
        "params": n_params,
        "architecture": f"1L Qwen3, d={MODEL_DIM}, {ATTENTION_HEADS}h/{KEY_VALUE_HEADS}kv, hd={HEAD_DIM}, ff={INTERMEDIATE_SIZE}",
        "tricks": ["Tied embeddings", f"RoPE theta={ROPE_THETA}"]
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    seq = _encode(a, b)
    digits = []
    for _ in range(OUTPUT_DIGITS):
        x = mx.array([seq], dtype=mx.int32)
        logits = model(x)
        d = int(mx.argmax(logits[0, -1, :]).item())
        seq.append(d)
        digits.append(d)
    # Digits are LSD-first, convert to integer
    return int("".join(str(d) for d in reversed(digits)))