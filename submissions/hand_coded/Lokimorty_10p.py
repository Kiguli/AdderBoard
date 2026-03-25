"""
AdderBoard submission candidate: 10-parameter qwen-derived adder.

Compression relative to accepted 12p:
- Tied gate family (2 params instead of 3)
- MLP up/down collapsed to one shared carry scalar

Keeps full 2-parameter Q projection (not phase-tied).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope

MODEL_LAYERS = 1
MODEL_DIM = 2
ATTENTION_HEADS = 1
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 2
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1

EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
DECODE_QUAD = 1e-3
DECODE_CURVATURE = 0.1

ROPE_PERIOD = 19.0
ROPE_FACTOR = ROPE_PERIOD / (2.0 * math.pi)
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
CARRY_ALPHA = 256.0 / CONST_NORM


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class UnitRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)


class SharedNormAttention(nn.Module):
    class StructuredQProj(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((2,), dtype=mx.float32)

        def __call__(self, x: mx.array) -> mx.array:
            q0 = x[..., 0] * self.weight[0]
            q1 = x[..., 0] * self.weight[1]
            return mx.stack([q0, q1], axis=-1)

    class StructuredKProj(nn.Module):
        def __call__(self, x: mx.array) -> mx.array:
            k0 = x[..., 0]
            k1 = mx.zeros_like(k0)
            return mx.stack([k0, k1], axis=-1)

    class StructuredVProj(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((1,), dtype=mx.float32)

        def __call__(self, x: mx.array) -> mx.array:
            v = self.weight[0]
            v0 = x[..., 1] * v
            v1 = mx.zeros_like(v0)
            return mx.stack([v0, v1], axis=-1)

    class StructuredOProj(nn.Module):
        def __call__(self, x: mx.array) -> mx.array:
            y0 = mx.zeros_like(x[..., 0])
            y1 = x[..., 0]
            return mx.stack([y0, y1], axis=-1)

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = (self.head_dim**-0.5) * (QK_NORM_SCALE * QK_NORM_SCALE)

        self.q_proj = SharedNormAttention.StructuredQProj()
        self.k_proj = SharedNormAttention.StructuredKProj()
        self.v_proj = SharedNormAttention.StructuredVProj()
        self.o_proj = SharedNormAttention.StructuredOProj()

        self.qk_norm = UnitRMSNorm(eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        bsz, seq_len, _ = x.shape
        queries = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.qk_norm(queries)
        keys = self.qk_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        attn = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        out = attn.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    class TiedGateProj(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((2,), dtype=mx.float32)

        def __call__(self, x: mx.array) -> mx.array:
            a = self.weight[0]
            c = self.weight[1]
            g0 = x[..., 0] * a + x[..., 1] * c
            g1 = x[..., 0] * (a - c / EMBED_CONST) + x[..., 1] * c
            return mx.stack([g0, g1], axis=-1)

    class SharedCarryProj(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.weight = mx.zeros((1,), dtype=mx.float32)

        def __call__(self, gate: mx.array, base: mx.array) -> mx.array:
            up = mx.broadcast_to(base[..., None], base.shape + (self.hidden_dim,))
            mix = swiglu(gate, up)
            y0 = mx.zeros_like(base)
            y1 = self.weight[0] * (mix[..., 1] - mix[..., 0])
            return mx.stack([y0, y1], axis=-1)

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_proj = MLP.TiedGateProj()
        self.carry_proj = MLP.SharedCarryProj(hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        return self.carry_proj(gate, x[..., 0])


class SharedNormBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = SharedNormAttention(args)
        self.mlp = MLP(args.intermediate_size)
        self.layernorm = UnitRMSNorm(eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.layernorm(x), mask, cache)
        return h + self.mlp(self.layernorm(h))


class ParametricDigitEmbedding(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = mx.zeros((2,), dtype=mx.float32)

    def table(self) -> mx.array:
        d = mx.arange(self.vocab_size, dtype=mx.float32)
        e0 = self.weight[0] - self.weight[1] * (d * d)
        e1 = -d
        return mx.stack([e0, e1], axis=-1)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.table()[tokens]

    def as_linear(self, x: mx.array) -> mx.array:
        return x @ self.table().T


class Core(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = ParametricDigitEmbedding(args.vocab_size)
        self.layers = [SharedNormBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = Core(args)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        return self.model.embed_tokens.as_linear(out)


def _build_model_args() -> ModelArgs:
    return ModelArgs(
        model_type="qwen10-freeq-tiedgate-sharedcarry",
        hidden_size=MODEL_DIM,
        num_hidden_layers=MODEL_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=ATTENTION_HEADS,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB_SIZE,
        tie_word_embeddings=True,
        num_key_value_heads=KEY_VALUE_HEADS,
        max_position_embeddings=2048,
        rope_theta=10000,
        head_dim=HEAD_DIM,
        rope_scaling={"type": "linear", "factor": ROPE_FACTOR},
    )


def _count_parameters(node) -> int:
    if isinstance(node, dict):
        return sum(_count_parameters(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return sum(_count_parameters(v) for v in node)
    if hasattr(node, "shape"):
        n = 1
        for dim in node.shape:
            n *= int(dim)
        return n
    return 0


def _encode_prompt(a: int, b: int) -> list[int]:
    a_digits = [int(c) for c in f"{a:010d}"][::-1]
    b_digits = [int(c) for c in f"{b:010d}"][::-1]
    return [0] + a_digits + [0] * 9 + b_digits + [0]


def _init_weights(model: Model) -> None:
    params = model.parameters()
    params["model"]["embed_tokens"]["weight"] = mx.array([EMBED_CONST, DECODE_QUAD], dtype=mx.float32)
    params["model"]["norm"]["weight"] = mx.array(
        [(DECODE_CURVATURE / DECODE_QUAD) / CONST_NORM, -(DIGIT_SCALE / 50.0)],
        dtype=mx.float32,
    )

    layer = params["model"]["layers"][0]
    layer["self_attn"]["v_proj"]["weight"] = mx.array([-22.0 * DIGIT_SCALE], dtype=mx.float32)
    layer["self_attn"]["q_proj"]["weight"] = mx.array([math.cos(PHI), -math.sin(PHI)], dtype=mx.float32)

    gate = np.zeros((2,), dtype=np.float32)
    gate[0] = CARRY_ALPHA * (-94.0) / CONST_NORM
    gate[1] = CARRY_ALPHA * DIGIT_SCALE
    layer["mlp"]["gate_proj"]["weight"] = mx.array(gate, dtype=mx.float32)

    layer["mlp"]["carry_proj"]["weight"] = mx.array(
        [(100.0 / CARRY_ALPHA) * (1.0 / CONST_NORM)],
        dtype=mx.float32,
    )

    model.update(params)
    mx.eval(model.parameters())


def _generate_output(model: Model, a: int, b: int) -> str:
    seq = _encode_prompt(a, b)
    for _ in range(OUTPUT_DIGITS):
        x = mx.array([seq], dtype=mx.int32)
        logits = model(x)
        d = int(np.array(mx.argmax(logits[:, -1, :], axis=-1), dtype=np.int32)[0])
        seq.append(d)
    return "".join(str(d) for d in seq[-OUTPUT_DIGITS:])


def build_model():
    model = Model(_build_model_args())
    _init_weights(model)
    metadata = {
        "name": "qwen10_freeq_tiedgate_sharedcarry",
        "author": "Lokimorty",
        "params": _count_parameters(model.parameters()),
        "architecture": "qwen-derived 1L h=2 with free-q, tied gate, shared carry scalar",
        "tricks": [
            "RoPE period-19 geometry",
            "tied carry hinge gate",
            "shared carry-scale scalar",
            "2-parameter embedding e(d)=[c0-c1*d^2,-d]",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")
    out = _generate_output(model, a, b)
    return int(out[::-1])
