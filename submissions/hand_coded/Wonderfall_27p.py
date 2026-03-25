import argparse
import math
import random

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

MODEL_DIM = 2
HEAD_DIM = 2
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
ROPE_SCALE = 1.0 / ROPE_FACTOR
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
CARRY_ALPHA = 256.0 / CONST_NORM


def _linear(x: mx.array, w: mx.array) -> mx.array:
    # x[..., in], w[out, in] -> x @ w.T
    return mx.matmul(x, mx.transpose(w))


def _rotate_2d(x: mx.array, theta: mx.array) -> mx.array:
    c = mx.cos(theta)
    s = mx.sin(theta)
    x0 = x[..., 0:1]
    x1 = x[..., 1:2]
    y0 = c * x0 - s * x1
    y1 = s * x0 + c * x1
    return mx.concatenate([y0, y1], axis=-1)


class ParameterlessRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)


class FactorizedQuadEmbedding(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_values = mx.zeros((vocab_size,), dtype=mx.float32)
        self.const0 = mx.zeros((1,), dtype=mx.float32)
        self.quad0 = mx.zeros((1,), dtype=mx.float32)

    def weight(self) -> mx.array:
        t = self.token_values
        col0 = self.const0 + self.quad0 * mx.square(t)
        col1 = t
        return mx.stack([col0, col1], axis=-1)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.weight()[tokens]

    def as_linear(self, x: mx.array) -> mx.array:
        w = self.weight()
        return mx.matmul(x, mx.transpose(w))


class TiedDenseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared dense matrices (swappable parameters).
        self.w_qk = mx.zeros((HEAD_DIM, MODEL_DIM), dtype=mx.float32)
        self.w_vo = mx.zeros((HEAD_DIM, MODEL_DIM), dtype=mx.float32)
        # Additional dense-path scalars (swappable).
        self.q_phase = mx.zeros((1,), dtype=mx.float32)
        self.v_scale = mx.ones((1,), dtype=mx.float32)

        self.rope = nn.RoPE(HEAD_DIM, traditional=False, base=10000.0, scale=ROPE_SCALE)
        self.scale = HEAD_DIM**-0.5

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        bsz, seqlen, _ = x.shape

        q = _rotate_2d(_linear(x, self.w_qk), self.q_phase)
        k = _linear(x, self.w_qk)
        v = self.v_scale * _linear(x, self.w_vo)

        q = q.reshape(bsz, seqlen, 1, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, 1, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, 1, HEAD_DIM).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seqlen, HEAD_DIM)

        # Shared VO matrix: output path uses transpose-side multiplication.
        return mx.matmul(out, self.w_vo)


class TiedDenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1_row0_col0 = mx.zeros((1,), dtype=mx.float32)
        self.w1_col1_scale = mx.zeros((1,), dtype=mx.float32)

    def _w1(self, const0: mx.array) -> mx.array:
        # Compressed tie:
        # row0 = [a, s*const0], row1 = [a-s, s*const0]
        a = self.w1_row0_col0
        s = self.w1_col1_scale
        col1 = s * const0
        row0 = mx.concatenate([a, col1], axis=0)
        row1 = mx.concatenate([a - s, col1], axis=0)
        return mx.stack([row0, row1], axis=0)

    def __call__(self, x: mx.array, w_vo: mx.array, const0: mx.array) -> mx.array:
        z = mx.maximum(_linear(x, self._w1(const0)), 0.0)
        # Cross-tie: reuse attention VO matrix as MLP second projection.
        return mx.matmul(z, w_vo)


class TiedDenseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = FactorizedQuadEmbedding(VOCAB_SIZE)
        self.input_norm = ParameterlessRMSNorm(eps=1e-6)
        self.attn = TiedDenseAttention()
        self.post_attn_norm = ParameterlessRMSNorm(eps=1e-6)
        self.mlp = TiedDenseMLP()
        self.final_norm = nn.RMSNorm(MODEL_DIM, eps=1e-6)

    def __call__(self, inputs: mx.array) -> mx.array:
        h = self.embed_tokens(inputs)
        mask = create_attention_mask(h, None)
        h = h + self.attn(self.input_norm(h), mask=mask)
        h = h + self.mlp(self.post_attn_norm(h), self.attn.w_vo, self.embed_tokens.const0)
        h = self.final_norm(h)
        return self.embed_tokens.as_linear(h)


def _validate_addends(a: int, b: int) -> None:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")


def _encode_addends_internal(a: int, b: int) -> list[int]:
    _validate_addends(a, b)
    a_digits = [int(c) for c in f"{a:010d}"][::-1]
    b_digits = [int(c) for c in f"{b:010d}"][::-1]
    return [0] + a_digits + [0] * 9 + b_digits + [0]


def _expected_output(a: int, b: int) -> str:
    _validate_addends(a, b)
    return str(a + b)[::-1].ljust(OUTPUT_DIGITS, "0")


def hand_set_weights(model: TiedDenseTransformer) -> None:
    params = tree_map(lambda x: mx.zeros_like(x), model.parameters())

    params["embed_tokens"]["token_values"] = mx.array(
        [-float(d) for d in range(VOCAB_SIZE)],
        dtype=mx.float32,
    )
    params["embed_tokens"]["const0"] = mx.array([EMBED_CONST], dtype=mx.float32)
    params["embed_tokens"]["quad0"] = mx.array([-DECODE_QUAD], dtype=mx.float32)
    params["final_norm"]["weight"] = mx.array(
        [(DECODE_CURVATURE / DECODE_QUAD) / CONST_NORM, -(DIGIT_SCALE / 50.0)],
        dtype=mx.float32,
    )

    # q/k shared matrix + phase reproduces 46-param q/k geometry.
    params["attn"]["w_qk"] = mx.array(
        [[QK_SCALE, 0.0], [0.0, 0.0]],
        dtype=mx.float32,
    )
    params["attn"]["q_phase"] = mx.array([-PHI], dtype=mx.float32)

    # Cross-tied construction:
    # choose w_vo = old_mlp_w2.T so MLP second projection is reused from attention.
    c = 100.0 / CARRY_ALPHA
    params["attn"]["w_vo"] = mx.array(
        [[0.0, -c], [0.0, c]],
        dtype=mx.float32,
    )
    v_scale_old = -22.0 * DIGIT_SCALE
    params["attn"]["v_scale"] = mx.array(
        [v_scale_old / (2.0 * c * c)],
        dtype=mx.float32,
    )

    params["mlp"]["w1_row0_col0"] = mx.array(
        [CARRY_ALPHA * (-94.0) / CONST_NORM],
        dtype=mx.float32,
    )
    params["mlp"]["w1_col1_scale"] = mx.array(
        [CARRY_ALPHA / CONST_NORM],
        dtype=mx.float32,
    )
    model.update(params)
    mx.eval(model.parameters())


def build_magic_model() -> TiedDenseTransformer:
    model = TiedDenseTransformer()
    hand_set_weights(model)
    return model


def _generate_output_batch(model: TiedDenseTransformer, addends: list[tuple[int, int]]) -> list[str]:
    internal = [_encode_addends_internal(a, b) for a, b in addends]
    for _ in range(OUTPUT_DIGITS):
        x = mx.array(internal, dtype=mx.int32)
        logits = model(x)
        next_digits = np.array(mx.argmax(logits[:, -1, :], axis=-1), dtype=np.int32)
        for seq, next_digit in zip(internal, next_digits):
            seq.append(int(next_digit))
    return ["".join(str(d) for d in seq[-OUTPUT_DIGITS:]) for seq in internal]


def run_self_test_batched(model: TiedDenseTransformer, num_tests: int, batch_size: int) -> None:
    rng = random.Random(123)
    tested = 0
    while tested < num_tests:
        cur_batch_size = min(batch_size, num_tests - tested)
        addends = []
        expected = []
        for _ in range(cur_batch_size):
            a = rng.randint(0, MAX_ADDEND)
            b = rng.randint(0, MAX_ADDEND)
            addends.append((a, b))
            expected.append(_expected_output(a, b))
        actual = _generate_output_batch(model, addends)
        for (a, b), exp, act in zip(addends, expected, actual):
            if act != exp:
                raise AssertionError(f"Mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}")
        tested += cur_batch_size
        print(f"self-test progress: {tested}/{num_tests}")


def run_edge_tests(model: TiedDenseTransformer) -> None:
    cases = [
        (0, 0),
        (0, 1),
        (1, 0),
        (9999999999, 0),
        (0, 9999999999),
        (9999999999, 1),
        (9999999999, 9999999999),
        (1234567890, 9876543210),
        (5000000000, 5000000000),
    ]
    for k in range(1, 11):
        cases.append((10**k - 1, 1))

    actual = _generate_output_batch(model, cases)
    expected = [_expected_output(a, b) for a, b in cases]
    for (a, b), exp, act in zip(cases, expected, actual):
        if act != exp:
            raise AssertionError(
                f"EDGE mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}"
            )
    print(f"edge-tests passed ({len(cases)} cases)")


def count_parameters(node) -> int:
    if isinstance(node, dict):
        return sum(count_parameters(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return sum(count_parameters(v) for v in node)
    if hasattr(node, "shape"):
        n = 1
        for dim in node.shape:
            n *= int(dim)
        return n
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tests", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_tests < 0:
        raise ValueError("--num-tests must be >= 0")

    model = build_magic_model()
    print(f"stored parameter count: {count_parameters(model.parameters())}")
    run_edge_tests(model)
    run_self_test_batched(model, args.num_tests, args.batch_size)
    print(f"self-test passed ({args.num_tests} random cases, batch size {args.batch_size})")


if __name__ == "__main__":
    main()
