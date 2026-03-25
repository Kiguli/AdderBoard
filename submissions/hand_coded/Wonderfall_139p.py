import argparse
import math
import random

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mlx_lm.models.qwen3 import Model, ModelArgs

MODEL_LAYERS = 1
MODEL_DIM = 3
ATTENTION_HEADS = 4
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 4
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1

# Use one large constant embedding component so RMSNorm scaling is near-constant.
EMBED_CONST = 1000.0
DIGIT_SCALE = EMBED_CONST / math.sqrt(MODEL_DIM)
CONST_NORM = math.sqrt(MODEL_DIM)
ALPHA = 20.0
QK_NORM_SCALE = 256.0
# Tied embedding decode: with h0≈1 and h2≈z, make logits ~ eps*z*d - (eps/2)*d^2.
DECODE_LINEAR_EPS = 5e-4
DECODE_QUAD = DECODE_LINEAR_EPS / 2.0


def build_model_args() -> ModelArgs:
    return ModelArgs(
        model_type="qwen3",
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
    )


def _validate_addends(a: int, b: int) -> None:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")


def _encode_addends_internal(a: int, b: int) -> list[int]:
    _validate_addends(a, b)
    prompt = f"{a:010d}{b:010d}"
    a = [int(c) for c in prompt[:10]]
    b = [int(c) for c in prompt[10:]]
    return [0] + list(reversed(a)) + [0] + [0] + list(reversed(b)) + [0]


def _expected_output(a: int, b: int) -> str:
    _validate_addends(a, b)
    return str(a + b)[::-1].ljust(OUTPUT_DIGITS, "0")


def _qvec(offset: int) -> tuple[float, float]:
    # With RoPE on head_dim=2, this peaks attention at the chosen relative offset.
    return (math.cos(offset), -math.sin(offset))


def hand_set_weights_better(model: Model) -> None:
    params = tree_map(lambda x: mx.zeros_like(x), model.parameters())

    # Embedding layout:
    # - dim0: large constant + quadratic term for tied-output decode bias
    # - dim1: exact token digit (used for carry logic)
    # - dim2: tiny digit channel (used for tied-output decode slope)
    params["model"]["embed_tokens"]["weight"] = mx.array(
        [
            [EMBED_CONST - DECODE_QUAD * (d * d), float(d), DECODE_LINEAR_EPS * float(d)]
            for d in range(10)
        ],
        dtype=mx.float32,
    )

    params["model"]["norm"]["weight"] = mx.array(
        [1.0 / CONST_NORM, 0.0, DIGIT_SCALE], dtype=mx.float32
    )

    layer = params["model"]["layers"][0]
    layer["input_layernorm"]["weight"] = mx.ones((MODEL_DIM,), dtype=mx.float32)
    layer["post_attention_layernorm"]["weight"] = mx.ones((MODEL_DIM,), dtype=mx.float32)

    attn = layer["self_attn"]
    attn["q_norm"]["weight"] = mx.array([QK_NORM_SCALE, QK_NORM_SCALE], dtype=mx.float32)
    attn["k_norm"]["weight"] = mx.array([QK_NORM_SCALE, QK_NORM_SCALE], dtype=mx.float32)
    attn["k_proj"]["weight"] = mx.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=mx.float32
    )
    attn["v_proj"]["weight"] = mx.array(
        [[0.0, DIGIT_SCALE, 0.0], [0.0, 0.0, 0.0]], dtype=mx.float32
    )

    q_prev_a = _qvec(23)
    q_prev_b = _qvec(11)
    q_cur_a = _qvec(22)
    q_cur_b = _qvec(10)
    attn["q_proj"]["weight"] = mx.array(
        [
            [q_prev_a[0], 0.0, 0.0],
            [q_prev_a[1], 0.0, 0.0],
            [q_prev_b[0], 0.0, 0.0],
            [q_prev_b[1], 0.0, 0.0],
            [q_cur_a[0], 0.0, 0.0],
            [q_cur_a[1], 0.0, 0.0],
            [q_cur_b[0], 0.0, 0.0],
            [q_cur_b[1], 0.0, 0.0],
        ],
        dtype=mx.float32,
    )

    # After attention:
    # - dim1 stores x = sum_prev - a_prev - b_prev
    # - dim2 stores s = a_cur + b_cur
    o_proj = np.zeros((MODEL_DIM, ATTENTION_HEADS * HEAD_DIM), dtype=np.float32)
    o_proj[1, 0] = -1.0
    o_proj[1, 2] = -1.0
    o_proj[2, 4] = 1.0
    o_proj[2, 6] = 1.0
    attn["o_proj"]["weight"] = mx.array(o_proj, dtype=mx.float32)

    # One-layer arithmetic:
    # c = I[x <= -9] where x = sum_prev - a_prev - b_prev
    # w = I[s + c >= 10], using high-margin linear separator m = -2*x + 20*s - 189
    # z = s + c - 10*w
    gate = np.zeros((INTERMEDIATE_SIZE, MODEL_DIM), dtype=np.float32)
    # c pair: relu(m_c) - relu(m_c - 1), m_c = -x - 8
    gate[0, 0] = ALPHA * (-8.0) / CONST_NORM
    gate[0, 1] = ALPHA * (-1.0) * DIGIT_SCALE
    gate[1, 0] = ALPHA * (-9.0) / CONST_NORM
    gate[1, 1] = ALPHA * (-1.0) * DIGIT_SCALE
    # w pair: relu(m + 1) - relu(m), m = -2*x + 20*s - 189
    gate[2, 0] = ALPHA * (-188.0) / CONST_NORM
    gate[2, 1] = ALPHA * (-2.0) * DIGIT_SCALE
    gate[2, 2] = ALPHA * (20.0) * DIGIT_SCALE
    gate[3, 0] = ALPHA * (-189.0) / CONST_NORM
    gate[3, 1] = ALPHA * (-2.0) * DIGIT_SCALE
    gate[3, 2] = ALPHA * (20.0) * DIGIT_SCALE
    layer["mlp"]["gate_proj"]["weight"] = mx.array(gate, dtype=mx.float32)

    up = np.zeros((INTERMEDIATE_SIZE, MODEL_DIM), dtype=np.float32)
    up[:, 0] = 1.0
    layer["mlp"]["up_proj"]["weight"] = mx.array(up, dtype=mx.float32)

    scale = 1.0 / (ALPHA * CONST_NORM)
    down = np.zeros((MODEL_DIM, INTERMEDIATE_SIZE), dtype=np.float32)
    down[2, 0] = 1.0 * scale
    down[2, 1] = -1.0 * scale
    down[2, 2] = -10.0 * scale
    down[2, 3] = 10.0 * scale
    layer["mlp"]["down_proj"]["weight"] = mx.array(down, dtype=mx.float32)

    model.update(params)
    mx.eval(model.parameters())


def build_magic_model() -> Model:
    model = Model(build_model_args())
    hand_set_weights_better(model)
    return model


def _generate_output_batch(model: Model, addends: list[tuple[int, int]]) -> list[str]:
    internal = [_encode_addends_internal(a, b) for a, b in addends]
    for _ in range(OUTPUT_DIGITS):
        x = mx.array(internal, dtype=mx.int32)
        logits = model(x)
        next_digits = np.array(mx.argmax(logits[:, -1, :], axis=-1), dtype=np.int32)
        for seq, next_digit in zip(internal, next_digits):
            seq.append(int(next_digit))
    return ["".join(str(d) for d in seq[-OUTPUT_DIGITS:]) for seq in internal]


def run_self_test_batched(model: Model, num_tests: int, batch_size: int) -> None:
    rng = random.Random(123)
    tested = 0
    while tested < num_tests:
        cur_batch_size = min(batch_size, num_tests - tested)
        addends = []
        expected = []
        for _ in range(cur_batch_size):
            a = rng.randint(0, 10**10 - 1)
            b = rng.randint(0, 10**10 - 1)
            addends.append((a, b))
            expected.append(_expected_output(a, b))
        actual = _generate_output_batch(model, addends)
        for (a, b), exp, act in zip(addends, expected, actual):
            if act != exp:
                raise AssertionError(f"Mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}")
        tested += cur_batch_size
        print(f"self-test progress: {tested}/{num_tests}")


def run_edge_tests(model: Model) -> None:
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
    print(f"parameter count: {count_parameters(model.parameters())}")
    run_edge_tests(model)
    run_self_test_batched(model, args.num_tests, args.batch_size)
    print(f"self-test passed ({args.num_tests} random cases, batch size {args.batch_size})")


if __name__ == "__main__":
    main()
