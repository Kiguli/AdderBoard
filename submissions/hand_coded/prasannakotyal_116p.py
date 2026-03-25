import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_ADDEND = 9_999_999_999
VOCAB_SIZE = 10
MODEL_DIM = 3
NUM_LAYERS = 1
NUM_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 2
ROPE_THETA = 10000.0
OUTPUT_DIGITS = 11

EMBED_CONST = 1000.0
DIGIT_SCALE = EMBED_CONST / math.sqrt(MODEL_DIM)
CONST_NORM = math.sqrt(MODEL_DIM)
ALPHA = 20.0
QK_NORM_SCALE = 256.0
DECODE_LINEAR_EPS = 5e-4
DECODE_QUAD = DECODE_LINEAR_EPS / 2.0
CARRY_SLOPE = -0.1


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def _rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (ROPE_THETA ** (idx / head_dim))
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs).to(dtype=dtype), torch.sin(freqs).to(dtype=dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    y = torch.empty_like(x)
    y[..., ::2] = y_even
    y[..., 1::2] = y_odd
    return y


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(MODEL_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(MODEL_DIM, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, MODEL_DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = _rope_cache(seq_len, HEAD_DIM, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if NUM_KV_HEADS != NUM_HEADS:
            repeat = NUM_HEADS // NUM_KV_HEADS
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, NUM_HEADS * HEAD_DIM)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, MODEL_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(MODEL_DIM)
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm(MODEL_DIM)
        self.mlp = MLP()

        # Share norm vectors to reduce unique parameter count.
        self.post_attention_layernorm.weight = self.input_layernorm.weight
        self.self_attn.k_norm.weight = self.self_attn.q_norm.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyQwenAdder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(MODEL_DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return torch.matmul(h, self.embed_tokens.weight.t())


def _qvec(offset: int) -> tuple[float, float]:
    return (math.cos(offset), -math.sin(offset))


def _set_handcrafted_weights(model: TinyQwenAdder) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        model.embed_tokens.weight.copy_(
            torch.tensor(
                [
                    [EMBED_CONST - DECODE_QUAD * (d * d), float(d), DECODE_LINEAR_EPS * float(d)]
                    for d in range(10)
                ],
                dtype=torch.float32,
            )
        )

        model.norm.weight.copy_(
            torch.tensor(
                [
                    1.0 / CONST_NORM,
                    CARRY_SLOPE * DIGIT_SCALE * DECODE_LINEAR_EPS,
                    DIGIT_SCALE,
                ],
                dtype=torch.float32,
            )
        )

        layer = model.layers[0]
        layer.input_layernorm.weight.fill_(1.0)

        attn = layer.self_attn
        attn.q_norm.weight.copy_(torch.tensor([QK_NORM_SCALE, QK_NORM_SCALE], dtype=torch.float32))
        attn.k_proj.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        )
        attn.v_proj.weight.copy_(
            torch.tensor(
                [[0.0, DIGIT_SCALE, 0.0], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        )

        q_prev_a = _qvec(23)
        q_prev_b = _qvec(11)
        q_cur_a = _qvec(22)
        q_cur_b = _qvec(10)
        attn.q_proj.weight.copy_(
            torch.tensor(
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
                dtype=torch.float32,
            )
        )

        o_proj = torch.zeros((MODEL_DIM, NUM_HEADS * HEAD_DIM), dtype=torch.float32)
        o_proj[1, 0] = -1.0
        o_proj[1, 2] = -1.0
        o_proj[2, 4] = 1.0
        o_proj[2, 6] = 1.0
        attn.o_proj.weight.copy_(o_proj)

        gate = torch.zeros((INTERMEDIATE_SIZE, MODEL_DIM), dtype=torch.float32)
        gate[0, 0] = ALPHA * (-188.0) / CONST_NORM
        gate[0, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate[0, 2] = ALPHA * (20.0) * DIGIT_SCALE
        gate[1, 0] = ALPHA * (-189.0) / CONST_NORM
        gate[1, 1] = ALPHA * (-2.0) * DIGIT_SCALE
        gate[1, 2] = ALPHA * (20.0) * DIGIT_SCALE
        layer.mlp.gate_proj.weight.copy_(gate)

        up = torch.zeros((INTERMEDIATE_SIZE, MODEL_DIM), dtype=torch.float32)
        up[0, 0] = 1.0
        up[1, 0] = 1.0
        layer.mlp.up_proj.weight.copy_(up)

        scale = 1.0 / (ALPHA * CONST_NORM)
        down = torch.zeros((MODEL_DIM, INTERMEDIATE_SIZE), dtype=torch.float32)
        down[2, 0] = -10.0 * scale
        down[2, 1] = 10.0 * scale
        layer.mlp.down_proj.weight.copy_(down)


def _count_unique_parameters(model: nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += int(p.numel())
    return total


def _validate_addends(a: int, b: int) -> None:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")


def _encode_addends_internal(a: int, b: int) -> list[int]:
    _validate_addends(a, b)
    prompt = f"{a:010d}{b:010d}"
    a_digits = [int(c) for c in prompt[:10]]
    b_digits = [int(c) for c in prompt[10:]]
    return [0] + list(reversed(a_digits)) + [0] + [0] + list(reversed(b_digits)) + [0]


def build_model():
    model = TinyQwenAdder()
    _set_handcrafted_weights(model)
    model.eval()

    metadata = {
        "name": "QwenStyle116SharedNorm",
        "author": "nino",
        "params": _count_unique_parameters(model),
        "architecture": "1-layer Qwen-style causal self-attention with tied embeddings",
        "tricks": [
            "handcrafted weights",
            "RoPE with head_dim=2",
            "shared RMSNorm vectors",
            "autoregressive decoding",
        ],
    }
    return model, metadata


def add(model: TinyQwenAdder, a: int, b: int) -> int:
    _validate_addends(a, b)
    device = next(model.parameters()).device
    seq = _encode_addends_internal(a, b)

    with torch.no_grad():
        for _ in range(OUTPUT_DIGITS):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)
            next_digit = int(torch.argmax(logits[0, -1, :]).item())
            seq.append(next_digit)

    generated = "".join(str(d) for d in seq[-OUTPUT_DIGITS:])
    return int(generated[::-1])