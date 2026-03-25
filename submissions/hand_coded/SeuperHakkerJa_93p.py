import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F



def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def rope_2d(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    RoPE for head_dim=2.
    x:   (B, H, T, 2)
    pos: (T,) angles (we use angle = position index)
    """
    c = torch.cos(pos).view(1, 1, -1, 1)
    s = torch.sin(pos).view(1, 1, -1, 1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)

def encode_prompt(a: int, b: int) -> list[int]:
    """
    Prompt layout:
      [0] + rev(a10) + [0] + [0] + rev(b10) + [0]
    """
    s = f"{a:010d}{b:010d}"
    A = [int(c) for c in s[:10]]
    B = [int(c) for c in s[10:]]
    return [0] + list(reversed(A)) + [0] + [0] + list(reversed(B)) + [0]

def expected_output(a: int, b: int) -> str:
    return str(a + b)[::-1].ljust(11, "0")



@dataclass(frozen=True)
class Config:
    vocab_size: int = 10
    hidden_size: int = 2
    num_attention_heads: int = 5
    head_dim: int = 2
    intermediate_size: int = 4

    # numeric margins
    embed_const: float = 1000.0
    decode_eps: float = 5e-4
    qk_scale: float = 256.0
    causal_mask_neg: float = -1e4

    # head order: self, prev_a, prev_b, cur_a, cur_b
    rope_offsets: tuple[float, float, float, float, float] = (0.0, 23.0, 11.0, 22.0, 10.0)


class Transformer(nn.Module):
    """
    Parameter count is 93 with the default Config.
    """

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size
        D = config.hidden_size
        H = config.num_attention_heads
        hd = config.head_dim
        M = config.intermediate_size

        # tied embedding/unembedding (10,2)
        self.embed_tokens = nn.Parameter(torch.zeros(V, D), requires_grad=False)

        # final scaling before tied decode (2,)
        self.out_scale = nn.Parameter(torch.ones(D), requires_grad=False)

        # attention projections 
        self.q_proj = nn.Parameter(torch.zeros(H * hd, D), requires_grad=False)  # (10,2)
        self.k_proj = nn.Parameter(torch.zeros(hd, D), requires_grad=False)      # (2,2)
        self.v_proj = nn.Parameter(torch.zeros(hd, D), requires_grad=False)      # (2,2)
        self.o_proj = nn.Parameter(torch.zeros(D, H * hd), requires_grad=False)  # (2,10)

        # causal mask constant
        self.causal_neg = nn.Parameter(
            torch.tensor([config.causal_mask_neg], dtype=torch.float32),
            requires_grad=False
        )

        # MLP: relu(xW1^T + b1)W2^T + b2
        self.w1 = nn.Parameter(torch.zeros(M, D), requires_grad=False)  # (4,2)
        self.b1 = nn.Parameter(torch.zeros(M), requires_grad=False)     # (4,)
        self.w2 = nn.Parameter(torch.zeros(D, M), requires_grad=False)  # (2,4)
        self.b2 = nn.Parameter(torch.zeros(D), requires_grad=False)     # (2,)

        self._init_weights()

    def _init_weights(self) -> None:
        cfg = self.config
        C = cfg.embed_const
        eps = cfg.decode_eps
        qk = cfg.qk_scale
        quad = eps / 2.0

        # Embedding: [C - (eps/2)*d^2, d]
        emb = [[C - quad * (d * d), float(d)] for d in range(10)]
        self.embed_tokens.data.copy_(torch.tensor(emb, dtype=torch.float32))

        # Output scaling for float32-safe tied decode
        self.out_scale.data.copy_(torch.tensor([1.0 / C, eps], dtype=torch.float32))

        # K uses only dim0, V uses only dim1
        self.k_proj.data.copy_(torch.tensor([[qk, 0.0], [0.0, 0.0]], dtype=torch.float32))
        self.v_proj.data.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32))

        # Q rows are RoPE steering vectors for each offset
        def qvec(off: float) -> tuple[float, float]:
            return math.cos(off), -math.sin(off)

        Q = torch.zeros_like(self.q_proj.data)
        for h, off in enumerate(cfg.rope_offsets):
            c, s = qvec(off)
            Q[2 * h + 0, 0] = c * qk
            Q[2 * h + 1, 0] = s * qk
        self.q_proj.data.copy_(Q)

        # o_proj wiring:
        # dim0 accumulates x = self - prev_a - prev_b (constant already lives in dim0)
        # dim1 becomes s = cur_a + cur_b by adding (-self + cur_a + cur_b) then residual adds self
        O = torch.zeros_like(self.o_proj.data)
        O[0, 0] = +1.0
        O[0, 2] = -1.0
        O[0, 4] = -1.0
        O[1, 0] = -1.0
        O[1, 6] = +1.0
        O[1, 8] = +1.0
        self.o_proj.data.copy_(O)

        # MLP implements carry and digit correction using ReLU hinge pairs
        self.w1.data.copy_(torch.tensor(
            [
                [-1.0,  0.0],   # (C-8) - u0
                [-1.0,  0.0],   # (C-9) - u0
                [-2.0, 20.0],   # m+1
                [-2.0, 20.0],   # m
            ],
            dtype=torch.float32
        ))
        self.b1.data.copy_(torch.tensor(
            [C - 8.0, C - 9.0, 2 * C - 188.0, 2 * C - 189.0],
            dtype=torch.float32
        ))

        self.w2.data.zero_()
        self.w2.data[1, 0] = +1.0
        self.w2.data[1, 1] = -1.0
        self.w2.data[1, 2] = -10.0
        self.w2.data[1, 3] = +10.0
        self.b2.data.zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B,T) tokens in [0..9]
        returns logits: (B,T,10)
        """
        cfg = self.config
        x = F.embedding(input_ids, self.embed_tokens)  # (B,T,2)
        B, T, _ = x.shape

        pos = torch.arange(T, device=x.device, dtype=x.dtype)

        # Q, K, V
        q = (x @ self.q_proj.t()).view(B, T, cfg.num_attention_heads, cfg.head_dim).permute(0, 2, 1, 3)
        k = (x @ self.k_proj.t()).view(B, T, 1, cfg.head_dim).permute(0, 2, 1, 3).expand(-1, cfg.num_attention_heads, -1, -1)
        v = (x @ self.v_proj.t()).view(B, T, 1, cfg.head_dim).permute(0, 2, 1, 3).expand(-1, cfg.num_attention_heads, -1, -1)

        q = rope_2d(q, pos)
        k = rope_2d(k, pos)

        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(cfg.head_dim)

        # causal mask
        upper = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype), diagonal=1)
        scores = scores + upper.view(1, 1, T, T) * self.causal_neg.to(dtype=x.dtype)

        w = F.softmax(scores, dim=-1)
        att = torch.einsum("bhts,bhsd->bhtd", w, v)
        att = att.permute(0, 2, 1, 3).contiguous().view(B, T, cfg.num_attention_heads * cfg.head_dim)

        # residual attention
        x = x + (att @ self.o_proj.t())

        # residual MLP
        h = F.relu(x @ self.w1.t() + self.b1)
        x = x + (h @ self.w2.t() + self.b2)

        # tied decode
        y = x * self.out_scale
        return y @ self.embed_tokens.t()



@torch.no_grad()
def generate(model: nn.Module, a: int, b: int, device: str = "cpu") -> str:
    seq = encode_prompt(a, b)
    x = torch.tensor([seq], dtype=torch.long, device=device)
    for _ in range(11):
        nxt = model(x)[:, -1].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    return "".join(str(d) for d in x[0, -11:].tolist())

def run_edge_tests(model: nn.Module, device: str) -> None:
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

    for a, b in cases:
        got = generate(model, a, b, device=device)
        exp = expected_output(a, b)
        if got != exp:
            raise AssertionError(f"EDGE mismatch a={a:010d} b={b:010d} exp={exp} got={got}")
    print(f"edge-tests passed ({len(cases)} cases)")

def run_random_tests(model: nn.Module, num_tests: int, seed: int, device: str) -> None:
    rng = random.Random(seed)
    for i in range(num_tests):
        a = rng.randint(0, 10**10 - 1)
        b = rng.randint(0, 10**10 - 1)
        got = generate(model, a, b, device=device)
        exp = expected_output(a, b)
        if got != exp:
            raise AssertionError(f"RANDOM mismatch a={a:010d} b={b:010d} exp={exp} got={got}")
        if (i + 1) % 256 == 0:
            print(f"self-test progress: {i+1}/{num_tests}")
    print(f"self-test passed ({num_tests} random cases)")

'''    
model = Transformer(Config()).to('mps')
print("params:", count_parameters(model))
>> 93

model = Transformer(Config()).to('mps')
run_random_tests(model, num_tests=10_000, seed=0, device='mps')
>> ...
>> self-test passed (10000 random cases)

run_edge_tests(model, 'mps')
>> edge-tests passed (19 cases)
