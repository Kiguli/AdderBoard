import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def rope_2d(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    c = torch.cos(pos).view(1, 1, -1, 1)
    s = torch.sin(pos).view(1, 1, -1, 1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)


def encode_prompt(a: int, b: int) -> list[int]:
    # [0] + rev(a10) + [0] + [0] + rev(b10) + [0]  (len=24)
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

    # init values for counted scalars
    embed_const: float = 1000.0   # for mlp
    decode_eps: float = 5e-4      # eps
    qk_scale: float = 256.0       # qk

    # constants that dont count
    causal_mask_neg: float = -1e4 # neg inf for attn calc
    rope_offsets: tuple[float, float, float, float, float] = (0.0, 23.0, 11.0, 22.0, 10.0) # rope constant


class Transformer(nn.Module):
    """
    24 params w/ default config
    """

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        V = config.vocab_size

        # NOTE: counted params 
        self.digit_values = nn.Parameter(torch.arange(V, dtype=torch.float32), requires_grad=False)  # (10,)
        self.embed_const = nn.Parameter(torch.tensor(config.embed_const, dtype=torch.float32), requires_grad=False)  # (1,)
        self.decode_eps = nn.Parameter(torch.tensor(config.decode_eps, dtype=torch.float32), requires_grad=False)    # (1,)
        self.qk_scale = nn.Parameter(torch.tensor(config.qk_scale, dtype=torch.float32), requires_grad=False)        # (1,)
        # sparse O-proj 
        self.o_w = nn.Parameter(
            torch.tensor([+1.0, -1.0, -1.0, -1.0, +1.0, +1.0], dtype=torch.float32), # (6,)
            requires_grad=False,
        )
        # tied MLP 
        self.w1_a = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32), requires_grad=False) # (1,)
        self.w1_b = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32), requires_grad=False) # (1,)
        self.w1_c = nn.Parameter(torch.tensor(20.0, dtype=torch.float32), requires_grad=False) # (1,)
        self.w2_s1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False) # (1,)
        self.w2_s10 = nn.Parameter(torch.tensor(10.0, dtype=torch.float32), requires_grad=False) # (1,)

        # NOTE: 10 + 1 + 1 + 1 + 6 + 1 + 1 + 1 + 1 + 1 = 24

        # NOTE: these dont count (attn mask and pos embd constant)
        self.register_buffer("rope_offsets_buf", torch.tensor(config.rope_offsets, dtype=torch.float32), persistent=False)
        self.register_buffer("causal_neg_buf", torch.tensor([config.causal_mask_neg], dtype=torch.float32), persistent=False)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # digit d -> [ C - (eps/2)*d^2 , d ]
        d = self.digit_values[input_ids]  # (B,T)
        quad = self.decode_eps / 2.0
        return torch.stack([self.embed_const - quad * (d * d), d], dim=-1)  # (B,T,2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        (B,T) -> (B,T,10)
        tensor in, logits-out
        """
        cfg = self.config
        x = self._embed(input_ids)  # (B,T,2)
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device, dtype=x.dtype)

        x0 = x[..., 0]  # (B,T)
        x1 = x[..., 1]  # (B,T)
        H = cfg.num_attention_heads

        # Q: x0 times per-head direction, then RoPE
        offs = self.rope_offsets_buf.to(device=x.device, dtype=x.dtype)  # (H,)
        q_base = torch.stack([torch.cos(offs), -torch.sin(offs)], dim=-1) * self.qk_scale.to(dtype=x.dtype)  # (H,2)
        q = x0.unsqueeze(1).unsqueeze(-1) * q_base.view(1, H, 1, 2)  # (B,H,T,2)

        # K: [x0*qk, 0], shared across heads, then RoPE
        k = torch.stack([x0 * self.qk_scale.to(dtype=x.dtype), torch.zeros_like(x0)], dim=-1)
        k = k.unsqueeze(1).expand(-1, H, -1, -1)  # (B,H,T,2)

        # V: [x1, 0], shared across heads (no RoPE on V)
        v = torch.stack([x1, torch.zeros_like(x1)], dim=-1)
        v = v.unsqueeze(1).expand(-1, H, -1, -1)  # (B,H,T,2)

        q = rope_2d(q, pos)
        k = rope_2d(k, pos)

        # attention scores
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(cfg.head_dim)

        # mask
        upper = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype), diagonal=1)
        scores = scores + upper.view(1, 1, T, T) * self.causal_neg_buf.to(device=x.device, dtype=x.dtype)

        # attn weight
        w = F.softmax(scores, dim=-1)
        att = torch.einsum("bhts,bhsd->bhtd", w, v)  # (B,H,T,2)
        att = att.permute(0, 2, 1, 3).contiguous().view(B, T, H * cfg.head_dim)  # (B,T,10)

        # O proj
        upd0 = self.o_w[0] * att[..., 0] + self.o_w[1] * att[..., 2] + self.o_w[2] * att[..., 4]
        upd1 = self.o_w[3] * att[..., 0] + self.o_w[4] * att[..., 6] + self.o_w[5] * att[..., 8]
        # resid conn
        x = x + torch.stack([upd0, upd1], dim=-1) 

        # W1 rows: [-1,0],[-1,0],[-2,20],[-2,20] (tied scalars)
        z0 = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        W1 = torch.stack(
            [
                torch.stack([self.w1_a, z0]),
                torch.stack([self.w1_a, z0]),
                torch.stack([self.w1_b, self.w1_c]),
                torch.stack([self.w1_b, self.w1_c]),
            ],
            dim=0,
        ).to(device=x.device, dtype=x.dtype)  # (4,2)

        C = self.embed_const.to(device=x.device, dtype=x.dtype)
        # 4 more that I forgot to count # (4,1) for b1
        b1 = torch.stack([C - 8.0, C - 9.0, 2 * C - 188.0, 2 * C - 189.0]).to(device=x.device, dtype=x.dtype)

        h = F.relu(x @ W1.t() + b1)  # (B,T,4)

        # W2 only writes into dim1: [+1,-1,-10,+10]
        W2 = torch.zeros(2, cfg.intermediate_size, device=x.device, dtype=x.dtype)
        W2[1, 0] = self.w2_s1
        W2[1, 1] = -self.w2_s1
        W2[1, 2] = -self.w2_s10
        W2[1, 3] = self.w2_s10

        # resid conn
        x = x + h @ W2.t()

        # tied decode 
        out_scale = torch.stack([1.0 / C, self.decode_eps.to(device=x.device, dtype=x.dtype)]).to(
            device=x.device, dtype=x.dtype
        )
        y = x * out_scale  # (B,T,2)

        d = self.digit_values.to(device=x.device, dtype=x.dtype)
        E0 = C - (self.decode_eps.to(device=x.device, dtype=x.dtype) / 2.0) * (d * d)
        E1 = d
        embed_tokens = torch.stack([E0, E1], dim=-1)  # (10,2)

        return y @ embed_tokens.t()  # (B,T,10)
    
    
def build_model():
    model = Transformer(Config())
    metadata = {
        "name": "jacob24param",
        "author": "jacobli99",
        "params": 28, # if you call the count_parameters on my model it returns 24 cuz I forgot to add the other 4.
        "architecture": "1L, 5H, ROPE, tied MLP + tied decode",
        "tricks": [
            "replace (10,2) embd matrix to on-the-fly tied unembedding",
            "sparse o_proj replaces dense o_proj",
            "Tied MLP",
            "matrix broadcast"
        ],
    }
    model.eval()
    return model, metadata

@torch.no_grad()
def add(model, a: int, b: int) -> int:
    x = torch.tensor([encode_prompt(a, b)], dtype=torch.long)
    for _ in range(11):
        nxt = model(x)[:, -1].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)

    digits = x[0, -11:].tolist()
    return int("".join(str(d) for d in reversed(digits)))