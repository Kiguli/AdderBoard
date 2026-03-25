"""AdderBoard submission: 39P trained Qwen-like adder.

1-layer decoder-only transformer with circular arc embedding, d=3, 1h/1kv,
hd=4, ff=2, RoPE theta=3, shared norms, tied K=V, tied Q/O readout,
anti-quarter shared QK norm, sparse SwiGLU gate, and 2x repeated shared block.

This is a standalone export of the trained checkpoint:
- checkpoint: 39p_tiekv_krowtie_o3qcol1tail3_gate12zero_qknormquarter_down2unit_repeatmix_sharednorms_s76.pt
- official verify.py: 99.91% (10001 / 10010)
- broad non-2025 panel: 99.96% (9996 / 10000)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 10
NUM_DIGITS = 10
SUM_DIGITS = 11
INPUT_LEN = 24
OUTPUT_LEN = SUM_DIGITS
TOTAL_LEN = INPUT_LEN + OUTPUT_LEN
MAX_ADDEND = 10**NUM_DIGITS - 1
RMS_EPS = 1e-6
ROPE_THETA = 3.0


METADATA = {
    "name": "39P trained Qwen-like",
    "author": "Lokimorty",
    "params": 39,
    "architecture": "1L decoder-only transformer + circular arc embedding, d=3, 1h/1kv, hd=4, ff=2, tieKV, tieQO-tail, shared RMSNorms, shared anti-quarter QK norm, repeat-mix shared block, RoPE theta=3",
    "tricks": [
        "Circular arc embedding (3 params instead of 30)",
        "Tied K=V with row-tied key/value projection",
        "Tied Q/O readout with a 3-parameter scaled tail",
        "Single shared RMSNorm across both block norms and final norm",
        "Single shared anti-quarter QK norm scalar",
        "Sparse 5-parameter SwiGLU gate",
        "Output-scaled tied-up MLP readout with unit carry path",
        "Two repeated applications of one shared transformer block with a learned repeat-mix scalar",
        "Trained checkpoint exported inline for auditability",
    ],
}


def encode(a: int, b: int) -> list[int]:
    left = f"{a:010d}"
    right = f"{b:010d}"
    return [0] + [int(ch) for ch in reversed(left)] + [0, 0] + [int(ch) for ch in reversed(right)] + [0]


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def rope_tables(head_dim: int, max_len: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_len, device=device, dtype=torch.float32)
    phase = torch.outer(pos, freq)
    return phase.cos(), phase.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    length = x.shape[2]
    cos = cos[:length].unsqueeze(0).unsqueeze(0)
    sin = sin[:length].unsqueeze(0).unsqueeze(0)
    even = x[..., ::2]
    odd = x[..., 1::2]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, weight: list[float]):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        return (x.float() * scale).to(x.dtype) * self.weight


class AntiQuarterNorm(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        y = (x.float() * scale).to(x.dtype)
        a = self.weight
        w = torch.stack([a, a * 0.25, a.new_zeros(()), -a])
        return y * w


class ArcEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(-26.32546615600586, dtype=torch.float32))
        self.start = nn.Parameter(torch.tensor(5.6956048011779785, dtype=torch.float32))
        self.stride = nn.Parameter(torch.tensor(6.395115852355957, dtype=torch.float32))

    def table(self) -> torch.Tensor:
        digits = torch.arange(VOCAB_SIZE, device=self.radius.device, dtype=self.radius.dtype)
        angle = self.start + digits * self.stride
        table = torch.zeros(VOCAB_SIZE, 3, device=self.radius.device, dtype=self.radius.dtype)
        table[:, 0] = self.radius * torch.cos(angle)
        table[:, 1] = self.radius * torch.sin(angle)
        return table


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Parameter(
            torch.tensor(
                [
                    [-1.2843137979507446, -0.477486252784729],
                    [4.059610843658447, 0.9027146100997925],
                    [-2.0226404666900635, -14.343998908996582],
                    [2.085467576980591, 0.37745726108551025],
                ],
                dtype=torch.float32,
            )
        )
        self.k_core = nn.Parameter(
            torch.tensor(
                [
                    [4.046281337738037, -0.8057977557182312],
                    [-1.1322791576385498, 0.8885217905044556],
                    [-0.20743858814239502, -2.1566951274871826],
                ],
                dtype=torch.float32,
            )
        )
        self.k_alpha_last = nn.Parameter(torch.tensor(1.1219857931137085, dtype=torch.float32))
        self.o_tail_scale = nn.Parameter(torch.tensor([0.21241770684719086, 0.1192014142870903, 0.14575722813606262], dtype=torch.float32))
        self.qk_norm = AntiQuarterNorm(4.671701431274414)

    def k_weight(self) -> torch.Tensor:
        return torch.cat([self.k_core, self.k_core[:1] * self.k_alpha_last], dim=0)

    def o_row3(self) -> torch.Tensor:
        tail = self.q_proj[1:, 1] * self.o_tail_scale
        return torch.cat([tail.new_zeros(1), tail], dim=0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q_raw = F.linear(x[..., :2], self.q_proj)
        k_raw = F.linear(x[..., :2], self.k_weight())
        v_raw = k_raw

        q = q_raw.view(batch, seq, 1, 4).transpose(1, 2)
        k = k_raw.view(batch, seq, 1, 4).transpose(1, 2)
        v = v_raw.view(batch, seq, 1, 4).transpose(1, 2)

        q = self.qk_norm(q)
        k = self.qk_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(4.0))
        scores = scores + mask[:seq, :seq]
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(batch, seq, 4)

        first = F.linear(out, self.q_proj.t())
        third = F.linear(out, self.o_row3().unsqueeze(0))
        return torch.cat([first, third], dim=-1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Parameter(
            torch.tensor(
                [
                    [-4.4253950119018555, 4.820530891418457, -9.961663246154785],
                    [0.35525211691856384, 0.39638543128967285, -0.22129130363464355],
                ],
                dtype=torch.float32,
            )
        )
        self.down_scale = nn.Parameter(torch.tensor([-0.7954375147819519, -1.7600338459014893], dtype=torch.float32))
        self.gate_row0 = nn.Parameter(torch.tensor([-0.13672497868537903, 0.07716552168130875, -0.7461947798728943], dtype=torch.float32))
        self.gate_row1_head = nn.Parameter(torch.tensor([0.6748242974281311, 0.9906638860702515], dtype=torch.float32))

    def gate_weight(self) -> torch.Tensor:
        row1 = torch.cat([self.gate_row1_head, self.gate_row1_head.new_zeros(1)])
        return torch.stack([self.gate_row0, row1], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.linear(x, self.up_proj)
        gate = F.linear(x, self.gate_weight())
        hidden = F.silu(gate) * up
        scale = torch.cat([self.down_scale, self.down_scale.new_zeros(1)])
        base = F.linear(hidden, self.up_proj.t() * scale.unsqueeze(-1))
        correction = hidden[..., -1:] * self.gate_weight()[-1]
        return base + correction


class Block(nn.Module):
    def __init__(self, norm: RMSNorm):
        super().__init__()
        self.ln1 = norm
        self.ln2 = norm
        self.attn = Attention()
        self.mlp = MLP()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ArcEmbedding()
        self.norm = RMSNorm([-25.52376365661621, -8.28279972076416, 3.0585718154907227])
        self.block = Block(self.norm)
        self.repeat_gain = nn.Parameter(torch.tensor(-1.2396835086401552e-05, dtype=torch.float32))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        table = self.embed.table()
        x = table[tokens]
        mask = causal_mask(tokens.shape[1], x.device)
        cos, sin = rope_tables(4, tokens.shape[1], ROPE_THETA, x.device)
        first = self.block(x, mask, cos, sin)
        second = self.block(first, mask, cos, sin)
        x = first + (second - first) * self.repeat_gain
        return F.linear(self.norm(x), table)


def build_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyAdder().to(device)
    model.eval()
    return model, METADATA


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    x = torch.tensor([encode(a, b)], dtype=torch.long, device=device)
    digits: list[int] = []
    with torch.no_grad():
        for _ in range(OUTPUT_LEN):
            logits = model(x)
            nxt = logits[0, -1].argmax().item()
            digits.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return sum(d * (10**i) for i, d in enumerate(digits))
