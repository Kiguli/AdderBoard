"""
TinyAdder: 50 learnable parameters
Hand-coded custom nanoGPT for 10-digit addition.

Architecture: d=4, 2 heads, head_dim=2, 1 layer, ReLU, sinusoidal PE (period=11)

Optimization path (130→72→51→39→50):
  130→72: Structure sharing (tied K/V across heads, rank-1 projections)
  72→51:  Weight tying (shared embed+V dir, merged head scalars)
  51→39:  Coefficient absorption (carry_coeff, wrap_coeff → 0p each)
  39→50:  Removed hardcoded dim indices, added learned routing (+11p)
    - O-proj: hardcoded dims → o_dir[2,4]=8p, removed o_scale[1] (net +7p)
    - MLP/Head: hardcoded dim3 → shared accum_dir[4]=4p (net +4p)

Parameter breakdown:
    digit_values        [10,1] =  10
    embed_dir              [4] =   4
    block.attn.K_proj    [2,4] =   8
    block.attn.q_angles    [2] =   2
    block.attn.o_dir     [2,4] =   8
    block.mlp.carry_dir    [4] =   4
    block.mlp.wrap_dir     [4] =   4
    block.mlp.carry_bias   [2] =   2
    block.mlp.wrap_bias    [2] =   2
    accum_dir              [4] =   4  (tied: MLP output + head input)
    head.lin               [1] =   1
    head.quad              [1] =   1
    TOTAL                      =  50
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 4
N_HEAD = 2
HEAD_DIM = 2
VOCAB = 10
THETA = 2 * math.pi / 11


class TinyAdder(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding: digit_values (param) * embed_dir (param)
        self.digit_values = nn.Parameter(torch.zeros(VOCAB, 1))
        self.embed_dir = nn.Parameter(torch.zeros(D_MODEL))

        # Attention
        self.K_proj = nn.Parameter(torch.zeros(HEAD_DIM, D_MODEL))
        self.q_angles = nn.Parameter(torch.zeros(N_HEAD))
        self.o_dir = nn.Parameter(torch.zeros(N_HEAD, D_MODEL))  # per-head output direction

        # MLP
        self.carry_dir = nn.Parameter(torch.zeros(D_MODEL))
        self.wrap_dir = nn.Parameter(torch.zeros(D_MODEL))
        self.carry_bias = nn.Parameter(torch.zeros(2))
        self.wrap_bias = nn.Parameter(torch.zeros(2))

        # Accumulator direction (tied: MLP output + head input)
        self.accum_dir = nn.Parameter(torch.zeros(D_MODEL))

        # Head
        self.head_lin = nn.Parameter(torch.zeros(1))
        self.head_quad = nn.Parameter(torch.zeros(1))

    def generate_pe(self, seq_len, device):
        pe = torch.zeros(seq_len, D_MODEL, device=device)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        amp = torch.where(pos <= 21, 100.0, 1.0)
        pe[:, 1] = amp * torch.sin(pos * THETA)
        pe[:, 2] = amp * torch.cos(pos * THETA)
        return pe

    def forward(self, idx):
        B, T = idx.size()
        device = idx.device

        # === Embedding + PE ===
        x = self.digit_values[idx] * self.embed_dir + self.generate_pe(T, device)

        # === Attention ===
        # K_proj shared for K and Q extraction (weight tying)
        pe_extracted = x @ self.K_proj.T                    # [B, T, hd]
        k = pe_extracted

        # V: scalar extraction using shared embed_dir
        v_scalar = (x * self.embed_dir).sum(-1)            # [B, T]

        # Q: rotation-parameterized from K_proj-extracted PE channels
        qs = []
        for h in range(N_HEAD):
            a = self.q_angles[h]
            c, s = torch.cos(a), torch.sin(a)
            q0 = -c * pe_extracted[:, :, 0] + s * pe_extracted[:, :, 1]
            q1 =  s * pe_extracted[:, :, 0] + c * pe_extracted[:, :, 1]
            qs.append(torch.stack([q0, q1], dim=-1))

        q = torch.stack(qs, dim=1)                         # [B, nh, T, hd]
        k = k.unsqueeze(1).expand(-1, N_HEAD, -1, -1)      # [B, nh, T, hd]

        # Attention weights (manual, since V is scalar not hd-dimensional)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)            # [B, nh, T, T]

        # Apply attention to scalar V directly (no padding needed)
        v_exp = v_scalar.unsqueeze(1).unsqueeze(-1).expand(-1, N_HEAD, -1, -1)  # [B, nh, T, 1]
        weighted_v = (attn_weights @ v_exp).squeeze(-1)     # [B, nh, T]

        # O-proj: per-head learned output direction
        attn_out = torch.zeros(B, T, D_MODEL, device=device, dtype=x.dtype)
        for h in range(N_HEAD):
            attn_out = attn_out + weighted_v[:, h, :, None] * self.o_dir[h]
        x = x + attn_out

        # === MLP: carry + wrap detection ===
        ci = (x * self.carry_dir).sum(-1)
        carry = F.relu(ci + self.carry_bias[0]) - F.relu(ci + self.carry_bias[1])

        wi = (x * self.wrap_dir).sum(-1)
        wrap = F.relu(wi + self.wrap_bias[1]) - F.relu(wi + self.wrap_bias[0])

        # MLP output: learned accum_dir (tied with head input)
        mlp_out = (carry + wrap).unsqueeze(-1) * self.accum_dir  # [B,T,1]*[4] → [B,T,4]
        x = x + mlp_out

        # === Head: parabolic logits ===
        d = torch.arange(VOCAB, device=device, dtype=x.dtype)
        z = (x * self.accum_dir).sum(-1, keepdim=True)    # tied with MLP output dir
        logits = self.head_lin * d * z + self.head_quad * d * d
        return logits[:, [-1], :]


# === Weight setting ===

def set_weights(model):
    with torch.no_grad():
        # Embedding: digit d → [d, 0, 0, 0]
        model.digit_values[:] = torch.arange(VOCAB).float().unsqueeze(1)
        model.embed_dir[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # K selects PE channels (dims 1,2)
        model.K_proj[:] = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        # Q rotation: head0→current pair (8θ), head1→previous pair (9θ)
        model.q_angles[:] = torch.tensor([8 * THETA, 9 * THETA])

        # O-proj: head0 → dim3 (scale 2.0), head1 → dim1 (scale 2.0)
        model.o_dir[:] = torch.tensor([
            [0.0, 0.0, 0.0, 2.0],  # head0 writes to dim3
            [0.0, 2.0, 0.0, 0.0],  # head1 writes to dim1
        ])

        # MLP carry: ci = -x0 + x1 = diff
        #   carry = relu(diff-0.5) - relu(diff-1.5) = 0 or 1
        model.carry_dir[:] = torch.tensor([-1.0, 1.0, 0.0, 0.0])
        model.carry_bias[:] = torch.tensor([-0.5, -1.5])

        # MLP wrap: wi = -10*x0 + 10*x1 + 1000*x3
        #   wrap = relu(wi-9055) - relu(wi-9045) = 0 or -10
        model.wrap_dir[:] = torch.tensor([-10.0, 10.0, 0.0, 1000.0])
        model.wrap_bias[:] = torch.tensor([-9045.0, -9055.0])

        # Accumulator direction: dim3 (tied for MLP output + head input)
        model.accum_dir[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        # Head: logits[d] = 2d*z - d^2 (parabola peaked at z)
        model.head_lin.fill_(2.0)
        model.head_quad.fill_(-1.0)


# === Submission interface ===

def build_model():
    model = TinyAdder()
    set_weights(model)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    metadata = {
        "name": f"TinyAdder-{params}p",
        "author": "lliu22",
        "params": params,
        "architecture": "1L custom GPT, d=4, 2h, hd=2, ReLU, sin PE(period=11)",
        "tricks": [
            "Factorized embedding: digit_values[10,1] * embed_dir[4]",
            "Tied embed_dir for embedding + V extraction (4p)",
            "Rotation-parameterized Q with K_proj-tied extraction (2p)",
            "Learned per-head O-projection directions (8p, absorbs o_scale)",
            "Tied accum_dir for MLP output + head input (4p)",
            "Absorbed carry/wrap coefficients into dir/bias rescaling",
            "Parabolic head: 2 scalars (lin=2, quad=-1)",
            "Fixed sinusoidal PE (period=11, 0p)",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    seq = [int(c) if c.isdigit() else 0
           for c in f"{a:010d}+{b:010d}="]
    with torch.no_grad():
        for _ in range(11):
            logits = model(torch.tensor([seq]))
            seq.append(logits[0, -1].argmax().item())
    return int("".join(str(d) for d in seq[22:])[::-1])


# === Self-test ===

if __name__ == "__main__":
    model, meta = build_model()
    print(f"Model: {meta['name']}")
    print(f"Parameters: {meta['params']}")
    print()
    for name, p in model.named_parameters():
        print(f"  {name:40s} {str(list(p.shape)):>12s} = {p.numel()}")
    print(f"  {'TOTAL':40s} {'':>12s} = {meta['params']}")
    print()

    tests = [
        (0, 0), (0, 1), (5, 5), (555, 445),
        (99999, 1), (9999999999, 1),
        (9999999999, 9999999999),
        (1234567890, 9876543210),
        (5000000000, 5000000000),
        (1111111111, 8888888889),
    ]
    passed = 0
    for a, b in tests:
        result = add(model, a, b)
        expected = a + b
        ok = result == expected
        passed += ok
        status = "OK" if ok else "FAIL"
        print(f"  {a:>13,d} + {b:>13,d} = {result:>14,d}  [{status}]")
    print(f"\n  {passed}/{len(tests)} passed")
