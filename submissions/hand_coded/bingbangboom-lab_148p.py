import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

MODEL_LAYERS = 2
MODEL_DIM = 5
ATTENTION_HEADS = 2
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 3
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1

LM_HEAD_WEIGHT = np.array([
    [5.5779090e00, 3.1322198e00, -4.0438358e02, 6.2589108e01, 9.9358273e-01],
    [5.0814748e00, 2.4687927e00, -3.1444955e02, 4.8671352e01, 7.7272820e-01],
    [3.6916721e00, 1.7657869e00, -2.2455742e02, 3.4757641e01, 5.5075526e-01],
    [1.4084998e00, 1.0232025e00, -1.3470717e02, 2.0847967e01, 3.2766387e-01],
    [-1.7680415e00, 2.4103954e-01, -4.4898785e01, 6.9423370e00, 1.0345399e-01],
    [-5.8379521e00, -5.8070201e-01, 4.4867714e01, -6.9592528e00, -1.2187435e-01],
    [-1.0801232e01, -1.4420221e00, 1.3459233e02, -2.0856800e01, -3.4832114e-01],
    [-1.6657881e01, -2.3429208e00, 2.2427509e02, -3.4750309e01, -5.7588643e-01],
    [-2.3407900e01, -3.2833982e00, 3.1391595e02, -4.8639774e01, -8.0457014e-01],
    [-3.1051287e01, -4.2634540e00, 4.0351492e02, -6.2525200e01, -1.0343723e00],
], dtype=np.float32)


def _factorize_lowrank(weight: np.ndarray, rank: int):
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    a = u[:, :rank] * s[:rank]
    b = vt[:rank, :]
    return a.astype(np.float32), b.astype(np.float32)


class Rank1Linear(nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(out_features))
        self.v = nn.Parameter(torch.zeros(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = (x * self.v).sum(dim=-1, keepdim=True)
        return s * self.u


class Rank1LinearSharedV(nn.Module):
    def __init__(self, out_features: int, shared_v: nn.Parameter):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(out_features))
        self.shared_v = shared_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = (x * self.shared_v).sum(dim=-1, keepdim=True)
        return s * self.u


class Rank1LinearSharedU(nn.Module):
    def __init__(self, in_features: int, shared_u: nn.Parameter):
        super().__init__()
        self.shared_u = shared_u
        self.v = nn.Parameter(torch.zeros(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = (x * self.v).sum(dim=-1, keepdim=True)
        return s * self.shared_u


class Rank1LinearSharedUV(nn.Module):
    def __init__(self, shared_u: nn.Parameter, shared_v: nn.Parameter):
        super().__init__()
        self.shared_u = shared_u
        self.shared_v = shared_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = (x * self.shared_v).sum(dim=-1, keepdim=True)
        return s * self.shared_u


class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(vocab_size, 2))
        self.B = nn.Parameter(torch.zeros(2, dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.A[ids] @ self.B


class LowRankLMHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(vocab_size, rank))
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.B.T) @ self.A.T


class SparseGateProj0(nn.Module):
    def __init__(self):
        super().__init__()
        self.W23 = nn.Parameter(torch.zeros(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x[..., :3]
        y2 = x3 @ self.W23.T
        pad = torch.zeros((*y2.shape[:-1], 1), dtype=y2.dtype, device=y2.device)
        return torch.cat([y2, pad], dim=-1)


class RMSNormNoWeight(nn.Module):
    def __init__(self, eps: float = 1e-6, scale: float = 1.0):
        super().__init__()
        self.eps = eps
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.scale * x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CompressedAttention(nn.Module):
    def __init__(
        self,
        shared_qk_v: nn.Parameter,
        shared_v_v: nn.Parameter,
        shared_o_v: nn.Parameter,
        shared_vproj_u: nn.Parameter,
    ):
        super().__init__()
        self.num_heads = ATTENTION_HEADS
        self.num_kv_heads = KEY_VALUE_HEADS
        self.head_dim = HEAD_DIM

        self.q_proj = Rank1LinearSharedV(self.num_heads * self.head_dim, shared_qk_v)
        self.k_proj = Rank1LinearSharedV(self.num_kv_heads * self.head_dim, shared_qk_v)
        self.v_proj = Rank1LinearSharedUV(shared_vproj_u, shared_v_v)
        self.o_proj = Rank1LinearSharedV(MODEL_DIM, shared_o_v)

        self.q_norm = RMSNormNoWeight(scale=16.0)
        self.k_norm = RMSNormNoWeight(scale=16.0)

    def forward(self, x, freqs_cis, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q.view(B, S, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(B, S, self.num_kv_heads, self.head_dim))
        v = v.view(B, S, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(output)


class CompressedLayer(nn.Module):
    def __init__(
        self,
        layer_idx,
        shared_qk_v: nn.Parameter,
        shared_v_v: nn.Parameter,
        shared_o_v: nn.Parameter,
        shared_vproj_u: nn.Parameter,
        shared_up_u: nn.Parameter,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNormNoWeight(scale=1.0)
        self.self_attn = CompressedAttention(shared_qk_v, shared_v_v, shared_o_v, shared_vproj_u)
        self.post_attention_layernorm = RMSNormNoWeight(scale=1.0)

        if layer_idx == 0:
            self.gate_proj = SparseGateProj0()
        else:
            self.gate_proj = nn.Linear(MODEL_DIM, INTERMEDIATE_SIZE, bias=False)

        self.up_proj = Rank1LinearSharedU(MODEL_DIM, shared_up_u)
        self.down_proj = Rank1Linear(MODEL_DIM, INTERMEDIATE_SIZE)

    def mlp_forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x, freqs_cis, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, mask)
        out = h + self.mlp_forward(self.post_attention_layernorm(h))
        return out


class Qwen3_148(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = FactorizedEmbedding(VOCAB_SIZE, MODEL_DIM)

        # Shared attention input vectors across both layers.
        self.shared_qk_v = nn.Parameter(torch.zeros(MODEL_DIM))
        self.shared_v_v = nn.Parameter(torch.zeros(MODEL_DIM))
        self.shared_o_v = nn.Parameter(torch.zeros(ATTENTION_HEADS * HEAD_DIM))

        # Fixed shared vectors (non-trainable params) to reduce trainable parameter count.
        self.shared_vproj_u = nn.Parameter(torch.tensor([1.0, 0.0]), requires_grad=False)
        self.shared_up_u = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=False)

        self.layers = nn.ModuleList([
            CompressedLayer(
                i,
                self.shared_qk_v,
                self.shared_v_v,
                self.shared_o_v,
                self.shared_vproj_u,
                self.shared_up_u,
            )
            for i in range(MODEL_LAYERS)
        ])
        self.norm = RMSNormNoWeight(scale=1.0)
        self.lm_head = LowRankLMHead(VOCAB_SIZE, MODEL_DIM, rank=2)

        self.freqs_cis = precompute_freqs_cis(HEAD_DIM, 2048)

    def forward(self, x):
        B, S = x.shape
        freqs_cis = self.freqs_cis[:S].to(x.device)

        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)

        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        return self.lm_head(h)


def set_weights(model):
    with torch.no_grad():
        # lm_head (lowrank_head2)
        a, b = _factorize_lowrank(LM_HEAD_WEIGHT, rank=2)
        model.lm_head.A.copy_(torch.from_numpy(a))
        model.lm_head.B.copy_(torch.from_numpy(b))

        # embed_tokens (embed2)
        model.embed_tokens.A.copy_(torch.tensor([[1.0, float(i)] for i in range(VOCAB_SIZE)]))
        model.embed_tokens.B.copy_(torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]))

        # shared vectors
        model.shared_qk_v.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
        model.shared_v_v.copy_(torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
        model.shared_o_v.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))
        model.shared_vproj_u.copy_(torch.tensor([1.0, 0.0]))
        model.shared_up_u.copy_(torch.tensor([1.0, 1.0, 1.0]))

        # layer 0
        l0 = model.layers[0]
        l0.self_attn.q_proj.u.copy_(torch.tensor([0.98502123, 0.17243294, 0.96630472, -0.25740093]))
        l0.self_attn.k_proj.u.copy_(torch.tensor([-0.31672141, -0.94851863]))
        l0.self_attn.o_proj.u.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))

        # layer 0 mlp (sparse_gate0 + rank1)
        l0.gate_proj.W23.copy_(torch.tensor([
            [-3.3532020e-01, -1.3412670e03, 6.0353305e04],
            [-1.3743691e01, -1.3418693e03, 6.0353277e04],
        ]))
        l0.up_proj.v.copy_(torch.tensor([1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0, 0.0]))
        l0.down_proj.u.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
        l0.down_proj.v.copy_(torch.tensor([1.0, -1.0, 0.0]))

        # layer 1
        l1 = model.layers[1]
        l1.self_attn.q_proj.u.copy_(torch.tensor([-0.25507239, 0.96692199, 0.17478994, 0.98460573]))
        l1.self_attn.k_proj.u.copy_(torch.tensor([0.32702553, -0.94501549]))
        l1.self_attn.o_proj.u.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))

        # layer 1 mlp (standard gate + rank1)
        l1.gate_proj.weight.copy_(torch.tensor([
            [-4.3951669e-01, 5.6323919e00, 4.9838150e-01, 1.3435575e03, 6.0357680e04],
            [-1.2112466e02, 3.2923722e-01, -5.0313854e00, 1.3449166e03, 6.0357438e04],
            [-1.3453412e02, -2.6000220e-01, -5.6458039e00, 1.3450677e03, 6.0357410e04],
        ]))
        l1.up_proj.v.copy_(torch.tensor([1.4899401e-02, 6.5471046e-04, 6.8268733e-04, -1.6779384e-04, 2.9817384e-05]))
        l1.down_proj.u.copy_(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
        l1.down_proj.v.copy_(torch.tensor([1.0, -10.0, 10.0]))


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


def _expected_output(a: int, b: int) -> str:
    _validate_addends(a, b)
    return str(a + b)[::-1].ljust(OUTPUT_DIGITS, "0")


def _generate_output_batch(model: Qwen3_148, addends: list[tuple[int, int]], device) -> list[str]:
    internal = [_encode_addends_internal(a, b) for a, b in addends]

    with torch.no_grad():
        for _ in range(OUTPUT_DIGITS):
            x = torch.tensor(internal, dtype=torch.long, device=device)
            logits = model(x)
            next_digits = logits[:, -1, :].argmax(dim=-1).cpu().numpy()
            for seq, next_digit in zip(internal, next_digits):
                seq.append(int(next_digit))

    return ["".join(str(d) for d in seq[-OUTPUT_DIGITS:]) for seq in internal]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def run_self_test_batched(model: Qwen3_148, num_tests: int, batch_size: int, device) -> None:
    rng = random.Random(123)
    tested = 0
    model.eval()
    while tested < num_tests:
        cur_batch_size = min(batch_size, num_tests - tested)
        addends = []
        expected = []
        for _ in range(cur_batch_size):
            a = rng.randint(0, 10**10 - 1)
            b = rng.randint(0, 10**10 - 1)
            addends.append((a, b))
            expected.append(_expected_output(a, b))
        actual = _generate_output_batch(model, addends, device)
        for (a, b), exp, act in zip(addends, expected, actual):
            if act != exp:
                raise AssertionError(f"Mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}")
        tested += cur_batch_size
        print(f"self-test progress: {tested}/{num_tests}")
    print(f"self-test passed ({num_tests} random cases, batch size {batch_size})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tests", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Qwen3_148().to(device)
    set_weights(model)
    trainable_params = count_parameters(model)
    frozen_params = count_frozen_parameters(model)
    total_params = trainable_params + frozen_params
    print(f"parameter count: {trainable_params} + {frozen_params} frozen = {total_params} total")

    run_self_test_batched(model, num_tests=args.num_tests, batch_size=args.batch_size, device=device)


if __name__ == "__main__":
    main()

    
    
# python "148_torch.py" --num-tests 10000 --batch-size 1024
# Using device: cuda
# parameter count: 143 + 5 frozen = 148 total
# self-test progress: 1024/10000
# self-test progress: 2048/10000
# self-test progress: 3072/10000
# self-test progress: 4096/10000
# self-test progress: 5120/10000
# self-test progress: 6144/10000
# self-test progress: 7168/10000
# self-test progress: 8192/10000
# self-test progress: 9216/10000
# self-test progress: 10000/10000
# self-test passed (10000 random cases, batch size 1024)

#written with help from GPT-5.3-Codex