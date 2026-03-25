"""
 Dynamic NanoGPT Adder: 130 params + 0 buffers

    transformer.wte.A                           [10, 1] =  10
    transformer.wte.B                            [1, 4] =   4
    transformer.h.0.attn.c_attn.weight          [12, 4] =  48
    transformer.h.0.attn.c_proj.weight           [4, 4] =  16
    transformer.h.0.mlp.c_fc.weight              [4, 4] =  16
    transformer.h.0.mlp.c_fc.bias                   [4] =   4
    transformer.h.0.mlp.c_proj.u                 [4, 1] =   4
    transformer.h.0.mlp.c_proj.v                 [1, 4] =   4
    lm_head.u                                   [10, 1] =  10
    lm_head.v                                    [1, 4] =   4
    lm_head.bias                                   [10] =  10
    TOTAL                                               = 130

============================================================
1. Original hand-picked tests
============================================================
              5 +             5 =            10  ✅
            555 +           445 =         1,000  ✅
         99,999 +             1 =       100,000  ✅
         19,492 +        23,919 =        43,411  ✅
  9,999,999,999 +             1 = 10,000,000,000  ✅
  1,234,567,890 +   987,654,321 = 2,222,222,211  ✅
              0 +             0 =             0  ✅
  1,111,111,111 + 8,888,888,889 = 10,000,000,000  ✅
  8/8 passed

============================================================
2. Boundary / edge cases
============================================================
              0 +             0 =             0  ✅
              0 +             1 =             1  ✅
              1 +             0 =             1  ✅
              0 +             9 =             9  ✅
              9 +             0 =             9  ✅
  9,999,999,999 +             0 = 9,999,999,999  ✅
              0 + 9,999,999,999 = 9,999,999,999  ✅
  9,999,999,999 + 9,999,999,999 = 19,999,999,998  ✅
  5,000,000,000 + 5,000,000,000 = 10,000,000,000  ✅
              1 + 9,999,999,999 = 10,000,000,000  ✅
  9,999,999,999 +             1 = 10,000,000,000  ✅
  11/11 passed

============================================================
3. Carry propagation chains (10^k - 1) + 1
============================================================
              9 +             1 =            10  ✅
             99 +             1 =           100  ✅
            999 +             1 =         1,000  ✅
          9,999 +             1 =        10,000  ✅
         99,999 +             1 =       100,000  ✅
        999,999 +             1 =     1,000,000  ✅
      9,999,999 +             1 =    10,000,000  ✅
     99,999,999 +             1 =   100,000,000  ✅
    999,999,999 +             1 = 1,000,000,000  ✅
  9,999,999,999 +             1 = 10,000,000,000  ✅
  10/10 passed

============================================================
4. Long carry chains (all 9s + small)
============================================================
  9,999,999,999 +             2 = 10,000,000,001  ✅
  9,999,999,999 +             9 = 10,000,000,008  ✅
  9,999,999,999 +            10 = 10,000,000,009  ✅
  9,999,999,999 +            99 = 10,000,000,098  ✅
  9,999,999,999 +           100 = 10,000,000,099  ✅
  9,999,999,990 +            10 = 10,000,000,000  ✅
  9,999,999,900 +           100 = 10,000,000,000  ✅
  9,999,999,000 +         1,000 = 10,000,000,000  ✅
  9,999,990,000 +        10,000 = 10,000,000,000  ✅
  9,999,900,000 +       100,000 = 10,000,000,000  ✅
  9,990,000,000 +    10,000,000 = 10,000,000,000  ✅
  9,900,000,000 +   100,000,000 = 10,000,000,000  ✅
  9,000,000,000 + 1,000,000,000 = 10,000,000,000  ✅
  13/13 passed

============================================================
5. No carry cases
============================================================
  1,111,111,111 + 2,222,222,222 = 3,333,333,333  ✅
  1,234,567,890 +             0 = 1,234,567,890  ✅
  1,000,000,000 + 2,000,000,000 = 3,000,000,000  ✅
          1,234 +         4,321 =         5,555  ✅
  1,010,101,010 + 2,020,202,020 = 3,030,303,030  ✅
  4,040,404,040 + 5,050,505,050 = 9,090,909,090  ✅
  6/6 passed

============================================================
6. Alternating carry patterns
============================================================
  5,050,505,050 + 5,050,505,050 = 10,101,010,100  ✅
  9,090,909,090 + 1,010,101,010 = 10,101,010,100  ✅
  1,919,191,919 + 8,181,818,181 = 10,101,010,100  ✅
  5,959,595,959 + 4,141,414,141 = 10,101,010,100  ✅
  8,282,828,282 + 2,828,282,828 = 11,111,111,110  ✅
  5/5 passed

============================================================
7. Powers of 10
============================================================
  64/64 passed

============================================================
8. Palindromic numbers
============================================================
  1,234,554,321 + 1,234,554,321 = 2,469,108,642  ✅
  9,876,556,789 +   123,443,210 = 9,999,999,999  ✅
  1,111,111,111 + 9,999,999,999 = 11,111,111,110  ✅
  5,678,876,543 + 4,321,123,456 = 9,999,999,999  ✅
  4/4 passed

============================================================
9. Commutativity: a+b == b+a
============================================================
  Commutative: 50/50, Correct: 50/50

============================================================
10. Random small numbers (0-99)
============================================================
  225/225 passed

============================================================
11. Random medium (1000-999999)
============================================================
  500/500 passed

============================================================
12. Large random stress test (10,000 cases)
============================================================
  Progress: 10,000/10,000 — 0 failures so far

  10000/10000 passed (0 failures)

============================================================
13. Single-digit-position carry tests
============================================================
  20/20 passed

============================================================
14. Previously-failing cases
============================================================
  9,999,999,999 +             0 = 9,999,999,999  ✅
              0 + 9,999,999,999 = 9,999,999,999  ✅
  4,040,404,040 + 5,050,505,050 = 9,090,909,090  ✅
  9,876,556,789 +   123,443,210 = 9,999,999,999  ✅
  5,678,876,543 + 4,321,123,456 = 9,999,999,999  ✅
        863,512 +       986,307 =     1,849,819  ✅
        642,029 +       250,362 =       892,391  ✅
        210,853 +       732,099 =       942,952  ✅
         93,028 +       100,535 =       193,563  ✅
        112,648 +       286,132 =       398,780  ✅
  10/10 passed

============================================================
GRAND TOTAL: 10926/10926 passed
============================================================
"""

"""130-param nanoGPT that adds any two 10-digit numbers.
Combines hand-coded weights with Rank-1, Factorized Embedding tricks, and dynamic PE.

FIX: Wrap-detection neuron biases changed from -9005/-9015 to -9045/-9055.

Mechanism at output position z_i:
  dim0 = z_{i-1} (previous output digit, from token embedding)
  dim1 = a_{i-1}+b_{i-1} (previous pair sum, from attention head 1 via o_proj)
  dim3 = a_i+b_i (current pair sum, from attention head 0 via o_proj)

  diff = dim1 - dim0 = (a_{i-1}+b_{i-1}) - z_{i-1} in {-1, 0, 9, 10}
  carry_in = I[diff >= 9]

  Wrap signal = 1000*(a_i+b_i) + 10*diff
    Max no-wrap: 9000  (sum=9, diff=0)
    Min wrap:    9090  (sum=9, diff=9)
    Gap = 90

  Original -9005: only 5 margin below 9000 -> float noise causes false carries.
  Fixed   -9045: 45 margin below, 35 above. Robust.
"""
import math
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# === Trick 1: Factorized Embeddings ===
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, rank=1):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(vocab_size, rank))
        self.B = nn.Parameter(torch.zeros(rank, emb_dim))
    
    def forward(self, x):
        return self.A[x] @ self.B

# === Trick 2: Rank-1 Linear Layers ===
class Rank1Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(out_features, 1))
        self.v = nn.Parameter(torch.zeros(1, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        out = (x @ self.v.T) @ self.u.T
        if self.bias is not None:
            out += self.bias
        return out

# === NanoGPT Core ===
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head, self.n_embd, self.dropout = config.n_head, config.n_embd, config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.mlp_hidden or 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, h, bias=True)
        self.gelu = nn.GELU()
        self.c_proj = Rank1Linear(h, config.n_embd, bias=config.bias) 

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.Identity()
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.Identity()
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

@dataclass
class GPTConfig:
    block_size: int = 35
    vocab_size: int = 10
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 4
    dropout: float = 0.0
    bias: bool = False
    mlp_hidden: int = 4

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = FactorizedEmbedding(config.vocab_size, config.n_embd, rank=1), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = Rank1Linear(config.n_embd, config.vocab_size, bias=True) 

    def generate_pe(self, seq_len, device):
        pe = torch.zeros(seq_len, self.config.n_embd, device=device)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        th = 2 * math.pi / 11
        amp = torch.where(positions <= 21, 100.0, 1.0)
        pe[:, 1] = amp * torch.sin(positions * th)
        pe[:, 2] = amp * torch.cos(positions * th)
        return pe

    def forward(self, idx, targets=None):
        seq_len = idx.size(1)
        x = self.transformer.wte(idx) + self.generate_pe(seq_len, idx.device)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(x[:, [-1], :]), None

# === Weight injection ===
def build_adder():
    config = GPTConfig()
    model = GPT(config)
    th, S, n = 2*math.pi/11, 100.0, 4

    with torch.no_grad():
        model.transformer.wte.A.zero_()
        model.transformer.wte.B.zero_()
        for v in range(10): model.transformer.wte.A[v, 0] = float(v)
        model.transformer.wte.B[0, 0] = 1.0

        model.transformer.h[0].mlp.gelu = nn.ReLU()

        w = torch.zeros(3*n, n)
        w[0,1] = -math.cos(8*th)*S; w[0,2] = math.sin(8*th)*S
        w[1,1] = math.sin(8*th)*S;  w[1,2] = math.cos(8*th)*S
        w[2,1] = -math.cos(9*th)*S; w[2,2] = math.sin(9*th)*S
        w[3,1] = math.sin(9*th)*S;  w[3,2] = math.cos(9*th)*S
        w[4,1] = w[6,1] = 1.0; w[5,2] = w[7,2] = 1.0
        w[8,0] = w[10,0] = 1.0
        model.transformer.h[0].attn.c_attn.weight.copy_(w)

        w = torch.zeros(n, n)
        w[3,0] = 2.0; w[1,2] = 2.0
        model.transformer.h[0].attn.c_proj.weight.copy_(w)

        fw, fb = torch.zeros(4, n), torch.zeros(4)
        fw[0,1]= 100; fw[0,0]=-100; fb[0]= -50
        fw[1,1]= 100; fw[1,0]=-100; fb[1]=-150
        fw[2,3]=1000; fw[2,1]=  10; fw[2,0]=-10; fb[2]=-9045   # FIX: was -9005
        fw[3,3]=1000; fw[3,1]=  10; fw[3,0]=-10; fb[3]=-9055   # FIX: was -9015
        model.transformer.h[0].mlp.c_fc.weight.copy_(fw)
        model.transformer.h[0].mlp.c_fc.bias.copy_(fb)

        model.transformer.h[0].mlp.c_proj.u.zero_()
        model.transformer.h[0].mlp.c_proj.v.zero_()
        model.transformer.h[0].mlp.c_proj.u[3, 0] = 1.0
        model.transformer.h[0].mlp.c_proj.v[0, :] = torch.tensor([0.01, -0.01, -1.0, 1.0])

        model.lm_head.u.zero_()
        model.lm_head.v.zero_()
        model.lm_head.bias.zero_()
        for v in range(10):
            model.lm_head.u[v, 0] = 2.0 * v
            model.lm_head.bias[v] = -float(v * v)
        model.lm_head.v[0, 3] = 1.0

    return model

# === Inference ===
def test(model, a, b, verbose=True):
    tok = {'+': 0, '=': 0}
    seq = [int(c) if c.isdigit() else tok[c] for c in f"{a:010d}+{b:010d}="]
    model.eval()
    with torch.no_grad():
        for _ in range(11):
            logits, _ = model(torch.tensor([seq]))
            seq.append(logits[0,-1].argmax().item())
    result = int("".join(str(t) for t in seq[22:])[::-1])
    ok = result == a + b
    if verbose:
        print(f"  {a:>13,d} + {b:>13,d} = {result:>13,d}  {'✅' if ok else '❌'}")
    return ok, result

def test_batch(model, pairs):
    tok = {'+': 0, '=': 0}
    seqs = []
    for a, b in pairs:
        seqs.append([int(c) if c.isdigit() else tok[c] for c in f"{a:010d}+{b:010d}="])
    model.eval()
    with torch.no_grad():
        for _ in range(11):
            logits, _ = model(torch.tensor(seqs))
            next_digits = logits[:, -1].argmax(dim=-1).tolist()
            for seq, d in zip(seqs, next_digits):
                seq.append(d)
    results = []
    for seq in seqs:
        results.append(int("".join(str(t) for t in seq[22:])[::-1]))
    return results

# === Comprehensive tests ===
def run_tests(model):
    params = sum(p.numel() for p in model.parameters())
    buf = sum(b.numel() for b in model.buffers())
    print(f"\n  Dynamic NanoGPT Adder: {params} params + {buf} buffers\n")
    for name, p in model.named_parameters():
        print(f"    {name:40s} {str(list(p.shape)):>10s} = {p.numel():>3d}")
    print(f"    {'TOTAL':40s} {'':>10s} = {params:>3d}\n")

    all_passed = 0
    all_total = 0

    # --- 1. Original hand-picked tests ---
    print("=" * 60)
    print("1. Original hand-picked tests")
    print("=" * 60)
    original = [
        (5, 5), (555, 445), (99999, 1), (19492, 23919),
        (9999999999, 1), (1234567890, 987654321),
        (0, 0), (1111111111, 8888888889),
    ]
    passed = sum(test(model, a, b)[0] for a, b in original)
    print(f"  {passed}/{len(original)} passed\n")
    all_passed += passed; all_total += len(original)

    # --- 2. Edge cases ---
    print("=" * 60)
    print("2. Boundary / edge cases")
    print("=" * 60)
    edge = [
        (0, 0), (0, 1), (1, 0), (0, 9), (9, 0),
        (9999999999, 0), (0, 9999999999),
        (9999999999, 9999999999),
        (5000000000, 5000000000),
        (1, 9999999999),
        (9999999999, 1),
    ]
    passed = sum(test(model, a, b)[0] for a, b in edge)
    print(f"  {passed}/{len(edge)} passed\n")
    all_passed += passed; all_total += len(edge)

    # --- 3. Carry chains ---
    print("=" * 60)
    print("3. Carry propagation chains (10^k - 1) + 1")
    print("=" * 60)
    carry_chain = [(10**k - 1, 1) for k in range(1, 11)]
    passed = sum(test(model, a, b)[0] for a, b in carry_chain)
    print(f"  {passed}/{len(carry_chain)} passed\n")
    all_passed += passed; all_total += len(carry_chain)

    # --- 4. Long carry chains ---
    print("=" * 60)
    print("4. Long carry chains (all 9s + small)")
    print("=" * 60)
    nines = [
        (9999999999, 2), (9999999999, 9), (9999999999, 10),
        (9999999999, 99), (9999999999, 100),
        (9999999990, 10), (9999999900, 100), (9999999000, 1000),
        (9999990000, 10000), (9999900000, 100000),
        (9990000000, 10000000), (9900000000, 100000000),
        (9000000000, 1000000000),
    ]
    passed = sum(test(model, a, b)[0] for a, b in nines)
    print(f"  {passed}/{len(nines)} passed\n")
    all_passed += passed; all_total += len(nines)

    # --- 5. No carry ---
    print("=" * 60)
    print("5. No carry cases")
    print("=" * 60)
    no_carry = [
        (1111111111, 2222222222),
        (1234567890, 0),
        (1000000000, 2000000000),
        (1234, 4321),
        (1010101010, 2020202020),
        (4040404040, 5050505050),
    ]
    passed = sum(test(model, a, b)[0] for a, b in no_carry)
    print(f"  {passed}/{len(no_carry)} passed\n")
    all_passed += passed; all_total += len(no_carry)

    # --- 6. Alternating carry ---
    print("=" * 60)
    print("6. Alternating carry patterns")
    print("=" * 60)
    alternating = [
        (5050505050, 5050505050),
        (9090909090, 1010101010),
        (1919191919, 8181818181),
        (5959595959, 4141414141),
        (8282828282, 2828282828),
    ]
    passed = sum(test(model, a, b)[0] for a, b in alternating)
    print(f"  {passed}/{len(alternating)} passed\n")
    all_passed += passed; all_total += len(alternating)

    # --- 7. Powers of 10 ---
    print("=" * 60)
    print("7. Powers of 10")
    print("=" * 60)
    powers = [(10**i, 10**j) for i in range(10) for j in range(10) if i + j <= 10]
    results = test_batch(model, powers)
    passed = sum(1 for (a, b), r in zip(powers, results) if r == a + b)
    failures = [(a, b, r) for (a, b), r in zip(powers, results) if r != a + b]
    if failures:
        for a, b, r in failures[:5]:
            print(f"  FAIL: {a} + {b} = {r} (expected {a+b})")
    print(f"  {passed}/{len(powers)} passed\n")
    all_passed += passed; all_total += len(powers)

    # --- 8. Palindromes ---
    print("=" * 60)
    print("8. Palindromic numbers")
    print("=" * 60)
    palindromes = [
        (1234554321, 1234554321),
        (9876556789, 123443210),
        (1111111111, 9999999999),
        (5678876543, 4321123456),
    ]
    passed = sum(test(model, a, b)[0] for a, b in palindromes)
    print(f"  {passed}/{len(palindromes)} passed\n")
    all_passed += passed; all_total += len(palindromes)

    # --- 9. Commutativity ---
    print("=" * 60)
    print("9. Commutativity: a+b == b+a")
    print("=" * 60)
    rng = random.Random(99)
    comm_pairs = [(rng.randint(0, 10**10-1), rng.randint(0, 10**10-1)) for _ in range(50)]
    forward_res = test_batch(model, comm_pairs)
    backward_res = test_batch(model, [(b, a) for a, b in comm_pairs])
    comm_ok = sum(1 for f, b in zip(forward_res, backward_res) if f == b)
    correct = sum(1 for (a, b), r in zip(comm_pairs, forward_res) if r == a + b)
    print(f"  Commutative: {comm_ok}/50, Correct: {correct}/50\n")
    all_passed += correct; all_total += 50

    # --- 10. Small numbers ---
    print("=" * 60)
    print("10. Random small numbers (0-99)")
    print("=" * 60)
    small_pairs = [(a, b) for a in range(0, 100, 7) for b in range(0, 100, 7)]
    results = test_batch(model, small_pairs)
    passed = sum(1 for (a, b), r in zip(small_pairs, results) if r == a + b)
    failures = [(a, b, r) for (a, b), r in zip(small_pairs, results) if r != a + b]
    if failures:
        for a, b, r in failures[:5]:
            print(f"  FAIL: {a} + {b} = {r} (expected {a+b})")
    print(f"  {passed}/{len(small_pairs)} passed\n")
    all_passed += passed; all_total += len(small_pairs)

    # --- 11. Medium numbers ---
    print("=" * 60)
    print("11. Random medium (1000-999999)")
    print("=" * 60)
    rng = random.Random(77)
    med_pairs = [(rng.randint(1000, 999999), rng.randint(1000, 999999)) for _ in range(500)]
    results = test_batch(model, med_pairs)
    passed = sum(1 for (a, b), r in zip(med_pairs, results) if r == a + b)
    failures = [(a, b, r) for (a, b), r in zip(med_pairs, results) if r != a + b]
    if failures:
        for a, b, r in failures[:5]:
            print(f"  FAIL: {a} + {b} = {r} (expected {a+b})")
    print(f"  {passed}/{len(med_pairs)} passed\n")
    all_passed += passed; all_total += len(med_pairs)

    # --- 12. Large stress test ---
    print("=" * 60)
    print("12. Large random stress test (10,000 cases)")
    print("=" * 60)
    rng = random.Random(42)
    batch_size = 512
    total, total_passed, total_failed = 10000, 0, 0
    first_failures = []
    for start in range(0, total, batch_size):
        cur = min(batch_size, total - start)
        pairs = [(rng.randint(0, 10**10-1), rng.randint(0, 10**10-1)) for _ in range(cur)]
        results = test_batch(model, pairs)
        for (a, b), r in zip(pairs, results):
            if r == a + b:
                total_passed += 1
            else:
                total_failed += 1
                if len(first_failures) < 10:
                    first_failures.append((a, b, r, a + b))
        done = start + cur
        if done % 2000 == 0 or done == total:
            print(f"  Progress: {done:,d}/{total:,d} — {total_failed} failures so far")
    if first_failures:
        print(f"\n  First failures:")
        for a, b, got, exp in first_failures:
            print(f"    {a:010d} + {b:010d} = {got} (expected {exp})")
    print(f"\n  {total_passed}/{total} passed ({total_failed} failures)\n")
    all_passed += total_passed; all_total += total

    # --- 13. Digit-position carry ---
    print("=" * 60)
    print("13. Single-digit-position carry tests")
    print("=" * 60)
    digit_carry = []
    for pos in range(10):
        a = 9 * (10 ** pos)
        b = 1 * (10 ** pos)
        digit_carry.append((a, b))
        a = int("9" * (pos + 1))
        b = 1
        digit_carry.append((a, b))
    results = test_batch(model, digit_carry)
    passed = sum(1 for (a, b), r in zip(digit_carry, results) if r == a + b)
    failures = [(a, b, r) for (a, b), r in zip(digit_carry, results) if r != a + b]
    if failures:
        for a, b, r in failures[:10]:
            print(f"  FAIL: {a} + {b} = {r} (expected {a+b})")
    print(f"  {passed}/{len(digit_carry)} passed\n")
    all_passed += passed; all_total += len(digit_carry)

    # --- 14. Previously-failing cases ---
    print("=" * 60)
    print("14. Previously-failing cases")
    print("=" * 60)
    prev_fail = [
        (9999999999, 0),
        (0, 9999999999),
        (4040404040, 5050505050),
        (9876556789, 123443210),
        (5678876543, 4321123456),
        (863512, 986307),
        (642029, 250362),
        (210853, 732099),
        (93028, 100535),
        (112648, 286132),
    ]
    passed = sum(test(model, a, b)[0] for a, b in prev_fail)
    print(f"  {passed}/{len(prev_fail)} passed\n")
    all_passed += passed; all_total += len(prev_fail)

    # --- Summary ---
    print("=" * 60)
    print(f"GRAND TOTAL: {all_passed}/{all_total} passed")
    print("=" * 60)


if __name__ == "__main__":
    model = build_adder()
    run_tests(model)
