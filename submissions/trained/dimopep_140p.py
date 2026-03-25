"""
AdderBoard Trained Submission: 140-param 1L decoder, d=4, 1h, hd=4, ff=4
Trained with autoregressive teacher-forced loss, AdamW lr=0.01, 20K steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


VOCAB_SIZE = 10
D_MODEL = 4
HEAD_DIM = 4
FF_DIM = 4
ROPE_THETA = 3.0
OUTPUT_DIGITS = 11


def apply_rope(x, theta=ROPE_THETA):
    _, seq_len, _, head_dim = x.shape
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(2)
    sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(2)
    dim_pairs = head_dim // 2
    x_pairs = x.reshape(*x.shape[:-1], dim_pairs, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    out0 = x0 * cos_vals - x1 * sin_vals
    out1 = x0 * sin_vals + x1 * cos_vals
    return torch.stack([out0, out1], dim=-1).reshape(x.shape)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight


class AdderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.norm1 = RMSNorm(D_MODEL)
        self.W_q = nn.Linear(D_MODEL, HEAD_DIM, bias=False)
        self.W_kv = nn.Linear(D_MODEL, HEAD_DIM, bias=False)  # tied K=V
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        self.norm2 = RMSNorm(D_MODEL)
        self.W_gate = nn.Linear(D_MODEL, FF_DIM, bias=False)
        self.W_up = nn.Linear(D_MODEL, FF_DIM, bias=False)
        self.W_down = nn.Linear(FF_DIM, D_MODEL, bias=False)
        self.norm_final = RMSNorm(D_MODEL)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        B, T, D = x.shape
        # Pre-norm attention
        h = self.norm1(x)
        q = self.W_q(h).reshape(B, T, 1, HEAD_DIM)
        kv = self.W_kv(h).reshape(B, T, 1, HEAD_DIM)
        k = v = kv
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q)
        k = apply_rope(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = HEAD_DIM ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, HEAD_DIM)
        # O = Q^T (tied)
        out = F.linear(out, self.W_q.weight.t())
        x = x + out
        # Pre-norm FFN (SwiGLU)
        h = self.norm2(x)
        h = F.silu(self.W_gate(h)) * self.W_up(h)
        h = self.W_down(h)
        x = x + h
        # Final norm + tied LM head
        x = self.norm_final(x)
        return x @ self.embedding.weight.T


STATE_DICT = {
    "embedding.weight": torch.tensor([[-1.730218768119812, -0.2547132074832916, 5.2851481437683105, -3.131179094314575], [-3.174604892730713, -0.49524009227752686, 3.2596633434295654, -3.096811056137085], [-3.930217742919922, -0.720429539680481, 2.0609347820281982, -2.3161725997924805], [-4.2807230949401855, -1.0139139890670776, 1.0809674263000488, -1.3698683977127075], [-4.3072919845581055, -1.3361276388168335, 0.09954871982336044, -0.5498083829879761], [-3.9868428707122803, -1.7041209936141968, -0.7639560699462891, 0.12342499196529388], [-3.3500146865844727, -2.068692922592163, -1.3606889247894287, 0.5828636884689331], [-2.440566301345825, -2.417412757873535, -1.6824067831039429, 0.9797969460487366], [-1.2624038457870483, -2.759227991104126, -1.7066621780395508, 1.2801145315170288], [0.6039751172065735, -3.1253647804260254, -1.2808130979537964, 0.8087077736854553]], dtype=torch.float32),
    "norm1.weight": torch.tensor([0.22397252917289734, 2.9709789752960205, 0.004627978429198265, 0.024981478229165077], dtype=torch.float32),
    "W_q.weight": torch.tensor([[0.9972899556159973, 1.3447751998901367, -0.7394230961799622, -1.0346050262451172], [-0.5944087505340576, -0.9044270515441895, -2.0017337799072266, -0.47218015789985657], [0.1680702418088913, 0.2559415400028229, -1.4458389282226562, -0.48717543482780457], [-0.680279552936554, -1.0666685104370117, -0.5049071907997131, -0.12875384092330933]], dtype=torch.float32),
    "W_kv.weight": torch.tensor([[-0.6151725053787231, -1.4711843729019165, 0.31121939420700073, -1.1036722660064697], [-0.4503699243068695, -0.26388832926750183, 0.35671520233154297, 1.052584171295166], [0.3745459318161011, -0.7169608473777771, 0.046804532408714294, 0.7246946692466736], [-0.74381422996521, -1.3578026294708252, 0.7606346011161804, -0.4730044901371002]], dtype=torch.float32),
    "q_norm.weight": torch.tensor([2.441133737564087, 1.9687837362289429, 0.8033236265182495, 3.624851942062378], dtype=torch.float32),
    "k_norm.weight": torch.tensor([2.280484199523926, 0.22510592639446259, -0.01126538123935461, 3.4231643676757812], dtype=torch.float32),
    "norm2.weight": torch.tensor([2.9156434535980225, 6.517229080200195, 3.1696534156799316, 1.6497373580932617], dtype=torch.float32),
    "W_gate.weight": torch.tensor([[0.6762492060661316, 1.383542537689209, -0.37185904383659363, 0.650637149810791], [4.506912708282471, 0.6688480973243713, 1.7770065069198608, -1.4233444929122925], [-1.5175119638442993, -2.9070382118225098, 1.5377386808395386, -0.8157817125320435], [-0.6493015885353088, -1.0984196662902832, 0.47089722752571106, -0.41704633831977844]], dtype=torch.float32),
    "W_up.weight": torch.tensor([[1.6318920850753784, 0.597070574760437, 2.720832109451294, -1.7925524711608887], [-3.993180990219116, -0.17637008428573608, -3.4726452827453613, 1.9285545349121094], [-0.7829828858375549, -1.368809461593628, 1.4623147249221802, 1.5931147336959839], [-1.5189820528030396, -3.0357723236083984, 2.272179365158081, 0.8424974679946899]], dtype=torch.float32),
    "W_down.weight": torch.tensor([[-2.013659954071045, 0.1503746509552002, 1.6445231437683105, -4.081696033477783], [0.3591371178627014, -0.05487572029232979, 0.059290580451488495, 0.07458747178316116], [2.2788405418395996, 0.8970686197280884, 1.4553635120391846, 1.3654996156692505], [-0.5225310325622559, -0.9923188090324402, -2.5637733936309814, 0.9559842348098755]], dtype=torch.float32),
    "norm_final.weight": torch.tensor([31.38949203491211, 2.7755320072174072, 25.098451614379883, 19.817163467407227], dtype=torch.float32),
}


def build_model():
    model = AdderModel()
    model.load_state_dict(STATE_DICT)
    model.eval()
    metadata = {
        "name": "140p Adder (d=4, tied KV/OQ/LM, QK-norm)",
        "author": "dimo",
        "params": 140,
        "architecture": "1L decoder, d=4, 1h/1kv, hd=4, ff=4, RoPE theta=3, SwiGLU",
        "tricks": ["tied K=V", "tied O=Q^T", "tied lm_head", "QK-norm", "autoregressive teacher-forced loss"],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    pa = f"{a:010d}"
    pb = f"{b:010d}"
    seq = [0] + [int(c) for c in reversed(pa)] + [0, 0] + [int(c) for c in reversed(pb)] + [0]
    device = next(model.parameters()).device
    x = torch.tensor([seq], dtype=torch.long, device=device)
    digits = []
    with torch.no_grad():
        for _ in range(OUTPUT_DIGITS):
            logits = model(x)
            d = logits[0, -1, :VOCAB_SIZE].argmax().item()
            digits.append(d)
            x = torch.cat([x, torch.tensor([[d]], dtype=torch.long, device=device)], dim=1)
    return int("".join(str(d) for d in reversed(digits)))
