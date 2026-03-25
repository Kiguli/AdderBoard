"""
Leaderboard metadata and submission URLs for all AdderBoard entries.
Parsed from the README.md leaderboard tables.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Category(str, Enum):
    HAND_CODED = "hand_coded"
    TRAINED = "trained"


class LinkType(str, Enum):
    GIST = "gist"
    REPO = "repo"


class VerificationTier(int, Enum):
    EXHAUSTIVE = 1   # Hand-coded ≤20 params
    SMT = 2          # ≤~200 params
    BOUNDS = 3       # Up to ~6000 params


@dataclass
class Submission:
    rank: int
    params: int
    accuracy: str
    author: str
    github_user: str
    category: Category
    architecture: str
    key_tricks: str
    link_url: str
    link_type: LinkType
    built_with: str = ""
    notes: str = ""

    @property
    def id(self) -> str:
        return f"{self.github_user}_{self.params}p"

    @property
    def tier(self) -> VerificationTier:
        if self.category == Category.HAND_CODED and self.params <= 20:
            return VerificationTier.EXHAUSTIVE
        elif self.params <= 200:
            return VerificationTier.SMT
        else:
            return VerificationTier.BOUNDS


# ── Hand-coded submissions ──────────────────────────────────────────

HAND_CODED: list[Submission] = [
    Submission(
        rank=1, params=6, accuracy="100%", author="zcbtrak",
        github_user="zcbtrak", category=Category.HAND_CODED,
        architecture="1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2",
        key_tricks="RoPE period-19, hardcoded Q_proj, folded norm, tied carry hinge gate",
        link_url="https://gist.github.com/zcbtrak/b9af065d6395a3ecd72e3b8d2e867ae9",
        link_type=LinkType.GIST,
        notes="Param count debated — see issue #75",
    ),
    Submission(
        rank=2, params=8, accuracy="100%", author="kswain98",
        github_user="kswain98", category=Category.HAND_CODED,
        architecture="1L Qwen-style decoder, d=2, 1h, hd=2, ff=2",
        key_tricks="RoPE period-19, phase-tied Q projection, coupled quadratic embedding",
        link_url="https://github.com/kswain98/AdderBoard",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=3, params=10, accuracy="100%", author="lokimorty",
        github_user="Lokimorty", category=Category.HAND_CODED,
        architecture="1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2",
        key_tricks="RoPE period-19, parametric tied embedding, gate tying via algebraic identity",
        link_url="https://gist.github.com/Lokimorty/d54e5c61997e00fb922b6692739a0f6c",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=4, params=12, accuracy="100%", author="lokimorty",
        github_user="Lokimorty", category=Category.HAND_CODED,
        architecture="1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2",
        key_tricks="RoPE period-19, parametric tied embedding, sparse attention/MLP, constructive carry hinge",
        link_url="https://gist.github.com/Lokimorty/cf27201cf6326b04c7328b6abe62248e",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=5, params=20, accuracy="100%", author="yieldthought",
        github_user="yieldthought", category=Category.HAND_CODED,
        architecture="1L decoder, d=2, 1h, hd=2",
        key_tricks="Quadratic tied embedding, RoPE-19 digit routing, sparse tied V/O, two-hinge ReLU MLP",
        link_url="https://gist.github.com/yieldthought/a48b8d690d31039fadddd2bf297cae99",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=6, params=27, accuracy="100%", author="Wonderfall",
        github_user="Wonderfall", category=Category.HAND_CODED,
        architecture="1L decoder, d=2, 1h, hd=2",
        key_tricks="Tied Q/K + V/O, factorized quadratic embedding, compressed MLP, RoPE period-19",
        link_url="https://gist.github.com/Wonderfall/bd1a2396eb9f846637f469d90823b48d",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=7, params=28, accuracy="100%", author="jacobli99",
        github_user="SeuperHakkerJa", category=Category.HAND_CODED,
        architecture="1L decoder, d=2, 5h (MQA), hd=2, ff=4",
        key_tricks="Tied parabolic decode, RoPE digit routing, sparse O-proj, tied MLP",
        link_url="https://gist.github.com/SeuperHakkerJa/da3050739bea97aabd86ee0d7d5ef689",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=8, params=31, accuracy="100%", author="Arch222",
        github_user="Arch222", category=Category.HAND_CODED,
        architecture="1L decoder, d=3, 4h/1kv, hd=2, ff=4",
        key_tricks="RoPE offset-targeted queries, sparse O-proj, SwiGLU carry detection, tied embed",
        link_url="https://github.com/Arch222/Addition_Final",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=9, params=33, accuracy="100%", author="fblissjr",
        github_user="fblissjr", category=Category.HAND_CODED,
        built_with="Claude Code + Gemini",
        architecture="1L decoder, d=3, 3h (d_head=1), ff=4",
        key_tricks="ALiBi prefix sum for carry, e^80 softmax anchoring, residual cancellation head",
        link_url="https://github.com/fblissjr/AdderBoard/blob/main/submission_1l.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=10, params=36, accuracy="100%", author="alexlitz",
        github_user="alexlitz", category=Category.HAND_CODED,
        architecture="2L decoder, d=5, 5h+1h",
        key_tricks="ALiBi slope=log(10), sparse embed, gated ReLU FFN, float64",
        link_url="https://gist.github.com/alexlitz/0d5efbccf443fb0e8136b8f5bd85140a",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=11, params=50, accuracy="100%", author="lichengliu03",
        github_user="lichengliu03", category=Category.HAND_CODED,
        architecture="1L custom GPT, d=4, 2h, hd=2",
        key_tricks="Factorized embed, rotation Q, tied embed+V dir, rank-1 MLP, parabolic head",
        link_url="https://github.com/lichengliu03/TinyAdder-50p",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=12, params=66, accuracy="100%", author="cosminscn",
        github_user="cosminscn", category=Category.HAND_CODED,
        architecture="1L nanoGPT, d=4, 2h",
        key_tricks="Rotation Q, sparse c_proj, parabolic lm_head, factorized embed, sinusoidal PE",
        link_url="https://gist.github.com/cosminscn/e4d028281378e16b18e61fca1163f9cb",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=13, params=87, accuracy="100%", author="bingbangboom-lab",
        github_user="bingbangboom-lab", category=Category.HAND_CODED,
        architecture="2L Qwen3, d=5, 2h/1kv, hd=2, ff=3",
        key_tricks="Cross-layer sharing, rank-1 projections, sparse gate, low-rank head",
        link_url="https://gist.github.com/bingbangboom-lab/ec367a6078e9ac2c5748dbbb78eae2a1",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=14, params=93, accuracy="100%", author="jacobli99",
        github_user="SeuperHakkerJa", category=Category.HAND_CODED,
        architecture="1L decoder, d=2, 5h (MQA), hd=2, ff=4",
        key_tricks="Tied parabolic decode, RoPE digit routing, ReLU carry detection",
        link_url="https://gist.github.com/SeuperHakkerJa/9d615964d2284a9a699b5a24cf19e69d",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=15, params=111, accuracy="100%", author="corbensorenson",
        github_user="corbensorenson", category=Category.HAND_CODED,
        built_with="Codex",
        architecture="1L decoder, d=3, 4h/1kv, hd=2, ff=2",
        key_tricks="Tied embed, RoPE, SwiGLU, GQA",
        link_url="https://github.com/corbensorenson/adderboard-submissions",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=16, params=116, accuracy="100%", author="nino",
        github_user="prasannakotyal", category=Category.HAND_CODED,
        architecture="1L Qwen3, d=3, 4h/1kv, hd=2",
        key_tricks="Tied embed, shared RMSNorm vectors, RoPE (hd=2)",
        link_url="https://gist.github.com/prasannakotyal/467d4c54564beba34d9d7edbd41c33dc",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=17, params=121, accuracy="100%", author="Wonderfall",
        github_user="Wonderfall", category=Category.HAND_CODED,
        built_with="Codex",
        architecture="1L Qwen3, d=3, 4h/1kv, hd=2, ff=2",
        key_tricks="Tied embed, RoPE digit routing, carry via final norm, SiLU wrap detection",
        link_url="https://gist.github.com/Wonderfall/7d6f49aa6703352f94d3d80b4cd31e15",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=18, params=130, accuracy="100%", author="cosminscn",
        github_user="cosminscn", category=Category.HAND_CODED,
        architecture="1L nanoGPT, d=4, 2h",
        key_tricks="Rank-1 linear, factorized embed, sinusoidal PE, ReLU carry detection, parabolic logit",
        link_url="https://gist.github.com/cosminscn/89c110dbae76ea0c873d67607e466f5b",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=19, params=130, accuracy="100%", author="Wonderfall",
        github_user="Wonderfall", category=Category.HAND_CODED,
        built_with="Codex",
        architecture="1L Qwen3, d=3, 4h/1kv, hd=2, ff=3",
        key_tricks="Tied embed, RoPE digit routing, SiLU carry logic",
        link_url="https://gist.github.com/Wonderfall/066df10de455cdc090900944bdc646cd",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=20, params=139, accuracy="100%", author="Wonderfall",
        github_user="Wonderfall", category=Category.HAND_CODED,
        built_with="GPT-5.2 Pro + Codex",
        architecture="1L Qwen3, d=3, 4h/1kv, hd=2",
        key_tricks="Tied embed, RoPE digit routing, SiLU carry logic",
        link_url="https://gist.github.com/Wonderfall/191bea43ff7f9316ac178b6c185d7165",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=21, params=148, accuracy="100%", author="bingbangboom-lab",
        github_user="bingbangboom-lab", category=Category.HAND_CODED,
        architecture="2L Qwen3, d=5, 2h/1kv, hd=2, ff=3",
        key_tricks="Rank-1 linear, factorized embed, sparse gate, param-free norm, cross-layer sharing",
        link_url="https://gist.github.com/bingbangboom-lab/3594f00a1aa0b668e70a92c396d0f0d1",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=22, params=177, accuracy="100%", author="xangma",
        github_user="xangma", category=Category.HAND_CODED,
        built_with="GPT + Codex",
        architecture="2L Qwen3, d=5, 2h/1kv, hd=2",
        key_tricks="Rank-1 linear, factorized embed, sparse gate, param-free norm, low-rank head",
        link_url="https://gist.github.com/xangma/1c2a1b2f9ca871b1f15646eed60d10ab",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=23, params=197, accuracy="~100%", author="xangma",
        github_user="xangma", category=Category.HAND_CODED,
        built_with="GPT + Codex",
        architecture="2L Qwen3, d=5, 2h/1kv, hd=2",
        key_tricks="Rank-1 linear, factorized embed, sparse gate, param-free norm",
        link_url="https://gist.github.com/xangma/c538a7a9d415f16e61f7bb26ae5cf6b0",
        link_type=LinkType.GIST,
        notes="Passed 8192 random tests; not independently verified on 10K suite",
    ),
]


# ── Trained submissions ─────────────────────────────────────────────

TRAINED: list[Submission] = [
    Submission(
        rank=1, params=36, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), K=rotation(Q), V=Q, tied O=Q^T, shared RMSNorms",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_36p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=2, params=39, accuracy="99.91%", author="lokimorty",
        github_user="Lokimorty", category=Category.TRAINED,
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding, tied K=V, shared RMSNorms, shared anti-quarter QK norm",
        link_url="https://gist.github.com/Lokimorty/b769726e4fd32ff2c5e08c7932a15f40",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=3, params=41, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), K=rotation(Q), V=Q, tied O=Q^T, shared RMSNorms",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_41p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=4, params=44, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), K=Q, V=Q, tied O=Q^T, shared RMSNorms, pure grokking",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_44p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=5, params=45, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), K=rotation(Q), V=Q, tied O=Q^T, shared RMSNorms",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_45p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=6, params=52, accuracy="100%", author="Enara Vijil",
        github_user="vijec", category=Category.TRAINED,
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding, tied K=V, tied O=Q^T, shared RMSNorms, Grokfast-EMA",
        link_url="https://github.com/vijec/AdderBoard-Submission/blob/main/submission-52/submission_52p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=7, params=55, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), K=αQ, gate=α·up, tied O=Q^T, shared block RMSNorms",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_55p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=8, params=57, accuracy="100%", author="evindor",
        github_user="evindor", category=Category.TRAINED,
        built_with="Claude Code + Codex",
        architecture="1L decoder, d=5(2+3), 1h, qk=4, hd=5, ff=2",
        key_tricks="Parametric circular embed, tied V/O, tied Q/K+phase, tied fc2=head_proj, rank-1 out",
        link_url="https://github.com/evindor/MicroAdder/tree/main/submission_57p",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=9, params=58, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), K=αQ, gate=α·up, tied O=Q^T",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_58p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=10, params=62, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), tied K=V, tied O=Q^T, tied lm_head",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_62p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=11, params=67, accuracy="100%", author="evindor",
        github_user="evindor", category=Category.TRAINED,
        built_with="Claude Code + Codex",
        architecture="1L decoder, d=5(2+3), 1h, qk=4, hd=5, ff=2",
        key_tricks="Parametric circular embed (3p), tied V/O, tied Q/K+phase, rank-1 out, carry-mix curriculum",
        link_url="https://github.com/evindor/MicroAdder/tree/main/submission_67p",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=12, params=83, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Tied embed, tied K=V, tied O=Q^T, shared all RMSNorms, iterated targeted FT",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_83p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=13, params=86, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Tied embed, tied K=V, tied O=Q^T, shared block RMSNorms, L-BFGS + targeted FT",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_86p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=14, params=89, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Tied embed, tied K=V, tied O=Q^T, RoPE, QK norms, 4-stage grokking-aware training",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_89p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=15, params=95, accuracy="99.03%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3 + circular arc embed, d=3, 1h/1kv, hd=4, ff=3, RoPE θ=3, SwiGLU",
        key_tricks="Circular arc embedding (3p), tied lm_head to dynamic embed, RoPE, QK norms",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_95p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=16, params=101, accuracy="100%", author="tbukic",
        github_user="tbukic", category=Category.TRAINED,
        built_with="SuperchargeAI + Claude Code",
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU",
        key_tricks="Tied embed, tied O=Q^T, RoPE, QK norms, cosine LR + targeted FT",
        link_url="https://github.com/tbukic/M10S-Transformer/blob/main/submissions/submission_101p.py",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=17, params=122, accuracy="99.95%", author="staghado",
        github_user="staghado", category=Category.TRAINED,
        architecture="1L Qwen3, d=3, 1h/1kv, hd=4, ff=3",
        key_tricks="Tied embed, RoPE θ=3",
        link_url="https://github.com/staghado/minimal-ten-digit-addition-transformer",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=18, params=140, accuracy="100%", author="dimopep",
        github_user="dimopep", category=Category.TRAINED,
        built_with="Claude Code",
        architecture="1L decoder, d=4, 1h/1kv, hd=4, ff=4, RoPE θ=3, SwiGLU",
        key_tricks="Tied K=V, tied O=Q^T, tied lm_head, QK-norm",
        link_url="https://gist.github.com/dimopep/27158a2b0ed983e32ee8f39af6e5a134",
        link_type=LinkType.GIST,
    ),
    Submission(
        rank=19, params=234, accuracy="99.91%", author="JackCai1206",
        github_user="JackCai1206", category=Category.TRAINED,
        built_with="Claude Code",
        architecture="1L decoder, d=6 (3 tok + 3 pos), 2h, hd=3, ff=2",
        key_tricks="Parametric spiral PE, split-head attn, shared XYZ pos, tied output head, LSB-first",
        link_url="https://github.com/JackCai1206/smallest-addition-transformer",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=20, params=262, accuracy="99.95%", author="lichengliu03",
        github_user="lichengliu03", category=Category.TRAINED,
        architecture="1L decoder, d=4, 1h, ff=8",
        key_tricks="Rank-3 factorization, shared-A tied-KV, RMSNorm, tied embed, curriculum learning",
        link_url="https://github.com/lichengliu03/TinyAdder-262p",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=21, params=275, accuracy="99.98%", author="ryanyord",
        github_user="ryanyord", category=Category.TRAINED,
        built_with="Gemini",
        architecture="1L decoder, d=4, 1h, ff=8, ranks=(3,3,2,2)",
        key_tricks="SVD truncation of 311p, tied embed, low-rank factorization, shareA_tieKV, RMSNorm",
        link_url="https://github.com/ryanyord/tiny-adder-275p",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=22, params=305, accuracy="99.98%", author="h3nock",
        github_user="h3nock", category=Category.TRAINED,
        architecture="1L decoder, d=4, 1h, ff=9",
        key_tricks="Low-rank factorization, shared-A tied-KV, RMSNorm, tied embed, learned PE, curriculum",
        link_url="https://github.com/h3nock/tiny-adder-lab/tree/exp/pe-sub512-search",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=23, params=311, accuracy="99.999%", author="rezabyt",
        github_user="rezabyt", category=Category.TRAINED,
        architecture="1L decoder, d=4, 1h, ff=8",
        key_tricks="Rank-3 factorization, shared-A tied-KV, RMSNorm, grokking",
        link_url="https://github.com/rezabyt/digit-addition-311p",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=24, params=456, accuracy="100%", author="yinglunz",
        github_user="yinglunz", category=Category.TRAINED,
        architecture="1L decoder, d=7, 1h, ff=14",
        key_tricks="Rank-3 factorization, shared-A tied-KV, rank-2 attn out, tied embed",
        link_url="https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=25, params=491, accuracy="99.97%", author="rezabyt",
        github_user="rezabyt", category=Category.TRAINED,
        architecture="1L decoder, d=7",
        key_tricks="Rank-3 factorization, RMSNorm, curriculum learning",
        link_url="https://github.com/rezabyt/digit-addition-491p",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=26, params=512, accuracy="99.988%", author="yinglunz",
        github_user="yinglunz", category=Category.TRAINED,
        architecture="1L decoder, d=7, 1h, ff=14",
        key_tricks="Rank-3 factorization",
        link_url="https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=27, params=777, accuracy="99.69%", author="Yeb Havinga",
        github_user="yhavinga", category=Category.TRAINED,
        built_with="Claude Code",
        architecture="1L decoder, d=7, 1h, ff=14",
        key_tricks="Tied embeddings, no FFN bias, curriculum learning",
        link_url="https://github.com/yhavinga/gpt-acc-jax",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=28, params=1644, accuracy="99.04%", author="anadim",
        github_user="anadim", category=Category.TRAINED,
        built_with="Codex",
        architecture="1L decoder, pair tokens",
        key_tricks="Pair token encoding (digit pairs as single tokens)",
        link_url="https://github.com/anadim/smallest-addition-transformer-codex",
        link_type=LinkType.REPO,
    ),
    Submission(
        rank=29, params=6080, accuracy="100%", author="anadim",
        github_user="anadim", category=Category.TRAINED,
        built_with="Claude Code",
        architecture="2L decoder, d=16, ff=48",
        key_tricks="Systematic scaling, found phase transition at d=16",
        link_url="https://github.com/anadim/smallest-addition-transformer-claude-code",
        link_type=LinkType.REPO,
    ),
]


ALL_SUBMISSIONS: list[Submission] = HAND_CODED + TRAINED


def get_submission(submission_id: str) -> Optional[Submission]:
    """Look up a submission by its ID (e.g. 'zcbtrak_6p')."""
    for s in ALL_SUBMISSIONS:
        if s.id == submission_id:
            return s
    return None


def get_by_tier(tier: VerificationTier) -> list[Submission]:
    """Get all submissions matching a verification tier."""
    return [s for s in ALL_SUBMISSIONS if s.tier == tier]


def get_by_category(category: Category) -> list[Submission]:
    """Get all submissions in a category."""
    return [s for s in ALL_SUBMISSIONS if s.category == category]
