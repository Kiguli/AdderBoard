# AdderBoard Formal Verification Status

Generated: 2026-03-26

## Hand-Coded Weights (23 submissions)

| Rank | Author | Params | Accuracy | Architecture | Verification | Method | Time | Counterexample | Notes |
|------|--------|-------:|----------|--------------|:------------:|--------|-----:|----------------|-------|
| 1 | zcbtrak | 6 | 100% | 1L Qwen-derived, d=2, 1h, hd=2, ff=2 | **PROVEN** | Structural algebraic | 26.5s | — | Rigorous proof via interval arithmetic over 1024 carry partitions |
| 2 | kswain98 | 8 | 100% | 1L Qwen-style, d=2, 1h, hd=2, ff=2 | **PROVEN** | Structural algebraic | 25.9s | — | Algebraically identical to zcbtrak (c=1000). Interval arithmetic over 1024 carry partitions. |
| 3 | lokimorty | 10 | 100% | 1L Qwen-derived, d=2, 1h, hd=2, ff=2 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 4 | lokimorty | 12 | 100% | 1L Qwen-derived, d=2, 1h, hd=2, ff=2 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 5 | yieldthought | 20 | 100% | 1L decoder, d=2, 1h, hd=2 | **PROVEN** | Structural algebraic | 26.6s | — | Position-only attention (RoPE-19), ReLU MLP. Interval arithmetic with token-dependent attention error bounds. |
| 6 | Wonderfall | 27 | 100% | 1L decoder, d=2, 1h, hd=2 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 7 | jacobli99 | 28 | 100% | 1L decoder, d=2, 5h (MQA), hd=2, ff=4 | UNKNOWN | Interval arithmetic | — | — | Formal verification attempted but INCONCLUSIVE: logit margins ~0.000244 (1 ULP at float32) are too thin for interval bounds (width ~1.73). No counterexample found in 102K+ samples. |
| 8 | Arch222 | 31 | 100% | 1L decoder, d=3, 4h/1kv, hd=2, ff=4 | NOT RUN | — | — | — | Missing module `tiny_transformer_adder` |
| 9 | fblissjr | 33 | 100% | 1L decoder, d=3, 3h (d_head=1), ff=4 | **PROVEN** | Structural algebraic | 1.0s | — | Q=K=0 fixed-mask attention, e^80 softmax anchoring (error < 1e-33, 24 orders of margin). 2-hinge ReLU exact step functions. Parabolic head gap >= 1. |
| 10 | alexlitz | 36 | 100% | 2L decoder, d=5, 5h+1h | FALSIFIED | Exhaustive sampling | 1.2s | 5193798811 + 1806201129 = 6999999940, model says 7000000000 | Carry propagation error |
| 11 | lichengliu03 | 50 | 100% | 1L custom GPT, d=4, 2h, hd=2 | TESTED | Exhaustive sampling | 795.8s | — | ~102K samples, not a proof |
| 12 | cosminscn | 66 | 100% | 1L nanoGPT, d=4, 2h | NOT RUN | — | — | — | Missing `build_model()` |
| 13 | bingbangboom-lab | 87 | 100% | 2L Qwen3, d=5, 2h/1kv, hd=2, ff=3 | NOT RUN | — | — | — | Missing `build_model()` |
| 14 | jacobli99 | 93 | 100% | 1L decoder, d=2, 5h (MQA), hd=2, ff=4 | NOT RUN | — | — | — | Syntax error in source |
| 15 | corbensorenson | 111 | 100% | 1L decoder, d=3, 4h/1kv, hd=2, ff=2 | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 16 | nino | 116 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2 | TESTED | Exhaustive sampling | 608.3s | — | ~102K samples, not a proof |
| 17 | Wonderfall | 121 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2, ff=2 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 18 | cosminscn | 130 | 100% | 1L nanoGPT, d=4, 2h | NOT RUN | — | — | — | Missing `build_model()` |
| 19 | Wonderfall | 130 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2, ff=3 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 20 | Wonderfall | 139 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 21 | bingbangboom-lab | 148 | 100% | 2L Qwen3, d=5, 2h/1kv, hd=2, ff=3 | NOT RUN | — | — | — | Missing `build_model()` |
| 22 | xangma | 177 | 100% | 2L Qwen3, d=5, 2h/1kv, hd=2 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 23 | xangma | 197 | ~100% | 2L Qwen3, d=5, 2h/1kv, hd=2 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |

## Trained Weights (29 submissions)

| Rank | Author | Params | Accuracy | Architecture | Verification | Method | Time | Counterexample | Notes |
|------|--------|-------:|----------|--------------|:------------:|--------|-----:|----------------|-------|
| 1 | tbukic | 36 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `sub50_sweep` |
| 2 | lokimorty | 39 | 99.91% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | FALSIFIED | Exhaustive | 97.2s | 5462119648 + 4228209436 = 9690329084, model says 9690339084 | Digit 6 wrong, ~0.02% failure rate, spans 13 carry patterns |
| 3 | tbukic | 41 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `sub50_sweep_obsolete` |
| 4 | tbukic | 44 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `sub50_sweep_obsolete` |
| 5 | tbukic | 45 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `sub50_sweep_obsolete` |
| 6 | Enara Vijil | 52 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `model` |
| 7 | tbukic | 55 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `experiments` |
| 8 | evindor | 57 | 100% | 1L decoder, d=5(2+3), 1h, qk=4, hd=5, ff=2 | NOT RUN | — | — | — | Missing checkpoint `checkpoint_57p.pt` |
| 9 | tbukic | 58 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `experiments` |
| 10 | tbukic | 62 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `minimal10digittransformer` |
| 11 | evindor | 67 | 100% | 1L decoder, d=5(2+3), 1h, qk=4, hd=5, ff=2 | NOT RUN | — | — | — | Missing checkpoint `checkpoint_67p.pt` |
| 12 | tbukic | 83 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `minimal10digittransformer` |
| 13 | tbukic | 86 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `minimal10digittransformer` |
| 14 | tbukic | 89 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `minimal10digittransformer` |
| 15 | tbukic | 95 | 99.03% | 1L Qwen3 + circular arc, d=3, 1h/1kv, hd=4, ff=3 | NOT RUN | — | — | — | Missing module `minimal10digittransformer` |
| 16 | tbukic | 101 | 100% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2 | NOT RUN | — | — | — | Missing module `minimal10digittransformer` |
| 17 | staghado | 122 | 99.95% | 1L Qwen3, d=3, 1h/1kv, hd=4, ff=3 | NOT RUN | — | — | — | Requires `mlx` (Apple only) |
| 18 | dimopep | 140 | 100% | 1L decoder, d=4, 1h/1kv, hd=4, ff=4 | TESTED | Exhaustive sampling | 728.8s | — | ~102K samples, not a proof |
| 19 | JackCai1206 | 234 | 99.91% | 1L decoder, d=6 (3+3), 2h, hd=3, ff=2 | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 20 | lichengliu03 | 262 | 99.95% | 1L decoder, d=4, 1h, ff=8 | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 21 | ryanyord | 275 | 99.98% | 1L decoder, d=4, 1h, ff=8 | ERROR | Bounds | 0.0s | — | auto_LiRPA not installed |
| 22 | h3nock | 305 | 99.98% | 1L decoder, d=4, 1h, ff=9 | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 23 | rezabyt | 311 | 99.999% | 1L decoder, d=4, 1h, ff=8 | NOT RUN | — | — | — | Missing module `src` |
| 24 | yinglunz | 456 | 100% | 1L decoder, d=7, 1h, ff=14 | NOT RUN | — | — | — | Missing module `src` |
| 25 | rezabyt | 491 | 99.97% | 1L decoder, d=7 | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 26 | yinglunz | 512 | 99.988% | 1L decoder, d=7, 1h, ff=14 | NOT RUN | — | — | — | Missing module `src` |
| 27 | Yeb Havinga | 777 | 99.69% | 1L decoder, d=7, 1h, ff=14 | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 28 | anadim | 1644 | 99.04% | 1L decoder, pair tokens | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 29 | anadim | 6080 | 100% | 2L decoder, d=16, ff=48 | NOT RUN | — | — | — | GitHub 404 (not fetched) |

## Statistics

| Metric | Count |
|--------|------:|
| Total submissions | 52 |
| **Formally proven (rigorous)** | **4** |
| Unknown (formal verification inconclusive, no counterexample) | 1 |
| Tested (exhaustive sampling, not a proof) | 4 |
| Falsified (counterexample found) | 2 |
| Not run (could not load) | 40 |
| Error during verification | 1 |

### Verification method key

| Label | Meaning |
|-------|---------|
| **PROVEN** | Rigorous mathematical proof that the model is correct for ALL 10^20 inputs. Uses structural algebraic analysis with sound interval arithmetic over 1024 carry partitions. |
| UNKNOWN | Formal verification was attempted but could not produce a proof (interval bounds too loose), AND no counterexample was found via exhaustive sampling. Correctness is neither proven nor disproven. |
| TESTED | Structured exhaustive sampling (~102K tests across 1024 carry partitions). Provides high confidence but is NOT a proof — only tests ~10^-15 of the input space. |
| FALSIFIED | Counterexample found that proves the model is incorrect. |

### Failure breakdown (41 not verified)

| Reason | Count |
|--------|------:|
| Requires `mlx` (Apple-only framework) | 9 |
| Missing repo-internal Python module | 15 |
| GitHub 404 (repo unavailable) | 8 |
| Missing `build_model()` function | 4 |
| Missing checkpoint file | 2 |
| Syntax error in source | 1 |
| `auto_LiRPA` not installed (Tier 3) | 1 |
| **Submissions that could be verified** | **11** |

### Formal verification details

All proofs use the **carry-partition method**: the 10^20 input space is divided into 1024 carry patterns (which digit positions produce a carry). Within each partition, attention weights are exact constants (position-only), and interval arithmetic propagates rigorous bounds through the network to prove the correct digit has the highest logit at every output position.

| Model | Technique | Key insight |
|-------|-----------|-------------|
| zcbtrak_6p | Interval arithmetic, SiLU gate state analysis | Position-only attention (K=[sqrt(2),0] after RMSNorm). SiLU gate deeply positive/negative -> bypass interval splitting. |
| kswain98_8p | Reuse zcbtrak proof | Algebraically identical to zcbtrak with c=1000: embed_w0=c, embed_w1=1/c, v_proj_w=-22c/sqrt(2). |
| yieldthought_20p | Interval arithmetic, ReLU MLP | Same RoPE-19 attention structure as zcbtrak. ReLU is piecewise-linear (simpler than SiLU). Token-dependent attention error bounded via hn[0] variation (~5.7e-5). |
| fblissjr_33p | Interval arithmetic, e^80 anchoring | Q=K=0 means attention is entirely from a fixed mask (0 attention params). e^80 softmax anchoring gives errors < 1e-33 with 24 orders of safety margin vs the 1e-10 carry threshold. 2-hinge ReLU gives exact 0/1 step functions. |
| jacobli99_28p | Attempted interval arithmetic | **INCONCLUSIVE**: decode_eps=5e-4 creates logit margins of exactly 1 ULP (2^-13 ~ 0.000244) at float32 magnitude ~1000. Interval bounds have width ~1.73, needing ~7000x tighter bounds. Empirically correct (102K+ tests pass) but not formally provable with current technique. |
