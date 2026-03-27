# AdderBoard Formal Verification Status

Generated: 2026-03-26

## Hand-Coded Weights (23 submissions)

| Rank | Author | Params | Accuracy | Architecture | Verification | Method | Time | Counterexample | Notes |
|------|--------|-------:|----------|--------------|:------------:|--------|-----:|----------------|-------|
| 1 | zcbtrak | 6 | 100% | 1L Qwen-derived, d=2, 1h, hd=2, ff=2 | **PROVEN** | Structural algebraic | 26.5s | — | Rigorous proof via interval arithmetic over 1024 carry partitions |
| 2 | kswain98 | 8 | 100% | 1L Qwen-style, d=2, 1h, hd=2, ff=2 | **PROVEN** | Structural algebraic | 25.9s | — | Algebraically identical to zcbtrak (c=1000). Interval arithmetic over 1024 carry partitions. |
| 3 | lokimorty | 10 | 100% | 1L Qwen-derived, d=2, 1h, hd=2, ff=2 | **PROVEN** | Structural algebraic | 27.4s | — | Algebraically identical to zcbtrak 6-param family. Interval arithmetic over 1024 carry partitions. |
| 4 | lokimorty | 12 | 100% | 1L Qwen-derived, d=2, 1h, hd=2, ff=2 | **PROVEN** | Structural algebraic | 27.5s | — | Algebraically identical to zcbtrak 6-param family. Interval arithmetic over 1024 carry partitions. |
| 5 | yieldthought | 20 | 100% | 1L decoder, d=2, 1h, hd=2 | **PROVEN** | Structural algebraic | 26.6s | — | Position-only attention (RoPE-19), ReLU MLP. Interval arithmetic with token-dependent attention error bounds. |
| 6 | Wonderfall | 27 | 100% | 1L decoder, d=2, 1h, hd=2 | **PROVEN** | Structural algebraic | 28.1s | — | Same RoPE-19 architecture as yieldthought. ReLU MLP with cross-tied w_vo. Interval arithmetic with hn[0] attention error bounds. |
| 7 | jacobli99 | 28 | 100% | 1L decoder, d=2, 5h (MQA), hd=2, ff=4 | UNKNOWN | Interval arithmetic | — | — | Formal verification attempted but INCONCLUSIVE: logit margins ~0.000244 (1 ULP at float32) are too thin for interval bounds (width ~1.73). No counterexample found in 102K+ samples. |
| 8 | Arch222 | 31 | 100% | 1L decoder, d=3, 4h/1kv, hd=2, ff=4 | NOT RUN | — | — | — | Missing module `tiny_transformer_adder` |
| 9 | fblissjr | 33 | 100% | 1L decoder, d=3, 3h (d_head=1), ff=4 | **PROVEN** | Structural algebraic | 1.0s | — | Q=K=0 fixed-mask attention, e^80 softmax anchoring (error < 1e-33, 24 orders of margin). 2-hinge ReLU exact step functions. Parabolic head gap >= 1. |
| 10 | alexlitz | 36 | 100% | 2L decoder, d=5, 5h+1h | FALSIFIED | Exhaustive sampling | 1.2s | 5193798811 + 1806201129 = 6999999940, model says 7000000000 | Carry propagation error |
| 11 | lichengliu03 | 50 | 100% | 1L custom GPT, d=4, 2h, hd=2 | **PROVEN** | Structural algebraic | 12.9s | — | Position-only attention (sinusoidal PE period=11). 2-hinge ReLU carry/wrap detection. Parabolic head with gap >= 1. |
| 12 | cosminscn | 66 | 100% | 1L nanoGPT, d=4, 2h | NOT RUN | — | — | — | Missing `build_model()` |
| 13 | bingbangboom-lab | 87 | 100% | 2L Qwen3, d=5, 2h/1kv, hd=2, ff=3 | NOT RUN | — | — | — | Missing `build_model()` |
| 14 | jacobli99 | 93 | 100% | 1L decoder, d=2, 5h (MQA), hd=2, ff=4 | NOT RUN | — | — | — | Syntax error in source |
| 15 | corbensorenson | 111 | 100% | 1L decoder, d=3, 4h/1kv, hd=2, ff=2 | NOT RUN | — | — | — | GitHub 404 (not fetched) |
| 16 | nino | 116 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2 | **PROVEN** | Structural algebraic | 29.9s | — | Algebraically identical to Wonderfall 121p (same d=3, ff=2, wrap-only MLP, CARRY_SLOPE=-0.1 output norm). |
| 17 | Wonderfall | 121 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2, ff=2 | **PROVEN** | Structural algebraic | 32.8s | — | Position-only GQA (QK_NORM_SCALE=256, score gap >42000). 2-hinge SiLU wrap pair. Carry via output norm dim1. |
| 18 | cosminscn | 130 | 100% | 1L nanoGPT, d=4, 2h | NOT RUN | — | — | — | Missing `build_model()` |
| 19 | Wonderfall | 130 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2, ff=3 | **PROVEN** | Structural algebraic | 35.1s | — | Position-only GQA (score gap >42000). 2-hinge SiLU wrap + always-on linear carry neuron. |
| 20 | Wonderfall | 139 | 100% | 1L Qwen3, d=3, 4h/1kv, hd=2, ff=4 | **PROVEN** | Structural algebraic | 35.9s | — | Position-only GQA (score gap >42000). Explicit carry pair + wrap pair (both 2-hinge SiLU). |
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
| 18 | dimopep | 140 | 100% | 1L decoder, d=4, 1h/1kv, hd=4, ff=4 | UNKNOWN | Interval arithmetic | — | — | Token-dependent attention (dense K matrix, score gap 0.007 << token variation 0.5). Formal verification infeasible; no counterexample found in 102K+ samples. |
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
| **Formally proven (rigorous)** | **12** |
| Unknown (formal verification inconclusive, no counterexample) | 2 |
| Tested (exhaustive sampling, not a proof) | 0 |
| Falsified (counterexample found) | 2 |
| Not run (could not load) | 35 |
| Error during verification | 1 |

### Verification method key

| Label | Meaning |
|-------|---------|
| **PROVEN** | Rigorous mathematical proof that the model is correct for ALL 10^20 inputs. Uses structural algebraic analysis with sound interval arithmetic over 1024 carry partitions. |
| UNKNOWN | Formal verification was attempted but could not produce a proof (interval bounds too loose), AND no counterexample was found via exhaustive sampling. Correctness is neither proven nor disproven. |
| TESTED | Structured exhaustive sampling (~102K tests across 1024 carry partitions). Provides high confidence but is NOT a proof — only tests ~10^-15 of the input space. |
| FALSIFIED | Counterexample found that proves the model is incorrect. |

### Failure breakdown (36 not verified)

| Reason | Count |
|--------|------:|
| Requires `mlx` (Apple-only framework) | 3 |
| Missing repo-internal Python module | 17 |
| GitHub 404 (repo unavailable) | 8 |
| Missing `build_model()` function | 4 |
| Missing checkpoint file | 2 |
| Syntax error in source | 1 |
| `auto_LiRPA` not installed (Tier 3) | 1 |
| **Submissions that could be verified** | **16** |

### Formal verification details

All proofs use the **carry-partition method**: the 10^20 input space is divided into 1024 carry patterns (which digit positions produce a carry). Within each partition, attention weights are exact constants (position-only), and interval arithmetic propagates rigorous bounds through the network to prove the correct digit has the highest logit at every output position.

| Model | Technique | Key insight |
|-------|-----------|-------------|
| zcbtrak_6p | Interval arithmetic, SiLU gate state analysis | Position-only attention (K=[sqrt(2),0] after RMSNorm). SiLU gate deeply positive/negative -> bypass interval splitting. |
| kswain98_8p | Reuse zcbtrak proof | Algebraically identical to zcbtrak with c=1000: embed_w0=c, embed_w1=1/c, v_proj_w=-22c/sqrt(2). |
| lokimorty_10p | Reuse zcbtrak proof | Algebraically identical to zcbtrak 6-param family (hardcoded params from mlx source). |
| lokimorty_12p | Reuse zcbtrak proof | Algebraically identical to zcbtrak 6-param family (hardcoded params from mlx source). |
| yieldthought_20p | Interval arithmetic, ReLU MLP | Same RoPE-19 attention structure as zcbtrak. ReLU is piecewise-linear (simpler than SiLU). Token-dependent attention error bounded via hn[0] variation (~5.7e-5). |
| wonderfall_27p | Interval arithmetic, ReLU MLP | Same d=2 RoPE-19 architecture as yieldthought. Cross-tied w_vo matrix (attention VO reused as MLP W2). Hardcoded params from mlx source. |
| fblissjr_33p | Interval arithmetic, e^80 anchoring | Q=K=0 means attention is entirely from a fixed mask (0 attention params). e^80 softmax anchoring gives errors < 1e-33 with 24 orders of safety margin vs the 1e-10 carry threshold. 2-hinge ReLU gives exact 0/1 step functions. |
| wonderfall_121p | Interval arithmetic, d=3 GQA | Position-only GQA with QK_NORM_SCALE=256 (score gap >42000, attention essentially exact). 2-hinge SiLU wrap pair, carry injected via output norm dim1. |
| wonderfall_130p | Interval arithmetic, d=3 GQA | Same d=3 attention. Wrap pair + always-on linear carry neuron (c ≈ 0.05-0.1x). |
| wonderfall_139p | Interval arithmetic, d=3 GQA | Same d=3 attention. Explicit carry pair + wrap pair (both 2-hinge SiLU, 4 neurons total). |
| lichengliu03_50p | Interval arithmetic, sinusoidal PE, 2-hinge ReLU | d=4, 2 heads with sinusoidal PE (period=11, amplitude 100 prompt / 1 output). Head 0 (angle 8θ) targets current digit pair, Head 1 (angle 9θ) targets previous. Score gap >= 11.2 → dominant/other ratio ~77000:1. V-sum conditioned on d_prev for tight bounds. 2-hinge ReLU carry (thresholds 0.5/1.5) and wrap (thresholds 9045/9055) give exact 0/1 or -10/0 steps. Parabolic head logit[d] = 2dz - d² with gap >= 1. |
| nino_116p | Reuse Wonderfall 121p proof | Algebraically identical to Wonderfall 121p: same d=3, ff=2, EMBED_CONST=1000, ALPHA=20, QK_NORM_SCALE=256, CARRY_SLOPE=-0.1. Same K_proj, V_proj, Q offsets (23,11,22,10), O_proj, MLP gates (-188/-189). ROPE_THETA=10000 is irrelevant since head_dim=2 → inv_freq = θ^0 = 1.0. |
| dimopep_140p | Attempted interval arithmetic | **INCONCLUSIVE**: Trained model with dense K matrix — attention is token-dependent (not position-only). Score gap between dominant and non-dominant positions is only 0.007, while token content variation contributes ~0.5. Softmax interval bounds explode, making formal verification infeasible. No counterexample found in 102K+ samples. |
| jacobli99_28p | Attempted interval arithmetic | **INCONCLUSIVE**: decode_eps=5e-4 creates logit margins of exactly 1 ULP (2^-13 ~ 0.000244) at float32 magnitude ~1000. Interval bounds have width ~1.73, needing ~7000x tighter bounds. Empirically correct (102K+ tests pass) but not formally provable with current technique. |
