# Formal Verification Report: dimopep — 140 params (trained)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: exhaustive
- Solve time: 728.8s
- All 1024 carry partitions verified with 102400 total tests

## Parameter Audit
- Claimed: 140 | Counted: 140 | Match

## Architecture Compliance
- Self-attention: No
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected

## Submission Info
- Architecture: 1L decoder, d=4, 1h/1kv, hd=4, ff=4, RoPE θ=3, SwiGLU
- Key tricks: Tied K=V, tied O=Q^T, tied lm_head, QK-norm
- Link: https://gist.github.com/dimopep/27158a2b0ed983e32ee8f39af6e5a134
- Verified: 2026-03-25 17:39