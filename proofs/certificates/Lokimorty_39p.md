# Formal Verification Report: lokimorty — 39 params (trained)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: smt_z3
- Solve time: 3.4s
- All 1024 carry partitions verified

## Parameter Audit
- Claimed: 39 | Counted: 39 | Match

## Architecture Compliance
- Self-attention: Yes
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: PASS

## Submission Info
- Architecture: 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU
- Key tricks: Circular arc embedding, tied K=V, shared RMSNorms, shared anti-quarter QK norm
- Link: https://gist.github.com/Lokimorty/b769726e4fd32ff2c5e08c7932a15f40
- Verified: 2026-03-25 16:13