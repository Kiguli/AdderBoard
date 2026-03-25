# Formal Verification Report: yieldthought — 20 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: exhaustive
- Solve time: 343.1s
- All 1024 carry partitions verified with 102400 total tests

## Parameter Audit
- Claimed: 20 | Counted: 20 | Match

## Architecture Compliance
- Self-attention: Yes
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: PASS

## Submission Info
- Architecture: 1L decoder, d=2, 1h, hd=2
- Key tricks: Quadratic tied embedding, RoPE-19 digit routing, sparse tied V/O, two-hinge ReLU MLP
- Link: https://gist.github.com/yieldthought/a48b8d690d31039fadddd2bf297cae99
- Verified: 2026-03-25 16:12