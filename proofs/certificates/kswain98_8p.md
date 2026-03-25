# Formal Verification Report: kswain98 — 8 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: exhaustive
- Solve time: 849.9s
- All 1024 carry partitions verified with 102400 total tests

## Parameter Audit
- Claimed: 8 | Counted: 2 | MISMATCH
- MISMATCH: claimed 8, counted 2 (diff=-6)

## Architecture Compliance
- Self-attention: No
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected

## Submission Info
- Architecture: 1L Qwen-style decoder, d=2, 1h, hd=2, ff=2
- Key tricks: RoPE period-19, phase-tied Q projection, coupled quadratic embedding
- Link: https://github.com/kswain98/AdderBoard
- Verified: 2026-03-25 16:07