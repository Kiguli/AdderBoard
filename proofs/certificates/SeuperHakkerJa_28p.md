# Formal Verification Report: jacobli99 — 28 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: exhaustive
- Solve time: 904.7s
- All 1024 carry partitions verified with 102400 total tests

## Parameter Audit
- Claimed: 28 | Counted: 24 | MISMATCH
- MISMATCH: claimed 28, counted 24 (diff=-4)

## Architecture Compliance
- Self-attention: No
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected

## Submission Info
- Architecture: 1L decoder, d=2, 5h (MQA), hd=2, ff=4
- Key tricks: Tied parabolic decode, RoPE digit routing, sparse O-proj, tied MLP
- Link: https://gist.github.com/SeuperHakkerJa/da3050739bea97aabd86ee0d7d5ef689
- Verified: 2026-03-25 16:52