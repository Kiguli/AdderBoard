# Formal Verification Report: fblissjr — 33 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: exhaustive
- Solve time: 349.9s
- All 1024 carry partitions verified with 102400 total tests

## Parameter Audit
- Claimed: 33 | Counted: 141 | MISMATCH
- MISMATCH: claimed 33, counted 141 (diff=108)

## Architecture Compliance
- Self-attention: No
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected

## Submission Info
- Architecture: 1L decoder, d=3, 3h (d_head=1), ff=4
- Key tricks: ALiBi prefix sum for carry, e^80 softmax anchoring, residual cancellation head
- Link: https://github.com/fblissjr/AdderBoard/blob/main/submission_1l.py
- Verified: 2026-03-25 16:58