# Formal Verification Report: alexlitz — 36 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: smt_z3
- Solve time: 3.5s
- All 1024 carry partitions verified

## Parameter Audit
- Claimed: 36 | Counted: -1 | MISMATCH
- Model has no named_parameters() — cannot enumerate

## Architecture Compliance
- Self-attention: No
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected

## Submission Info
- Architecture: 2L decoder, d=5, 5h+1h
- Key tricks: ALiBi slope=log(10), sparse embed, gated ReLU FFN, float64
- Link: https://gist.github.com/alexlitz/0d5efbccf443fb0e8136b8f5bd85140a
- Verified: 2026-03-25 16:13