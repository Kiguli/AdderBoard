# Formal Verification Report: lichengliu03 — 50 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: smt_z3
- Solve time: 4.4s
- All 1024 carry partitions verified

## Parameter Audit
- Claimed: 50 | Counted: 50 | Match

## Architecture Compliance
- Self-attention: No
- forward() clean: No
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected
- Issue: forward() contains task-specific logic
- Warning: forward(): Suspicious variable 'carry' (looks like explicit carry state)

## Submission Info
- Architecture: 1L custom GPT, d=4, 2h, hd=2
- Key tricks: Factorized embed, rotation Q, tied embed+V dir, rank-1 MLP, parabolic head
- Link: https://github.com/lichengliu03/TinyAdder-50p
- Verified: 2026-03-25 16:13