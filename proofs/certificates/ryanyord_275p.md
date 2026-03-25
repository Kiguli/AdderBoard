# Formal Verification Report: ryanyord — 275 params (trained)

## Result: INCONCLUSIVE

## Verification Method
- Method: bounds_propagation
- Solve time: 0.0s
- auto_LiRPA not installed. Install with: pip install auto_LiRPA

## Parameter Audit
- Claimed: 275 | Counted: -1 | MISMATCH
- Model has no named_parameters() — cannot enumerate

## Architecture Compliance
- Self-attention: No
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected

## Submission Info
- Architecture: 1L decoder, d=4, 1h, ff=8, ranks=(3,3,2,2)
- Key tricks: SVD truncation of 311p, tied embed, low-rank factorization, shareA_tieKV, RMSNorm
- Link: https://github.com/ryanyord/tiny-adder-275p
- Verified: 2026-03-25 16:13