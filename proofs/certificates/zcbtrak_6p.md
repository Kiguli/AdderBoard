# Formal Verification Report: zcbtrak — 6 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: exhaustive
- Solve time: 121.5s
- All 1024 carry partitions verified with 102400 total tests

## Parameter Audit
- Claimed: 6 | Counted: -1 | MISMATCH
- Model has no named_parameters() — cannot enumerate

## Architecture Compliance
- Self-attention: No
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: FAIL
- Issue: No self-attention mechanism detected

## Submission Info
- Architecture: 1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2
- Key tricks: RoPE period-19, hardcoded Q_proj, folded norm, tied carry hinge gate
- Link: https://gist.github.com/zcbtrak/b9af065d6395a3ecd72e3b8d2e867ae9
- Verified: 2026-03-25 15:53