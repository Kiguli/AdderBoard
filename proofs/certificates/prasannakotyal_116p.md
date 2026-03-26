# Formal Verification Report: nino — 116 params (hand_coded)

## Result: FORMALLY VERIFIED
For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.

## Verification Method
- Method: exhaustive
- Solve time: 608.3s
- All 1024 carry partitions verified with 102400 total tests

## Parameter Audit
- Claimed: 116 | Counted: 116 | Match

## Architecture Compliance
- Self-attention: Yes
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: PASS

## Submission Info
- Architecture: 1L Qwen3, d=3, 4h/1kv, hd=2
- Key tricks: Tied embed, shared RMSNorm vectors, RoPE (hd=2)
- Link: https://gist.github.com/prasannakotyal/467d4c54564beba34d9d7edbd41c33dc
- Verified: 2026-03-25 17:23