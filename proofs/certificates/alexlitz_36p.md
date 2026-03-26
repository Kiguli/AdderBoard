# Formal Verification Report: alexlitz — 36 params (hand_coded)

## Result: FALSIFIED

## Counterexample

| Input a | Input b | Expected (a+b) | Model Output |
|---------|---------|----------------|--------------|
| 5193798811 | 1806201129 | 6999999940 | 7000000000 |

## Failure Analysis
- **Wrong digit(s)**: Positions [1, 2, 3, 4, 5, 6, 7, 8, 9] (from MSB)
- **Failure type**: Wrong at positions [1, 2, 3, 4, 5, 6, 7, 8, 9]
- **Carry pattern**: 0000000001
- **Pattern**: Carry pattern 0000000001 — failures span 5 carry patterns

## Failure Region
- Found 17 additional failing inputs in neighborhood search
- Estimated failure rate: 0.0000% of input space

## Additional Counterexamples

| a | b | Expected | Model Output |
|---|---|----------|--------------|
| 5193798841 | 9706201123 | 14899999964 | 14900000000 |
| 5193798551 | 1806201119 | 6999999670 | 7000000000 |
| 5193798821 | 1806201129 | 6999999950 | 7000000000 |
| 5193798812 | 1806201129 | 6999999941 | 7000000000 |
| 5193798811 | 1806201124 | 6999999935 | 7000000000 |
| 193798830 | 1606201129 | 1799999959 | 1800000000 |
| 5193798811 | 9806201120 | 14999999931 | 15000000000 |
| 6193798611 | 1806201199 | 7999999810 | 8000000000 |
| 5193798851 | 1806201129 | 6999999980 | 7000000000 |
| 5193798811 | 9806201029 | 14999999840 | 15000000000 |

## Verification Method
- Method: exhaustive
- Solve time: 1.2s
- Carry pattern: 0000000001

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
- Verified: 2026-03-25 17:00