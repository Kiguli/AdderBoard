# Formal Verification Report: lokimorty — 39 params (trained)

## Result: FALSIFIED

## Counterexample

| Input a | Input b | Expected (a+b) | Model Output |
|---------|---------|----------------|--------------|
| 5462119648 | 4228209436 | 9690329084 | 9690339084 |

## Failure Analysis
- **Wrong digit(s)**: Positions [6] (from MSB)
- **Failure type**: Single-digit boundary (position 6)
- **Carry pattern**: 0001001101
- **Pattern**: Carry pattern 0001001101 — failures span 13 carry patterns

## Failure Region
- Found 50 additional failing inputs in neighborhood search
- Estimated failure rate: 0.0200% of input space

## Additional Counterexamples

| a | b | Expected | Model Output |
|---|---|----------|--------------|
| 5462119648 | 7728309436 | 13190429084 | 13190439084 |
| 5452119648 | 4268209432 | 9720329080 | 9720339080 |
| 462119618 | 4228209432 | 4690329050 | 4690339050 |
| 5462119668 | 1228209431 | 6690329099 | 6690339099 |
| 5862119448 | 228009634 | 6090129082 | 6090139082 |
| 5462119648 | 2208209436 | 7670329084 | 7670339084 |
| 5418119645 | 4348709836 | 9766829481 | 9766839481 |
| 5762119648 | 9428209736 | 15190329384 | 15190339384 |
| 5162119621 | 4221209936 | 9383329557 | 9383339557 |
| 5412119648 | 528209436 | 5940329084 | 5940339084 |

## Verification Method
- Method: exhaustive
- Solve time: 97.2s
- Carry pattern: 0001001101

## Parameter Audit
- Claimed: 39 | Counted: 39 | Match

## Architecture Compliance
- Self-attention: Yes
- forward() clean: Yes
- add() clean: Yes
- Autoregressive: Yes
- Overall: PASS

## Submission Info
- Architecture: 1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE θ=3, SwiGLU
- Key tricks: Circular arc embedding, tied K=V, shared RMSNorms, shared anti-quarter QK norm
- Link: https://gist.github.com/Lokimorty/b769726e4fd32ff2c5e08c7932a15f40
- Verified: 2026-03-25 17:27