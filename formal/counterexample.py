"""
Counterexample analysis for falsified AdderBoard submissions.

When formal verification finds a counterexample (a, b) where model(a,b) != a+b,
this module:
1. Confirms the failure empirically
2. Classifies the failure type
3. Searches the neighborhood to map the failure region
4. Estimates the failure rate
5. Finds additional counterexamples
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CounterexampleAnalysis:
    """Full analysis of a counterexample."""
    primary: tuple[int, int]  # (a, b) — the original counterexample
    expected: int
    model_output: int

    # Classification
    failure_type: str  # From taxonomy
    wrong_digits: list[int]  # Positions (from MSB) where output differs
    carry_pattern: str  # Binary string of carry pattern

    # Neighborhood analysis
    additional_counterexamples: list[tuple[int, int, int, int]] = field(default_factory=list)
    # Each: (a, b, expected, model_output)

    neighborhood_tested: int = 0
    neighborhood_failures: int = 0
    estimated_failure_rate: float = 0.0

    # Pattern analysis
    failure_pattern: str = ""  # Human-readable description
    notes: list[str] = field(default_factory=list)


def confirm_counterexample(
    module: Any, model: Any, a: int, b: int
) -> Optional[tuple[int, int]]:
    """
    Confirm a counterexample by actually running the model.
    Returns (expected, actual) if confirmed, None if the model actually succeeds.
    """
    expected = a + b
    try:
        result = module.add(model, a, b)
        if result != expected:
            return (expected, result)
        return None
    except Exception as e:
        logger.warning("Model raised exception on (%d, %d): %s", a, b, e)
        return (expected, -1)


def _find_wrong_digits(expected: int, actual: int) -> list[int]:
    """Find which digit positions differ between expected and actual."""
    exp_str = str(expected).zfill(11)
    act_str = str(actual).zfill(11) if isinstance(actual, int) else str(actual).zfill(11)
    wrong = []
    for i, (e, a) in enumerate(zip(exp_str, act_str)):
        if e != a:
            wrong.append(i)
    return wrong


def _compute_carry_pattern(a: int, b: int) -> str:
    """Compute the binary carry pattern for a + b.
    Bit i represents whether digit position 10^i produces a carry-out.
    Returned as a 10-char binary string (MSB=10^9 on left, LSB=units on right)."""
    mask = 0
    carry = 0
    for pos in range(10):  # pos 0 = units, pos 9 = 10^9
        a_d = (a // (10 ** pos)) % 10
        b_d = (b // (10 ** pos)) % 10
        if a_d + b_d + carry >= 10:
            mask |= (1 << pos)
            carry = 1
        else:
            carry = 0
    return format(mask, "010b")


def _classify_failure(
    a: int, b: int, expected: int, actual: int, wrong_digits: list[int]
) -> str:
    """Classify the failure type."""
    if not isinstance(actual, int) or actual < 0:
        return "Exception/Invalid output"

    if expected >= 10_000_000_000 and actual < 10_000_000_000:
        return "10→11 digit overflow"

    if expected < 10_000_000_000 and actual >= 10_000_000_000:
        return "Spurious overflow"

    # Check carry propagation
    carry_pattern = _compute_carry_pattern(a, b)
    consecutive_carries = max(
        (len(s) for s in carry_pattern.split("0") if s), default=0
    )
    if consecutive_carries >= 3 and wrong_digits:
        return f"Carry propagation (chain={consecutive_carries})"

    if len(wrong_digits) == 1:
        return f"Single-digit boundary (position {wrong_digits[0]})"

    if all(d == 0 for d in [a % 10, b % 10]):
        return "Leading zeros"

    return f"Wrong at positions {wrong_digits}"


def _search_neighborhood(
    module: Any, model: Any, a: int, b: int,
    radius: int = 100, samples: int = 1000,
) -> list[tuple[int, int, int, int]]:
    """Search around a counterexample for more failures."""
    rng = random.Random(a ^ b)
    failures = []

    for _ in range(samples):
        # Perturb each digit independently
        a_perturbed = a
        b_perturbed = b
        for pos in range(10):
            if rng.random() < 0.3:  # 30% chance to change each digit
                factor = 10 ** pos
                a_digit = (a_perturbed // factor) % 10
                new_digit = rng.randint(0, 9)
                a_perturbed += (new_digit - a_digit) * factor

            if rng.random() < 0.3:
                factor = 10 ** pos
                b_digit = (b_perturbed // factor) % 10
                new_digit = rng.randint(0, 9)
                b_perturbed += (new_digit - b_digit) * factor

        a_perturbed = max(0, min(9999999999, a_perturbed))
        b_perturbed = max(0, min(9999999999, b_perturbed))

        expected = a_perturbed + b_perturbed
        try:
            result = module.add(model, a_perturbed, b_perturbed)
            if result != expected:
                failures.append((a_perturbed, b_perturbed, expected, result))
        except Exception:
            failures.append((a_perturbed, b_perturbed, expected, -1))

    return failures


def _search_by_carry_pattern(
    module: Any, model: Any, carry_pattern: str, samples: int = 500
) -> list[tuple[int, int, int, int]]:
    """Search for failures with the same carry pattern."""
    target_mask = int(carry_pattern, 2)
    rng = random.Random(target_mask)
    failures = []
    tested = 0

    for _ in range(samples * 10):
        a = rng.randint(0, 9999999999)
        b = rng.randint(0, 9999999999)

        # Check if carry pattern matches
        mask = 0
        carry = 0
        for pos in range(9, -1, -1):
            a_d = (a // (10 ** pos)) % 10
            b_d = (b // (10 ** pos)) % 10
            if a_d + b_d + carry >= 10:
                mask |= (1 << (9 - pos))
                carry = 1
            else:
                carry = 0

        if mask != target_mask:
            continue

        tested += 1
        expected = a + b
        try:
            result = module.add(model, a, b)
            if result != expected:
                failures.append((a, b, expected, result))
        except Exception:
            failures.append((a, b, expected, -1))

        if tested >= samples:
            break

    return failures


def analyze_counterexample(
    module: Any, model: Any,
    a: int, b: int, expected: int, model_output: int,
) -> CounterexampleAnalysis:
    """
    Perform full analysis of a counterexample.
    """
    wrong_digits = _find_wrong_digits(expected, model_output)
    carry_pattern = _compute_carry_pattern(a, b)
    failure_type = _classify_failure(a, b, expected, model_output, wrong_digits)

    # Search neighborhood
    neighborhood_failures = _search_neighborhood(module, model, a, b)

    # Search by carry pattern
    pattern_failures = _search_by_carry_pattern(module, model, carry_pattern)

    all_additional = []
    seen = {(a, b)}
    for ce in neighborhood_failures + pattern_failures:
        key = (ce[0], ce[1])
        if key not in seen:
            seen.add(key)
            all_additional.append(ce)

    # Estimate failure rate from random sampling
    rng = random.Random(42)
    random_tested = 10000
    random_failures = 0
    for _ in range(random_tested):
        ra = rng.randint(0, 9999999999)
        rb = rng.randint(0, 9999999999)
        try:
            result = module.add(model, ra, rb)
            if result != ra + rb:
                random_failures += 1
        except Exception:
            random_failures += 1

    failure_rate = random_failures / random_tested

    # Build pattern description
    pattern_desc = f"Carry pattern {carry_pattern}"
    if len(all_additional) > 0:
        common_patterns = set()
        for ce_a, ce_b, _, _ in all_additional[:20]:
            common_patterns.add(_compute_carry_pattern(ce_a, ce_b))
        if len(common_patterns) == 1:
            pattern_desc += " — all failures share this pattern"
        else:
            pattern_desc += f" — failures span {len(common_patterns)} carry patterns"

    analysis = CounterexampleAnalysis(
        primary=(a, b),
        expected=expected,
        model_output=model_output,
        failure_type=failure_type,
        wrong_digits=wrong_digits,
        carry_pattern=carry_pattern,
        additional_counterexamples=all_additional[:50],  # Cap at 50
        neighborhood_tested=1000 + 500,
        neighborhood_failures=len(all_additional),
        estimated_failure_rate=failure_rate,
        failure_pattern=pattern_desc,
    )

    logger.info(
        "Counterexample analysis for (%d, %d): type=%s, %d additional failures, rate=%.4f%%",
        a, b, failure_type, len(all_additional), failure_rate * 100,
    )

    return analysis
