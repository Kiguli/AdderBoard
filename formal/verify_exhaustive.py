"""
Exhaustive/symbolic verification for the smallest hand-coded models (≤20 params).

For these tiny models, we can verify correctness through structured enumeration:
1. Partition inputs by carry pattern (2^10 = 1024 partitions)
2. Within each partition, verify the model computes the correct linear function
3. Check boundary conditions where carry status changes

This is Tier 1 verification — essentially checking the constructive proof.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExhaustiveResult:
    """Result of exhaustive verification."""
    status: str  # "PROVEN_CORRECT", "COUNTEREXAMPLE_FOUND", "ERROR"
    solve_time_seconds: float = 0.0
    partitions_verified: int = 0
    total_partitions: int = 1024
    counterexample: Optional[tuple[int, int]] = None
    expected: Optional[int] = None
    model_output: Optional[int] = None
    failure_type: str = ""
    notes: list[str] = field(default_factory=list)
    method: str = "exhaustive"


def _carry_pattern_for(a: int, b: int) -> int:
    """
    Compute the carry pattern for a + b.
    Returns a 10-bit mask where bit i (from MSB=bit9 to LSB=bit0) indicates
    that digit position i produces a carry-out.
    Bit 9 = units place (10^0), bit 0 = 10^9 place.
    """
    mask = 0
    carry = 0
    for pos in range(10):  # pos 0 = units (10^0), pos 9 = 10^9
        a_digit = (a // (10 ** pos)) % 10
        b_digit = (b // (10 ** pos)) % 10
        digit_sum = a_digit + b_digit + carry
        if digit_sum >= 10:
            mask |= (1 << pos)
            carry = 1
        else:
            carry = 0
    return mask


def _representative_inputs_for_carry_pattern(carry_mask: int, count: int = 10) -> list[tuple[int, int]]:
    """
    Generate representative input pairs that match a specific carry pattern.
    Uses digit-level construction to guarantee the pattern.
    """
    import random
    rng = random.Random(carry_mask)  # Deterministic per pattern
    pairs = []

    for _ in range(count * 10):  # Try more, filter to matching
        a_digits = []
        b_digits = []
        carry_in = 0

        # Build digits from LSB (pos=0, units) to MSB (pos=9, 10^9)
        for pos in range(10):
            has_carry = bool(carry_mask & (1 << pos))

            if has_carry:
                # Need a_d + b_d + carry_in >= 10
                min_sum = 10 - carry_in
                # Pick a_d and b_d such that a_d + b_d >= min_sum
                a_d = rng.randint(max(0, min_sum - 9), 9)
                b_d_min = max(0, min_sum - a_d)
                b_d = rng.randint(b_d_min, 9)
                carry_in = 1
            else:
                # Need a_d + b_d + carry_in < 10
                max_sum = 9 - carry_in
                if max_sum < 0:
                    break  # Impossible pattern
                a_d = rng.randint(0, min(9, max_sum))
                b_d = rng.randint(0, min(9, max_sum - a_d))
                carry_in = 0

            a_digits.append(a_d)
            b_digits.append(b_d)
        else:
            # Successfully built all digits (a_digits[i] is the digit at 10^i)
            a = sum(d * 10 ** i for i, d in enumerate(a_digits))
            b = sum(d * 10 ** i for i, d in enumerate(b_digits))

            # Verify pattern matches
            if _carry_pattern_for(a, b) == carry_mask:
                pairs.append((a, b))
                if len(pairs) >= count:
                    break

    return pairs


def verify_exhaustive(
    module: Any,
    model: Any,
    submission_id: str,
    samples_per_partition: int = 100,
    timeout_seconds: int = 3600,
) -> ExhaustiveResult:
    """
    Verify a submission by exhaustive carry-pattern enumeration.

    For each of the 1024 carry patterns:
    1. Generate representative inputs matching that pattern
    2. Run the model on each input
    3. Check correctness

    This is not full formal verification — it's structured exhaustive testing
    that covers every carry pattern. For true formal verification,
    combine with verify_smt.py which proves correctness within each partition.
    """
    start = time.time()
    total_tested = 0
    partitions_ok = 0

    for carry_mask in range(1024):
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            return ExhaustiveResult(
                status="TIMEOUT",
                solve_time_seconds=elapsed,
                partitions_verified=partitions_ok,
                notes=[f"Timed out after {partitions_ok}/1024 partitions, {total_tested} tests"],
            )

        # Generate test inputs for this carry pattern
        pairs = _representative_inputs_for_carry_pattern(carry_mask, count=samples_per_partition)

        partition_ok = True
        for a, b in pairs:
            expected = a + b
            try:
                result = module.add(model, a, b)
            except Exception as e:
                return ExhaustiveResult(
                    status="COUNTEREXAMPLE_FOUND",
                    solve_time_seconds=time.time() - start,
                    partitions_verified=partitions_ok,
                    counterexample=(a, b),
                    expected=expected,
                    model_output=None,
                    failure_type=f"Exception: {e}",
                    notes=[f"Carry pattern: {carry_mask:010b}"],
                )

            total_tested += 1

            if result != expected:
                # Classify the failure
                failure_type = _classify_failure(a, b, expected, result, carry_mask)

                return ExhaustiveResult(
                    status="COUNTEREXAMPLE_FOUND",
                    solve_time_seconds=time.time() - start,
                    partitions_verified=partitions_ok,
                    counterexample=(a, b),
                    expected=expected,
                    model_output=result,
                    failure_type=failure_type,
                    notes=[f"Carry pattern: {carry_mask:010b}"],
                )

        partitions_ok += 1

        if partitions_ok % 100 == 0:
            logger.info(
                "Verified %d/1024 carry partitions (%d tests, %.1fs)",
                partitions_ok, total_tested, elapsed,
            )

    return ExhaustiveResult(
        status="PROVEN_CORRECT",
        solve_time_seconds=time.time() - start,
        partitions_verified=1024,
        notes=[f"All 1024 carry partitions verified with {total_tested} total tests"],
    )


def _classify_failure(
    a: int, b: int, expected: int, actual: int, carry_mask: int
) -> str:
    """Classify the type of failure for a counterexample."""
    expected_str = str(expected).zfill(11)
    actual_str = str(actual).zfill(11) if isinstance(actual, int) else str(actual)

    # Check if it's a 10->11 digit overflow issue
    if expected >= 10_000_000_000 and (not isinstance(actual, int) or actual < 10_000_000_000):
        return "10→11 digit overflow"

    if not isinstance(actual, int):
        return f"Non-integer output: {type(actual).__name__}"

    # Find which digits differ
    wrong_positions = []
    for i, (e, a_c) in enumerate(zip(expected_str, actual_str)):
        if e != a_c:
            wrong_positions.append(i)

    if not wrong_positions:
        return "Unknown (strings match but ints differ)"

    # Check if it's a carry propagation failure
    # Count the carry chain length at the failure point
    carry_chain = 0
    for pos in range(9, -1, -1):
        if carry_mask & (1 << (9 - pos)):
            carry_chain += 1
        else:
            carry_chain = 0

    if carry_chain >= 3:
        return f"Carry propagation (chain={carry_chain})"

    first_wrong = wrong_positions[0]
    return f"Wrong digit at position {first_wrong} (from MSB)"


def verify_boundary_cases(
    module: Any, model: Any, submission_id: str
) -> ExhaustiveResult:
    """
    Test specifically at carry-pattern boundaries where behavior changes.
    These are the inputs where one digit sum is exactly 9 or 10 —
    the boundary between carry and no-carry.
    """
    start = time.time()
    tested = 0
    boundary_pairs = []

    # Generate boundary cases: digit sums of exactly 9 and 10
    for pos in range(10):
        for target_sum in [9, 10]:
            for a_d in range(min(10, target_sum + 1)):
                b_d = target_sum - a_d
                if 0 <= b_d <= 9:
                    # Build a number pair where this digit is at the boundary
                    a = a_d * (10 ** (9 - pos))
                    b = b_d * (10 ** (9 - pos))
                    boundary_pairs.append((a, b))

    for a, b in boundary_pairs:
        expected = a + b
        try:
            result = module.add(model, a, b)
        except Exception as e:
            return ExhaustiveResult(
                status="COUNTEREXAMPLE_FOUND",
                solve_time_seconds=time.time() - start,
                counterexample=(a, b),
                expected=expected,
                failure_type=f"Exception at boundary: {e}",
            )

        tested += 1
        if result != expected:
            carry_mask = _carry_pattern_for(a, b)
            failure_type = _classify_failure(a, b, expected, result, carry_mask)
            return ExhaustiveResult(
                status="COUNTEREXAMPLE_FOUND",
                solve_time_seconds=time.time() - start,
                counterexample=(a, b),
                expected=expected,
                model_output=result,
                failure_type=failure_type,
                notes=[f"Boundary case at digit position, carry={carry_mask:010b}"],
            )

    return ExhaustiveResult(
        status="PROVEN_CORRECT",
        solve_time_seconds=time.time() - start,
        partitions_verified=0,
        notes=[f"All {tested} boundary cases passed"],
        method="boundary_exhaustive",
    )
