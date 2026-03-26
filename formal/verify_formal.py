"""
Generic formal verification driver for AdderBoard submissions.

Provides the carry-partition loop infrastructure that all model-specific
verifiers share. Each model implements a ModelVerifier that handles the
model-specific interval propagation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .verify_smt import SMTVerificationResult

logger = logging.getLogger(__name__)

OUTPUT_DIGITS = 11


# ── Carry partition helpers (shared across all models) ─────────────────

def carry_in_at(carry_mask: int, pos: int) -> int:
    """Carry-in to digit position pos given carry_mask."""
    if pos == 0:
        return 0
    return 1 if (carry_mask & (1 << (pos - 1))) else 0


def possible_output_digits(carry_mask: int, pos: int) -> list[int]:
    """All possible output digits at position pos for this carry partition."""
    if pos == 10:
        return [1 if (carry_mask & (1 << 9)) else 0]
    c_in = carry_in_at(carry_mask, pos)
    c_out = 1 if (carry_mask & (1 << pos)) else 0
    digits = set()
    for a_d in range(10):
        for b_d in range(10):
            s = a_d + b_d + c_in
            if (c_out and s >= 10) or (not c_out and s < 10):
                digits.add(s % 10)
    return sorted(digits)


def digit_sum_for_target(carry_mask: int, pos: int, target: int) -> int:
    """The a[pos]+b[pos] value that produces target digit at this position."""
    c_in = carry_in_at(carry_mask, pos)
    c_out = 1 if (carry_mask & (1 << pos)) else 0
    if c_out:
        return target + 10 - c_in
    else:
        return target - c_in


class ModelVerifier(Protocol):
    """Protocol for model-specific verifiers."""

    def verify_digit(
        self,
        carry_mask: int,
        digit_pos: int,
        target_digit: int,
        prev_output_digits: list[int],
    ) -> tuple[bool, str]:
        """Verify that the model outputs target_digit at digit_pos for ALL
        inputs matching this carry partition.

        Returns (proven, reason_if_failed).
        """
        ...


def verify_model(
    verifier: ModelVerifier,
    timeout_seconds: int = 3600,
    method_name: str = "structural_algebraic",
) -> SMTVerificationResult:
    """
    Run formal verification using carry-partition enumeration.

    For each of 1024 carry patterns, verifies all 11 output digit positions
    using the model-specific verifier.
    """
    start = time.time()
    total_partitions = 1024
    verified = 0
    inconclusive = []

    for carry_mask in range(total_partitions):
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            return SMTVerificationResult(
                status="TIMEOUT",
                solve_time_seconds=elapsed,
                method=method_name,
                notes=[f"Verified {verified}/{total_partitions}, "
                       f"{len(inconclusive)} inconclusive"],
            )

        partition_ok = True
        fail_reason = ""
        prev_output_digits: list[int] = []

        for digit_pos in range(OUTPUT_DIGITS):
            possible_targets = possible_output_digits(carry_mask, digit_pos)

            all_targets_ok = True
            for target in possible_targets:
                ok, reason = verifier.verify_digit(
                    carry_mask, digit_pos, target, prev_output_digits,
                )
                if not ok:
                    all_targets_ok = False
                    fail_reason = reason
                    break

            if not all_targets_ok:
                partition_ok = False
                break

            if len(possible_targets) == 1:
                prev_output_digits.append(possible_targets[0])
            else:
                prev_output_digits.append(-1)  # Varies

        if partition_ok:
            verified += 1
        else:
            inconclusive.append((carry_mask, fail_reason))

        if (verified + len(inconclusive)) % 128 == 0:
            logger.info(
                "Progress: %d verified, %d inconclusive / %d checked (%.1fs)",
                verified, len(inconclusive),
                verified + len(inconclusive), time.time() - start,
            )

    elapsed = time.time() - start

    if inconclusive:
        return SMTVerificationResult(
            status="INCONCLUSIVE",
            solve_time_seconds=elapsed,
            method=method_name,
            notes=[
                f"Verified {verified}/{total_partitions} carry partitions",
                f"{len(inconclusive)} inconclusive partitions",
                f"First failure: partition {inconclusive[0][0]:010b}: {inconclusive[0][1]}",
            ],
        )

    return SMTVerificationResult(
        status="PROVEN_CORRECT",
        solve_time_seconds=elapsed,
        method=method_name,
        notes=[f"All {total_partitions} carry partitions formally verified"],
    )


# ── Dispatcher to model-specific verifiers ─────────────────────────────

def verify_submission(
    submission_id: str,
    model: Any,
    module: Any,
    timeout_seconds: int = 3600,
) -> SMTVerificationResult:
    """Dispatch to the appropriate model-specific verifier."""

    if submission_id == "kswain98_8p":
        from .verifiers.kswain98 import create_verifier
        v = create_verifier(model)
        return verify_model(v, timeout_seconds, "structural_algebraic")

    elif submission_id == "yieldthought_20p":
        from .verifiers.yieldthought import create_verifier
        v = create_verifier(model)
        return verify_model(v, timeout_seconds, "structural_algebraic")

    elif submission_id == "SeuperHakkerJa_28p":
        from .verifiers.seuperhakkerja import create_verifier
        v = create_verifier(model)
        return verify_model(v, timeout_seconds, "structural_algebraic")

    elif submission_id == "fblissjr_33p":
        from .verifiers.fblissjr import create_verifier
        v = create_verifier(model)
        return verify_model(v, timeout_seconds, "structural_algebraic")

    elif submission_id == "lichengliu03_50p":
        from .verifiers.lichengliu03 import create_verifier
        v = create_verifier(model)
        return verify_model(v, timeout_seconds, "structural_algebraic")

    elif submission_id == "prasannakotyal_116p":
        from .verifiers.prasannakotyal import create_verifier
        v = create_verifier(model)
        return verify_model(v, timeout_seconds, "structural_algebraic")

    elif submission_id == "dimopep_140p":
        from .verifiers.dimopep import create_verifier
        v = create_verifier(model)
        return verify_model(v, timeout_seconds, "interval_propagation")

    else:
        return SMTVerificationResult(
            status="ERROR",
            notes=[f"No formal verifier available for {submission_id}"],
        )
