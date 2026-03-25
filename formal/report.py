"""
Generate verification reports: per-submission certificates and aggregate summary table.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import Submission, Category, ALL_SUBMISSIONS

logger = logging.getLogger(__name__)

PROOFS_DIR = Path(__file__).parent.parent / "proofs"
CERTS_DIR = PROOFS_DIR / "certificates"
CE_DIR = PROOFS_DIR / "counterexamples"
SUMMARY_PATH = PROOFS_DIR / "summary.md"


def _status_emoji(status: str) -> str:
    if "PROVEN" in status or "VERIFIED" in status:
        return "FORMALLY VERIFIED"
    elif "FALSIFIED" in status or "COUNTEREXAMPLE" in status:
        return "FALSIFIED"
    elif "TIMEOUT" in status:
        return "TIMEOUT"
    else:
        return "INCONCLUSIVE"


def generate_certificate(
    submission: Submission,
    param_result: Any,
    arch_result: Any,
    verification_result: Any,
    counterexample_analysis: Optional[Any] = None,
) -> Path:
    """Generate a per-submission verification report."""
    CERTS_DIR.mkdir(parents=True, exist_ok=True)

    status = getattr(verification_result, "status", "UNKNOWN")
    normalized = _status_emoji(status)
    filename = f"{submission.id}.md"
    path = CERTS_DIR / filename

    lines = []
    lines.append(f"# Formal Verification Report: {submission.author} — {submission.params} params ({submission.category.value})")
    lines.append("")

    # Result
    if normalized == "FORMALLY VERIFIED":
        lines.append(f"## Result: FORMALLY VERIFIED")
        lines.append(f"For all (a, b) in [0, 9999999999]^2, this model correctly computes a + b.")
    elif normalized == "FALSIFIED":
        lines.append(f"## Result: FALSIFIED")
    elif normalized == "TIMEOUT":
        lines.append(f"## Result: TIMEOUT")
        lines.append("Solver exceeded time limit before reaching a conclusion.")
    else:
        lines.append(f"## Result: INCONCLUSIVE")
    lines.append("")

    # Counterexample details (if falsified)
    if normalized == "FALSIFIED" and counterexample_analysis is not None:
        ce = counterexample_analysis
        a, b = ce.primary
        lines.append("## Counterexample")
        lines.append("")
        lines.append("| Input a | Input b | Expected (a+b) | Model Output |")
        lines.append("|---------|---------|----------------|--------------|")
        lines.append(f"| {a} | {b} | {ce.expected} | {ce.model_output} |")
        lines.append("")

        lines.append("## Failure Analysis")
        lines.append(f"- **Wrong digit(s)**: Positions {ce.wrong_digits} (from MSB)")
        lines.append(f"- **Failure type**: {ce.failure_type}")
        lines.append(f"- **Carry pattern**: {ce.carry_pattern}")
        lines.append(f"- **Pattern**: {ce.failure_pattern}")
        lines.append("")

        if ce.additional_counterexamples:
            lines.append("## Failure Region")
            lines.append(f"- Found {len(ce.additional_counterexamples)} additional failing inputs in neighborhood search")
            lines.append(f"- Estimated failure rate: {ce.estimated_failure_rate:.4%} of input space")
            lines.append("")

            lines.append("## Additional Counterexamples")
            lines.append("")
            lines.append("| a | b | Expected | Model Output |")
            lines.append("|---|---|----------|--------------|")
            for ca, cb, cexp, cout in ce.additional_counterexamples[:10]:
                lines.append(f"| {ca} | {cb} | {cexp} | {cout} |")
            lines.append("")
    elif normalized == "FALSIFIED" and verification_result is not None:
        ce = getattr(verification_result, "counterexample", None)
        if ce:
            a, b = ce
            expected = getattr(verification_result, "expected", a + b)
            actual = getattr(verification_result, "model_output", "?")
            lines.append("## Counterexample")
            lines.append("")
            lines.append("| Input a | Input b | Expected (a+b) | Model Output |")
            lines.append("|---------|---------|----------------|--------------|")
            lines.append(f"| {a} | {b} | {expected} | {actual} |")
            lines.append("")

    # Verification method
    lines.append("## Verification Method")
    method = getattr(verification_result, "method", "unknown")
    solve_time = getattr(verification_result, "solve_time_seconds", 0)
    lines.append(f"- Method: {method}")
    lines.append(f"- Solve time: {solve_time:.1f}s")
    for note in getattr(verification_result, "notes", []):
        lines.append(f"- {note}")
    lines.append("")

    # Parameter audit
    if param_result is not None:
        lines.append("## Parameter Audit")
        match_str = "Match" if param_result.match else "MISMATCH"
        lines.append(f"- Claimed: {param_result.claimed} | Counted: {param_result.counted} | {match_str}")
        if param_result.tied_groups:
            for group in param_result.tied_groups:
                lines.append(f"- Tied: {' = '.join(group)}")
        if param_result.excluded_pe:
            lines.append(f"- Excluded PE: {param_result.excluded_pe}")
        for note in param_result.notes:
            lines.append(f"- {note}")
        lines.append("")

    # Architecture compliance
    if arch_result is not None:
        lines.append("## Architecture Compliance")
        lines.append(f"- Self-attention: {'Yes' if arch_result.has_self_attention else 'No'}")
        lines.append(f"- forward() clean: {'Yes' if arch_result.forward_clean else 'No'}")
        lines.append(f"- add() clean: {'Yes' if arch_result.add_clean else 'No'}")
        lines.append(f"- Autoregressive: {'Yes' if arch_result.is_autoregressive else 'Unknown'}")
        lines.append(f"- Overall: {arch_result.overall}")
        for issue in arch_result.issues:
            lines.append(f"- Issue: {issue}")
        for warning in arch_result.warnings:
            lines.append(f"- Warning: {warning}")
        lines.append("")

    # Metadata
    lines.append("## Submission Info")
    lines.append(f"- Architecture: {submission.architecture}")
    lines.append(f"- Key tricks: {submission.key_tricks}")
    lines.append(f"- Link: {submission.link_url}")
    lines.append(f"- Verified: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Certificate written: %s", path)
    return path


def generate_summary_table(results: list[dict[str, Any]]) -> Path:
    """
    Generate the master verification table (proofs/summary.md).
    results: list of dicts with keys matching the table columns.
    """
    PROOFS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# AdderBoard Formal Verification Results")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Hand-coded section
    lines.append("## Hand-Coded Weights")
    lines.append("")
    lines.append("| Rank | Author | Params | Formal Status | Counterexample (a, b) | Expected | Model Output | Failure Type |")
    lines.append("|------|--------|--------|---------------|----------------------|----------|--------------|--------------|")

    hc_results = [r for r in results if r.get("category") == "hand_coded"]
    for r in sorted(hc_results, key=lambda x: x.get("params", 0)):
        status = _status_emoji(r.get("status", "UNKNOWN"))
        ce = r.get("counterexample", "—")
        expected = r.get("expected", "—")
        model_out = r.get("model_output", "—")
        failure = r.get("failure_type", "—")
        lines.append(
            f"| {r.get('rank', '?')} | {r.get('author', '?')} | {r.get('params', '?')} "
            f"| {status} | {ce} | {expected} | {model_out} | {failure} |"
        )

    lines.append("")

    # Trained section
    lines.append("## Trained Weights")
    lines.append("")
    lines.append("| Rank | Author | Params | Formal Status | Counterexample (a, b) | Expected | Model Output | Failure Type |")
    lines.append("|------|--------|--------|---------------|----------------------|----------|--------------|--------------|")

    tr_results = [r for r in results if r.get("category") == "trained"]
    for r in sorted(tr_results, key=lambda x: x.get("params", 0)):
        status = _status_emoji(r.get("status", "UNKNOWN"))
        ce = r.get("counterexample", "—")
        expected = r.get("expected", "—")
        model_out = r.get("model_output", "—")
        failure = r.get("failure_type", "—")
        lines.append(
            f"| {r.get('rank', '?')} | {r.get('author', '?')} | {r.get('params', '?')} "
            f"| {status} | {ce} | {expected} | {model_out} | {failure} |"
        )

    lines.append("")

    # Summary statistics
    total = len(results)
    verified = sum(1 for r in results if "PROVEN" in r.get("status", "") or "VERIFIED" in r.get("status", ""))
    falsified = sum(1 for r in results if "FALSIFIED" in r.get("status", "") or "COUNTEREXAMPLE" in r.get("status", ""))
    timeout = sum(1 for r in results if "TIMEOUT" in r.get("status", ""))
    inconclusive = total - verified - falsified - timeout

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total submissions**: {total}")
    lines.append(f"- **Formally verified**: {verified}")
    lines.append(f"- **Falsified**: {falsified}")
    lines.append(f"- **Timeout**: {timeout}")
    lines.append(f"- **Inconclusive**: {inconclusive}")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary table written: %s", SUMMARY_PATH)
    return SUMMARY_PATH
