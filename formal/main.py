"""
CLI entry point for the AdderBoard formal verification framework.

Usage:
    python -m formal.main fetch --all
    python -m formal.main prereq --all
    python -m formal.main verify --submission "zcbtrak_6p" --tier 1
    python -m formal.main verify --category hand_coded
    python -m formal.main verify --all
    python -m formal.main report
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    ALL_SUBMISSIONS, HAND_CODED, TRAINED,
    Category, VerificationTier,
    get_submission, get_by_tier, get_by_category,
)

logger = logging.getLogger("formal")


def cmd_fetch(args):
    """Fetch submissions from GitHub."""
    from .fetch import fetch_all, fetch_submission, get_cached_path

    if args.all:
        results = fetch_all(force=args.force)
        ok = sum(1 for v in results.values() if v is not None)
        print(f"Fetched {ok}/{len(results)} submissions")
    elif args.submission:
        sub = get_submission(args.submission)
        if sub is None:
            print(f"Unknown submission: {args.submission}")
            sys.exit(1)
        path = fetch_submission(sub, force=args.force)
        print(f"{'OK' if path else 'FAILED'}: {path or 'could not fetch'}")
    else:
        print("Specify --all or --submission <id>")


def cmd_prereq(args):
    """Run prerequisite checks (param count + arch compliance)."""
    from .fetch import get_cached_path
    from .extract import extract
    from .param_counter import count_params
    from .arch_checker import check_compliance

    submissions = _resolve_submissions(args)

    for sub in submissions:
        path = get_cached_path(sub)
        if path is None:
            print(f"[SKIP] {sub.id}: not fetched")
            continue

        print(f"\n{'='*60}")
        print(f"Checking: {sub.id} ({sub.params}p, {sub.category.value})")
        print(f"{'='*60}")

        try:
            spec = extract(path, sub.id, sub.author, sub.params, sub.category.value)

            # Param count
            param_result = count_params(spec._model, sub.params)
            match_str = "MATCH" if param_result.match else "MISMATCH"
            print(f"  Params: claimed={param_result.claimed}, counted={param_result.counted} [{match_str}]")
            for note in param_result.notes:
                print(f"    {note}")

            # Architecture
            arch_result = check_compliance(spec._model, path)
            print(f"  Architecture: {arch_result.overall}")
            for issue in arch_result.issues:
                print(f"    Issue: {issue}")
            for warning in arch_result.warnings:
                print(f"    Warning: {warning}")

        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_verify(args):
    """Run formal verification."""
    from .fetch import get_cached_path
    from .extract import extract
    from .param_counter import count_params
    from .arch_checker import check_compliance
    from .verify_exhaustive import verify_exhaustive, verify_boundary_cases
    from .verify_smt import verify_full as verify_smt
    from .verify_bounds import verify_by_region
    from .counterexample import analyze_counterexample, confirm_counterexample
    from .report import generate_certificate

    submissions = _resolve_submissions(args)
    all_results = []

    for sub in submissions:
        path = get_cached_path(sub)
        if path is None:
            print(f"[SKIP] {sub.id}: not fetched — run 'fetch' first")
            all_results.append({
                "id": sub.id, "author": sub.author, "params": sub.params,
                "rank": sub.rank, "category": sub.category.value,
                "status": "INCONCLUSIVE", "notes": "Not fetched",
            })
            continue

        print(f"\n{'='*60}")
        print(f"Verifying: {sub.id} ({sub.params}p, {sub.category.value}, tier {sub.tier.value})")
        print(f"{'='*60}")

        try:
            spec = extract(path, sub.id, sub.author, sub.params, sub.category.value)
            param_result = count_params(spec._model, sub.params)
            arch_result = check_compliance(spec._model, path)

            # Select verification tier
            tier = args.tier if args.tier else sub.tier.value

            if tier == 1:
                print("  Running Tier 1: Exhaustive carry-pattern verification...")
                vresult = verify_exhaustive(spec._module, spec._model, sub.id)
            elif tier == 2:
                print("  Running Tier 2: SMT verification...")
                vresult = verify_smt(spec, timeout_seconds=args.timeout)
            else:
                print("  Running Tier 3: Bound propagation verification...")
                vresult = verify_by_region(spec._model, spec._module, sub.id, timeout_seconds=args.timeout)

            # Analyze counterexample if found
            ce_analysis = None
            if hasattr(vresult, "counterexample") and vresult.counterexample:
                a, b = vresult.counterexample
                expected = getattr(vresult, "expected", a + b)
                actual = getattr(vresult, "model_output", None)
                if actual is not None:
                    print(f"  COUNTEREXAMPLE: {a} + {b} = {expected}, model says {actual}")
                    ce_analysis = analyze_counterexample(
                        spec._module, spec._model, a, b, expected, actual
                    )

            # Generate certificate
            cert_path = generate_certificate(sub, param_result, arch_result, vresult, ce_analysis)

            status = getattr(vresult, "status", "UNKNOWN")
            solve_time = getattr(vresult, "solve_time_seconds", 0)
            print(f"  Result: {status} ({solve_time:.1f}s)")
            print(f"  Certificate: {cert_path}")

            result_dict = {
                "id": sub.id, "author": sub.author, "params": sub.params,
                "rank": sub.rank, "category": sub.category.value,
                "status": status,
            }
            if ce_analysis:
                a, b = ce_analysis.primary
                result_dict["counterexample"] = f"({a}, {b})"
                result_dict["expected"] = str(ce_analysis.expected)
                result_dict["model_output"] = str(ce_analysis.model_output)
                result_dict["failure_type"] = ce_analysis.failure_type
            all_results.append(result_dict)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "id": sub.id, "author": sub.author, "params": sub.params,
                "rank": sub.rank, "category": sub.category.value,
                "status": "ERROR", "notes": str(e),
            })

    # Generate summary if we verified multiple
    if len(all_results) > 1:
        from .report import generate_summary_table
        summary_path = generate_summary_table(all_results)
        print(f"\nSummary table: {summary_path}")


def cmd_report(args):
    """Generate aggregate report from existing certificates."""
    from .report import generate_summary_table
    # Scan certificates directory and build results
    certs_dir = Path(__file__).parent.parent / "proofs" / "certificates"
    if not certs_dir.exists():
        print("No certificates found — run 'verify' first")
        return

    results = []
    for sub in ALL_SUBMISSIONS:
        cert = certs_dir / f"{sub.id}.md"
        if cert.exists():
            content = cert.read_text()
            status = "UNKNOWN"
            if "FORMALLY VERIFIED" in content:
                status = "PROVEN_CORRECT"
            elif "FALSIFIED" in content:
                status = "COUNTEREXAMPLE_FOUND"
            elif "TIMEOUT" in content:
                status = "TIMEOUT"
            results.append({
                "id": sub.id, "author": sub.author, "params": sub.params,
                "rank": sub.rank, "category": sub.category.value,
                "status": status,
            })
        else:
            results.append({
                "id": sub.id, "author": sub.author, "params": sub.params,
                "rank": sub.rank, "category": sub.category.value,
                "status": "INCONCLUSIVE",
            })

    summary_path = generate_summary_table(results)
    print(f"Summary: {summary_path}")


def _resolve_submissions(args):
    """Resolve which submissions to process from CLI args."""
    if getattr(args, "all", False):
        return ALL_SUBMISSIONS
    elif getattr(args, "submission", None):
        sub = get_submission(args.submission)
        if sub is None:
            print(f"Unknown submission: {args.submission}")
            sys.exit(1)
        return [sub]
    elif getattr(args, "category", None):
        cat = Category(args.category)
        return get_by_category(cat)
    elif getattr(args, "tier", None):
        return get_by_tier(VerificationTier(args.tier))
    else:
        print("Specify --all, --submission <id>, --category <cat>, or --tier <n>")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="formal",
        description="AdderBoard Formal Verification Framework",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Download submissions")
    p_fetch.add_argument("--all", action="store_true")
    p_fetch.add_argument("--submission", type=str)
    p_fetch.add_argument("--force", action="store_true")

    # prereq
    p_prereq = subparsers.add_parser("prereq", help="Run prerequisite checks")
    p_prereq.add_argument("--all", action="store_true")
    p_prereq.add_argument("--submission", type=str)
    p_prereq.add_argument("--category", type=str)

    # verify
    p_verify = subparsers.add_parser("verify", help="Run formal verification")
    p_verify.add_argument("--all", action="store_true")
    p_verify.add_argument("--submission", type=str)
    p_verify.add_argument("--category", type=str)
    p_verify.add_argument("--tier", type=int, choices=[1, 2, 3])
    p_verify.add_argument("--timeout", type=int, default=3600, help="Timeout per submission (seconds)")

    # report
    p_report = subparsers.add_parser("report", help="Generate aggregate report")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "prereq":
        cmd_prereq(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
