"""
Verify parameter counts for AdderBoard submissions.
Counts unique parameters after weight tying/deduplication,
excluding fixed positional encodings per contest rules.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParamCountResult:
    """Result of parameter counting verification."""
    claimed: int
    counted: int
    match: bool
    breakdown: dict[str, int]  # param_name -> scalar count
    tied_groups: list[list[str]]  # groups of param names sharing storage
    excluded_pe: list[str]  # param names excluded as fixed positional encodings
    notes: list[str]


def _is_fixed_positional_encoding(name: str, param: np.ndarray, model: Any) -> bool:
    """
    Heuristic: detect if a parameter is a fixed (non-learned) positional encoding.
    Fixed/sinusoidal PE should NOT be counted per AdderBoard rules.
    Learned PE SHOULD be counted.
    """
    name_lower = name.lower()

    # RoPE frequencies are typically named inv_freq, freqs, theta
    # These are fixed and should not be counted
    if any(kw in name_lower for kw in ("inv_freq", "rope_freq", "cos_cached", "sin_cached")):
        return True

    # Check if it's registered as a buffer (not a parameter)
    try:
        import torch
        if hasattr(model, "named_buffers"):
            buffer_names = {n for n, _ in model.named_buffers()}
            if name in buffer_names:
                return True
    except ImportError:
        pass

    # Sinusoidal pattern detection: check if values match sin/cos pattern
    if param.ndim == 2 and "position" in name_lower and "embed" in name_lower:
        # Check if it's a sinusoidal table (non-learned)
        if not _param_requires_grad(name, model):
            return True

    return False


def _param_requires_grad(name: str, model: Any) -> bool:
    """Check if a named parameter requires gradient (i.e., is learned)."""
    try:
        import torch
        for pname, param in model.named_parameters():
            if pname == name:
                return param.requires_grad
    except (ImportError, StopIteration):
        pass
    return True  # Default: assume it's learned


def count_params(model: Any, claimed: int) -> ParamCountResult:
    """
    Count unique parameters in a model and compare to claimed count.

    Handles:
    - Weight tying (shared data_ptr in PyTorch)
    - Fixed positional encodings (excluded per rules)
    - Hardcoded constants registered as parameters
    """
    try:
        import torch
    except ImportError:
        return ParamCountResult(
            claimed=claimed, counted=-1, match=False,
            breakdown={}, tied_groups=[], excluded_pe=[],
            notes=["PyTorch not available — cannot count parameters"],
        )

    seen_ptrs: dict[int, list[str]] = {}
    unique_params: dict[str, np.ndarray] = {}
    excluded_pe: list[str] = []
    notes: list[str] = []

    # Collect all named parameters
    if hasattr(model, "named_parameters"):
        all_named = list(model.named_parameters())
    else:
        notes.append("Model has no named_parameters() — cannot enumerate")
        return ParamCountResult(
            claimed=claimed, counted=-1, match=False,
            breakdown={}, tied_groups=[], excluded_pe=[], notes=notes,
        )

    for name, param in all_named:
        ptr = param.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs[ptr] = [name]
        else:
            seen_ptrs[ptr].append(name)

    # Identify tied groups and unique parameters
    tied_groups = [names for names in seen_ptrs.values() if len(names) > 1]

    for ptr, names in seen_ptrs.items():
        canonical = names[0]
        param_tensor = None
        for pname, p in all_named:
            if pname == canonical:
                param_tensor = p
                break

        if param_tensor is None:
            continue

        arr = param_tensor.detach().cpu().numpy()

        # Check if this is a fixed positional encoding
        if _is_fixed_positional_encoding(canonical, arr, model):
            excluded_pe.append(canonical)
            if len(names) > 1:
                notes.append(f"Excluded PE group: {names}")
            continue

        unique_params[canonical] = arr

    # Count unique scalars
    breakdown = {name: arr.size for name, arr in unique_params.items()}
    counted = sum(breakdown.values())

    if tied_groups:
        for group in tied_groups:
            notes.append(f"Tied: {' = '.join(group)}")

    if excluded_pe:
        notes.append(f"Excluded as fixed PE: {excluded_pe}")

    match = counted == claimed
    if not match:
        notes.append(f"MISMATCH: claimed {claimed}, counted {counted} (diff={counted - claimed})")

    return ParamCountResult(
        claimed=claimed,
        counted=counted,
        match=match,
        breakdown=breakdown,
        tied_groups=tied_groups,
        excluded_pe=excluded_pe,
        notes=notes,
    )
