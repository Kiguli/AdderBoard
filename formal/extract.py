"""
Extract model weights, architecture, and tokenization from submissions
into a standardized ModelSpec representation.
"""

import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """Describes a single layer in the model."""
    name: str
    type: str  # "attention", "mlp", "norm", "embedding", "lm_head", "other"
    params: dict[str, np.ndarray] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """Standardized representation of an AdderBoard submission model."""
    submission_id: str
    author: str
    claimed_params: int
    category: str

    # Architecture
    num_layers: int = 0
    d_model: int = 0
    num_heads: int = 0
    head_dim: int = 0
    ff_dim: int = 0
    vocab_size: int = 0
    max_seq_len: int = 0
    activation: str = ""  # "relu", "gelu", "swiglu", "silu", etc.
    has_rope: bool = False
    has_alibi: bool = False
    has_rmsnorm: bool = False

    # Layers
    layers: list[LayerInfo] = field(default_factory=list)

    # All unique parameters as flat numpy arrays
    unique_params: dict[str, np.ndarray] = field(default_factory=dict)
    counted_params: int = 0

    # Raw module reference (for running inference)
    _module: Any = None
    _model: Any = None

    def total_scalar_params(self) -> int:
        """Count total unique scalar parameters."""
        return sum(p.size for p in self.unique_params.values())


def _load_module(submission_path: Path) -> Any:
    """Load a submission Python file as a module."""
    spec = importlib.util.spec_from_file_location("submission", str(submission_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {submission_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_pytorch_params(model: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """
    Extract unique parameters from a PyTorch model.
    Returns: (all_params, unique_params, unique_scalar_count)
    Handles weight tying by checking data_ptr().
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not available — cannot extract parameters")
        return {}, {}, 0

    all_params = {}
    seen_ptrs: dict[int, str] = {}  # data_ptr -> first name
    unique_params = {}

    if hasattr(model, "named_parameters"):
        param_iter = model.named_parameters()
    elif hasattr(model, "parameters"):
        param_iter = ((f"param_{i}", p) for i, p in enumerate(model.parameters()))
    else:
        logger.warning("Model has no parameters() method")
        return {}, {}, 0

    for name, param in param_iter:
        arr = param.detach().cpu().numpy()
        all_params[name] = arr

        if hasattr(param, "data_ptr"):
            ptr = param.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = name
                unique_params[name] = arr
            else:
                logger.debug("Tied: %s -> %s (same data_ptr)", name, seen_ptrs[ptr])
        else:
            # Non-tensor parameter — treat as unique
            unique_params[name] = arr

    unique_count = sum(p.size for p in unique_params.values())
    return all_params, unique_params, unique_count


def _detect_architecture(model: Any) -> dict[str, Any]:
    """Inspect a model to detect architectural features."""
    info: dict[str, Any] = {
        "has_attention": False,
        "has_rope": False,
        "has_alibi": False,
        "has_rmsnorm": False,
        "activation": "unknown",
        "num_layers": 0,
        "d_model": 0,
    }

    model_str = str(type(model))
    source = ""
    try:
        import inspect
        source = inspect.getsource(type(model))
    except (TypeError, OSError):
        pass

    # Check for attention
    if hasattr(model, "self_attn") or hasattr(model, "attention") or hasattr(model, "attn"):
        info["has_attention"] = True
    for name, mod in getattr(model, "named_modules", lambda: [])():
        mod_type = type(mod).__name__.lower()
        if "attention" in mod_type or "multihead" in mod_type:
            info["has_attention"] = True
        if "rmsnorm" in mod_type:
            info["has_rmsnorm"] = True
        if "rope" in mod_type or "rotary" in mod_type:
            info["has_rope"] = True

    # Check source code for patterns
    source_lower = source.lower()
    if "rope" in source_lower or "rotary" in source_lower:
        info["has_rope"] = True
    if "alibi" in source_lower:
        info["has_alibi"] = True
    if "rmsnorm" in source_lower or "rms_norm" in source_lower:
        info["has_rmsnorm"] = True
    if "swiglu" in source_lower or "silu" in source_lower:
        info["activation"] = "swiglu"
    elif "gelu" in source_lower:
        info["activation"] = "gelu"
    elif "relu" in source_lower:
        info["activation"] = "relu"

    return info


def extract(submission_path: Path, submission_id: str, author: str,
            claimed_params: int, category: str) -> ModelSpec:
    """
    Extract a ModelSpec from a submission file.
    Loads the model, extracts weights, detects architecture.
    """
    mod = _load_module(submission_path)

    if not hasattr(mod, "build_model"):
        raise ValueError(f"Submission {submission_id} missing build_model()")
    if not hasattr(mod, "add"):
        raise ValueError(f"Submission {submission_id} missing add()")

    model, metadata = mod.build_model()

    # Extract parameters
    all_params, unique_params, counted = _extract_pytorch_params(model)

    # Detect architecture
    arch_info = _detect_architecture(model)

    spec = ModelSpec(
        submission_id=submission_id,
        author=author,
        claimed_params=claimed_params,
        category=category,
        unique_params=unique_params,
        counted_params=counted,
        has_rope=arch_info.get("has_rope", False),
        has_alibi=arch_info.get("has_alibi", False),
        has_rmsnorm=arch_info.get("has_rmsnorm", False),
        activation=arch_info.get("activation", "unknown"),
        _module=mod,
        _model=model,
    )

    logger.info(
        "Extracted %s: claimed=%d, counted=%d unique scalars, arch=%s",
        submission_id, claimed_params, counted, arch_info,
    )
    return spec
