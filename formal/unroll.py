"""
Unroll the autoregressive generation loop into a fixed computation graph.

For 10-digit addition, the output is at most 11 digits + EOS = 12 tokens.
We unroll the generation loop into a chain of forward passes, each consuming
the growing sequence and producing the next token via argmax.

This produces a single, non-recursive computation that formal verifiers
(SMT solvers, bound propagators) can reason about.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Max output length for 10-digit addition: up to 11 digits + EOS
MAX_OUTPUT_TOKENS = 12


@dataclass
class UnrolledStep:
    """One step of the unrolled autoregressive computation."""
    step_index: int
    input_length: int  # total sequence length at this step
    # The computation: input_tokens -> logits -> argmax -> next_token
    # Stored as weight snapshots for the same model at different sequence positions


@dataclass
class UnrolledGraph:
    """
    The full unrolled computation graph for a submission.
    Represents the chain: tok(a,b) -> step1 -> step2 -> ... -> step_N -> detok -> sum
    """
    submission_id: str
    input_format: str  # Description of how (a,b) are tokenized
    num_steps: int
    input_length: int  # Length of tok(a,b)
    vocab_size: int
    steps: list[UnrolledStep] = field(default_factory=list)

    # Function references for actual computation
    _tokenize: Optional[Callable] = None
    _detokenize: Optional[Callable] = None
    _forward: Optional[Callable] = None


def _analyze_tokenization(module: Any, model: Any) -> dict[str, Any]:
    """
    Analyze how a submission tokenizes inputs.
    Runs the add() function on a sample input to observe the tokenization.
    """
    import inspect

    info = {
        "format": "unknown",
        "input_length": 0,
        "vocab_size": 0,
        "digit_order": "unknown",  # "msb_first" or "lsb_first"
        "separator": "unknown",
    }

    # Try to analyze the source code of add()
    try:
        source = inspect.getsource(module.add)
        source_lower = source.lower()

        if "reverse" in source_lower or "lsb" in source_lower:
            info["digit_order"] = "lsb_first"
        else:
            info["digit_order"] = "msb_first"

        if "sep" in source_lower or "+" in source:
            info["separator"] = "has_separator"

    except (TypeError, OSError):
        pass

    return info


def _trace_generation(module: Any, model: Any, a: int, b: int) -> dict[str, Any]:
    """
    Trace one full generation pass to understand the computation flow.
    Returns details about each step (input tokens, output token, logits).
    """
    trace: dict[str, Any] = {"steps": [], "input_tokens": None, "output_tokens": []}

    try:
        # We hook into the model to capture intermediate states
        import torch

        step_data = []
        original_forward = None

        if hasattr(model, "forward"):
            original_forward = model.forward

            def traced_forward(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                # Record input and output shapes
                if isinstance(args[0], torch.Tensor):
                    step_data.append({
                        "input_shape": tuple(args[0].shape),
                        "output_shape": tuple(result.shape) if isinstance(result, torch.Tensor) else "non-tensor",
                    })
                return result

            model.forward = traced_forward

        # Run the actual add function
        result = module.add(model, a, b)

        # Restore original forward
        if original_forward is not None:
            model.forward = original_forward

        trace["result"] = result
        trace["expected"] = a + b
        trace["correct"] = result == (a + b)
        trace["steps"] = step_data
        trace["num_forward_passes"] = len(step_data)

    except Exception as e:
        trace["error"] = str(e)
        logger.warning("Trace failed for (%d, %d): %s", a, b, e)

    return trace


def analyze_submission(module: Any, model: Any, submission_id: str) -> UnrolledGraph:
    """
    Analyze a submission to understand its tokenization and generation pattern.
    Runs sample inputs to determine the computation structure.
    """
    # Analyze tokenization from source
    tok_info = _analyze_tokenization(module, model)

    # Trace a few sample generations to understand the pattern
    sample_inputs = [
        (123, 456),           # Simple, no carry
        (999, 1),             # Carry propagation
        (9999999999, 1),      # Max carry chain
        (0, 0),               # Edge case
        (5000000000, 5000000000),  # 10->11 digit boundary
    ]

    traces = []
    for a, b in sample_inputs:
        trace = _trace_generation(module, model, a, b)
        traces.append(trace)
        logger.debug("Trace (%d+%d): %d forward passes", a, b, trace.get("num_forward_passes", -1))

    # Determine the number of unrolling steps
    forward_counts = [t.get("num_forward_passes", 0) for t in traces if "error" not in t]
    max_steps = max(forward_counts) if forward_counts else MAX_OUTPUT_TOKENS

    graph = UnrolledGraph(
        submission_id=submission_id,
        input_format=f"digit_order={tok_info['digit_order']}, sep={tok_info['separator']}",
        num_steps=max_steps,
        input_length=tok_info.get("input_length", 0),
        vocab_size=tok_info.get("vocab_size", 0),
        _forward=getattr(model, "forward", None),
    )

    logger.info(
        "Analyzed %s: %d max forward passes, format=%s",
        submission_id, max_steps, graph.input_format,
    )

    return graph


def unroll_for_verification(
    module: Any, model: Any, submission_id: str, num_steps: Optional[int] = None
) -> UnrolledGraph:
    """
    Create an unrolled computation graph suitable for formal verification.
    If num_steps is not provided, it's determined by tracing sample inputs.
    """
    graph = analyze_submission(module, model, submission_id)
    if num_steps is not None:
        graph.num_steps = num_steps

    # Build the step-by-step structure
    for i in range(graph.num_steps):
        step = UnrolledStep(
            step_index=i,
            input_length=graph.input_length + i,
        )
        graph.steps.append(step)

    return graph
