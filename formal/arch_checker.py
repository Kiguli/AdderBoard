"""
Architecture compliance checker for AdderBoard submissions.
Verifies that submissions are genuine autoregressive transformers
with no task-specific heuristics in forward() or add().
"""

import ast
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Patterns forbidden inside forward()
FORBIDDEN_FORWARD_PATTERNS = {
    "carry": "Explicit carry variable",
    "digit_pair": "Digit pairing logic",
    "num_digits": "Digit count reference",
    "string_to_int": "String manipulation",
    "int_to_string": "String manipulation",
    "str(": "String conversion in computation",
}

# Patterns forbidden inside add() (direct arithmetic shortcuts)
FORBIDDEN_ADD_PATTERNS = {
    "a + b": "Direct Python addition",
    "a+b": "Direct Python addition",
    "operator.add": "Operator module addition",
    "sum(": "Built-in sum",
}


@dataclass
class ComplianceResult:
    """Result of architecture compliance check."""
    has_self_attention: bool = False
    forward_clean: bool = False
    add_clean: bool = False
    is_autoregressive: bool = False
    overall: str = "UNKNOWN"  # "PASS", "FAIL", "REVIEW_NEEDED"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ast_findings: list[str] = field(default_factory=list)


class _ForbiddenPatternVisitor(ast.NodeVisitor):
    """AST visitor that looks for forbidden patterns in function bodies."""

    def __init__(self):
        self.findings: list[str] = []
        self._current_func: str = ""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old = self._current_func
        self._current_func = node.name
        self.generic_visit(node)
        self._current_func = old

    # Check for explicit carry variables
    def visit_Assign(self, node: ast.Assign):
        if self._current_func == "forward":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id.lower()
                    if "carry" in name:
                        self.findings.append(
                            f"forward(): Suspicious variable '{target.id}' (looks like explicit carry state)"
                        )
                    if "digit" in name and ("index" in name or "pair" in name):
                        self.findings.append(
                            f"forward(): Suspicious variable '{target.id}' (looks like digit indexing)"
                        )
        self.generic_visit(node)

    # Check for digit-specific conditionals in forward
    def visit_Compare(self, node: ast.Compare):
        if self._current_func == "forward":
            # Check for comparisons like `if digit == 9`
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int):
                    if 0 <= comparator.value <= 9:
                        # Could be digit-specific branching
                        self.findings.append(
                            f"forward(): Comparison with digit constant {comparator.value}"
                        )
        self.generic_visit(node)

    # Check for eval/exec
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec"):
                self.findings.append(
                    f"{self._current_func}(): Uses {node.func.id}() — dynamic code execution"
                )
        self.generic_visit(node)

    # Check for direct a+b in add()
    def visit_BinOp(self, node: ast.BinOp):
        if self._current_func == "add" and isinstance(node.op, ast.Add):
            # Check if this looks like a + b (the function parameters)
            if (isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)):
                if {node.left.id, node.right.id} == {"a", "b"}:
                    self.findings.append(
                        "add(): Direct 'a + b' arithmetic — model not doing the work"
                    )
        self.generic_visit(node)


def _check_ast(source_code: str) -> list[str]:
    """Parse source code and check for forbidden patterns via AST."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return [f"AST parse error: {e}"]

    visitor = _ForbiddenPatternVisitor()
    visitor.visit(tree)
    return visitor.findings


def _check_self_attention(model: Any) -> bool:
    """Check if model contains a self-attention mechanism."""
    # Check named modules
    if hasattr(model, "named_modules"):
        for name, mod in model.named_modules():
            mod_type = type(mod).__name__.lower()
            if any(kw in mod_type for kw in ("attention", "multihead", "selfattn", "mha")):
                return True

    # Check source code for attention patterns
    try:
        source = inspect.getsource(type(model))
        source_lower = source.lower()
        attention_markers = [
            "softmax", "attention", "q_proj", "k_proj", "v_proj",
            "query", "key", "value", "attn_weight",
        ]
        if sum(1 for m in attention_markers if m in source_lower) >= 2:
            return True
    except (TypeError, OSError):
        pass

    return False


def _check_autoregressive(source_code: str) -> tuple[bool, list[str]]:
    """
    Check if the add() function uses autoregressive generation.
    Look for: token-by-token loop, feeding output back as input.
    """
    warnings = []
    source_lower = source_code.lower()

    # Look for generation loop patterns
    has_loop = "for " in source_lower or "while " in source_lower
    has_argmax = "argmax" in source_lower
    has_token_append = any(kw in source_lower for kw in ("append", "cat(", "concat", "torch.cat"))

    if has_loop and (has_argmax or has_token_append):
        return True, warnings

    # Could be a single-pass model or unusual generation
    if not has_loop:
        warnings.append("No generation loop detected in add() — may not be autoregressive")

    return bool(has_loop), warnings


def check_compliance(model: Any, source_path: Path) -> ComplianceResult:
    """
    Run all architecture compliance checks on a submission.
    """
    result = ComplianceResult()

    # Read source
    try:
        source_code = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        result.issues.append(f"Cannot read source: {e}")
        result.overall = "FAIL"
        return result

    # 1. Check for self-attention
    result.has_self_attention = _check_self_attention(model)
    if not result.has_self_attention:
        result.issues.append("No self-attention mechanism detected")

    # 2. AST analysis for forbidden patterns
    ast_findings = _check_ast(source_code)
    result.ast_findings = ast_findings
    if ast_findings:
        result.warnings.extend(ast_findings)

    # 3. Check forward() is clean
    critical_findings = [f for f in ast_findings if "forward()" in f and "carry" in f.lower()]
    result.forward_clean = len(critical_findings) == 0
    if not result.forward_clean:
        result.issues.append("forward() contains task-specific logic")

    # 4. Check add() doesn't use direct arithmetic
    add_direct = [f for f in ast_findings if "add():" in f and "Direct" in f]
    result.add_clean = len(add_direct) == 0
    if not result.add_clean:
        result.issues.append("add() uses direct arithmetic instead of model")

    # 5. Check autoregressive generation
    is_ar, ar_warnings = _check_autoregressive(source_code)
    result.is_autoregressive = is_ar
    result.warnings.extend(ar_warnings)

    # Determine overall status
    if not result.has_self_attention:
        result.overall = "FAIL"
    elif not result.forward_clean or not result.add_clean:
        result.overall = "FAIL"
    elif result.warnings:
        result.overall = "REVIEW_NEEDED"
    else:
        result.overall = "PASS"

    return result
