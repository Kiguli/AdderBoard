"""
Tests using toy models with known properties to verify the framework.
"""

import sys
import pytest
import math

torch = pytest.importorskip("torch")
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from formal.verify_exhaustive import verify_exhaustive, verify_boundary_cases


class PerfectAdder:
    """A 'model' that always returns the correct answer (uses Python addition)."""
    pass


class BrokenAdder:
    """A 'model' that fails on long carry chains (>= 5 consecutive carries)."""
    pass


class _PerfectModule:
    """Module with build_model() and add() for the perfect adder."""
    @staticmethod
    def build_model():
        return PerfectAdder(), {"name": "perfect", "author": "test", "params": 0}

    @staticmethod
    def add(model, a, b):
        return a + b


class _BrokenModule:
    """Module that fails on carry chains >= 5."""
    @staticmethod
    def build_model():
        return BrokenAdder(), {"name": "broken", "author": "test", "params": 0}

    @staticmethod
    def add(model, a, b):
        correct = a + b
        # Simulate carry propagation failure
        carry_chain = 0
        max_chain = 0
        carry = 0
        for pos in range(10):
            a_d = (a // (10 ** pos)) % 10
            b_d = (b // (10 ** pos)) % 10
            if a_d + b_d + carry >= 10:
                carry_chain += 1
                carry = 1
            else:
                max_chain = max(max_chain, carry_chain)
                carry_chain = 0
                carry = 0
        max_chain = max(max_chain, carry_chain)

        if max_chain >= 5:
            # Drop the carry at position 5 — return wrong answer
            return correct - 100000  # Off by 10^5
        return correct


def test_perfect_adder_passes():
    """A correct model should pass exhaustive verification."""
    mod = _PerfectModule()
    model = PerfectAdder()
    result = verify_exhaustive(mod, model, "test_perfect", samples_per_partition=10)
    assert result.status == "PROVEN_CORRECT"
    assert result.partitions_verified == 1024


def test_broken_adder_caught():
    """A model with carry chain failures should be caught."""
    mod = _BrokenModule()
    model = BrokenAdder()
    result = verify_exhaustive(mod, model, "test_broken", samples_per_partition=50)
    # Should find a counterexample in partitions with long carry chains
    assert result.status == "COUNTEREXAMPLE_FOUND"
    assert result.counterexample is not None
    a, b = result.counterexample
    assert mod.add(model, a, b) != a + b


def test_boundary_cases_perfect():
    """Perfect adder should pass boundary checks."""
    mod = _PerfectModule()
    model = PerfectAdder()
    result = verify_boundary_cases(mod, model, "test_boundary")
    assert result.status == "PROVEN_CORRECT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
