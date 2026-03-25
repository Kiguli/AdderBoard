"""Tests for unroll.py — verify autoregressive unrolling."""

import sys
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from formal.verify_exhaustive import _carry_pattern_for, _representative_inputs_for_carry_pattern


def test_carry_pattern_no_carry():
    """1 + 1 = 2, no carries."""
    assert _carry_pattern_for(1, 1) == 0


def test_carry_pattern_simple():
    """5 + 5 = 10, carry at rightmost digit (units place = bit 0)."""
    pattern = _carry_pattern_for(5, 5)
    # Units digit: 5+5=10, carry. Bit 0 should be set.
    assert pattern & 1 == 1


def test_carry_pattern_full():
    """9999999999 + 1 = 10000000000, carries propagate all the way."""
    pattern = _carry_pattern_for(9999999999, 1)
    # Units: 9+1=10 carry, tens: 9+0+1=10 carry, ..., all 10 positions carry
    assert pattern == 0b1111111111  # All 10 bits set


def test_carry_pattern_max():
    """9999999999 + 9999999999 = 19999999998."""
    pattern = _carry_pattern_for(9999999999, 9999999999)
    # All digit pairs sum to 18 or 19, all carry
    assert pattern == 0b1111111111


def test_representative_inputs():
    """Generated inputs should match their target carry pattern."""
    for mask in [0, 1, 0b1111111111, 0b1010101010, 0b0000011111]:
        pairs = _representative_inputs_for_carry_pattern(mask, count=5)
        for a, b in pairs:
            assert 0 <= a <= 9999999999
            assert 0 <= b <= 9999999999
            assert _carry_pattern_for(a, b) == mask, (
                f"Carry pattern mismatch for ({a}, {b}): "
                f"expected {mask:010b}, got {_carry_pattern_for(a, b):010b}"
            )


def test_zero_inputs():
    """0 + 0 has no carries."""
    assert _carry_pattern_for(0, 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
