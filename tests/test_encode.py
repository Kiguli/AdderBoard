"""Tests for encode.py — verify SMT encoding is faithful."""

import sys
import pytest
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

z3 = pytest.importorskip("z3")
from formal.encode import (
    encode_linear,
    encode_relu,
    encode_argmax,
    encode_digit_input,
    encode_addition_spec,
)


def test_encode_linear_identity():
    """Identity matrix should preserve input."""
    solver = z3.Solver()
    x = [z3.Real("x0"), z3.Real("x1")]
    weight = np.eye(2)
    bias = np.zeros(2)

    out = encode_linear(solver, x, weight, bias, "id")

    # Set input
    solver.add(x[0] == 3.0, x[1] == 7.0)
    assert solver.check() == z3.sat

    model = solver.model()
    assert float(model[out[0]].as_fraction()) == pytest.approx(3.0)
    assert float(model[out[1]].as_fraction()) == pytest.approx(7.0)


def test_encode_linear_scale():
    """Diagonal weight matrix should scale inputs."""
    solver = z3.Solver()
    x = [z3.Real("x0"), z3.Real("x1")]
    weight = np.array([[2.0, 0.0], [0.0, 3.0]])
    bias = np.array([1.0, -1.0])

    out = encode_linear(solver, x, weight, bias, "scale")
    solver.add(x[0] == 5.0, x[1] == 4.0)
    assert solver.check() == z3.sat

    model = solver.model()
    # out[0] = 2*5 + 1 = 11, out[1] = 3*4 - 1 = 11
    assert float(model[out[0]].as_fraction()) == pytest.approx(11.0)
    assert float(model[out[1]].as_fraction()) == pytest.approx(11.0)


def test_encode_relu():
    """ReLU should zero out negatives."""
    solver = z3.Solver()
    x = [z3.Real("x0"), z3.Real("x1")]
    solver.add(x[0] == -3.0, x[1] == 5.0)

    out = encode_relu(solver, x, "relu")
    assert solver.check() == z3.sat

    model = solver.model()
    assert float(model[out[0]].as_fraction()) == pytest.approx(0.0)
    assert float(model[out[1]].as_fraction()) == pytest.approx(5.0)


def test_encode_argmax():
    """Argmax should return the index of the max element."""
    solver = z3.Solver()
    x = [z3.Real(f"x{i}") for i in range(4)]
    solver.add(x[0] == 1.0, x[1] == 5.0, x[2] == 3.0, x[3] == 2.0)

    idx = encode_argmax(solver, x, "am")
    assert solver.check() == z3.sat

    model = solver.model()
    assert model[idx].as_long() == 1  # x[1]=5 is the max


def test_digit_input_constraints():
    """Digit inputs should be constrained to [0, 9]."""
    solver = z3.Solver()
    a_digits, b_digits = encode_digit_input(solver, "test", num_digits=3)

    # Try to set a digit to 10 — should be unsat
    solver.push()
    solver.add(a_digits[0] == 10)
    assert solver.check() == z3.unsat
    solver.pop()

    # Set valid digits
    solver.add(a_digits[0] == 5, a_digits[1] == 3, a_digits[2] == 7)
    assert solver.check() == z3.sat


def test_addition_spec_finds_mismatch():
    """Spec should be satisfiable when output doesn't match a+b."""
    solver = z3.Solver()
    a_digits, b_digits = encode_digit_input(solver, "spec", num_digits=2)

    # a = 12, b = 34, so a+b = 46
    solver.add(a_digits[0] == 1, a_digits[1] == 2)
    solver.add(b_digits[0] == 3, b_digits[1] == 4)

    wrong_output = z3.Int("wrong_out")
    solver.add(wrong_output == 99)  # Wrong answer

    encode_addition_spec(solver, a_digits, b_digits, wrong_output, num_digits=2)

    # Should be SAT because 99 != 46
    assert solver.check() == z3.sat


def test_addition_spec_correct_unsat():
    """Spec should be unsatisfiable when output equals a+b."""
    solver = z3.Solver()
    a_digits, b_digits = encode_digit_input(solver, "spec2", num_digits=2)

    solver.add(a_digits[0] == 1, a_digits[1] == 2)
    solver.add(b_digits[0] == 3, b_digits[1] == 4)

    correct_output = z3.Int("correct_out")
    solver.add(correct_output == 46)  # Correct answer

    encode_addition_spec(solver, a_digits, b_digits, correct_output, num_digits=2)

    # Should be UNSAT because 46 == 46 (we assert output != a+b, but output IS a+b)
    assert solver.check() == z3.unsat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
