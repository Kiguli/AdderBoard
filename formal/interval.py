"""
Rigorous interval arithmetic for formal verification of neural networks.

Each value is represented as a closed interval [lo, hi] that is guaranteed
to contain the true value. All operations produce sound over-approximations:
the result interval always contains every possible output.

This module provides interval versions of the operations used in the
zcbtrak-family AdderBoard models: embedding lookup, RMSNorm, RoPE attention,
SiLU gating, and linear projections.
"""

import math
from typing import Optional


class Interval:
    """
    A closed real interval [lo, hi] with rigorous arithmetic.

    All operations return intervals that are guaranteed to contain
    every possible result — i.e., they are sound over-approximations.
    """
    __slots__ = ('lo', 'hi')

    def __init__(self, lo: float, hi: Optional[float] = None):
        if hi is None:
            hi = lo
        assert lo <= hi, f"Invalid interval: [{lo}, {hi}]"
        self.lo = float(lo)
        self.hi = float(hi)

    # ── Arithmetic ────────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Interval(self.lo + other, self.hi + other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Interval(self.lo - other, self.hi - other)
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Interval(other - self.hi, other - self.lo)
        return other.__sub__(self)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other >= 0:
                return Interval(self.lo * other, self.hi * other)
            else:
                return Interval(self.hi * other, self.lo * other)
        products = [
            self.lo * other.lo, self.lo * other.hi,
            self.hi * other.lo, self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other > 0:
                return Interval(self.lo / other, self.hi / other)
            elif other < 0:
                return Interval(self.hi / other, self.lo / other)
            else:
                raise ZeroDivisionError("Division by zero scalar")
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError(f"Division by interval containing zero: {other}")
        return self * Interval(1.0 / other.hi, 1.0 / other.lo)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Interval(other, other) / self
        return other / self

    def __pow__(self, n):
        if isinstance(n, int) and n == 2:
            return self.square()
        raise NotImplementedError(f"Only integer power 2 is supported, got {n}")

    def square(self):
        """x² with proper handling of intervals crossing zero."""
        if self.lo >= 0:
            return Interval(self.lo ** 2, self.hi ** 2)
        elif self.hi <= 0:
            return Interval(self.hi ** 2, self.lo ** 2)
        else:
            return Interval(0.0, max(self.lo ** 2, self.hi ** 2))

    # ── Comparisons (for bound checking, not for branching) ──────────

    def strictly_positive(self) -> bool:
        return self.lo > 0

    def strictly_negative(self) -> bool:
        return self.hi < 0

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0

    def __repr__(self):
        return f"[{self.lo:.6g}, {self.hi:.6g}]"

    # ── Set operations ────────────────────────────────────────────────

    def hull(self, other: 'Interval') -> 'Interval':
        """Smallest interval containing both."""
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def intersect(self, other: 'Interval') -> Optional['Interval']:
        """Intersection, or None if disjoint."""
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return None
        return Interval(lo, hi)


# ── Transcendental functions ──────────────────────────────────────────────

def iv_exp(x: Interval) -> Interval:
    """exp(x) — monotonically increasing, so just evaluate at endpoints."""
    lo = math.exp(max(x.lo, -700))   # Avoid underflow
    hi = math.exp(min(x.hi, 700))    # Avoid overflow
    return Interval(lo, hi)


def iv_sqrt(x: Interval) -> Interval:
    """sqrt(x) for non-negative intervals."""
    assert x.lo >= 0 or x.lo > -1e-15, f"sqrt of negative interval: {x}"
    return Interval(math.sqrt(max(0.0, x.lo)), math.sqrt(x.hi))


def iv_cos(x: Interval) -> Interval:
    """
    cos(x) with rigorous bounds.
    For narrow intervals (width < pi), evaluates endpoints + checks for extrema.
    """
    if x.width > 2 * math.pi:
        return Interval(-1.0, 1.0)

    c_lo = math.cos(x.lo)
    c_hi = math.cos(x.hi)
    lo = min(c_lo, c_hi)
    hi = max(c_lo, c_hi)

    # Check if a multiple of pi is inside the interval (cos extrema)
    # cos has maxima at 2k*pi and minima at (2k+1)*pi
    k_start = math.ceil(x.lo / math.pi)
    k_end = math.floor(x.hi / math.pi)
    for k in range(int(k_start), int(k_end) + 1):
        pt = k * math.pi
        if x.lo <= pt <= x.hi:
            c = math.cos(pt)
            lo = min(lo, c)
            hi = max(hi, c)

    return Interval(lo, hi)


def iv_sin(x: Interval) -> Interval:
    """sin(x) with rigorous bounds."""
    # sin(x) = cos(x - pi/2)
    return iv_cos(Interval(x.lo - math.pi / 2, x.hi - math.pi / 2))


def iv_sigmoid(x: Interval) -> Interval:
    """sigmoid(x) = 1/(1+exp(-x)) — monotonically increasing."""
    def _sig(v):
        if v > 500:
            return 1.0
        if v < -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-v))
    return Interval(_sig(x.lo), _sig(x.hi))


def iv_silu(x: Interval) -> Interval:
    """
    SiLU(x) = x * sigmoid(x).

    SiLU has a global minimum at x ≈ -1.2785 where SiLU ≈ -0.2785.
    It is monotonically decreasing for x < -1.2785 and increasing for x > -1.2785.
    """
    SILU_MIN_X = -1.2784645427610737

    def _silu(v):
        if v > 500:
            return v
        if v < -500:
            return 0.0
        return v / (1.0 + math.exp(-v))

    vals = [_silu(x.lo), _silu(x.hi)]
    # Check if the minimum is inside the interval
    if x.lo <= SILU_MIN_X <= x.hi:
        vals.append(_silu(SILU_MIN_X))

    return Interval(min(vals), max(vals))


def iv_max(a: Interval, b: float) -> Interval:
    """max(a, b) where b is a constant."""
    return Interval(max(a.lo, b), max(a.hi, b))


# ── Interval vector/matrix operations ────────────────────────────────────

def iv_dot(a: list[Interval], b: list[Interval]) -> Interval:
    """Dot product of two interval vectors."""
    assert len(a) == len(b)
    result = Interval(0.0)
    for ai, bi in zip(a, b):
        result = result + ai * bi
    return result


def iv_dot_const(a: list[Interval], b: list[float]) -> Interval:
    """Dot product of interval vector with constant vector."""
    assert len(a) == len(b)
    result = Interval(0.0)
    for ai, bi in zip(a, b):
        result = result + ai * bi
    return result


def iv_rms_norm(x: list[Interval], eps: float = 1e-6) -> list[Interval]:
    """
    RMSNorm: x / sqrt(mean(x²) + eps)
    Returns interval bounds on each component of the normalized vector.
    """
    n = len(x)
    # Compute mean of squares
    sum_sq = Interval(0.0)
    for xi in x:
        sum_sq = sum_sq + xi.square()
    mean_sq = sum_sq / n

    # rms = sqrt(mean_sq + eps)
    rms = iv_sqrt(mean_sq + eps)

    # Normalized: x / rms
    return [xi / rms for xi in x]


def iv_softmax(logits: list[Interval]) -> list[Interval]:
    """
    Softmax with interval arithmetic.

    Uses the max-subtraction trick for numerical stability.
    Returns interval bounds on each softmax probability.
    """
    n = len(logits)

    # Find a sound upper bound on the max logit for stability
    max_hi = max(l.hi for l in logits)
    max_lo = max(l.lo for l in logits)

    # Shift logits by subtracting max_hi (sound: all shifted logits ≤ 0)
    shifted = [Interval(l.lo - max_hi, l.hi - max_lo) for l in logits]

    # Compute exp of shifted logits
    exps = [iv_exp(s) for s in shifted]

    # Compute sum of exponentials
    exp_sum = Interval(0.0)
    for e in exps:
        exp_sum = exp_sum + e

    # Each softmax value = exp_i / sum
    return [e / exp_sum for e in exps]


def iv_softmax_dominant(logits: list[Interval]) -> Optional[int]:
    """
    Check if one logit is provably the largest (i.e., would get the highest
    softmax probability). Returns the dominant index, or None if inconclusive.
    """
    n = len(logits)
    for i in range(n):
        is_dominant = True
        for j in range(n):
            if i == j:
                continue
            # Check if logit[i] is always > logit[j]
            if logits[i].lo <= logits[j].hi:
                is_dominant = False
                break
        if is_dominant:
            return i
    return None


# ── Additional operations for general model verification ───────────────

def iv_relu(x: Interval) -> Interval:
    """ReLU(x) = max(x, 0) — exact piecewise-linear interval."""
    return Interval(max(x.lo, 0.0), max(x.hi, 0.0))


def iv_matmul(W: list[list[float]], x: list[Interval]) -> list[Interval]:
    """Matrix-vector multiply: constant matrix W times interval vector x.

    W is a list of rows, each row a list of floats.
    Returns a list of Interval results.
    """
    result = []
    for row in W:
        assert len(row) == len(x)
        acc = Interval(0.0)
        for wij, xj in zip(row, x):
            acc = acc + xj * wij
        result.append(acc)
    return result


def iv_matmul_bias(W: list[list[float]], x: list[Interval],
                   b: list[float]) -> list[Interval]:
    """Matrix-vector multiply with bias: W @ x + b."""
    out = iv_matmul(W, x)
    return [oi + bi for oi, bi in zip(out, b)]


def iv_rms_norm_weighted(x: list[Interval], weights: list[float],
                         eps: float = 1e-6) -> list[Interval]:
    """RMSNorm with per-dimension weights: (x / sqrt(mean(x²) + eps)) * w."""
    normed = iv_rms_norm(x, eps)
    return [ni * wi for ni, wi in zip(normed, weights)]
