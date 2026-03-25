"""Tests for param_counter.py — verify we count parameters correctly."""

import sys
import pytest
import numpy as np

torch = pytest.importorskip("torch")
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from formal.param_counter import count_params


class TinyLinear(nn.Module):
    """Simple model with known param count."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=True)  # 3*2 + 2 = 8 params

    def forward(self, x):
        return self.fc(x)


class TiedWeightModel(nn.Module):
    """Model with weight tying — encoder and decoder share weights."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)  # 40 params
        self.proj = nn.Linear(4, 10, bias=False)  # 40 params, but tied
        self.proj.weight = self.embed.weight  # Tied! Only 40 unique

    def forward(self, x):
        return self.proj(self.embed(x))


class ModelWithBuffer(nn.Module):
    """Model with a non-parameter buffer (like RoPE frequencies)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)  # 8 params
        # Register a buffer — should NOT be counted
        self.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))

    def forward(self, x):
        return self.fc(x)


def test_simple_model():
    model = TinyLinear()
    result = count_params(model, claimed=8)
    assert result.counted == 8
    assert result.match is True


def test_tied_weights():
    model = TiedWeightModel()
    result = count_params(model, claimed=40)
    # embed.weight and proj.weight share data_ptr
    # PyTorch may deduplicate in named_parameters() itself,
    # so tied_groups may be empty if only one name is reported.
    assert result.counted == 40
    assert result.match is True


def test_tied_weights_wrong_claim():
    model = TiedWeightModel()
    result = count_params(model, claimed=80)  # Wrong: counts both as separate
    assert result.counted == 40
    assert result.match is False


def test_buffer_excluded():
    model = ModelWithBuffer()
    result = count_params(model, claimed=8)
    # inv_freq is a buffer, not a parameter — should be excluded
    assert result.counted == 8
    assert result.match is True


def test_no_parameters():
    model = nn.Module()
    result = count_params(model, claimed=0)
    assert result.counted == 0
    assert result.match is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
