"""Test evaluation module."""

import babyseg
import pytest
import torch


def test_select_dtype_values():
    """Test selecting data types."""
    expected = {
            0: torch.uint8,
           -1: torch.int16,
        2**15: torch.uint16,
        2**16: torch.int32
    }
    for x, y in expected.items():
        x = torch.tensor(x)
        assert babyseg.eval.select_dtype(x) == y


def test_select_dtype_long():
    """Test if values requiring long integers raise an error."""
    x = torch.tensor(2**32)
    with pytest.raises(ValueError):
        babyseg.eval.select_dtype(x)


def test_select_dtype_illegal():
    """Test if selecting types for illegal input data raises errors."""
    for dtype in (torch.float32, torch.complex64):
        x = torch.tensor(1, dtype=dtype)
        with pytest.raises(TypeError):
            babyseg.eval.select_dtype(x)
