"""Test initialization."""

import babyseg


def test_version_exists():
    """Test if version string exists."""
    assert hasattr(babyseg, '__version__')


def test_version_valid():
    """Test version string is valid."""
    v = babyseg.__version__
    assert isinstance(v, str)
    assert v
    assert not v.startswith('.')
    assert not v.endswith('.')
    assert '..' not in v
    assert all(c.isdigit() or c == '.' for c in v)
