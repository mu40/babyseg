"""Tests for package import and initialization."""

import babyseg
import pytest
import subprocess
import sys


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


def test_pytorch_free_import():
    """Test if lazy package import does not pull in PyTorch."""
    code = '\n'.join((
        'import babyseg',
        'import sys',
        'if any(f.startswith("torch") for f in sys.modules):',
        '    for name, module in sys.modules.items():',
        '        print(name, module)',
        '    exit(1)',
    ))
    p = subprocess.run((sys.executable, '-c', code))
    assert p.returncode == 0, 'lazy import pulls in PyTorch'


@pytest.mark.parametrize('module', ['config', 'data'])
def test_pytorch_free_submodule(module):
    """Test importing PyTorch-free submodules without PyTorch."""
    code = '\n'.join((
        f'import babyseg.{module}',
        'import sys',
        'if any(f.startswith("torch") for f in sys.modules):',
        '    for name, module in sys.modules.items():',
        '        print(name, module)',
        '    exit(1)',
    ))
    p = subprocess.run((sys.executable, '-c', code))
    assert p.returncode == 0, 'PyTorch among imported modules'
