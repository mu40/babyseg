"""Tests for environment install."""

import subprocess
import venv


def test_clone_environment(tmp_path):
    """Test environment install from requirements file."""
    venv.EnvBuilder(with_pip=True).create(tmp_path)
    pip = tmp_path / 'bin' / 'pip'
    p = subprocess.run((pip, 'install', '-r', 'requirements.txt'), env={})
    assert p.returncode == 0
