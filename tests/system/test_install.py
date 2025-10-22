"""Tests for requirement install."""

import subprocess
import sys


def test_install_requirements(tmp_path):
    """Test install from requirements file."""
    # Paths.
    venv = tmp_path
    pip = venv / 'bin' / 'pip'

    # Helper.
    def run(*f):
        p = subprocess.run(f, env=dict(PIP_CACHE_DIR=str(venv / 'cache')))
        assert not p.returncode

    # Virtual environment.
    run(sys.executable, '-m', 'venv', venv)
    run(pip, 'install', '-r', 'requirements.txt')
