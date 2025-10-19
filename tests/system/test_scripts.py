"""System tests for scripts."""

import pathlib
import subprocess


MODEL = 'babyseg.v1.pt'


def test_download_path(tmp_path):
    """Test downloading checkpoints to default path."""
    call = pathlib.Path('scripts/download.py').absolute()
    p = subprocess.run(call, cwd=tmp_path)
    assert not p.returncode

    # Expect checkpoint in relative checkpoints folder.
    f = tmp_path / 'checkpoints' / MODEL
    assert f.exists()


def test_download_repeat(tmp_path):
    """Test repeated checkpoint download with and without overriding."""
    call = ('scripts/download.py', '-o', tmp_path)
    p = subprocess.run(call)
    assert not p.returncode

    # Expect checkpoint in specified folder.
    f = tmp_path / MODEL
    t = f.stat().st_mtime
    assert f.exists()

    # Expect default skipping does not change modification time.
    p = subprocess.run(call)
    assert not p.returncode
    assert f.stat().st_mtime == t

    # Expect overriding results in higher modification time.
    p = subprocess.run((*call, '-f'))
    assert not p.returncode
    assert f.stat().st_mtime > t
