"""System tests for scripts."""

import pathlib
import subprocess


def test_download(tmp_path):
    """Test repeated checkpoint download."""
    def run():
        call = pathlib.Path('scripts/download.py').absolute()
        return subprocess.run(call, cwd=tmp_path).returncode

    # Expect checkpoint in specified folder.
    assert run() == 0
    files = list(tmp_path.glob('checkpoints/*.pt'))
    assert files

    # Expect to skip existing files, thus no change in modification time.
    f = files[0]
    t = f.stat().st_mtime
    assert run() == 0
    assert f.stat().st_mtime == t
