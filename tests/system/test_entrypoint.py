"""System tests for scripts."""

import babyseg
import docker.entrypoint
import pytest


@pytest.fixture
def setup_env(monkeypatch):
    """Set BABYSEG_HOME environment variable for testing."""
    monkeypatch.setenv('BABYSEG_HOME', '.')


def test_home(monkeypatch, capteesys):
    """Test if absence of `BABYSEG_HOME` raises an error."""
    monkeypatch.delenv('BABYSEG_HOME', raising=False)
    with pytest.raises(SystemExit):
        assert 0 != docker.entrypoint.main(argv=[])

    f = capteesys.readouterr()
    assert 'environment variable' in f.err


def test_usage(setup_env, capteesys):
    """Test printing usage without arguments."""
    with pytest.raises(SystemExit):
        assert 0 == docker.entrypoint.main(argv=[])

    f = capteesys.readouterr()
    assert 'usage' in f.out


def test_help(setup_env, capteesys):
    """Test printing the help text."""
    with pytest.raises(SystemExit):
        assert 0 == docker.entrypoint.main(argv=['-h'])

    f = capteesys.readouterr()
    assert 'positional arguments' in f.out


def test_version(setup_env, capteesys):
    """Test printing the version number."""
    with pytest.raises(SystemExit):
        assert 0 == docker.entrypoint.main(argv=['-V'])

    f = capteesys.readouterr()
    assert f.out.strip() == babyseg.__version__


def test_image_missing(setup_env, capteesys):
    """Test if passing no input image raises an error."""
    with pytest.raises(SystemExit):
        assert 0 != docker.entrypoint.main(argv=['-g'])

    f = capteesys.readouterr()
    assert 'required: image' in f.err
