"""Tests for container entry point."""

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
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=[])

    f = capteesys.readouterr()
    assert 'environment variable' in f.err
    assert e.value.code != 0


def test_usage(setup_env, capteesys):
    """Test printing usage without arguments."""
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=[])

    f = capteesys.readouterr()
    assert 'usage' in f.out
    assert e.value.code == 0


def test_help(setup_env, capteesys):
    """Test printing the help text."""
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['-h'])

    f = capteesys.readouterr()
    assert 'positional arguments' in f.out
    assert e.value.code == 0


def test_version(setup_env, capteesys):
    """Test printing the version number."""
    with pytest.raises(SystemExit) as e :
        docker.entrypoint.main(argv=['-V'])

    f = capteesys.readouterr()
    assert f.out.strip() == babyseg.__version__
    assert e.value.code == 0


def test_image_missing(setup_env, capteesys):
    """Test if passing no input image raises an error."""
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['-g'])

    f = capteesys.readouterr()
    assert 'required: image' in f.err
    assert e.value.code != 0


@pytest.mark.parametrize('flag', ('-o', '-l', '-p'))
def test_mgz_output(setup_env, capteesys, flag):
    """Test if specifying non-NIfTI outputs raises an error."""
    out = 'out.mgz'
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['in.nii', flag, out])

    f = capteesys.readouterr()
    assert flag in f.err
    assert out in f.err
    assert e.value.code != 0


def test_mgz_input(setup_env, capteesys):
    """Test if specifying non-NIfTI inputs raises an error."""
    inp = 'in.mgz'
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['-o', 'out.nii', inp])

    f = capteesys.readouterr()
    assert inp in f.err
    assert e.value.code != 0
