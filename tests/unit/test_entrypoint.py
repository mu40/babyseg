"""Tests for entry point options and error handling."""

import babyseg
import docker.entrypoint
import pytest
import torch
import unittest


@pytest.fixture()
def babyseg_home(monkeypatch, tmp_path):
    """Set up minimal mock environment for testing."""
    home = tmp_path
    config = home / 'config'
    checkpoints = home / 'checkpoints'
    config.mkdir()
    checkpoints.mkdir()
    (config / 'babyseg.1.json').touch()
    (checkpoints / 'babyseg.1.pt').touch()
    monkeypatch.setenv('BABYSEG_HOME', str(home))
    return home


def test_home(monkeypatch, capteesys):
    """Test if absence of `BABYSEG_HOME` raises an error."""
    monkeypatch.delenv('BABYSEG_HOME', raising=False)
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=[])

    f = capteesys.readouterr()
    assert 'environment variable' in f.err
    assert e.value.code > 0


def test_usage(babyseg_home, capteesys):
    """Test printing usage without arguments."""
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=[])

    f = capteesys.readouterr()
    assert 'usage' in f.out
    assert e.value.code == 0


def test_help(babyseg_home, capteesys):
    """Test printing the help text."""
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['-h'])

    f = capteesys.readouterr()
    assert 'positional arguments' in f.out
    assert e.value.code == 0


def test_version(babyseg_home, capteesys):
    """Test printing the version number."""
    with pytest.raises(SystemExit) as e :
        docker.entrypoint.main(argv=['-V'])

    f = capteesys.readouterr()
    assert f.out.strip() == babyseg.__version__
    assert e.value.code == 0


def test_image_missing(babyseg_home, capteesys):
    """Test if passing no input image raises an error."""
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['-g'])

    f = capteesys.readouterr()
    assert 'required: image' in f.err
    assert e.value.code > 0


@pytest.mark.parametrize('flag', ('-o', '-l', '-p'))
def test_mgz_output(babyseg_home, capteesys, flag):
    """Test if specifying non-NIfTI outputs raises an error."""
    out = 'out.mgz'
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['in.nii', flag, out])

    f = capteesys.readouterr()
    assert flag in f.err
    assert out in f.err
    assert e.value.code > 0


def test_mgz_input(babyseg_home, capteesys):
    """Test if specifying non-NIfTI inputs raises an error."""
    inp = 'in.mgz'
    with pytest.raises(SystemExit) as e:
        docker.entrypoint.main(argv=['-o', 'out.nii', inp])

    f = capteesys.readouterr()
    assert inp in f.err
    assert e.value.code > 0


def test_threads(babyseg_home, monkeypatch):
    """Test setting the number of intraop threads on the CPU."""
    mock = unittest.mock.Mock(side_effect=RuntimeError('END'))
    monkeypatch.setattr(torch, 'set_num_threads', mock)

    threads = 7
    with pytest.raises(RuntimeError, match='END'):
        docker.entrypoint.main(argv=['-j', f'{threads}', 'in.nii'])

    mock.assert_called_once_with(threads)
