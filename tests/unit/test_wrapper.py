"""Tests for the wrapper script."""

import docker.wrapper
import os
import pathlib
import pytest
import subprocess
import unittest.mock


TOOLS_SIF = {'apptainer', 'singularity'}
TOOLS_ALL = {'docker', 'podman', *TOOLS_SIF}


@pytest.fixture(scope='session')
def docker_name():
    """Provide a validated Docker image name for tests."""
    name = os.getenv('BABYSEG_DOCKER_NAME')
    if not name or '/' not in name:
        pytest.fail('invalid or missing BABYSEG_DOCKER_NAME')

    return name


@pytest.fixture
def sif_factory(docker_name):
    """Return factory to construct SIF file paths."""
    def sif_file(folder, tag, touch=False):
        f = pathlib.Path(docker_name).name
        f = pathlib.Path(folder) / f'{f}_{tag}.sif'
        if touch:
            f.touch()

        return f

    return sif_file


@pytest.fixture
def mock_factory(monkeypatch, tmp_path):
    """Return factory to create a mock container tool that logs calls."""
    mock = unittest.mock.create_autospec(subprocess.run)
    monkeypatch.setattr(subprocess, 'run', mock)

    def f(name, set_path=True, set_tool=True, code=0):
        tool = tmp_path / name
        tool.touch(mode=0o755)
        monkeypatch.setenv('PATH', str(tmp_path))
        monkeypatch.setenv('BABYSEG_TOOL', tool.name)

        mock.return_value = subprocess.CompletedProcess(
            args='',
            returncode=code,
        )
        return tool, mock

    return f


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_shell(tool, tmp_path, sif_factory, monkeypatch):
    """Verify entry point shebang, executable bit, CLI argument handling."""
    code = 123
    arg = ('-A', '-B', 'file.json')

    # Mock tool.
    tool = tmp_path / tool
    tool.write_text(f'#!/bin/sh\necho "$@" >/dev/stderr\nexit {code:d}\n')
    tool.chmod(0o755)

    # Create SIF file, so SIF tools skip the `pull` call.
    sif = sif_factory(tmp_path, docker.wrapper.TAG, touch=True)
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))
    monkeypatch.setenv('BABYSEG_TOOL', str(tool))

    call = (docker.wrapper.__file__, *arg)
    p = subprocess.run(call, capture_output=True, text=True)
    assert p.returncode == code
    assert p.stderr.strip().endswith(' '.join(arg))


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_tool_auto(mock_factory, tool):
    """Test auto-selecting tools from `PATH`."""
    tool, mock = mock_factory(tool)
    assert docker.wrapper.main(argv=[]) == 0
    mock.assert_called()
    (call,) = mock.call_args.args
    assert call[0] == tool


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_tool_absolute(mock_factory, tool, monkeypatch):
    """Test running a tool from absolute path when not in `PATH`."""
    tool, mock = mock_factory(tool)
    monkeypatch.delenv('PATH')
    monkeypatch.setenv('BABYSEG_TOOL', str(tool))

    assert docker.wrapper.main(argv=[]) == 0
    mock.assert_called()
    (call,) = mock.call_args.args
    assert call[0] == tool


def test_tool_missing(monkeypatch, capteesys):
    """Test if setting a tool that does not exist raises an error."""
    monkeypatch.setenv('BABYSEG_TOOL', '/some/missing/tool')
    assert docker.wrapper.main(argv=[]) > 0
    f = capteesys.readouterr()
    assert 'cannot locate' in f.err


def test_tool_unknown(mock_factory, capteesys):
    """Test if setting an unknown existing tool raises an error."""
    mock_factory('some-fancy-tool')
    assert docker.wrapper.main(argv=[]) > 0
    f = capteesys.readouterr()
    assert 'unknown container tool' in f.err


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_user(mock_factory, tool):
    """Test which tools specify user and group."""
    tool, mock = mock_factory(tool)
    assert docker.wrapper.main(argv=[]) == 0

    # Expect UID, GID in the final `run` call only for Docker.
    mock.assert_called()
    (call,) = mock.call_args.args
    assert (f'{os.getuid()}:{os.getgid()}' in call) == (tool.name == 'docker')


def test_docker_run(mock_factory, monkeypatch, docker_name):
    """Test the presence of flags in a Docker call."""
    tool, mock = mock_factory('docker')
    tag = 'latest'
    monkeypatch.setenv('BABYSEG_TAG', tag)

    # Expect a single `run` call.
    assert docker.wrapper.main(argv=[]) == 0
    mock.assert_called_once()

    (call,) = mock.call_args.args
    assert call[:2] == (tool, 'run')
    assert '--rm' in call
    assert f'{os.getcwd()}:/mnt' in call
    assert call[-1] == f'{docker_name}:{tag}'


@pytest.mark.parametrize('tool', TOOLS_SIF)
def test_sif_directory_error(mock_factory, tool, monkeypatch, capteesys):
    """Test if not pointing `BABYSEG_SIF` to a directory raises an error."""
    # Mock tool to avoid pulls.
    tool, _ = mock_factory(tool)

    # Expect failure on an existing file.
    monkeypatch.setenv('BABYSEG_SIF', str(tool))
    assert docker.wrapper.main(argv=[]) > 0
    assert 'directory' in capteesys.readouterr().err

    # Expect failure on a path that does not exist.
    monkeypatch.setenv('BABYSEG_SIF', str(tool.parent / 'missing'))
    assert docker.wrapper.main(argv=[]) > 0
    assert 'directory' in capteesys.readouterr().err

    # Expect success on a directory.
    monkeypatch.setenv('BABYSEG_SIF', str(tool.parent))
    assert docker.wrapper.main(argv=[]) == 0


@pytest.mark.parametrize('tool', TOOLS_SIF)
def test_sif_pull(mock_factory, tool, monkeypatch, docker_name, sif_factory):
    """Test behavior when the SIF file is missing and `BABYSEG_SIF` unset."""
    tool, mock = mock_factory(tool)
    tag = 'absent'
    monkeypatch.setenv('BABYSEG_TAG', tag)

    # Expect `pull` and `run` calls for SIF tools.
    assert docker.wrapper.main(argv=[]) == 0
    assert mock.call_count == 2

    # Default image path and URL.
    d = pathlib.Path(docker.wrapper.__file__).parent
    sif = sif_factory(d, tag, touch=False)
    hub = f'docker://{docker_name}:{tag}'

    # Expect valid `pull` arguments.
    (first,) = mock.call_args_list[0].args
    assert first == (tool, 'pull', sif, hub)


@pytest.mark.parametrize('tool', TOOLS_SIF)
def test_sif_run(mock_factory, tool, monkeypatch, sif_factory):
    """Test behavior when the SIF file exists."""
    tool, mock = mock_factory(tool)
    sif = sif_factory(tool.parent, docker.wrapper.TAG, touch=True)
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))

    # Expect `run` call  only.
    assert docker.wrapper.main(argv=[]) == 0
    mock.assert_called_once()

    # Expect valid `run` arguments.
    (call,) = mock.call_args.args
    assert call[:2] == (tool, 'run')
    assert '--pwd' in call
    assert '/mnt' in call
    assert f'{os.getcwd()}:/mnt' in call
    assert call[-1] == sif


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_bind_mount(mock_factory, tool, monkeypatch):
    """Test explicit bind mount of `/mnt` inside container."""
    tool, mock = mock_factory(tool)
    d = '/a/b/c'
    monkeypatch.setenv('BABYSEG_MNT', d)

    # Expect one or two calls.
    assert docker.wrapper.main(argv=[]) == 0
    mock.assert_called()

    # Expect bind mount in final `run` call.
    (call,) = mock.call_args.args
    assert f'{d}:/mnt' in call


@pytest.mark.parametrize('tool', TOOLS_SIF)
@pytest.mark.parametrize('tag', ['1.2.3-cu130', '9.9'])
def test_sif_gpu(mock_factory, tool, monkeypatch, tag):
    """Test enabling GPU support via image tag."""
    tool, mock = mock_factory(tool)
    monkeypatch.setenv('BABYSEG_TAG', tag)
    assert docker.wrapper.main(argv=[]) == 0

    # Expect GPU flag when `-cu` in tag.
    mock.assert_called()
    (call,) = mock.call_args.args
    assert ('-cu' in tag) == ('--nv' in call)


@pytest.mark.parametrize('tool', TOOLS_SIF)
def test_error_code_on_pull(mock_factory, tool, monkeypatch):
    """Test if failure on SIF image `pull` returns the correct code."""
    # Set BABYSEG_SIF to a directory without SIF images.
    tool, mock = mock_factory(tool)
    monkeypatch.setenv('BABYSEG_SIF', str(tool.parent))
    code = 7
    mock.return_value = subprocess.CompletedProcess(args='', returncode=code)

    # Expect exit after failed pull.
    assert docker.wrapper.main(argv=[]) == code
    mock.assert_called_once()


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_error_code_on_run(mock_factory, tool, monkeypatch, sif_factory):
    """Test if failure on `run` returns the correct code."""
    tool, mock = mock_factory(tool)
    code = 13
    mock.return_value = subprocess.CompletedProcess(args='', returncode=code)
    sif = sif_factory(tool.parent, docker.wrapper.TAG, touch=True)
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))

    # Expect single call as SIF file exists.
    assert docker.wrapper.main(argv=[]) == code
    mock.assert_called_once()


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_arguments(mock_factory, tool, sif_factory, monkeypatch):
    """Test if the wrapper forwards input arguments."""
    # Create SIF file, so SIF tools skip the `pull` call.
    argv = ('-a', '-B','file.json')
    tool, mock = mock_factory(tool)
    sif = sif_factory(tool.parent, docker.wrapper.TAG, touch=True)
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))

    assert docker.wrapper.main(argv) == 0
    mock.assert_called_once()
    (call,) = mock.call_args.args
    print(call)
    assert call[-len(argv):] == argv
