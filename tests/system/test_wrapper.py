"""Tests for the wrapper script."""

import os
import pathlib
import pytest
import subprocess


TOOLS_SIF = {'apptainer', 'singularity'}
TOOLS_ALL = {'docker', 'podman', *TOOLS_SIF}


@pytest.fixture
def mock_tool(monkeypatch, tmp_path):
    """Return factory to create a mock container tool that logs calls."""
    def f(name, set_path=True, set_tool=True, code=0):
        # Empty file.
        log = tmp_path / f'{name}.log'
        log.write_text('')

        # Mock tool.
        tool = tmp_path / name
        tool.write_text(
            '#!/bin/sh\n'
            f'echo "$0" "$@" | tee -a "{log}"\n'
            f'exit {code:d}\n'
        )
        tool.chmod(0o755)

        # Environment.
        if set_path:
            path = os.getenv('PATH')
            monkeypatch.setenv('PATH', f'{tmp_path}:{path}')
        if set_tool:
            monkeypatch.setenv('BABYSEG_TOOL', tool.name)

        return tool, log

    return f


def run_wrapper():
    """Run the container script without relying on `PATH`."""
    return subprocess.run('docker/wrapper.py')


def test_environment(monkeypatch):
    """Test the test environment."""
    name = os.getenv('BABYSEG_DOCKER_NAME')
    assert name is not None
    assert not name.startswith('/')
    assert not name.endswith('/')
    assert '/' in name


def construct_sif_file(folder, tag, touch=False):
    """Construct SIF file path from folder, tag, and `BABYSEG_DOCKER_NAME`."""
    f = os.getenv('BABYSEG_DOCKER_NAME')
    f = pathlib.Path(f).name
    f = pathlib.Path(folder) / f'{f}_{tag}.sif'
    if touch:
        f.touch()

    return f


@pytest.mark.parametrize('name', TOOLS_ALL)
def test_tool_auto(mock_tool, name):
    """Test auto-selecting tools from `PATH`."""
    tool, log = mock_tool(name)

    # Expect no log initially.
    assert not log.read_text()
    assert not run_wrapper().returncode
    assert log.read_text().startswith(str(tool))


@pytest.mark.parametrize('name', TOOLS_ALL)
def test_tool_absolute(mock_tool, monkeypatch, name):
    """Test running a tool from absolute path when not in `PATH`."""
    tool, log = mock_tool(name, set_path=False, set_tool=False)
    monkeypatch.setenv('BABYSEG_TOOL', str(tool))

    assert not run_wrapper().returncode
    assert log.read_text().startswith(str(tool))


def test_tool_missing(monkeypatch):
    """Test if setting a tool that does not exist raises an error."""
    monkeypatch.setenv('BABYSEG_TOOL', '/an/unknown/tool')
    assert run_wrapper().returncode


def test_tool_unknown(mock_tool):
    """Test if setting an unknown but existing tool raises an error."""
    mock_tool('fancy-container-tool')
    assert run_wrapper().returncode


@pytest.mark.parametrize('name', TOOLS_ALL)
def test_user(mock_tool, monkeypatch, name):
    """Test which tools specify user and group."""
    tag = '0.0'
    tool, log = mock_tool(name)

    # Point to existing SIF file.
    sif = construct_sif_file(tool.parent, tag, touch=True)
    monkeypatch.setenv('BABYSEG_TAG', tag)
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))
    assert not run_wrapper().returncode

    # Expect UID, GID setting only for Docker.
    is_docker = name == 'docker'
    is_user = f'{os.getuid()}:{os.getgid()}' in log.read_text().split()
    assert is_user == is_docker


def test_docker_run(mock_tool, monkeypatch):
    """Test the presence of flags in a Docker call."""
    tag = '0.0'
    tool, log = mock_tool('docker')
    monkeypatch.setenv('BABYSEG_TAG', tag)
    assert not run_wrapper().returncode

    out = log.read_text().split()
    assert out[0] == str(tool)
    assert out[1] == 'run'
    assert '--rm' in out
    assert f'{os.getcwd()}:/mnt' in out
    assert out[-1] == os.getenv('BABYSEG_DOCKER_NAME') + f':{tag}'


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_sif_directory_error(mock_tool, monkeypatch, name):
    """Test if not pointing `BABYSEG_SIF` to a directory raises an error."""
    tool, log = mock_tool(name)

    # Expect failure on an existing file
    d = log
    monkeypatch.setenv('BABYSEG_SIF', str(d))
    assert run_wrapper().returncode != 0

    # Expect failure on a directory that does not exist.
    d = tool.parent / 'a' / 'b'
    monkeypatch.setenv('BABYSEG_SIF', str(d))
    assert run_wrapper().returncode != 0

    # Expect success on an existing directory.
    d = tool.parent / 'existing'
    d.mkdir()
    monkeypatch.setenv('BABYSEG_SIF', str(d))
    assert run_wrapper().returncode == 0


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_sif_file_absent(mock_tool, monkeypatch, name):
    """Test behavior when the SIF file is missing and `BABYSEG_SIF` unset."""
    tag = 'absent'
    tool, log = mock_tool(name)
    monkeypatch.setenv('BABYSEG_TAG', tag)
    p = run_wrapper()
    assert not p.returncode

    # Expect default SIF path alongside script.
    image = os.getenv('BABYSEG_DOCKER_NAME')
    d = pathlib.Path(p.args).absolute().parent
    sif = construct_sif_file(d, tag, touch=False)
    hub = f'docker://{image}:{tag}'

    # Expect two calls; `pull` first.
    out = log.read_text().splitlines()
    assert len(out) == 2
    assert out[0].split() == [str(tool), 'pull', str(sif), hub]


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_sif_file_present(mock_tool, monkeypatch, name):
    """Test behavior when the SIF file exists."""
    tag = '9.0'
    tool, log = mock_tool(name)
    sif = construct_sif_file(tool.parent, tag, touch=True)
    monkeypatch.setenv('BABYSEG_TAG', tag)
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))
    p = run_wrapper()
    assert not p.returncode

    # Expect `run` call  only.
    assert len(log.read_text().splitlines()) == 1

    run = log.read_text().split()
    assert run[:2] == [str(tool), 'run']
    assert '--pwd' in run
    assert '/mnt' in run
    assert f'{os.getcwd()}:/mnt' in run
    assert run[-1] == str(sif)


@pytest.mark.parametrize('name', TOOLS_ALL)
def test_bind_mount(mock_tool, monkeypatch, name):
    """Test explicit bind mount of `/mnt` inside container."""
    d = '/a/b/c'
    _, log = mock_tool(name)
    monkeypatch.setenv('BABYSEG_MNT', d)

    assert not run_wrapper().returncode
    assert f'{d}:/mnt' in log.read_text().split()


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_gpu(mock_tool, monkeypatch, name):
    """Test enabling GPU support via image tag."""
    tool, log = mock_tool(name)
    monkeypatch.setenv('BABYSEG_SIF', str(tool.parent))

    # Expect GPU enabled when `-cu` in tag.
    for tag in ('1.2.3-cu130', '9.9'):
        log.write_text('')
        sif = construct_sif_file(os.getenv('BABYSEG_SIF'), tag, touch=True)
        monkeypatch.setenv('BABYSEG_TAG', tag)
        assert not run_wrapper().returncode

        is_gpu_image = '-cu' in tag
        is_gpu_enabled = '--nv' in log.read_text().split()
        assert is_gpu_image == is_gpu_enabled


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_error_code_on_pull(mock_tool, monkeypatch, name):
    """Test if failure on SIF image `pull` returns the correct code."""
    code = 7
    mock_tool(name, code=code)
    monkeypatch.setenv('BABYSEG_TAG', '2.0-missing')
    assert run_wrapper().returncode == code


@pytest.mark.parametrize('name', TOOLS_ALL - TOOLS_SIF)
def test_error_code_on_run(mock_tool, monkeypatch, name):
    """Test if failure on `run` returns the correct code."""
    code = 13
    mock_tool(name, code=code)
    assert run_wrapper().returncode == code
