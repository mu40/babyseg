"""System tests for the wrapper script."""

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
    """Test if setting an unknown tool that does exist raises an error."""
    mock_tool('fancy-container-tool')
    assert run_wrapper().returncode


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
    assert f'{os.getuid()}:{os.getgid()}' in out
    assert f'{os.getcwd()}:/mnt' in out
    assert out[-1] == os.getenv('BABYSEG_DOCKER_NAME') + f':{tag}'


def test_podman_run(mock_tool, monkeypatch):
    """Test the absence of flags in a Podman call."""
    tag = '0.0'
    _, log = mock_tool('podman')
    monkeypatch.setenv('BABYSEG_TAG', tag)
    assert not run_wrapper().returncode

    # Expect no UID, GID setting for Podman.
    assert f'{os.getuid()}:{os.getgid()}' not in log.read_text()


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_sif_absent(mock_tool, monkeypatch, name):
    """Test the behavior when default SIF file is missing."""
    tag = 'absent'
    tool, log = mock_tool(name)
    monkeypatch.setenv('BABYSEG_TAG', tag)
    p = run_wrapper()
    assert not p.returncode

    # Expect default SIF path alongside script.
    image = os.getenv('BABYSEG_DOCKER_NAME')
    d = pathlib.Path(p.args).absolute().parent
    sif = d / (pathlib.Path(image).name + f'_{tag}.sif')
    url = f'docker://{image}:{tag}'

    # Expect two calls; `pull` first.
    out = log.read_text().splitlines()
    assert len(out) == 2
    assert out[0].split() == [str(tool), 'pull', str(sif), url]


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_sif_present(mock_tool, monkeypatch, name):
    """Test behavior when a user-provided SIF file exists."""
    tool, log = mock_tool(name)
    sif = tool.parent / 'present.sif'
    sif.touch()
    monkeypatch.setenv('BABYSEG_SIF', str(sif))
    p = run_wrapper()
    assert not p.returncode

    # Expect `run` only.
    assert len(log.read_text().splitlines()) == 1

    # Expect No GPU flag `--nv` with regular image name.
    run = log.read_text().split()
    assert run[0] == str(tool)
    assert '--pwd' in run
    assert '/mnt' in run
    assert f'{os.getcwd()}:/mnt' in run
    assert run[-1] == str(sif)
    assert '--nv' not in run


@pytest.mark.parametrize('name', TOOLS_ALL)
def test_bind_mount(mock_tool, monkeypatch, name):
    """Test explicit bind mount of `/mnt` inside container."""
    d = '/a/b/c'
    _, log = mock_tool(name)
    monkeypatch.setenv('SUBJECTS_DIR', d)

    assert not run_wrapper().returncode
    assert f'{d}:/mnt' in log.read_text().split()


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_gpu_environment(mock_tool, monkeypatch, name):
    """Test enabling Apptainer GPU support via environment variable."""
    _, log = mock_tool(name)
    monkeypatch.setenv('BABYSEG_GPU', '1')

    assert not run_wrapper().returncode
    assert '--nv' in log.read_text().split()


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_gpu_filename(mock_tool, monkeypatch, name):
    """Test enabling Apptainer GPU support via image name."""
    tool, log = mock_tool(name)

    # Expect GPU support for `-cu` or `-gpu` in file name.
    for tag in ('1.2.3-cu130', '0.0-gpu'):
        # Mock image.
        sif = tool.parent / f'image_{tag}.sif'
        sif.touch()

        monkeypatch.setenv('BABYSEG_SIF', str(sif))
        assert not run_wrapper().returncode
        assert '--nv' in log.read_text().split()


@pytest.mark.parametrize('name', TOOLS_SIF)
def test_error_code_pull(mock_tool, monkeypatch, name):
    """Test if failure on SIF image `pull` returns the correct code."""
    code = 7
    mock_tool(name, code=code)
    monkeypatch.setenv('BABYSEG_TAG', '2.0-missing')
    assert run_wrapper().returncode == code


@pytest.mark.parametrize('name', TOOLS_ALL - TOOLS_SIF)
def test_error_code_run(mock_tool, monkeypatch, name):
    """Test if failure on `run` returns the correct code."""
    code = 13
    mock_tool(name, code=code)
    assert run_wrapper().returncode == code
