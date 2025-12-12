"""Tests for the wrapper script."""

import docker.wrapper
import os
import pathlib
import pytest
import subprocess
import unittest.mock


TOOLS_SIF = {'apptainer', 'singularity'}
TOOLS_ALL = {'docker', 'podman', *TOOLS_SIF}


@pytest.fixture
def tool(monkeypatch, tmp_path, request):
    """Create a mock container tool that logs calls."""
    name = request.param
    path = tmp_path / name
    path.touch(mode=0o755)

    tool = unittest.mock.create_autospec(subprocess.run)
    tool.name = name
    tool.path = path
    tool.return_value = subprocess.CompletedProcess(args='', returncode=0)

    # Clear environment variables to ensure the wrapper uses defaults.
    monkeypatch.delenv('BABYSEG_MNT', raising=False)
    monkeypatch.delenv('BABYSEG_SIF', raising=False)
    monkeypatch.delenv('BABYSEG_TAG', raising=False)
    monkeypatch.setenv('BABYSEG_TOOL', name)
    monkeypatch.setenv('PATH', str(tmp_path))
    monkeypatch.setattr(subprocess, 'run', tool)
    return tool


def test_docker_name():
    """Validate the Docker Hub image name."""
    name = docker.wrapper.IMAGE
    assert isinstance(name, str)
    assert name
    assert not name.startswith('/')
    assert not name.endswith('/')


@pytest.mark.parametrize('tag', ['1', '7-cu5', '/root-cu/image.sif'])
def test_is_cuda_image(tag):
    """Test determining whether an image tag is CUDA-enabled."""
    is_cuda = docker.wrapper.is_cuda_image(tag)
    assert isinstance(is_cuda, bool)
    assert is_cuda == ('-cu' in pathlib.Path(tag).name)


@pytest.mark.parametrize('dtype', [str, pathlib.Path])
def test_sif_image_path(dtype):
    """Test SIF image path construction from directory and tag."""
    base = pathlib.Path(docker.wrapper.IMAGE).name
    tag = 'test-tag'
    folder = dtype('a/b/')
    expected = pathlib.Path(folder) / f'{base}_{tag}.sif'
    assert docker.wrapper.sif_image_path(folder, tag) == expected


@pytest.mark.parametrize('tool', TOOLS_ALL)
def test_shell(tool, tmp_path, monkeypatch):
    """Verify entry-point shebang, executable bit, CLI argument handling."""
    code = 123
    arg = ('-A', '-B', 'file.json')

    # Mock tool.
    tool = tmp_path / tool
    tool.write_text(f'#!/bin/sh\necho "$@" >/dev/stderr\nexit {code:d}\n')
    tool.chmod(0o755)

    # Create dummy SIF file, so SIF tools skip the `pull` call.
    sif = docker.wrapper.sif_image_path(tmp_path, docker.wrapper.TAG)
    sif.touch()
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))
    monkeypatch.setenv('BABYSEG_TOOL', str(tool))

    call = (docker.wrapper.__file__, *arg)
    p = subprocess.run(call, capture_output=True, text=True)
    assert p.returncode == code
    assert p.stderr.strip().endswith(' '.join(arg))


@pytest.mark.parametrize('tool', TOOLS_ALL, indirect=True)
def test_tool_auto(tool):
    """Test auto-selecting tools from `PATH`."""
    assert docker.wrapper.main(argv=[]) == 0
    tool.assert_called()
    (call,) = tool.call_args.args
    assert call[0] == tool.path


@pytest.mark.parametrize('tool', TOOLS_ALL, indirect=True)
def test_tool_absolute(tool, monkeypatch):
    """Test running a tool from absolute path when not in `PATH`."""
    monkeypatch.delenv('PATH')
    monkeypatch.setenv('BABYSEG_TOOL', str(tool.path))

    assert docker.wrapper.main(argv=[]) == 0
    tool.assert_called()
    (call,) = tool.call_args.args
    assert call[0] == tool.path


def test_tool_missing(monkeypatch, capteesys):
    """Verify that a missing tool raises an error."""
    monkeypatch.setenv('BABYSEG_TOOL', '/some/missing/tool')
    assert docker.wrapper.main(argv=[]) > 0
    f = capteesys.readouterr()
    assert 'cannot locate' in f.err


@pytest.mark.parametrize('tool', ['unknown-tool'], indirect=True)
def test_tool_unknown(tool, capteesys):
    """Verify that an unknown existing tool raises an error."""
    assert docker.wrapper.main(argv=[]) > 0
    f = capteesys.readouterr()
    assert 'unknown container tool' in f.err


@pytest.mark.parametrize('tool', TOOLS_ALL, indirect=True)
def test_user(tool):
    """Test which tools specify user and group."""
    assert docker.wrapper.main(argv=[]) == 0

    # Expect UID, GID in the final `run` call only for Docker.
    tool.assert_called()
    (call,) = tool.call_args.args
    assert (f'{os.getuid()}:{os.getgid()}' in call) == (tool.name == 'docker')


@pytest.mark.parametrize('tool', ['docker'], indirect=True)
def test_docker_run(tool, monkeypatch):
    """Test the presence of flags in a Docker call."""
    tag = 'latest'
    monkeypatch.setenv('BABYSEG_TAG', tag)

    # Expect a single `run` call.
    assert docker.wrapper.main(argv=[]) == 0
    tool.assert_called_once()

    (call,) = tool.call_args.args
    assert call[:2] == (tool.path, 'run')
    assert '--rm' in call
    assert f'{os.getcwd()}:/mnt' in call
    assert call[-1] == f'{docker.wrapper.IMAGE}:{tag}'


@pytest.mark.parametrize('tool', TOOLS_SIF, indirect=True)
def test_sif_directory_error(tool, monkeypatch, capteesys):
    """Test if not pointing `BABYSEG_SIF` to a directory raises an error."""
    # Expect failure on an existing file.
    monkeypatch.setenv('BABYSEG_SIF', str(tool.path))
    assert docker.wrapper.main(argv=[]) > 0
    assert 'directory' in capteesys.readouterr().err

    # Expect failure on a path that does not exist.
    monkeypatch.setenv('BABYSEG_SIF', str(tool.path.parent / 'missing'))
    assert docker.wrapper.main(argv=[]) > 0
    assert 'directory' in capteesys.readouterr().err

    # Expect success on a directory.
    monkeypatch.setenv('BABYSEG_SIF', str(tool.path.parent))
    assert docker.wrapper.main(argv=[]) == 0


@pytest.mark.parametrize('tool', TOOLS_SIF, indirect=True)
def test_sif_pull(tool, monkeypatch):
    """Test behavior when the SIF file is missing and `BABYSEG_SIF` unset."""
    tag = 'absent'
    monkeypatch.setenv('BABYSEG_TAG', tag)

    # Expect `pull` and `run` calls for SIF tools.
    assert docker.wrapper.main(argv=[]) == 0
    assert tool.call_count == 2

    # Default image path and URL.
    d = pathlib.Path(docker.wrapper.__file__).parent
    sif = docker.wrapper.sif_image_path(d, tag)
    hub = f'docker://{docker.wrapper.IMAGE}:{tag}'

    # Expect valid `pull` arguments.
    (first,) = tool.call_args_list[0].args
    assert first == (tool.path, 'pull', sif, hub)


@pytest.mark.parametrize('tool', TOOLS_SIF, indirect=True)
def test_sif_run(tool, monkeypatch):
    """Test behavior when the SIF file exists."""
    sif = docker.wrapper.sif_image_path(tool.path.parent, docker.wrapper.TAG)
    sif.touch()
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))

    # Expect `run` call only.
    assert docker.wrapper.main(argv=[]) == 0
    tool.assert_called_once()

    # Expect valid `run` arguments.
    (call,) = tool.call_args.args
    assert call[:2] == (tool.path, 'run')
    assert '--pwd' in call
    assert '/mnt' in call
    assert f'{os.getcwd()}:/mnt' in call
    assert call[-1] == sif


@pytest.mark.parametrize('tool', TOOLS_ALL, indirect=True)
def test_bind_mount(tool, monkeypatch):
    """Test explicit bind mount of `/mnt` inside container."""
    d = '/a/b/c'
    monkeypatch.setenv('BABYSEG_MNT', d)

    # Expect one or two calls.
    assert docker.wrapper.main(argv=[]) == 0
    tool.assert_called()

    # Expect bind mount in final `run` call.
    (call,) = tool.call_args.args
    assert f'{d}:/mnt' in call


@pytest.mark.parametrize('tool', TOOLS_SIF, indirect=True)
@pytest.mark.parametrize('tag', ['1.2.3-cu130', '9.9'])
def test_sif_gpu(tool, monkeypatch, tag):
    """Test enabling GPU support via image tag."""
    monkeypatch.setenv('BABYSEG_TAG', tag)
    assert docker.wrapper.main(argv=[]) == 0

    # Expect GPU flag when `-cu` in tag.
    tool.assert_called()
    (call,) = tool.call_args.args
    assert ('--nv' in call) == docker.wrapper.is_cuda_image(tag)


@pytest.mark.parametrize('tool', TOOLS_SIF, indirect=True)
def test_error_code_on_pull(tool, monkeypatch):
    """Test if failure on SIF image `pull` returns the correct code."""
    # Set BABYSEG_SIF to a directory without SIF images.
    code = 7
    tool.return_value = subprocess.CompletedProcess(args='', returncode=code)
    monkeypatch.setenv('BABYSEG_SIF', str(tool.path.parent))

    # Expect exit after failed pull.
    assert docker.wrapper.main(argv=[]) == code
    tool.assert_called_once()


@pytest.mark.parametrize('tool', TOOLS_ALL, indirect=True)
def test_error_code_on_run(tool, monkeypatch):
    """Test if failure on `run` returns the correct code."""
    code = 13
    tool.return_value = subprocess.CompletedProcess(args='', returncode=code)
    sif = docker.wrapper.sif_image_path(tool.path.parent, docker.wrapper.TAG)
    sif.touch()
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))

    # Expect single call as SIF file exists.
    assert docker.wrapper.main(argv=[]) == code
    tool.assert_called_once()


@pytest.mark.parametrize('tool', TOOLS_ALL, indirect=True)
def test_arguments(tool, monkeypatch):
    """Test if the wrapper forwards input arguments."""
    # Create SIF file, so SIF tools skip the `pull` call.
    argv = ('-a', '-B', 'file.json')
    sif = docker.wrapper.sif_image_path(tool.path.parent, docker.wrapper.TAG)
    sif.touch()
    monkeypatch.setenv('BABYSEG_SIF', str(sif.parent))

    assert docker.wrapper.main(argv) == 0
    tool.assert_called_once()
    (call,) = tool.call_args.args
    assert call[-len(argv):] == argv
