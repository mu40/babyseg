#!/usr/bin/env python3
"""Ease setup and use of BabySeg containers.

Pulls a container from Docker Hub and mounts the host directory set by
environment variable `BABYSEG_MNT` to `/mnt` in the container, which serves as
its working directory. If unset, `BABYSEG_MNT` defaults to the current
directory. Thus, BabySeg can access relative paths under it without setting
`BABYSEG_MNT` explicitly.

"""

import os
import pathlib
import shutil
import signal
import subprocess
import sys


# Settings. Adjust the values below to control the container version, local
# image folder, and preferred container tools. You can override them by setting
# environment variables `BABYSEG_TAG`, `BABYSEG_SIF`, and `BABYSEG_TOOL`.

# Container version tag. Must exist.
TAG = '0.0'

# Local directory for storing Apptainer, Singularity images.
SIF = pathlib.Path(__file__).parent

# Container tool preference. Checked left to right.
TOOLS = ('apptainer', 'singularity', 'docker', 'podman')

# Image name on Docker Hub.
IMAGE = 'freesurfer/babyseg'


def error(message, code=1):
    """Print a message to standard error, propagating an exit code.

    Parameters
    ----------
    message : str
        Error message.
    code : int, optional
        Exit code.

    Returns
    -------
    int
        Exit code.

    """
    print(f'ERROR: {message}', file=sys.stderr)
    return code


def read_env(name, default):
    """Return an environment variable if set, otherwise a default value.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : object
        Fallback value.

    Returns
    -------
    object
        Environment variable or fallback value.

    """
    value = os.getenv(name)
    if value:
        print(f'Applying environment variable {name}="{value}"')
        return value

    return default


def is_cuda_image(tag):
    """Determine if a tag or path indicates a CUDA-enabled image.

    Parameters
    ----------
    tag : str or os.PathLike
        Tag or path.

    Returns
    -------
    bool
        If the tag or base name contains the string '-cu'.

    """
    return '-cu' in pathlib.Path(tag).name


def sif_image_path(sif_dir, tag):
    """Construct SIF image path for a given tag.

    Parameters
    ----------
    sif_dir : str or os.PathLike
        Directory containing SIF image.
    tag : str
        Tag appended to the hard-coded base name.

    Returns
    -------
    pathlib.Path
        Image path.

    """
    base = pathlib.Path(IMAGE).name
    return pathlib.Path(sif_dir) / f'{base}_{tag}.sif'


def main(argv=None):
    """Entry point for command-line execution.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, defaults to `sys.argv[1:]`.

    """
    # Environment variables. Override settings above.
    host = read_env('BABYSEG_MNT', os.getcwd())
    tag = read_env('BABYSEG_TAG', TAG)
    sif = read_env('BABYSEG_SIF', SIF)
    sif = sif_image_path(sif, tag)

    tools = read_env('BABYSEG_TOOL', TOOLS)
    if isinstance(tools, str):
        tools = (tools,)

    # Report version. Avoid errors when piping, for example, to `head`.
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, handler=signal.SIG_DFL)
    hub = 'https://hub.docker.com/u/freesurfer'
    print(f'Running BabySeg version "{tag}" from {hub}')

    # Find a container system.
    for tool in tools:
        tool = shutil.which(tool)
        if tool:
            tool = pathlib.Path(tool)
            tool_name = tool.stem.lower()
            print(f'Selected "{tool}" to manage containers')
            break

    else:
        return error(f'cannot locate container tool {tools}')

    # Bind path and image URL. Mount BABYSEG_MNT as /mnt inside the container,
    # which we made the working directory when building the image. Docker and
    # Podman require absolute paths.
    host = pathlib.Path(host).absolute()
    print(f'Will bind /mnt in container to BABYSEG_MNT="{host}"')

    image = f'{IMAGE}:{tag}'
    if tool_name != 'docker':
        image = f'docker://{image}'

    # Run Docker using the UID and GID of the host user. This user will own
    # bind mounts in the container, preventing outputs owned by root. Root
    # inside rootless Podman containers maps to the host user, which is what we
    # want. If we set UID and GID inside the container to the non-root host
    # user, as for Docker, these would get remapped according to /etc/subuid
    # outside, causing permission problems. Pretty-print help text with `-t`.
    if tool_name in ('docker', 'podman'):
        arg = ('run', '--rm', '-v', f'{host}:/mnt', image)
        if sys.stdout.isatty():
            arg = (*arg[:-1], '-t', arg[-1])
        if 'docker' in tool_name:
            # Windows does not provide getuid/getgid; Docker can run without -u.
            if hasattr(os, 'getuid') and hasattr(os, 'getgid'):
                arg = (*arg[:-1], '-u', f'{os.getuid()}:{os.getgid()}', arg[-1])

    # For Apptainer or Singularity, the users inside and outside the container
    # are the same. The working directory is also the same, unless we set it.
    elif tool_name in ('apptainer', 'singularity'):
        if not sif.parent.is_dir():
            return error('variable BABYSEG_SIF does not point to a directory')

        if not sif.exists():
            call = (tool, 'pull', sif, image)
            print(f'Cannot find image "{sif}", pulling it')
            print('Command:', *call)
            p = subprocess.run(call)
            if p.returncode != 0:
                return p.returncode

        arg = ('run', '--pwd', '/mnt', '-e', '-B', f'{host}:/mnt', sif)
        if is_cuda_image(tag):
            arg = (arg[0], '--nv', *arg[1:])

    else:
        return error(f'cannot set up unknown container tool "{tool}"')

    # Default to command-line arguments if none provided.
    if argv is None:
        argv = sys.argv[1:]

    # Summary, launch.
    print('Command:', tool, *arg)
    print('BabySeg arguments:', *argv)
    p = subprocess.run((tool, *arg, *argv))
    return p.returncode


if __name__ == '__main__':
    raise SystemExit(main())
