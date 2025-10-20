#!/usr/bin/env python3
"""Ease setup and use of BabySeg containers.

Pulls a container from Docker Hub and mounts the host directory set by
environment variable `SUBJECTS_DIR` to `/mnt` in the container, which serves as
its working directory. If unset, `SUBJECTS_DIR` defaults to your current
directory. Thus, BabySeg can access relative paths under it without requiring
you to set `SUBJECTS_DIR` explicitly.

The editable settings below control the container version, local image path,
and preferred container tools. Override them by setting environment variables
`BABYSEG_TAG`, `BABYSEG_SIF`, and `BABYSEG_TOOL`.
"""

# Container version tag.
tag = '0.0'

# Local image path for Apptainer, Singularity.
sif_file = ''

# Container tool preference. Checked left to right.
tools = ('docker', 'apptainer', 'singularity', 'podman')


# ruff: noqa: E402
import os
import shutil
import signal
import subprocess
import sys
# ruff: enable: E402


# Environment variables. Override settings above.
def env(key, default):
    """Return environment variable or default value."""
    value = os.getenv(key)
    if value:
        print(f'Applying environment variable {key}={value}')
        return value
    return default

tag = env('BABYSEG_TAG', tag)
f = os.path.join(os.path.dirname(__file__), f'babyseg_{tag}.sif')
sif_file = env('BABYSEG_SIF', sif_file if sif_file else f)
tools = env('BABYSEG_TOOL', tools)
if isinstance(tools, str):
    tools = (tools,)


# Report version. Avoid errors when piping, for example, to `head`.
signal.signal(signal.SIGPIPE, handler=signal.SIG_DFL)
hub = 'https://hub.docker.com/u/freesurfer'
print(f'Running BabySeg version {tag} from {hub}')


# Find a container system.
for tool in tools:
    path = shutil.which(tool)
    if path:
        print(f'Using {path} to manage containers')
        break

if not path:
    print(f'Cannot find container tool {tools} in PATH', file=sys.stderr)
    exit(1)


# Bind path and image URL. Mount SUBJECTS_DIR as /mnt inside the container,
# which we made the working directory when building the image. Docker and
# Podman require absolute paths.
host = os.getenv('SUBJECTS_DIR', os.getcwd())
host = os.path.abspath(host)
print(f'Will bind /mnt in container to SUBJECTS_DIR="{host}"')

image = f'freesurfer/babyseg:{tag}'
if tool != 'docker':
    image = f'docker://{image}'


# Run Docker containers with the UID and GID of the host user. This user will
# own bind mounts inside the container, preventing output files owned by root.
# Root inside rootless Podman containers maps to the non-root host user, which
# is what we want. If we set UID and GID inside the container to the non-root
# host user as for Docker, then these would get remapped according to
# /etc/subuid outside, causing problems with read and write permissions.
if tool in ('docker', 'podman'):
    arg = ('run', '--rm', '-v', f'{host}:/mnt')

    # Pretty-print help text.
    if sys.stdout.isatty():
        arg = (*arg, '-t')
    if tool == 'docker':
        arg = (*arg, '-u', f'{os.getuid()}:{os.getgid()}')

    arg = (*arg, image)


# For Apptainer or Singularity, the users inside and outside the container are
# the same. The working directory is also the same, unless we set it.
if tool in ('apptainer', 'singularity'):
    arg = ('run', '--pwd', '/mnt', '-e', f'-B{host}:/mnt', sif_file)
    if '-cu' in tag:
        arg = (arg[0], '--nv', *arg[1:])

    if not os.path.isfile(sif_file):
        print(f'Cannot find image {sif_file}, pulling it', file=sys.stderr)
        p = subprocess.run((tool, 'pull', sif_file, image))
        if p.returncode:
            exit(p.returncode)


# Summary, launch.
print('Command:', ' '.join((tool, *arg)))
print('BabySeg arguments:', *sys.argv[1:])
p = subprocess.run((tool, *arg, *sys.argv[1:]))
exit(p.returncode)
