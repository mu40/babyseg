#!/bin/sh

# Set up a virtual Python environment for development, update requirements.

set -e
ENV='.venv'
IND='https://download.pytorch.org/whl/cu129'


if [ ! -f .gitignore ]; then
    echo "ERROR: not in the top-level repository directory"
    exit 1
fi


# Virtual environment.
if [ ! -d "$ENV" ]; then
    python=$(
        find /usr/bin/ /usr/local/bin/ -name 'python*' |
        grep 'python[0-9.]*$' |
        sort -V |
        tail -n1
    )
    "$python" -m venv "$ENV"
    . "$ENV/bin/activate"

    # Packages.
    pip install -U pip setuptools
    pip install -i "$IND" torch
    pip install \
        https://github.com/dalcalab/voxel/archive/24cb8b10d698dd1dbce14426080c954a50b27858.zip \
        https://github.com/mu40/katy/archive/741f772c5ec8c2d1598dddae061e40e8c9a46722.zip \
        nibabel \
        pytest \
        ruff \
        shellcheck-py \
        typos \
    && :

    # Requirements.
    { echo "--extra-index-url $IND"; pip freeze; } >requirements.txt
fi


# Git hooks.
if [ -d .git/hooks ]; then
    cp -v hooks/* .git/hooks
    if f=$(command -v commit-msg.py 2>/dev/null); then
        ln -vsfn "$f" .git/hooks/commit-msg
    fi
fi


# Environment manager.
cat >.envrc <<EOF
[ -d "$ENV" ] && . "$ENV/bin/activate"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export BABYSEG_DOCKER_NAME='freesurfer/babyseg'
EOF
