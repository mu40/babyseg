#!/bin/sh

# Set up a virtual Python environment for development, update requirements.

set -e
venv_dir='.venv'
pt_index='https://download.pytorch.org/whl/cu129'
python="${1:-python3}"


if [ ! -f .gitignore ]; then
    echo "ERROR: not in the top-level repository directory"
    exit 1
fi


# Virtual environment.
if ! "$venv_dir/bin/python" -c 'import pip' 2>/dev/null; then
    "$python" -m venv --clear "$venv_dir"
    . "$venv_dir/bin/activate"

    # Packages.
    pip install -U pip setuptools
    pip install -i "$pt_index" torch
    pip install -e '.[dev]'

    # Requirements.
    { echo "--extra-index-url $pt_index"; pip freeze; } >requirements.txt
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
venv_dir='$venv_dir'
if [ -d "\$venv_dir/bin" ]; then
    export VIRTUAL_ENV="\$PWD/\$venv_dir"
    PATH_add "\$VIRTUAL_ENV/bin"
fi
export PYTHONPATH="\$PWD"
export CUDA_VISIBLE_DEVICES=0
export BABYSEG_DOCKER_NAME='freesurfer/babyseg'
EOF
