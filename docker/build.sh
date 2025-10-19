#!/bin/sh

# Set up a virtual Python environment for development, update requirements.


# Environment.
set -e
: "${BABYSEG_DOCKER_NAME:?environment variable unset}"
if [ ! -f .gitignore ]; then
    echo "ERROR: not in the top-level repository directory"
    exit 1
fi


# Arguments.
if [ $# -ne 1 ]; then
    echo "Usage: $(basename "$0") PYTORCH_PLATFORM"
    exit 1
fi
platform="$1"


# Platform.
if ! curl -sfI "https://download.pytorch.org/whl/$platform" >/dev/null; then
    echo "ERROR: PyTorch platform \"$platform\" does not exist"
    exit 1
fi


# Checkpoint.
scripts/download.py
checkpoint=$(find checkpoints/ -name 'babyseg.*.pt' | sort -V | tail -n1)


# Version tag.
version=$(python -c 'import babyseg; print(babyseg.__version__)')
tag="$BABYSEG_DOCKER_NAME:$version"
[ "$platform" = cpu ] || tag="${tag}-${platform}"


# Build, cleanup.
docker build \
    -f "docker/Dockerfile" \
    -t "$tag" \
    --build-arg "CHECKPOINT=$checkpoint" \
    --build-arg "PLATFORM=$platform" \
    .

docker image prune -f
