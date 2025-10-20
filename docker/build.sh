#!/bin/sh

# Build a Docker image using the latest checkpoint and given PyTorch platform.


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


# Checkpoint.
scripts/download.py
checkpoint=$(find checkpoints/ -name 'babyseg.*.pt' | sort -V | tail -n1)


# Version tag.
tag=$(python -c 'import babyseg; print(babyseg.__version__)')
[ "$platform" = cpu ] || tag="${tag}-${platform}"


# Build, cleanup.
docker build \
    -f "docker/Dockerfile" \
    -t "$BABYSEG_DOCKER_NAME:$tag" \
    --build-arg "CHECKPOINT=$checkpoint" \
    --build-arg "PLATFORM=$platform" \
    .

docker image prune -f
