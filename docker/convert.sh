#!/bin/sh

# Convert a Docker image to a SIF container for Apptainer or Singularity.


# Environment.
set -e
: "${BABYSEG_DOCKER_NAME:?environment variable unset}"


# Arguments.
if [ $# -ne 2 ] || ! [ -d "$2" ] ; then
    echo "Usage: $(basename "$0") PYTORCH_PLATFORM OUT_DIR"
    exit 1
fi
platform="$1"
out_dir="$2"


# Version tag.
version=$(python -c 'import babyseg; print(babyseg.__version__)')
tag="$BABYSEG_DOCKER_NAME:$version"
[ "$platform" = cpu ] || tag="${tag}-${platform}"


# Build, cleanup.
out="$(basename "$tag" | tr ':' '_')"
out="$out_dir/$out.sif"
apptainer build -f "$out" "docker-daemon://$tag"
