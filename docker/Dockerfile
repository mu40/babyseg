# Base image with Python and BabySeg on path (eventually).
FROM python:3.11-slim AS base
ENV FREESURFER_HOME="/freesurfer"
ENV PYTHONUSERBASE="$FREESURFER_HOME/env"
ENV PATH="$FREESURFER_HOME:$PATH"

# Intermediate stage with build requirements not needed in final image.
FROM base AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential

# BabySeg from local folder.
COPY --chmod=0755 segment.py $FREESURFER_HOME/
COPY --chmod=0644 jane/config/defaults.json *.json *.pt *.zip $FREESURFER_HOME/

RUN --mount=type=cache,target=/root/.cache \
    pip3 install --user -i https://download.pytorch.org/whl/cpu torch==2.8.0

RUN --mount=type=cache,target=/root/.cache \
    pip3 install --user \
        $FREESURFER_HOME/jane.zip \
        nibabel \
        https://github.com/mu40/katy/archive/05b9e9d2d66967b9db0cc9373cf69f918a940308.zip \
        https://github.com/dalcalab/voxel/archive/24cb8b10d698dd1dbce14426080c954a50b27858.zip 

# Build artifacts.
RUN python3 -V >python.txt
RUN python3 -m pip freeze >requirements.txt


FROM scratch AS export
COPY --from=build *.txt /


# Only copy files needed, once, to avoid unnecessary layers.
FROM base
COPY --from=build $FREESURFER_HOME $FREESURFER_HOME
ENTRYPOINT ["babyseg.py"]
