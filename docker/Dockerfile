# Base image with Python and BabySeg on `PATH`.
FROM python:3.11-slim AS base
ENV BABYSEG_HOME="/babyseg"
ENV PATH="$BABYSEG_HOME:$PATH"
ENV PYTHONUSERBASE="$BABYSEG_HOME/env"


# Intermediate stage.
FROM base AS copy

ARG CHECKPOINT
COPY --chmod=0755 docker/entrypoint.py $BABYSEG_HOME/
COPY --chmod=0644 $CHECKPOINT $BABYSEG_HOME/checkpoints/
COPY --chmod=0644 babyseg/ $BABYSEG_HOME/babyseg/
COPY --chmod=0644 config/ $BABYSEG_HOME/config/
COPY --chmod=0644 LICENSE $BABYSEG_HOME/
RUN chmod -R +X $BABYSEG_HOME

ARG PLATFORM=cpu
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -i https://download.pytorch.org/whl/$PLATFORM \
        torch==2.9.0

RUN --mount=type=cache,target=/root/.cache \
    pip install --user \
        https://github.com/dalcalab/voxel/archive/24cb8b10d698dd1dbce14426080c954a50b27858.zip \
        https://github.com/mu40/katy/archive/741f772c5ec8c2d1598dddae061e40e8c9a46722.zip \
        nibabel \
    && :

WORKDIR /artifacts
RUN python -V >python.txt
RUN pip freeze >requirements.txt


# Build artifact stage.
FROM scratch AS export
COPY --from=copy /artifacts/*.txt /


# Copy once to avoid unnecessary layers.
FROM base
COPY --from=copy $BABYSEG_HOME $BABYSEG_HOME
WORKDIR /mnt
ENTRYPOINT ["entrypoint.py"]
