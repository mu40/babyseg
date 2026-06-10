# BabySeg container images

This document describes how to run BabySeg using container images from [Docker Hub](https://hub.docker.com/r/freesurfer/babyseg).

## Requirements

We provide a [wrapper script](https://get.babyseg.io) that facilitates setup and use of BabySeg containers.
It requires **Python** and supports any of the container platforms: **Docker**, **Podman**, **Apptainer**, or **Singularity**.

## Initial setup

On first run, the script pulls the latest image.
**Apptainer** or **Singularity** store it as a SIF file in the directory containing the script.
Separately, they create a cache under your home directory.
If your home quota is low, redirect the cache to a different path:

```sh
d=$(mktemp -d)
export APPTAINER_CACHEDIR="$d"
export APPTAINER_TMPDIR="$d"
```

Download and run the script, which auto-detects container tools in your `PATH`:

```sh
curl -Lo babyseg get.babyseg.io
chmod +x babyseg
./babyseg -h
```

## Optional settings

You can control BabySeg's behavior by setting environment variables or by editing the top of the script.
Environment variables take precedence.
For example, to use a GPU image in your home directory, downloading it if it does not exist:

```sh
export BABYSEG_TAG="0.0-cu126"
export BABYSEG_SIF="$HOME"
./babyseg
```

To use Docker, even if Apptainer is also installed:

```sh
BABYSEG_TOOL=docker ./babyseg
```

| Variable       | Purpose                                                                      | Default                                                       |
|:---------------|:-----------------------------------------------------------------------------|:--------------------------------------------------------------|
| `BABYSEG_MNT`  | Define the working directory inside the container                            | Your current working directory                                |
| `BABYSEG_SIF`  | Control where the tool stores and looks for  Apptainer or Singularity images | The directory containing the BabySeg script                   |
| `BABYSEG_TAG`  | Select a newer or GPU image tag                                              | Latest CPU-only tag at time of download                       |
| `BABYSEG_TOOL` | Find a container tool by name or by absolute path                            | First found of `apptainer`, `singularity`, `docker`, `podman` |

## Path resolution

For convenience, BabySeg temporarily mounts the host directory set in environment variable `BABYSEG_MNT` to `/mnt` inside the container, which serves as its working directory.
If you do not set `BABYSEG_MNT`, it defaults to your current directory.
This enables BabySeg to access relative paths under your working directory without requiring you to set `BABYSEG_MNT`.

## Usage examples

Change into or set BabySeg's working directory to `~/data`:

```sh
export BABYSEG_MNT=~/data
```

Segment image `~/data/in.nii.gz`, saving the label map as `~/data/out.nii.gz`:

```sh
./babyseg -o out.nii.gz in.nii.gz
```

Estimate a label map from several same-subject registered images in `~/data/bert/`:

```sh
./babyseg -o out.nii bert/t1.nii bert/t2.nii bert/flair.nii
```

Output probability maps `~/data/prob.nii` instead of a label map:

```sh
./babyseg -p prob.nii in.nii
```

Enable GPU acceleration for SIF images with `-cu` in the tag *and* file name:

```sh
./babyseg -go out.nii in.nii
```

Display help:

```sh
./babyseg -h
```

## Tags and changes

Simple tags such as `0.0` indicate CPU-only images.
GPU-enabled image tags end in `-cu` followed by the CUDA version.
The default `latest` tag points to the latest CPU-only image.

* [`0.0`, `0.0-cu126`](https://github.com/mu40/babyseg/blob/071785c26be04bff357bbaa27627715932141807/docker/Dockerfile)\
\- release initial images
