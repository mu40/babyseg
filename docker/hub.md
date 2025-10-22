# BabySeg

BabySeg is a brain segmentation tool for infants and young children, developed to delineate anatomical structures in MRI without preprocessing.
The tool can integrate information from multiple image volumes of variable size, resolution, and contrast in any order, provided that (1) their header geometries are correct, and (2) they are properly aligned in world space.


## Requirements

Segmenting a single image can use about 24 GB of memory on the CPU or under 2 GB of memory with a 24-GB GPU, depending on image size.
We provide a [wrapper script](https://w3id.org/babyseg/get) that facilitates setup and use of BabySeg containers.
It requires **Python 3** and supports any of the container platforms: **Docker**, **Podman**, **Apptainer**, or **Singularity**.


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

```
curl -Lo babyseg https://w3id.org/babyseg/get
chmod +x babyseg
./babyseg
```


## Optional settings

You can control BabySeg's behavior either by setting environment variables or by editing the top of the script.
Environment variables take precedence.
For example, to use an existing SIF file in your home directory:

```
export BABYSEG_SIF="$HOME/babyseg_0.0.sif"
./babyseg
```

To use Apptainer, even if Docker is also installed:
```
BABYSEG_SIF=apptainer ./babyseg
```

| Variable       | Purpose                                             | Default |
|:---------------|:----------------------------------------------------|:--------------------------------------------------------------|
| `BABYSEG_SIF`  | Specify the Apptainer or Singularity image path     | File `babyseg_${BABYSEG_TAG}.sif` in the script's directory   |
| `BABYSEG_TAG`  | Select a newer or GPU image tag                     | Latest CPU-only tag                                           |
| `BABYSEG_TOOL` | Find a container tool in `PATH` or by absolute path | First found of `docker`, `apptainer`, `singularity`, `podman` |

Additionally, `SUBJECTS_DIR` controls the working directory in the container as follows.


## Path resolution

For convenience, BabySeg temporarily mounts the host directory set in environment variable `SUBJECTS_DIR` to `/mnt` inside the container, which serves as its working directory.
If unset, `SUBJECTS_DIR` defaults to your current directory.
This enables BabySeg to access relative paths under your working directory without your setting `SUBJECTS_DIR`.


## Usage examples

Change into or set BabySeg's working directory to `~/data`:

```sh
export SUBJECTS_DIR=~/data
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


## Segmented structures

| Index | Label value | Structure name              |
|:-----:|:-----------:|:----------------------------|
| 1     | 0           | Unknown                     |
| 2     | 2           | Left-Cerebral-White-Matter  |
| 3     | 3           | Left-Cerebral-Cortex        |
| 4     | 4           | Left-Lateral-Ventricles     |
| 5     | 8           | Left-Cerebellum             |
| 6     | 10          | Left-Thalamus               |
| 7     | 11          | Left-Caudate                |
| 8     | 16          | Brain-Stem                  |
| 9     | 17          | Left-Hippocampus            |
| 10    | 18          | Left-Amygdala               |
| 11    | 28          | Left-VentralDC              |
| 12    | 41          | Right-Cerebral-White-Matter |
| 13    | 42          | Right-Cerebral-Cortex       |
| 14    | 43          | Right-Lateral-Ventricles    |
| 15    | 47          | Right-Cerebellum            |
| 16    | 49          | Right-Thalamus              |
| 17    | 50          | Right-Caudate               |
| 18    | 53          | Right-Hippocampus           |
| 19    | 54          | Right-Amygdala              |
| 20    | 60          | Right-VentralDC             |
| 21    | 169         | Left-Basal-Ganglia          |
| 22    | 176         | Right-Basal-Ganglia         |


## Tags and changes

Simple tags such as `0.0` indicate CPU-only images.
GPU-enabled image tags end in `-cu` followed by the CUDA version.
The default `latest` tag points to the latest CPU-only image.

* [`0.0`, `0.0-cu126`](https://github.com/mu40/babyseg/blob/071785c26be04bff357bbaa27627715932141807/docker/Dockerfile)\
\- release initial images


## Support

Post questions or bug reports to the [FreeSurfer mailing list](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSupport) or on [GitHub](https://github.com/mu40/babyseg).
