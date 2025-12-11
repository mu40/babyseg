# BabySeg

BabySeg is a brain segmentation tool for infants and young children, developed to delineate anatomical structures in MRI without preprocessing.
The tool can integrate information from multiple **NIfTI image** volumes of variable contrast, shape, and resolution in any order, provided that (1) their **header geometries** are correct, and (2) they are **properly aligned** in world space.


## Running BabySeg

The recommended way to run BabySeg is in a [container](https://hub.docker.com/r/freesurfer/babyseg).


## Attribution

If you find this work useful, please cite [the BabySeg paper](https://arxiv.org/abs/2512.05114):

```bibtex
@inproceedings{hoffmann2025deep,
  title={{Deep infant brain segmentation from multi-contrast MRI}},
  author={Hoffmann, Malte and Z{\"o}llei, Lilla and Dalca, Adrian V},
  booktitle={{Asilomar Conference on Signals, Systems, and Computers}},
  year={2025},
  publisher={IEEE}
}
```
