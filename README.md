# BabySeg

BabySeg is a brain segmentation tool for infants and young children, designed to delineate anatomical structures in MRI without preprocessing.
The tool can integrate information from multiple NIfTI image volumes of variable contrast, shape, and resolution in any order, provided that (1) their header geometries are correct, and (2) they are properly aligned in world space.


## Running BabySeg

The recommended way to run BabySeg is in a [container](docker/README.md).


## Attribution

If you find this work useful, please cite the BabySeg [paper](https://arxiv.org/abs/2512.05114):

```bibtex
@inproceedings{hoffmann2025deep,
  title={{Deep infant brain segmentation from multi-contrast MRI}},
  author={Hoffmann, Malte and Z{\"o}llei, Lilla and Dalca, Adrian V},
  booktitle={{Asilomar Conference on Signals, Systems, and Computers}},
  year={2025},
  publisher={IEEE}
}
```


## Support

Read the [FAQ](doc/faq.md), post questions to the FreeSurfer [mailing list](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSupport), or file bugs on GitHub.
