# BabySeg

BabySeg is a brain segmentation tool for infants and young children, designed to delineate anatomical structures in MRI without preprocessing.
The tool can integrate information from multiple NIfTI image volumes of variable contrast, shape, and resolution in any order, provided that (1) their header geometries are correct, and (2) they are properly aligned in world space.


## Running BabySeg

The recommended way to run BabySeg is in a [container](docker/README.md).


## Video

Watch a 30-minute [video](https://www.youtube.com/watch?v=UZwUZQXhnBo) recording of a talk introducing the motivation and methodology behind BabySeg.


## Attribution

If you find this work useful, please cite the relevant papers below.

BabySeg [method](https://arxiv.org/abs/2512.05114):

```bibtex
@inproceedings{hoffmann2025deep,
  title={{Deep infant brain segmentation from multi-contrast MRI}},
  author={Hoffmann, Malte and Z{\"o}llei, Lilla and Dalca, Adrian V},
  booktitle={{Asilomar Conference on Signals, Systems, and Computers}},
  pages={974--981},
  year={2025},
  publisher={IEEE}
}
```

Data [engine](https://arxiv.org/abs/2507.13458):

```bibtex
@article{hoffmann2025domain,
  title={Domain-randomized deep learning for neuroimage analysis},
  author={Hoffmann, Malte},
  journal={IEEE Signal Processing Magazine},
  volume={42},
  number={4},
  pages={78--90},
  year={2025},
  publisher={IEEE}
}
```

## Support

Read the [FAQ](doc/faq.md), post questions to the FreeSurfer [mailing list](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSupport), or file bugs on GitHub.
