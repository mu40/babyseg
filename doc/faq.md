# FAQ

This FAQ covers common questions about inputs, usage, and performance.


## What age range does BabySeg support?

BabySeg supports the first five years of postnatal development, including pre-term newborns.


## Do I need to specify the child's age?

No. BabySeg does not require age information.


## Is there a special mode for newborns?

No. BabySeg uses the same model for newborns and older children.


## Does BabySeg require a specific input format?

BabySeg requires 3D NIfTI inputs with valid headers.
When you provide multiple inputs, they must be aligned in world space.


## Do I need to preprocess the input images?

No. BabySeg does not require preprocessing.


## Can I run BabySeg on data processed with other pipelines?

Yes. You can run BabySeg on previously processed images.
However, repeated resampling degrades image quality.


## What if the segmentation looks wrong?

Ensure inputs have valid headers and are aligned in world space.
Low image quality, motion, and other artifacts can degrade results.
Consider trying a different input combination.


## Which MRI sequences does BabySeg support?

BabySeg does not make any assumptions about MRI sequence or contrast.
While 3D T1- and T2-weighted sequences typically yield the best results, you can run the tool on any sequence.


## What resolution does BabySeg expect?

BabySeg accepts any voxel size, as it resamples data internally.


## Can I provide more than one input image?

Yes. You can pass as many images from the same session as memory allows.
Their order does not matter, but they must be aligned in world space.


## Should I provide more than one input image?

We recommend trying several input combinations to select the best result.
Passing complementary contrasts can slightly boost accuracy, as long as they are well aligned.


## Do all inputs need the same resolution or volume shape?

No. You can pass a combination of inputs with any shape and resolution.


## Do input or output filenames matter?

No. Name input and output files however you like.


## Does the tool require a GPU?

No. BabySeg runs fine on the CPU but will be faster on the GPU.


## How do I get the tool to use the GPU?

GPU use requires a GPU-enabled container image.
See [docker/README.md](../docker/README.md) for details.


## How much memory does BabySeg need?

Segmenting a single image can take ~24 GB of memory on the CPU or <2 GB of memory with a recent 24-GB GPU, depending on image size.


## How long does BabySeg take?

Segmenting a single image can take ~30 seconds on the GPU or ~2 minutes on a CPU using a single thread, depending on hardware.


## Is BabySeg deterministic?

Yes. Given the same inputs and environment, BabySeg produces the same output.


## What labels does the tool segment?

See [labels/README.md](../labels/README.md) for a list of anatomical structures.


## How do I compute label-wise statistics from the segmentation?

You can save per-region statistics to `out.txt` using `mri_segstats` from [FreeSurfer](https://freesurfer.net) with BabySeg's [lookup table](../labels/BabySegColorLUT.txt):

```sh
mri_segstats --seg babyseg.nii --ctab BabySegColorLUT.txt --o out.txt
```
