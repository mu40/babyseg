"""Tests for full segmentation using container images."""

import babyseg
import docker.wrapper
import os
import pathlib
import pytest
import shutil
import torch


from test_segment import *  # noqa: F403


ROOT = pathlib.Path(__file__).resolve().parents[2]
SIF = os.getenv('BABYSEG_SIF', str(ROOT))
TAG = os.getenv('BABYSEG_TAG', babyseg.__version__)
TOOLS_SIF = {'apptainer', 'singularity'}
TOOLS_ALL = {'docker', 'podman', *TOOLS_SIF}
TOOLS_GPU = {'apptainer'}


def combinations():
    """Generate container tool-device combinations."""
    devices = ('cpu', 'cuda')

    for tool in TOOLS_ALL:
        marks = (pytest.mark.container, getattr(pytest.mark, tool))
        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                continue

            if device == 'cuda' and tool not in TOOLS_GPU:
                continue

            if device == 'cuda' and not docker.wrapper.is_cuda_image(TAG):
                continue

            yield pytest.param(f'{tool}-{device}', marks=marks)


@pytest.fixture(scope='module', params=combinations())
def tool(request):
    """Configure environment to use container tool for segmentation."""
    name, device = request.param.split('-')
    if not shutil.which(name):
        pytest.skip(f'{name} not found')

    # Environment. Keep copy to restore, `BABYSEG_MNT` for changes applied in
    # `segment`. Cannot use `monkeypatch` in module-scoped fixture.
    keys = ('BABYSEG_SIF', 'BABYSEG_TAG', 'BABYSEG_TOOL', 'BABYSEG_MNT')
    env = {k: os.environ.get(k) for k in keys}
    os.environ['BABYSEG_SIF'] = SIF
    os.environ['BABYSEG_TAG'] = TAG
    os.environ['BABYSEG_TOOL'] = name

    image = docker.wrapper.sif_image_path(SIF, TAG)
    if name in TOOLS_SIF and not image.is_file():
        pytest.skip(f'{image.name} not found')

    # GPU flag for GPU-enabled images if tool in `TOOLS_GPU`.
    runner = docker.wrapper.main
    if device == 'cuda':
        def runner(argv):
            return docker.wrapper.main(argv=['-g', *argv])

    try:
        yield runner

    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k)

            else:
                os.environ[k] = v


@pytest.fixture(scope='module')
def segment(paths, tool, tmp_path_factory):
    """Segment and add the output path."""
    # Output path.
    tmp_path = tmp_path_factory.mktemp(pathlib.Path(__file__).stem)
    paths = dict(paths)
    paths['pred'] = tmp_path / 'out.label_map.nii.gz'

    # Working directory in the container.
    shutil.copy(paths['image'], tmp_path)
    os.environ['BABYSEG_MNT'] = str(tmp_path)
    assert tool(argv=['-o', paths['pred'].name, paths['image'].name]) == 0

    return paths
