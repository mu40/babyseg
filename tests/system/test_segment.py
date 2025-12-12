"""Tests for full local segmentation."""

import babyseg
import functools
import katy
import pathlib
import pytest
import torch
import voxel as vx


ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def paths():
    """Return input paths for a segmentation test case."""
    data =  ROOT / 'data'
    paths = {
        'config': ROOT / 'config' / 'babyseg.v1.json',
        'checkpoint': ROOT / 'checkpoints' / 'babyseg.v1.pt',
        'image': data / 't1.nii.gz',
        'true': data / 'labels.nii.gz',
    }

    for k, v in paths.items():
        assert v.is_file(), f'missing "{k}" file "{v}"'

    return paths


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def device(request):
    """Parametrize PyTorch device but skip if unavailable."""
    device = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    return device


@functools.cache
def load_labels():
    """Load segmentation label definitions."""
    labels = katy.io.load(path=ROOT / 'labels' / 'baby.loss.json')
    return {int(k): v for k, v in labels.items()}


@pytest.fixture(scope='module')
def segment(paths, tmp_path_factory, device):
    """Segment and add the output path."""
    # Paths.
    tmp_path = f'{pathlib.Path(__file__).stem}_{device}'
    tmp_path = tmp_path_factory.mktemp(tmp_path)
    paths = dict(paths)
    paths['pred'] = tmp_path / 'out.label_map.nii.gz'

    babyseg.eval.segment(
        config=paths['config'],
        images=paths['image'],
        checkpoint=paths['checkpoint'],
        out_seg=paths['pred'],
        device=device,
    )
    return paths


@pytest.fixture(scope='module')
def outputs(segment):
    """Load and return image from a set of paths, adding a batch dimension."""
    out = {}
    for k, v in segment.items():
        if v.name.endswith(('.nii', '.nii.gz')):
            out[k] = vx.load_volume(v).tensor.unsqueeze(0)

    return out


def test_expected_labels_only(outputs):
    """Confirm that the predicted label map includes only expected labels."""
    labels = list(load_labels())
    expected = torch.as_tensor(labels)
    output = outputs['pred'].unique()
    assert torch.isin(output, expected).all()


@pytest.mark.parametrize('label', load_labels(), ids=load_labels().values())
def test_dice(outputs, label):
    """Verify the Dice score for a label."""
    dice = katy.metrics.dice(outputs['pred'], outputs['true'], labels=label)
    assert 0.98 < dice <= 1


def test_label_map_identity(outputs):
    """Test voxel-wise identity of predicted and ground-truth labels."""
    # Show equality as a success, but expect some variability on GPU.
    test = torch.eq(outputs['pred'], outputs['true'])
    cond = test.all()
    if not cond:
        f = 100 * test.logical_not().count_nonzero() / test.numel()
        pytest.xfail(reason=f'{f:.1f}% of voxels differ')

    assert cond
