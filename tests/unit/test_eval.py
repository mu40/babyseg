"""Test evaluation module."""

import babyseg
import katy
import pytest
import torch
import voxel as vx


@pytest.mark.parametrize('value, expected', [
    (    0, torch.uint8),
    (   -1, torch.int16),
    (2**15, torch.uint16),
    (2**16, torch.int32),
])
def test_select_dtype_values(value, expected):
    """Test selecting data types."""
    x = torch.tensor(value)
    assert babyseg.eval.select_dtype(x) == expected


def test_select_dtype_long():
    """Test if values requiring long integers raise an error."""
    x = torch.tensor(2**32)
    with pytest.raises(ValueError):
        babyseg.eval.select_dtype(x)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_select_dtype_illegal(dtype):
    """Test if selecting types for illegal input data raises errors."""
    x = torch.tensor(1, dtype=dtype)
    with pytest.raises(TypeError):
        babyseg.eval.select_dtype(x)


@pytest.fixture
def config(monkeypatch):
    """Set up mock segmentation environment, return configuration."""
    # Mock configuration.
    labels = 3
    config = {
        'eval': {'orientation': 'LIA', 'spacing': 1, 'divisor': 2},
        'synthesis': {'kwargs': {'optimize': 'some_path'}},
    }

    class Model:
        def eval(self):
            return self

        def __call__(self, x):
            """Return a tensor with `labels` channels, propagating NaN."""
            return x.sum(dim=1, keepdim=True).expand(-1, labels, *x.shape[2:])

    def load_model(*args, **kwargs):
        return Model()

    monkeypatch.setattr(katy.io, 'load', lambda _: range(labels))
    monkeypatch.setattr(babyseg.config, 'load_model', load_model)
    return config


@pytest.fixture
def inputs(monkeypatch):
    """Map paths to segmentation inputs and feed them to `vx.load_volume`."""
    inputs = {}
    monkeypatch.setattr(vx, '_load_volume', vx.load_volume, raising=False)
    monkeypatch.setattr(vx, 'load_volume', lambda f: inputs[f])
    return inputs


def test_segment_without_outputs():
    """Test if segmenting without specifying outputs raises an error."""
    with pytest.raises(ValueError) as e:
        babyseg.eval.segment(config={}, images=[])

    assert 'output' in str(e.value)


def test_segment_eval_mode(monkeypatch):
    """Test if segmentation sets evaluation mode."""
    class Model:
        pass

    def load_model(*args, **kwargs):
        return Model()

    monkeypatch.setattr(babyseg.config, 'load_model', load_model)
    with pytest.raises(AttributeError) as e:
        babyseg.eval.segment(config={}, images=[], out_seg='out.nii')

    assert e.value.name == 'eval'


def test_segment_output_probabilities(config, inputs, tmp_path):
    """Test output probability map properties."""
    size = (2, 2, 2)
    inputs['in.nii'] = vx.Volume(torch.ones(1, *size))
    images = tuple(inputs)
    f = tmp_path / 'out.nii.gz'

    babyseg.eval.segment(config, images, out_prob=f)
    out = vx._load_volume(f)
    assert out.dtype == torch.float32
    assert out.num_channels > 1
    assert out.baseshape == size
    assert out.min() >= 0
    assert out.max() <= 1


def test_segment_output_labels(config, inputs, tmp_path):
    """Test output label map properties."""
    size = (1, 2, 2, 2)
    inputs['in.nii'] = vx.Volume(torch.ones(size))
    images = tuple(inputs)
    f = tmp_path / 'out.nii.gz'

    babyseg.eval.segment(config, images, out_seg=f)
    out = vx._load_volume(f)
    assert out.dtype == torch.uint8
    assert out.shape == size


def test_segment_nan(config, inputs, tmp_path):
    """Test if NaN inputs will propagate to the probability map."""
    x = vx.Volume(torch.ones(1, 2, 2, 2))
    x[0, 0, 0, 0] = torch.nan
    images = 'in.nii'
    inputs[images] = x
    f = tmp_path / 'out.nii.gz'

    babyseg.eval.segment(config, images, out_prob=f)
    assert not vx._load_volume(f).isnan().any()


def test_segment_multiple_inputs(config, inputs, tmp_path):
    """Test if segmenting multiple inputs yields one output label map."""
    inputs['a.nii'] = vx.Volume(torch.ones(1, 2, 2, 2))
    inputs['b.nii'] = vx.Volume(torch.ones(3, 1, 1, 1))
    inputs['c.nii'] = vx.Volume(torch.ones(1, 3, 3, 3))
    images = tuple(inputs)
    f = tmp_path / 'out.nii.gz'

    babyseg.eval.segment(config, images, out_seg=f)
    out = vx._load_volume(f)
    assert out.num_channels == 1
