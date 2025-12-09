"""Test evaluation module."""

import babyseg
import katy
import pytest
import torch
import unittest.mock
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
        'eval': {'orientation': 'LIA', 'spacing': 1.5, 'divisor': 2},
        'synthesis': {'kwargs': {'optimize': 'labels.json'}},
    }

    # Return tensors with `labels` channels, propagating NaN.
    def forward(x):
        return x.sum(dim=1, keepdim=True).expand(-1, labels, *x.shape[2:])

    model = unittest.mock.Mock(side_effect=forward)
    load_model = unittest.mock.Mock(return_value=model)
    monkeypatch.setattr(babyseg.config, 'load_model', load_model)
    monkeypatch.setattr(katy.io, 'load', lambda _: range(labels))
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
    with pytest.raises(ValueError, match='output'):
        babyseg.eval.segment(config={}, images=[])


def test_segment_load_config(monkeypatch):
    """Test if segmentation loads a specified configuration."""
    mock = unittest.mock.Mock(side_effect=RuntimeError('LOAD'))
    monkeypatch.setattr(babyseg.config, 'load', mock)

    image = 'in.nii'
    config = 'config.json'
    with pytest.raises(RuntimeError, match='LOAD'):
        babyseg.eval.segment(config, image)

    mock.assert_called_once_with(config)


def test_segment_load_model(monkeypatch):
    """Test if segmentation passes settings to model setup."""
    mock = unittest.mock.create_autospec(babyseg.config.load_model)
    mock.side_effect = RuntimeError('LOAD_MODEL')
    monkeypatch.setattr(babyseg.config, 'load_model', mock)

    checkpoint = 'checkpoint.pt'
    device = 'GPU'
    with pytest.raises(RuntimeError, match='LOAD_MODEL'):
        babyseg.eval.segment(
            config={},
            images='in.nii',
            out_seg='out.nii',
            checkpoint=checkpoint,
            device=device,
        )

    mock.assert_called_once()
    args, kwargs = mock.call_args
    received = (*args, *kwargs.values())
    assert checkpoint in received
    assert device in received


def test_segment_eval_mode(monkeypatch):
    """Test if segmentation sets evaluation mode."""
    model = unittest.mock.Mock()
    model.eval.side_effect = RuntimeError('EVAL')

    mock = unittest.mock.Mock(return_value=model)
    monkeypatch.setattr(babyseg.config, 'load_model', mock)

    with pytest.raises(RuntimeError, match='EVAL'):
        babyseg.eval.segment(config={}, images=[], out_seg='out.nii')

    model.eval.assert_called_with()


def test_segment_output_probabilities(config, inputs, tmp_path):
    """Test output probability map properties."""
    x = vx.Volume(torch.ones(1, 2, 2, 2))
    inputs['in.nii'] = x
    images = tuple(inputs)
    f = tmp_path / 'out.nii.gz'

    babyseg.eval.segment(config, images, out_prob=f)
    out = vx._load_volume(f)

    # Expect probabilities.
    assert out.dtype == torch.float32
    assert out.min() >= 0
    assert out.max() <= 1

    # Expect same geometry as input, more than one channel.
    assert out.num_channels > 1
    assert out.baseshape == x.shape[1:]
    assert out.geometry.orientation == x.geometry.orientation
    assert out.geometry.spacing.equal(x.geometry.spacing)


def test_segment_output_labels(config, inputs, tmp_path):
    """Test output label map properties."""
    x = vx.Volume(torch.ones(1, 2, 2, 2))
    inputs['in.nii'] = x
    images = tuple(inputs)
    f = tmp_path / 'out.nii.gz'

    babyseg.eval.segment(config, images, out_seg=f)
    out = vx._load_volume(f)

    # Expect smallest integer type for small constant-valued input.
    assert out.dtype == torch.uint8

    # Expect same geometry as input, one channel.
    assert out.shape == x.shape
    assert out.geometry.orientation == x.geometry.orientation
    assert out.geometry.spacing.equal(x.geometry.spacing)


def test_segment_output_lead(config, inputs, tmp_path):
    """Test conformed lead image properties."""
    size = (1, 2, 2, 2)
    images = 'in.nii'
    inputs[images] = vx.Volume(torch.ones(size))
    f = tmp_path / 'out.nii.gz'

    babyseg.eval.segment(config, images, out_lead=f)
    out = vx._load_volume(f)

    # Expect hard coded type and shape, number of input channels.
    assert out.dtype == torch.float32
    assert out.baseshape == (128, 128, 128)
    assert out.num_channels == size[0]

    # Expect configuration geometry.
    assert out.geometry.orientation == config['eval']['orientation']
    assert out.geometry.spacing.eq(config['eval']['spacing']).all()


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
