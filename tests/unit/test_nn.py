"""Tests for neural network module."""

import babyseg
import pytest
import torch


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_conv_shape(ndim):
    """Test output shape of convolution parametrized by dimension."""
    batch, i, o = 1, 2, 3
    space = (4, 5, 6)[:ndim]
    x = torch.empty(batch, i, *space)

    layer = babyseg.nn.Conv(
        ndim=ndim,
        in_channels=i,
        out_channels=o,
        kernel_size=3,
        padding=1,
        groups=1,
    )

    y = layer(x)
    assert y.shape == (batch, o, *space)


def test_group_conv_shape():
    """Test shape of 2D group convolution."""
    ndim = 2
    batch, group, i, o = 1, 5, 2, 3
    space = [4] * ndim
    x = torch.empty(batch, group, i, *space)

    layer = babyseg.nn.GroupConv(
        ndim=ndim,
        in_channels=i,
        out_channels=o,
        kernel_size=3,
        padding='same',
        dilation=1,
    )

    assert layer(x).shape == (batch, group, o, *space)


def test_group_max_pool_shape():
    """Test shape of group-aware max pooling."""
    ndim = 2
    batch, group, channels = 2, 5, 3
    kernel = 2
    space = [4] * ndim
    x = torch.empty(batch, group, channels, *space)

    layer = babyseg.nn.GroupMaxPool(
        ndim=ndim,
        kernel_size=[kernel] * ndim,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
    )

    expected = (batch, group, channels, *(s // kernel for s in space))
    assert layer(x).shape == expected


def test_group_upsample_shape():
    """Test shape of group-aware upsampling."""
    ndim = 1
    batch, group, channels = 5, 3, 4
    scale = 2
    space = [1] * ndim
    x = torch.empty(batch, group, channels, *space)

    layer = babyseg.nn.GroupUpsample(scale_factor=scale, mode='nearest')
    expected = (batch, group, channels, *(s * scale for s in space))
    assert layer(x).shape == expected


@pytest.mark.parametrize('activation', ['ReLU', torch.nn.ReLU])
def test_group_net_shape(activation):
    """Test shape of group-aware U-Net."""
    ndim = 2
    batch, out = 1, 3
    space = [2] * ndim
    x = torch.empty(batch, 4, *space)

    model = babyseg.nn.GroupNet(
        ndim=ndim,
        out=out,
        enc=(1,),
        dec=(1,),
        add=(1,),
        rep=1,
        act=activation,
        clip=(0, 1),
    )

    assert model(x).shape == (batch, out, *space)


def test_group_net_invalid_clip():
    """Test instantiation group network with scalar clipping argument."""
    ndim = 3
    with pytest.raises(ValueError, match='clip'):
        babyseg.nn.GroupNet(ndim, clip=1)
