"""Neural network module."""

import babyseg as bs
import katy as kt
import logging
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Conv(nn.Module):
    """A wrapper to `torch.nn.Conv{N}d` that takes `N` as an argument."""

    def __init__(self, ndim, *args, **kwargs):
        """Initialize the operation with arguments to `torch.nn.Conv{N}d`.

        Parameters
        ----------
        ndim : {1, 2, 3}
            Dimensionality `N`.
        *args : tuple, optional
            Positional arguments.
        **kwargs : dict, optional
            Keyword arguments.

        """
        super().__init__()
        self.conv = getattr(nn, f'Conv{ndim}d')(*args, **kwargs)

    def forward(self, x, /):
        """Apply the operation."""
        return self.conv(x)


class GroupConv(nn.Module):
    """Group convolution with cross-talk along the group dimension."""

    def __init__(self, ndim, in_channels, out_channels, *args, **kwargs):
        """Initialize the operation with arguments to `torch.nn.Conv{N}d`.

        Parameters
        ----------
        ndim : {1, 2, 3}
            Dimensionality `N`.
        in_channels : int
            Group convolution input channels.
        out_channels : int
            Group convolution output channels.
        *args : tuple, optional
            Positional arguments.
        **kwargs : dict, optional
            Keyword arguments.

        """
        super().__init__()
        self.conv = getattr(nn, f'Conv{ndim}d')
        self.conv_b = self.conv(in_channels, out_channels, *args, **kwargs)
        self.conv_m = self.conv(in_channels, out_channels, *args, **kwargs)

    def forward(self, x, /):
        """Apply the operation across `G` groups.

        Parameters
        ----------
        x : (B, G, in_channels, *size) torch.Tensor
            Input of `N`-element spatial `size`.

        Returns
        -------
        (B, G, out_channels, ...) torch.Tensor
            Output of spatial size depending on arguments.

        """
        m = x.mean(dim=1)  # B, C, *X
        y = x.view(-1, *x.shape[2:])  # BG, C, *X
        m = self.conv_m(m)  # B, C', *X'
        y = self.conv_b(y)  # BG, C', *X'
        m = m.unsqueeze(1)  # B, 1, C', *X'
        y = y.view(*x.shape[:2], *y.shape[1:])  # B, G, C', *X'
        return 0.5 * (y + m)


class GroupMaxPool(nn.Module):
    """Max pooling in `N` dimensions for tensors with group entries."""

    def __init__(self, ndim, *args, **kwargs):
        """Initialize the operation with arguments to `torch.nn.MaxPool{N}d`.

        Parameters
        ----------
        ndim : {1, 2, 3}
            Dimensionality `N`.
        *args : tuple, optional
            Positional arguments.
        **kwargs : dict, optional
            Keyword arguments.

        """
        super().__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(*args, **kwargs)

    def forward(self, x, /):
        """Apply the operation across `G` groups.

        Parameters
        ----------
        x : (B, G, C, *size) torch.Tensor
            Input of `N`-element spatial `size`.

        Returns
        -------
        (B, G, C, ...) torch.Tensor
            Output of spatial size depending on arguments.

        """
        y = x.view(-1, *x.shape[2:])  # BG, C, *X
        y = self.pool(y)  # BG, C, *X'
        return y.view(*x.shape[:2], *y.shape[1:])  # B, G, C, *X'


class GroupUpsample(nn.Upsample):
    """Version of `torch.nn.Upsample` tensors with group entries."""

    def forward(self, x, /):
        """Apply the operation across `G` groups.

        Parameters
        ----------
        x : (B, G, C, *size) torch.Tensor
            Input of `N`-element spatial `size`.

        Returns
        -------
        (B, G, C, ...) torch.Tensor
            Output of spatial size depending on arguments.

        """
        y = x.view(-1, *x.shape[2:])  # BG, C, *X
        y = super().forward(y)  # BG, C, *X'
        return y.view(*x.shape[:2], *y.shape[1:])  # B, G, C, *X'


class GroupNet(nn.Module):
    """An `N`-dimensional U-Net using group convolutions."""

    def __init__(
        self,
        dim=3,
        inp=1,
        out=1,
        *,
        enc=(24, 48, 96, 192, 384),
        dec=(384, 192, 96, 48, 24),
        add=(),
        rep=1,
        act=nn.ELU,
        conv=GroupConv,
        clip=(0.01, 0.99),
    ):
        """Initialize the model.

        Parameters
        ----------
        dim : int, optional
            Number of spatial dimensions `N`.
        inp : int, optional
            Number of input channels.
        out : int, optional
            Number of output channels.
        enc : sequence of int, optional
            Number of encoding convolutional filters at each level.
        dec : sequence of int, optional
            Number of decoding convolutional filters at each level.
        add : sequence of int, optional
            Number of additional convolutional filters at the final level.
        rep : int, optional
            Number of repeats for each convolutional operation.
        act : str or nn.Module or type, optional
            Activation function after each convolution.
        conv : str, optional
            Cross-convolution layer.
        clip : tuple of float, optional
            Clip at min-max quantiles in [0, 1] before normalizing.

        """
        super().__init__()
        mode = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}[dim]
        self.clip = clip

        # Blocks.
        conv = bs.config.build(
            conv,
            dim,
            kernel_size=3,
            padding='same',
            instance=False,
        )
        make_activation = kt.models.make_activation

        # Encoder.
        n_inp = inp
        enc = list(enc)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for n_out in enc:
            level = []
            for _ in range(rep):
                level.append(conv(n_inp, n_out))
                level.append(make_activation(act))
                n_inp = n_out
            self.enc.append(nn.Sequential(*level))
            self.down.append(GroupMaxPool(dim, kernel_size=2))

        # Decoder.
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for n_out in dec:
            level = []
            for _ in range(rep):
                level.append(conv(n_inp, n_out))
                level.append(make_activation(act))
                n_inp = n_out
            n_inp += enc.pop()
            self.dec.append(nn.Sequential(*level))
            self.up.append(GroupUpsample(scale_factor=2, mode=mode))

        # Additional convolutions.
        level = []
        for n_out in add:
            for _ in range(rep):
                level.append(conv(n_inp, n_out))
                level.append(make_activation(act))
                n_inp = n_out

        # Softmax over channels. After batch and group dimensions.
        level.append(conv(n_inp, out))
        level.append(nn.Softmax(dim=2))
        self.add = nn.Sequential(*level)

    def forward(self, x, /):
        """Define the computation performed by the model call.

        Parameters
        ----------
        x : (B, inp, ...) torch.Tensor
            Input tensor.

        Returns
        -------
        out : (B, out, ...) torch.Tensor
            Model output.

        """
        # Group dimension from channels: (B, G, 1, *space). Normalization.
        with torch.no_grad():
            x = x.unsqueeze(2)
            dim = range(2, x.ndim)
            x = kt.utility.normalize(x, dim, *self.clip)

        # Encoding convolutions.
        enc = []
        for conv, down in zip(self.enc, self.down, strict=True):
            x = conv(x)
            enc.append(x)
            x = down(x)

        # Decoding convolutions.
        for conv, up in zip(self.dec, self.up, strict=True):
            x = conv(x)
            x = torch.cat([enc.pop(), up(x)], dim=2)

        # Summarize groups, final convolutions. Drop singleton group.
        x = x.mean(dim=1, keepdim=True)
        return self.add(x).squeeze(1)
