import torch
import torch.nn as nn
from typing import Literal
import functools
import numpy as np
from scipy.special import comb


# Helper function from Fangda Li's repository
def get_filter(filt_size: int = 3) -> torch.Tensor:

    """Return a 2D filter
        1.Creates a 2D filter of size `filt_size` x `filt_size` with the
          Pascal's triangle coefficients.
        2. The filter is normalized to sum to 1.

    Parameters:
        filt_size (int) -- the size of the filter (default 3)
    """

    def pascal_row(n):
        return np.array(
            [comb(n, k, exact=True) for k in range(n + 1)], dtype=np.float32
            )

    a = pascal_row(filt_size - 1)
    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


# Helper function from Fangda Li's repository
def get_pad_layer(
        pad_type: Literal['zero', 'reflect', 'replicate'] = 'zero'
        ) -> nn.Module:
    if pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    elif pad_type == 'reflect':
        PadLayer = nn.ReflectionPad2d
    elif pad_type == 'replicate':
        PadLayer = nn.ReplicationPad2d
    else:
        raise NotImplementedError('Pad type [%s] not recognized.' % pad_type)
    return PadLayer


# Helper function from Fangda Li's repository
def get_norm_layer(
        norm_type: Literal['batch', 'instance', 'none'] = 'instance'
        ) -> nn.Module:
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none' or norm_type is None:
        norm_layer = nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# Modified version of Downsample from Fangda Li's repository
# Changes made: Utilization of separable convolution in the forward pass
class Downsample(nn.Module):
    def __init__(
            self,
            channels: int,
            pad_type: Literal['zero', 'reflect', 'replicate'] = 'reflect',
            filt_size: int = 3,
            stride: int = 2,
            pad_off: int = 0
            ) -> None:
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.stride = stride
        self.channels = channels
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2) + pad_off,
            int(np.ceil(1. * (filt_size - 1) / 2)) + pad_off,
            int(1. * (filt_size - 1) / 2) + pad_off,
            int(np.ceil(1. * (filt_size - 1) / 2)) + pad_off
            ]
        self.off = int((self.stride - 1) / 2.)

        filt = get_filter(filt_size=self.filt_size)
        filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

        # Precompute 1D filters for separable convolutions
        filt_1d_vert = filt[0, 0, :, 0].view(1, 1, -1, 1)
        filt_1d_horiz = filt[0, 0, 0, :].view(1, 1, 1, -1)

        self.register_buffer(
            'filt_1d_vert',
            filt_1d_vert.repeat(self.channels, 1, 1, 1),
            persistent=False
            )
        self.register_buffer(
            'filt_1d_horiz',
            filt_1d_horiz.repeat(self.channels, 1, 1, 1),
            persistent=False
            )

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]

        # Option 1: Using the conv2d function
        # return nn.functional.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

        # Option 2: Apply separable convolutions
        # Vertical pass
        out = nn.functional.conv2d(
            self.pad(inp),
            self.filt_1d_vert,
            stride=(self.stride, 1),
            groups=inp.shape[1]
            )

        # Horizontal pass
        out = nn.functional.conv2d(
            out,
            self.filt_1d_horiz,
            stride=(1, self.stride),
            groups=inp.shape[1]
            )

        return out
