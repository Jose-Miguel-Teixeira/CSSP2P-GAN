import torch
import torch.nn.functional as F
import torch.nn as nn
# from collections import deque
import math
import numbers


class GaussianSmoothing(nn.Module):
    """
    Apply Gaussian smoothing on a tensor using a depthwise convolution.

    Args:
        channels (int): Number of channels of the input tensor.
        kernel_size (int or sequence): Size of the Gaussian kernel (e.g., 5 or [5, 5]).
        sigma (float or sequence): Standard deviation of the Gaussian kernel (e.g., 1 or [1, 1]).
        dim (int): Number of dimensions of the data (1, 2, or 3).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # Create a multi-dimensional Gaussian kernel
        kernel = 1
        # meshgrids will generate a grid for each dimension
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing='ij')
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.
            kernel *= 1/(std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) ** 2)/(2 * std ** 2))

        # Normalize the kernel so that its sum is 1
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight [out_channels, in_channels/groups, kH, kW]
        kernel = kernel.view(1, 1, *kernel.shape)
        kernel = kernel.repeat(channels, *([1] * (kernel.dim()-1)))

        self.register_buffer(
            'weight',
            kernel,
            persistent=False
            )
        self.groups = channels

        # Compute padding to keep input dimensions unchanged ("same" convolution)
        self.padding = tuple(k // 2 for k in kernel_size)

    def forward(self, input):
        # Input is assumed to be of shape (batch_size, channels, height, width)
        return F.conv2d(input, weight=self.weight, groups=self.groups, padding=self.padding)
