import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import sys
import numpy as np
import random
import os
import warnings
from omegaconf import DictConfig
from typing import List, Dict, Tuple
import hydra
from torchvision.transforms import v2


def setup(
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        matmul_precision: Literal[
            'medium',
            'high',
            'highest'
            ] = 'medium',
        deterministic: bool = True,
        benchmarking: bool = False,
        device: Literal['cpu', 'gpu'] = 'cpu',
        verbose: bool = True,
) -> str:
    """
    Configures the PyTorch environment with the given settings.

    Args:
        seed (int): Random seed for reproducibility.
        dtype (torch.dtype): Default data type for tensors.
        matmul_precision (Literal): Precision level for matrix multiplications.
        deterministic (bool): Whether to enforce deterministic computations.
        benchmarking (bool): If True, enables benchmarking for performance.
        device (str): Target accelerator (e.g., 'gpu', 'cpu').
        verbose (bool): Whether to print configuration details.
        mixed_precision (bool): Enable mixed precision training.

    Returns:
        str: The name of the chosen accelerator ('cpu', 'cuda', or 'mps').
    """

    # Seed everything for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

    # Set the default data type
    torch.set_default_dtype(dtype)

    # Enforce deterministic computations
    torch.use_deterministic_algorithms(deterministic)

    # Set the precision level for matrix multiplications
    torch.set_float32_matmul_precision(matmul_precision)

    enabled_benchmarking = False

    match device:
        case 'cpu':
            accelerator = 'cpu'
        case 'gpu':

            if sys.platform.startswith('darwin'):  # macOS
                if torch.backends.mps.is_available():
                    accelerator = 'mps'
                else:
                    accelerator = 'cpu'
                    warnings.warn("MPS is not available on macOS. Using CPU instead.")

            else:  # Linux or Windows
                if torch.cuda.is_available():
                    accelerator = 'cuda'
                    print(f"\nGPU: {torch.cuda.get_device_name()}.")

                    # Configure benchmarking for performance (if allowed)
                    if benchmarking:
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.enabled = True
                        enabled_benchmarking = True

                else:
                    torch.set_default_tensor_type(torch.FloatTensor)
                    accelerator = 'cpu'
                    raise ValueError(
                        "CUDA is not available on this device."
                        "Please switch to 'cpu' mode."
                    )
        case _:
            raise ValueError(
                f"Invalid device: {device}."
                "Valid options are 'cpu' and 'gpu'."
                )

    if verbose:
        print("=== Training Setup ===")
        print(f"Seed: {seed}")
        print(f"Default dtype: {dtype}")
        print(f"Matmul Precision precision: {matmul_precision}")
        print(f"Deterministic: {deterministic}")
        print(f"Device: {accelerator}")
        print(f"CUDNN Benchmarking: {'Enabled' if enabled_benchmarking else 'Disabled'}\n")
        print("====================================")
    return accelerator


def get_callbacks(
        cfg: DictConfig,
        callbacks_key: str,
        verbose: bool = True
        ) -> List:
    """
    Returns a list of PyTorch Lightning callbacks based on the given configuration.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        List: List of PyTorch Lightning callbacks.
    """
    dict_config = cfg.get(callbacks_key, None)

    if dict_config is None:
        warnings.warn(f"No callbacks found in the configuration under '{callbacks_key}'.")
        return []

    callbacks = []
    if verbose:
        print("\n=== Callbacks ===")
    for key, value in dict_config.items():
        if value._target_.split('.')[-1] == 'ModelCheckpoint':
            callback = hydra.utils.instantiate(value)(
                dirpath=os.path.join(cfg.hydra.run.dir, 'checkpoints'),
            )
        elif value._target_.split('.')[-1] == 'PredictionWriter':
            callback = hydra.utils.instantiate(value)(
                output_dir=os.path.join(cfg.hydra.run.dir, 'predictions'),
            )
        else:
            callback = hydra.utils.instantiate(value)
        callbacks.append(callback)
        if verbose:
            print(f"\n{key}: {callback}")
            for attr, val in vars(callback).items():
                print(f"{attr}: {val}")
    else:
        return callbacks


def get_transforms(
        cfg: DictConfig,
        transforms_key: List[str],
        verbose: bool = True
        ) -> Dict[str, v2.Compose]:

    transforms = {}

    for key in transforms_key:
        dict_config = cfg.get(key, None)

        if dict_config is None:
            raise ValueError(
                f"The key: `{key}` was not found in the configuration."
                )
        elif not dict_config:  # Empty dictionary
            warnings.warn(
                f"No transforms were found in the configuration under {key}."
                )
            transforms[key] = None
            continue

        transforms_list = []
        if verbose:
            print(f"\n=== {key} Transforms ===")
        for dict_key, value in dict_config.items():
            transform = hydra.utils.instantiate(value)
            transforms_list.append(transform)
            if verbose:
                print(f"\n{dict_key}: {transform}")
                for attr, val in vars(transform).items():
                    print(f"{attr}: {val}")
        else:
            if transforms_list:
                transforms[key] = v2.Compose(transforms_list)
            else:
                transforms[key] = None

    return transforms


def jensenshannon(p, q, base=None, axis=0, keepdims=False):
    """
    Compute the Jensen-Shannon distance between two probability distributions.
    This function is the PyTorch equivalent of the NumPy implementation.

    The Jensen-Shannon distance between two probability vectors p and q is defined as:

        sqrt((D(p || m) + D(q || m)) / 2)

    where m = (p + q) / 2 and D(·||·) is the Kullback-Leibler divergence.

    If the input tensors do not sum to 1 along the given axis, they are normalized.

    Parameters
    ----------
    p : torch.Tensor
        Left probability vector.
    q : torch.Tensor
        Right probability vector.
    base : float, optional
        The logarithm base to use. If given, the divergence is divided by log(base).
    axis : int, optional
        Axis along which the probabilities sum to 1 (default: 0).
    keepdims : bool, optional
        If True, the reduced dimensions are retained with size 1 (default: False).

    Returns
    -------
    torch.Tensor
        The Jensen-Shannon distance between p and q.
    """
    # Ensure inputs are tensors of floating point type
    p = torch.as_tensor(p, dtype=torch.float)
    q = torch.as_tensor(q, dtype=torch.float)

    # Normalize the probability vectors along the specified axis
    p = p / torch.sum(p, dim=axis, keepdim=True)
    q = q / torch.sum(q, dim=axis, keepdim=True)

    # Compute the pointwise mean
    m = (p + q) / 2.0

    # Compute the relative entropy (KL divergence) components.
    # Use torch.where to avoid NaNs: 0 * log(0/m) is defined as 0.
    left = torch.where(p > 0, p * torch.log(p / m), torch.zeros_like(p))
    right = torch.where(q > 0, q * torch.log(q / m), torch.zeros_like(q))

    # Sum along the specified axis
    left_sum = torch.sum(left, dim=axis, keepdim=keepdims)
    right_sum = torch.sum(right, dim=axis, keepdim=keepdims)

    js = left_sum + right_sum

    # Optionally adjust for a specific logarithm base
    if base is not None:
        js = js / torch.log(torch.tensor(base, dtype=js.dtype, device=js.device))

    # Return the Jensen-Shannon distance (square root of half the divergence)
    return torch.sqrt(js / 2.0)


def L1_distance(p, q):
    return torch.abs(p - q).sum(dim=1)


def L2_distance(p, q, eps=1e-8):
    return torch.sqrt(((p - q) ** 2).sum(dim=1) + eps)


class HED(nn.Module):

    # References:
    # - A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
    #   staining by color deconvolution.," Analytical and quantitative
    #   cytology and histology / the International Academy of Cytology [and]
    #   American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001
    # - skimage.color.rgb2hed

    def __init__(self) -> None:
        super(HED, self).__init__()

        # HED to RGB matrix
        self.register_buffer(
            'rgb_from_hed',
            torch.tensor(
                [[0.65, 0.70, 0.29],
                 [0.07, 0.99, 0.11],
                 [0.27, 0.57, 0.78]]
            ),
            persistent=False
        )

        # RGB to HED matrix
        self.register_buffer(
            'hed_from_rgb',
            torch.linalg.inv(self.rgb_from_hed),
            persistent=False
        )

    def rgb2hed(
            self,
            x: torch.Tensor,
            clip: Tuple[float, float] = None,
            ) -> torch.Tensor:

        x_rgb = torch.maximum(
            input=x,
            other=torch.tensor(
                data=1e-6,
                dtype=x.dtype,
                device=x.device
                )
            )

        log_adjust = torch.log(
            torch.tensor(
                data=1e-6,
                dtype=x.dtype,
                device=x.device,
                )
            )

        # Clamping the input to a minimum of 1e-6 and dividing by the
        # log_adjust avoids negative infite values
        x_rgb = torch.log(x_rgb) / log_adjust
        x_hed = torch.einsum(
            'bchw,cd->bdhw',
            x_rgb,
            self.hed_from_rgb.to(x.dtype)
            )

        if clip is not None:
            if clip[0] < clip[1]:
                x_hed = torch.clamp(x_hed, min=clip[0], max=clip[1])
            else:
                raise ValueError(
                    'clip[0] must be less than clip[1]'
                    f"Got clip[0]={clip[0]} and clip[1]={clip[1]}"
                    )

        return x_hed

    def hed2rgb(
            self,
            x: torch.Tensor,
            ) -> torch.Tensor:
        log_adjust = - torch.log(
            torch.tensor(
                data=1e-6,
                dtype=x.dtype,
                device=x.device,
                )
            )
        x_adjusted = x * log_adjust
        x_rgb = - torch.einsum(
            'bchw,cd->bdhw',
            x_adjusted,
            self.rgb_from_hed.to(x.dtype)
            )
        x_rgb = torch.exp(x_rgb)
        x_rgb = torch.clamp(x_rgb, min=0.0, max=1.0)

        return x_rgb

    def forward(
            self,
            x: torch.Tensor,
            clip: Tuple[float, float] = None,
            ) -> torch.Tensor:

        return self.rgb2hed(x, clip=clip)


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size: int, sigma: float, channels: int = 1):
        """
        Gaussian filter for 2D images and batched tensors.

        :param kernel_size: Size of the Gaussian kernel.
        :param sigma: Standard deviation of the Gaussian distribution.
        :param channels: Number of input channels (for multi-channel input support).
        """
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.padding = kernel_size // 2

        # Create the 2D Gaussian kernel
        self.register_buffer(
            'kernel',
            self._create_gaussian_kernel_2d(),
            persistent=False
        )

    def _create_gaussian_kernel_2d(self):
        """Creates a 2D Gaussian kernel."""
        ax = torch.arange(self.kernel_size) - (self.kernel_size - 1) / 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-0.5 * (xx.pow(2) + yy.pow(2)) / self.sigma**2)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.channels, 1, 1, 1)  # Shape: (C, 1, K, K)
        return kernel

    def forward(self, x):
        """Applies Gaussian filtering to batched 2D images."""
        c = x.shape[1]
        if c != self.channels:
            raise ValueError(
                f"Input channels ({c}) do not match filter channels ({self.channels})"
                )
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)
