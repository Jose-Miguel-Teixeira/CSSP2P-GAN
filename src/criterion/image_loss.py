import torch
import torch.nn as nn
from typing import Tuple, Optional, Sequence, Union, Literal
from torchmetrics.functional.image.ssim import (
    _ssim_check_inputs,
    _multiscale_ssim_update,
)
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torch.nn import functional as F


def _css_update(
        preds: torch.Tensor,
        target: torch.Tensor,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        data_range: Optional[Union[float, tuple[float, float]]] = None,
        k: float = 0.03,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

    is_3d = preds.ndim == 5

    if not isinstance(kernel_size, Sequence):
        kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 3 * [sigma] if is_3d else 2 * [sigma]

    if len(kernel_size) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(kernel_size) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )
    if len(sigma) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(sigma) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(preds.max() - preds.min(), target.max() - target.min())  # type: ignore[call-overload]
    elif isinstance(data_range, tuple):
        preds = torch.clamp(preds, min=data_range[0], max=data_range[1])
        target = torch.clamp(target, min=data_range[0], max=data_range[1])
        data_range = data_range[1] - data_range[0]

    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype

    gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in sigma]

    if gaussian_kernel:
        pad_h = (gauss_kernel_size[0] - 1) // 2
        pad_w = (gauss_kernel_size[1] - 1) // 2
    else:
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2

    if is_3d:
        pad_d = (kernel_size[2] - 1) // 2
        preds = _reflection_pad_3d(preds, pad_d, pad_w, pad_h)
        target = _reflection_pad_3d(target, pad_d, pad_w, pad_h)
        if gaussian_kernel:
            kernel = _gaussian_kernel_3d(channel, gauss_kernel_size, sigma, dtype, device)
    else:
        preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        if gaussian_kernel:
            kernel = _gaussian_kernel_2d(channel, gauss_kernel_size, sigma, dtype, device)

    if not gaussian_kernel:
        kernel = torch.ones((channel, 1, *kernel_size), dtype=dtype, device=device) / torch.prod(
            torch.tensor(kernel_size, dtype=dtype, device=device)
        )

    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)

    outputs = F.conv3d(input_list, kernel, groups=channel) if is_3d else F.conv2d(input_list, kernel, groups=channel)

    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    # Calculate the variance of the predicted and target images, should be non-negative
    sigma_pred_sq = torch.clamp(output_list[2] - mu_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(output_list[3] - mu_target_sq, min=0.0)
    sigma_pred_target = output_list[4] - mu_pred_target

    c = pow(k * data_range, 2)  # stability constant

    upper = 2 * sigma_pred_target.to(dtype) + c
    lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c

    css_idx_full_image = upper / lower

    return css_idx_full_image.reshape(css_idx_full_image.shape[0], -1).mean(-1)


class CSSLoss(nn.Module):
    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum"] = "elementwise_mean",
        data_range: Optional[Union[float, tuple[float, float]]] = None,
        k: float = 0.03,
    ) -> None:
        super(CSSLoss, self).__init__()

        valid_reduction = ("elementwise_mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")

        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k = k

    def forward(
            self,
            preds: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:
        preds, target = _ssim_check_inputs(preds, target)
        similarity = _css_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k,
        )
        similarity = (similarity + 1) / 2  # Normalize the similarity index to [0, 1]
        similarity = - torch.log(similarity + 1e-8)

        if self.reduction == 'sum':
            loss = similarity.sum()
        elif self.reduction == 'elementwise_mean':
            loss = similarity.sum() / preds.shape[0]
        else:
            loss = similarity
        return loss
