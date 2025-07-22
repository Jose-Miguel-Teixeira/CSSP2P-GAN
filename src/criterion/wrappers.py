import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Literal, List

# Project Imports
from criterion.utils import GaussianSmoothing


class PyramidLoss(nn.Module):
    def __init__(
            self,
            loss_fn: nn.Module,
            channels: int,
            scales: List[int],
            weights: List[float] = None,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            ) -> None:
        """
        Args:
            loss_fn (torch.nn.Module): The loss function to be used.
            channels (int): The number of channels in the input tensor.
            scales (List[int]): The list of scales at which the loss is computed (starting from 0 to highest scale).
            weights (List[float]): The weights for each scale. Length must match the length of `scales`.
            reduction (str): The method used to reduce the loss. Options are 'mean', 'sum', or 'none'.
        """

        super(PyramidLoss, self).__init__()
        self.loss_fn = loss_fn

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(
                "Reduction method not supported. Should be one of ['mean', 'sum', 'none']"
                f"Got {reduction}"
            )
        self.reduction = reduction

        if not isinstance(scales, list):
            scales = list(scales)
        if not all(x < y for x, y in zip(scales, scales[1:])):
            raise ValueError(
                "Scales should be in ascending order"
                f"Got {scales}"
            )
        self.scales = scales

        if weights is None:
            # If weights are not provided, set them to 1.0 for all scales
            weights = torch.tensor([1.0] * len(scales))
        else:
            if not all(isinstance(w, float) for w in weights):
                raise ValueError(
                    "Weights must be a list of floats."
                    f" Got {weights} with types {[type(w) for w in weights]}"
                )
            if len(weights) != len(scales):
                raise ValueError(
                    "The length of weights must be the same as the length of scales."
                    f" Got weights length {len(weights)} and scales length {len(scales)}"
                )
            weights = torch.tensor(weights)

        self.register_buffer(
            'weights',
            weights,
            persistent=False
        )

        self.gaussian_smoothing = GaussianSmoothing(
            channels=channels,
            kernel_size=3,
            sigma=1
            )
        self.mean_pool = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1
            )

    def _down_scale(self, x):
        with torch.no_grad():
            for _ in range(4):
                x = self.gaussian_smoothing(x)
            x = self.mean_pool(x)

        return x

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            ) -> torch.Tensor:
        loss_list = []

        if 0 in self.scales:
            loss_list.append(self.loss_fn(input, target))

        for i in range(1, self.scales[-1] + 1):
            input = self._down_scale(input)
            target = self._down_scale(target)
            if i in self.scales:
                loss_list.append(self.loss_fn(input, target))

        loss = torch.stack(loss_list)

        if self.reduction == 'mean':
            loss = torch.sum(loss * self.weights) / torch.sum(self.weights)
        elif self.reduction == 'sum':
            loss = torch.sum(loss * self.weights)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(
                "Reduction method not supported. Should be one of ['mean', 'sum', 'none']"
                f"Got {self.reduction}"
            )

        return loss


class PatchLoss(nn.Module):
    def __init__(
            self,
            loss_fn: nn.Module,
            patch_size: int,
            ) -> None:

        super(PatchLoss, self).__init__()

        self.loss_fn = loss_fn
        self.patch_size = patch_size

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            ) -> torch.Tensor:

        batch_size, channels, height, width = input.shape

        # * Compute number of patches along height and width
        patches_height = height // self.patch_size
        patches_width = width // self.patch_size

        # * Unfold the input and target tensors
        # ** [B, C, H, W] -> [B, C * patch_size * patch_size, L]
        # L is the number of patches
        input_patches = F.unfold(
            input,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            )
        target_patches = F.unfold(
            target,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            )

        # ** Reshape into 2D patch grid [B, C, patch_size, patch_size, num_patches_h, num_patches_w]
        input_patches = input_patches.reshape(
            batch_size,
            channels,
            self.patch_size,
            self.patch_size,
            patches_height,
            patches_width
            )
        target_patches = target_patches.reshape(
            batch_size,
            channels,
            self.patch_size,
            self.patch_size,
            patches_height,
            patches_width
            )

        # ** Compute mean over patches [B, C, 1, 1, num_patches_h, num_patches_w]
        input_mean = input_patches.mean(dim=[2, 3])  # Averaging over patch pixels
        target_mean = target_patches.mean(dim=[2, 3])

        # * Compute the Loss
        loss = self.loss_fn(input_mean, target_mean)

        return loss

# * Utilize this second version to utilize Pyramidal approach for
# * the adversarial loss.

# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from typing import Literal, List, Optional

# # Project Imports
# from criterion.utils import GaussianSmoothing


# class PyramidLoss(nn.Module):
#     def __init__(
#             self,
#             loss_fn: nn.Module,
#             channels: int,
#             scales: List[int],
#             weights: List[float] = None,
#             reduction: Literal['mean', 'sum', 'none'] = 'mean',
#             ) -> None:
#         """
#         Args:
#             loss_fn (torch.nn.Module): The loss function to be used.
#             channels (int): The number of channels in the input tensor.
#             scales (List[int]): The list of scales at which the loss is computed (starting from 0 to highest scale).
#             weights (List[float]): The weights for each scale. Length must match the length of `scales`.
#             reduction (str): The method used to reduce the loss. Options are 'mean', 'sum', or 'none'.
#         """

#         super(PyramidLoss, self).__init__()
#         self.loss_fn = loss_fn

#         if reduction not in ['mean', 'sum', 'none']:
#             raise ValueError(
#                 "Reduction method not supported. Should be one of ['mean', 'sum', 'none']"
#                 f"Got {reduction}"
#             )
#         self.reduction = reduction

#         if not isinstance(scales, list):
#             scales = list(scales)
#         if not all(x < y for x, y in zip(scales, scales[1:])):
#             raise ValueError(
#                 "Scales should be in ascending order"
#                 f"Got {scales}"
#             )
#         self.scales = scales

#         if weights is None:
#             # If weights are not provided, set them to 1.0 for all scales
#             weights = torch.tensor([1.0] * len(scales))
#         else:
#             if not all(isinstance(w, float) for w in weights):
#                 raise ValueError(
#                     "Weights must be a list of floats."
#                     f" Got {weights} with types {[type(w) for w in weights]}"
#                 )
#             if len(weights) != len(scales):
#                 raise ValueError(
#                     "The length of weights must be the same as the length of scales."
#                     f" Got weights length {len(weights)} and scales length {len(scales)}"
#                 )
#             weights = torch.tensor(weights)

#         self.register_buffer(
#             'weights',
#             weights,
#             persistent=False
#         )

#         self.gaussian_smoothing = GaussianSmoothing(
#             channels=channels,
#             kernel_size=3,
#             sigma=1
#             )
#         self.mean_pool = nn.AvgPool2d(
#             kernel_size=3,
#             stride=2,
#             padding=1
#             )

#     def _down_scale(self, x):
#         with torch.no_grad():
#             for _ in range(4):
#                 x = self.gaussian_smoothing(x)
#             x = self.mean_pool(x)

#         return x

#     def forward(
#             self,
#             input: torch.Tensor,
#             target: Optional[torch.Tensor] = None,
#             **kwargs
#             ) -> torch.Tensor:

#         input = input.clone()
#         if target is not None:
#             target = target.clone()

#         loss_list = []

#         if 0 in self.scales:
#             if target is not None:
#                 loss_list.append(self.loss_fn(input, target, **kwargs))
#             else:
#                 loss_list.append(self.loss_fn(input, **kwargs))

#         for i in range(1, self.scales[-1] + 1):
#             input = self._down_scale(input)
#             if target is not None:
#                 target = self._down_scale(target)
#             if i in self.scales:
#                 if target is not None:
#                     loss_list.append(self.loss_fn(input, target, **kwargs))
#                 else:
#                     loss_list.append(self.loss_fn(input, **kwargs))

#         loss = torch.stack(loss_list)

#         if self.reduction == 'mean':
#             loss = torch.sum(loss * self.weights) / torch.sum(self.weights)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss * self.weights)
#         elif self.reduction == 'none':
#             pass
#         else:
#             raise ValueError(
#                 "Reduction method not supported. Should be one of ['mean', 'sum', 'none']"
#                 f"Got {self.reduction}"
#             )

#         return loss


# class PatchLoss(nn.Module):
#     def __init__(
#             self,
#             loss_fn: nn.Module,
#             patch_size: int,
#             ) -> None:

#         super(PatchLoss, self).__init__()

#         self.loss_fn = loss_fn
#         self.patch_size = patch_size

#     def forward(
#             self,
#             input: torch.Tensor,
#             target: torch.Tensor,
#             ) -> torch.Tensor:

#         batch_size, channels, height, width = input.shape

#         # * Compute number of patches along height and width
#         patches_height = height // self.patch_size
#         patches_width = width // self.patch_size

#         # * Unfold the input and target tensors
#         # ** [B, C, H, W] -> [B, C * patch_size * patch_size, L]
#         # L is the number of patches
#         input_patches = F.unfold(
#             input,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#             )
#         target_patches = F.unfold(
#             target,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#             )

#         # ** Reshape into 2D patch grid [B, C, patch_size, patch_size, num_patches_h, num_patches_w]
#         input_patches = input_patches.reshape(
#             batch_size,
#             channels,
#             self.patch_size,
#             self.patch_size,
#             patches_height,
#             patches_width
#             )
#         target_patches = target_patches.reshape(
#             batch_size,
#             channels,
#             self.patch_size,
#             self.patch_size,
#             patches_height,
#             patches_width
#             )

#         # ** Compute mean over patches [B, C, 1, 1, num_patches_h, num_patches_w]
#         input_mean = input_patches.mean(dim=[2, 3])  # Averaging over patch pixels
#         target_mean = target_patches.mean(dim=[2, 3])

#         # * Compute the Loss
#         loss = self.loss_fn(input_mean, target_mean)

#         return loss
