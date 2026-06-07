import torch
from torchmetrics import Metric
import warnings
from typing import (
    Tuple,
    Dict,
    List,
    Optional,
    )


# * Metrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    )


@torch.no_grad()
def normalize_range(
        image: torch.Tensor,
        input_range: Tuple[float, float],
        target_range: Tuple[float, float] = (0, 1)
        ) -> torch.Tensor:

    current_min, current_max = input_range
    target_min, target_max = target_range

    if current_max <= current_min:
        raise ValueError(
            "The input range must have max value greater than min value."
        )

    image = torch.clamp(image, current_min, current_max)

    image = (
        (image - current_min) / (current_max - current_min) *
        (target_max - target_min) + target_min
        )

    return image


def is_image_in_range(
        image: torch.Tensor,
        data_range: Tuple[int, int],
        ) -> bool:
    min_val, max_val = data_range
    return torch.all((image >= min_val) & (image <= max_val))


class StainMetrics(Metric):
    """
    Composite metric wrapper for image-to-image stain translation tasks.

    The class manages a configurable collection of torchmetrics image metrics
    and exposes a unified ``update`` / ``compute`` / ``reset`` interface.
    Supported metric names are:
    - ``psnr``
    - ``ssim``
    - ``msssim``
    - ``fid``
    - ``kid``
    - ``lpips``

    During ``update``, predicted images are optionally normalized to
    ``target_range`` when they are outside that range. This normalization uses
    ``input_range`` and is required for consistent metric computation
    when model outputs are produced in a different scale.

    Args:
        input_range (Optional[Tuple[int, int]]): Value range of ``y_pred``
            before normalization. Required only when predictions may fall
            outside ``target_range``.
        target_range (Tuple[int, int]): Expected intensity range used for
            range-sensitive metrics and normalization output.
        compute_on_cpu (bool): If True, compute metric state updates on CPU.
        metrics (Optional[List[str]]): Subset of metrics to compute. If None,
            all available metrics are enabled.
        **kwargs: Optional metric-specific settings:
            - ``kid_subset_size`` (int)
            - ``kid_subsets`` (int)
            - ``lpips_net_type`` (str)

    Raises:
        ValueError: If ``target_range`` is invalid or requested
            metric names are unsupported.
    """
    def __init__(
            self,
            input_range: Optional[Tuple[int, int]] = None,
            target_range: Tuple[int, int] = (0, 1),
            compute_on_cpu: bool = False,
            metrics: Optional[List[str]] = None,
            **kwargs
            ) -> None:
        super().__init__()

        self.input_range = input_range
        self.target_range = target_range
        min_target_range, max_target_range = target_range

        if max_target_range <= min_target_range:
            raise ValueError(
                "The target range must have max value greater than min value."
                )

        available_metrics: Dict[str, Metric] = {
            'psnr': PeakSignalNoiseRatio(
                data_range=target_range,
                compute_on_cpu=compute_on_cpu
                ),
            'ssim': StructuralSimilarityIndexMeasure(
                data_range=target_range,
                compute_on_cpu=compute_on_cpu
                ),
            'msssim': MultiScaleStructuralSimilarityIndexMeasure(
                data_range=target_range,
                compute_on_cpu=compute_on_cpu
                ),
            'fid': FrechetInceptionDistance(
                normalize=True,
                compute_on_cpu=compute_on_cpu
                ),
            'kid': KernelInceptionDistance(
                normalize=True,
                compute_on_cpu=compute_on_cpu,
                subset_size=kwargs.get('kid_subset_size', 100),
                subsets=kwargs.get('kid_subsets', 20)
                ),
            'lpips': LearnedPerceptualImagePatchSimilarity(
                net_type=kwargs.get('lpips_net_type', 'alex'),
                compute_on_cpu=compute_on_cpu
            ),
        }

        self.metrics = {}
        if metrics is None:
            warnings.warn(
                "No metrics were provided. All metrics will be computed."
                )
            selected_metrics = available_metrics.keys()
        else:
            selected_metrics = set(metrics)
            if not selected_metrics.issubset(available_metrics.keys()):
                raise ValueError(
                    "The selected metrics must be a subset of the available metrics."
                    "Available metrics are: 'psnr', 'ssim', 'msssim', 'fid', 'kid', 'lpips'."
                    )

        for metric_name in selected_metrics:
            self.metrics[metric_name] = available_metrics[metric_name]

    def to(self, device) -> 'StainMetrics':
        if isinstance(device, str):
            if device not in ('cpu', 'cuda', 'mps'):
                raise ValueError("The device must be one of 'cpu', 'cuda', or 'mps'.")
            device = torch.device(device)
        elif isinstance(device, torch.device):
            if device.type not in ('cpu', 'cuda', 'mps'):
                raise ValueError("The device must be one of 'cpu', 'cuda', or 'mps'.")
        else:
            raise ValueError("The device must be a string or a torch.device instance.")

        for metric in self.metrics.values():
            metric.to(device)

        return self

    def update(
            self,
            y_pred: torch.Tensor,
            y: torch.Tensor,
            ) -> None:

        y_pred = y_pred.clone().detach()
        y = y.clone()

        if not is_image_in_range(y_pred, self.target_range):
            if self.input_range is None:
                raise ValueError(
                    "The input range must be provided to normalize the input image."
                    )
            y_pred = normalize_range(
                image=y_pred,
                input_range=self.input_range,
                target_range=self.target_range
                )

        for metric_name, metric in self.metrics.items():
            if metric_name in ('fid', 'kid'):
                metric.update(y_pred, real=False)
                metric.update(y, real=True)
            else:
                metric.update(y_pred, y)

    def compute(self) -> Dict[str, float]:
        return {
            name: metric.compute() for name, metric in self.metrics.items()
            }

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
