import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchmetrics import Metric
from typing import Literal
import torch.nn.functional as F
from typing import (
    Tuple,
    Optional
    )

# Project Imports
from utils import (
    jensenshannon,
    L1_distance,
    L2_distance
    )


class HistogramDistance(Metric):
    """
    Histogram-based distance metric for pathology image distributions.

    This metric compares predicted and target images by first converting each
    sample in a batch into a normalized histogram, then computing a
    per-sample distance between histogram pairs. Supported distances are
    ``L1``, ``L2``, and Jensen-Shannon distance.

    Histogram construction behavior:
    - If ``range`` is provided, all samples use shared fixed bin edges.
    - If ``range`` is None, each sample uses its own min/max range.
    - Histograms are normalized to sum to 1 per sample.

    Accumulation behavior:
    - ``update`` accumulates the sum of per-sample distances.
    - ``compute`` returns either the accumulated sum (``reduction='sum'``)
      or the average distance per sample (``reduction='mean'``).

    Args:
        nbins (int): Number of histogram bins.
        range (Optional[Tuple]): Fixed histogram value range
            ``(min, max)``. If None, sample-wise min/max is used.
        reduction (Literal['sum', 'mean']): Reduction applied in ``compute``.
        distance_function (Literal['L1', 'L2', 'jensenshannon']): Distance
            used between predicted and target histograms.
        base (float): Logarithm base for Jensen-Shannon distance.
        **kwargs: Additional keyword arguments forwarded to ``Metric``.

    Raises:
        ValueError: If ``range`` is invalid, ``reduction`` is unsupported,
            or ``distance_function`` is unsupported.
    """
    def __init__(
            self,
            nbins: int = 150,
            range: Optional[Tuple] = None,
            reduction: Literal['sum', 'mean'] = 'mean',
            distance_function: Literal[
                'L1',
                'L2',
                'jensenshannon'
                ] = 'jensenshannon',
            base: float = 2.0,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        self.add_state(
            "sum_distances",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
            )
        self.add_state(
            "num_batches",
            default=torch.tensor(0),
            dist_reduce_fx="sum"
            )
        self.nbins = nbins
        self.range = range
        if self.range is not None and self.range[1] <= self.range[0]:
            raise ValueError(
                "The second element of range must be greater than the first element."
                f"Got range={self.range}.")
        if reduction not in ['sum', 'mean']:
            raise ValueError("Reduction must be either 'sum' or 'mean'")
        self.reduction = reduction
        match distance_function:
            case 'L1':
                self.distance_function = L1_distance
            case 'L2':
                self.distance_function = L2_distance
            case 'jensenshannon':
                self.distance_function = jensenshannon
            case _:
                raise ValueError(
                    "Distance function must be one of 'L1', 'L2', or 'jensenshannon'"
                    )
        self.base = base

    def compute_histogram(self, x, eps=1e-8):
        """
        Compute a normalized histogram for each sample in a batch.

        Parameters
        ----------
        x : torch.Tensor
            preds tensor of shape [batch, ...].
        eps : float, optional
            Small epsilon to avoid division by zero.

        Returns
        -------
        torch.Tensor
            Tensor of shape [batch, bins] containing normalized histograms
            (i.e. probability distributions that sum to 1 along each row).
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1).contiguous()

        if self.range is not None:  # Use vectorized computation
            min_val, max_val = self.range

            # Create bin edges (bins+1 edges)
            bin_edges = torch.linspace(min_val, max_val, self.nbins + 1, device=x.device, dtype=x.dtype)

            # bucketize returns indices in [1, bins+1]; subtract 1 for 0-indexed bins.
            bin_idx = torch.bucketize(x_flat, bin_edges, right=False) - 1

            # Clamp any indices outside [0, bins-1]
            bin_idx = bin_idx.clamp(0, self.nbins - 1)

            # One-hot encode along a new last dimension: shape [batch, num_pixels, bins]
            one_hot = F.one_hot(bin_idx, num_classes=self.nbins).float()

            # Sum over pixel dimension to get histogram per sample: shape [batch, bins]
            hist = one_hot.sum(dim=1)
        else:
            # Compute histogram for each sample individually.
            hist_list = []
            for i in range(batch_size):
                sample = x_flat[i]
                sample_min = sample.min()
                sample_max = sample.max()
                # If the sample is constant, avoid division-by-zero
                if sample_min == sample_max:
                    # Define a histogram with all mass in the first bin.
                    hist_sample = torch.zeros(self.nbins, device=x.device, dtype=x.dtype)
                    hist_sample[0] = 1.0
                else:
                    bin_edges = torch.linspace(sample_min, sample_max, self.nbins + 1, device=x.device, dtype=x.dtype)
                    sample_bin_idx = torch.bucketize(sample, bin_edges, right=False) - 1
                    sample_bin_idx = sample_bin_idx.clamp(0, self.nbins - 1)
                    sample_one_hot = F.one_hot(sample_bin_idx, num_classes=self.nbins).float()
                    hist_sample = sample_one_hot.sum(dim=0)
                hist_list.append(hist_sample)
            hist = torch.stack(hist_list, dim=0)

        # Normalize histograms so each sums to 1.
        hist = hist / (hist.sum(dim=1, keepdim=True) + eps)
        return hist

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            ) -> None:
        hist_preds = self.compute_histogram(preds)
        hist_target = self.compute_histogram(target)
        if self.distance_function == jensenshannon:
            distance = self.distance_function(
                hist_preds,
                hist_target,
                base=self.base,
                axis=1
                )
        else:
            distance = self.distance_function(hist_preds, hist_target)
        self.sum_distances += distance.sum()
        self.num_batches += preds.size(0)

    def compute(self) -> torch.Tensor:
        if self.reduction == 'sum':
            return self.sum_distances
        return self.sum_distances / self.num_batches
