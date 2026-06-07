import torch
from torchvision.transforms import v2
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    )
from tqdm import tqdm
import os
import csv
import glob
from PIL import Image
import warnings

# Project Imports
from metrics.image import CSS
from metrics.pathology import HistogramDistance
from utils import HED

METRICS = {
        'ssim': StructuralSimilarityIndexMeasure(
            data_range=(0, 1),
        ),
        'psnr': PeakSignalNoiseRatio(
            data_range=(0, 1),
        ),
        'msssim': MultiScaleStructuralSimilarityIndexMeasure(
            data_range=(0, 1),
        ),
        'css': CSS(
            data_range=(0, 1),
        ),
        'lpips': LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',
        ),
        'fid': FrechetInceptionDistance(
            normalize=True,
        ),
        'kid': KernelInceptionDistance(
            normalize=True,
            subset_size=100,
            subsets=20,
        ),
        'jsd': HistogramDistance(
            nbins=150,
            reduction='mean',
            distance_function='jensenshannon',
        ),
    }


def calculate_metrics(
    real_dir: str,
    fake_dir: str,
    he_dir: str,
    results_dir: str,
    device: str = 'cuda',
        ) -> None:
    """
    Compute image quality metrics for generated IHC images and save results.

    The function iterates over generated images in ``fake_dir`` and matches
    them by filename with corresponding real IHC images (``real_dir``) and
    HE images (``he_dir``). For each image triplet, it computes per-image
    metrics (SSIM, PSNR, MSSSIM, LPIPS, JSD, CSS), writes per-image rows to a
    CSV file, then writes dataset-level summary statistics (mean/std) and
    distribution metrics (FID, KID).

    Notes:
    - ``JSD`` is computed on the DAB channel obtained through HED conversion.
    - ``CSS`` is computed between fake IHC and HE images.
    - If ``device='cuda'`` but CUDA is unavailable, computation falls back to
      CPU with a warning.

    Args:
        real_dir (str): Directory containing real/reference IHC images.
        fake_dir (str): Directory containing generated IHC images to evaluate.
        he_dir (str): Directory containing corresponding HE images.
        results_dir (str): Output directory where ``metrics.csv`` is written.
        device (str): Target device for metric computation (for example,
            ``'cuda'`` or ``'cpu'``).

    Returns:
        None: Results are written to disk and printed to stdout.
    """

    global METRICS

    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn("CUDA is not available. Using CPU instead.")
        device = 'cpu'

    # Send metrics to device
    for metric in METRICS.values():
        metric.to(device)

    # Instantiate HED and send to device
    ToHED = HED().to(device)

    # Lists to store metrics
    ssim_list = []
    psnr_list = []
    msssim_list = []
    lpips_list = []
    jsd_list = []
    css_list = []

    # Transform to convert PIL Image to Tensor
    ToTensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True
            ),
    ])

    fake_list = sorted(glob.glob(os.path.join(fake_dir, '*.jpg')))
    print(f"Found {len(fake_list)} images in '{fake_dir}'.")
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, 'metrics.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Image', 'SSIM', 'PSNR', 'MSSSIM', 'LPIPS', 'JSD', 'CSS']
            )

        for i in tqdm(fake_list):
            base_name = os.path.basename(i)
            real_path = os.path.join(real_dir, base_name)
            fake_path = os.path.join(fake_dir, base_name)
            he_path = os.path.join(he_dir, base_name)

            # Read Images
            real = Image.open(real_path).convert('RGB')
            fake = Image.open(fake_path).convert('RGB')
            he = Image.open(he_path).convert('RGB')

            # Convert to Tensor
            real = ToTensor(real).unsqueeze(0).to(device)
            fake = ToTensor(fake).unsqueeze(0).to(device)
            he = ToTensor(he).unsqueeze(0).to(device)

            # Compute DAB Channel
            real_DAB = ToHED(real)[:, 2, :, :]
            fake_DAB = ToHED(fake)[:, 2, :, :]

            for metric_name, metric in METRICS.items():
                if metric_name in ('fid', 'kid'):
                    metric.update(fake, real=False)
                    metric.update(real, real=True)
                elif metric_name == 'jsd':
                    metric.update(preds=real_DAB, target=fake_DAB)
                elif metric_name == 'lpips':
                    metric.update(img1=fake, img2=real)
                elif metric_name == 'css':
                    metric.update(preds=fake, target=he)
                else:
                    metric.update(preds=fake, target=real)

            ssim = METRICS['ssim'].compute().item()
            METRICS['ssim'].reset()
            psnr = METRICS['psnr'].compute().item()
            METRICS['psnr'].reset()
            msssim = METRICS['msssim'].compute().item()
            METRICS['msssim'].reset()
            lpips = METRICS['lpips'].compute().item()
            METRICS['lpips'].reset()
            jsd = METRICS['jsd'].compute().item()
            METRICS['jsd'].reset()
            css = METRICS['css'].compute().item()
            METRICS['css'].reset()

            ssim_list.append(ssim)
            psnr_list.append(psnr)
            msssim_list.append(msssim)
            lpips_list.append(lpips)
            jsd_list.append(jsd)
            css_list.append(css)

            writer.writerow(
                [base_name, ssim, psnr, msssim, lpips, jsd, css]
                )

        ssim_mean = sum(ssim_list)/len(ssim_list)
        ssim_std = torch.std(torch.tensor(ssim_list)).item()
        psnr_mean = sum(psnr_list)/len(psnr_list)
        psnr_std = torch.std(torch.tensor(psnr_list)).item()
        msssim_mean = sum(msssim_list)/len(msssim_list)
        msssim_std = torch.std(torch.tensor(msssim_list)).item()
        lpips_mean = sum(lpips_list)/len(lpips_list)
        lpips_std = torch.std(torch.tensor(lpips_list)).item()
        jsd_mean = sum(jsd_list)/len(jsd_list)
        jsd_std = torch.std(torch.tensor(jsd_list)).item()
        css_mean = sum(css_list)/len(css_list)
        css_std = torch.std(torch.tensor(css_list)).item()

        writer.writerow(['Mean', ssim_mean, psnr_mean, msssim_mean, lpips_mean, jsd_mean, css_mean])
        writer.writerow(['Std', ssim_std, psnr_std, msssim_std, lpips_std, jsd_std, css_std])

        fid = METRICS['fid'].compute().item()
        METRICS['fid'].reset()
        kid_mean, kid_std = METRICS['kid'].compute()
        METRICS['kid'].reset()
        writer.writerow(['FID', fid])
        writer.writerow(['KID', kid_mean.item(), kid_std.item()])

    print("Metrics Results:")
    print(f"  SSIM: {ssim_mean} ± {ssim_std}")
    print(f"  PSNR: {psnr_mean} ± {psnr_std}")
    print(f"  MSSSIM: {msssim_mean} ± {msssim_std}")
    print(f"  LPIPS: {lpips_mean} ± {lpips_std}")
    print(f"  JSD: {jsd_mean} ± {jsd_std}")
    print(f"  CSS: {css_mean} ± {css_std}")
    print(f"  FID: {fid}")
    print(f"  KID: {kid_mean} ± {kid_std}")
