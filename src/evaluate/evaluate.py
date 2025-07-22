import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from omegaconf import DictConfig
import hydra

# Project Imports
from benchmark import calculate_metrics


@hydra.main(
    config_path="../conf/",
    config_name="evaluate_config.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    calculate_metrics(
        fake_dir=cfg.predictions_dir,
        real_dir=cfg.target_dir,
        he_dir=cfg.HE_dir,
        results_dir=os.path.join(
            '/'.join(cfg.predictions_dir.split('/')[:-1]),
            cfg.phase,
            'metrics',
            ),
        device=cfg.device,
        )


if __name__ == "__main__":
    main()
