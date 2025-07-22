import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import DictConfig, OmegaConf
from time import time

# Project Imports
from utils import (
    get_callbacks,
    get_transforms,
    setup,
    )


def run(cfg: DictConfig) -> None:
    """
    Main function to run the training of the GAN.

    Args:
        cfg (DictConfig): Configuration object.
    """

    # Set the current working directory as the run directory
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({'hydra': {'run': {'dir': HydraConfig.get().run.dir}}})
        )
    OmegaConf.set_struct(cfg, True)

    callbacks = get_callbacks(
        cfg=cfg,
        callbacks_key="callbacks",
        verbose=cfg.general.verbose
        )

    transforms = get_transforms(
        cfg=cfg,
        transforms_key=['HE_transforms', 'IHC_transforms'],
        verbose=cfg.general.verbose
    )

    accelerator = setup(
        seed=cfg.general.seed,
        dtype=torch.float32,
        matmul_precision=cfg.general.matmul_precision,
        deterministic=cfg.general.deterministic,
        benchmarking=cfg.general.benchmarking,
        device=cfg.general.device,
        verbose=cfg.general.verbose
        )

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=False,
        accelerator=accelerator,
        precision=cfg.general.trainer_precision
    )

    module = hydra.utils.instantiate(cfg.module)(cfg)

    if cfg.train.resume_from_checkpoint is None:
        raise ValueError(
            "Provide a checkpoint to resume training from."
            )

    datamodule = hydra.utils.instantiate(cfg.datamodule)(
        **transforms,
    )

    trainer.predict(
        module,
        datamodule=datamodule,
        ckpt_path=cfg.train.resume_from_checkpoint,
        return_predictions=False,
    )


@hydra.main(
        config_path="../conf/",
        config_name='gan_predict.yaml',
        version_base=None
        )
def main(cfg: DictConfig) -> None:
    """
    Main function to run the training of the GAN.

    Args:
        cfg (DictConfig): Configuration object.
    """
    tic = time()
    run(cfg)
    toc = time()
    print(f"Execution time: {(toc - tic) / 3600:.2f} hours.")


if __name__ == "__main__":
    main()
