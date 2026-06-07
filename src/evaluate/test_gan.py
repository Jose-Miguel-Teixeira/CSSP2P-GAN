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
    Run GAN model evaluation on the test split from a saved checkpoint.

    This routine instantiates callbacks, transforms, logger, accelerator,
    trainer, module, and datamodule from the Hydra configuration, then
    executes PyTorch Lightning testing via ``trainer.test``.

    Side effects:
    - Injects ``hydra.run.dir`` into the config after temporarily disabling
      OmegaConf struct mode.
    - Creates ``<run_dir>/logs`` when a logger is configured.
    - Runs model evaluation and writes outputs through configured callbacks.

    Args:
        cfg (DictConfig): Hydra-composed test configuration. Expected
            sections include ``general``, ``callbacks``, ``trainer``,
            ``module``, ``datamodule``, and ``train.resume_from_checkpoint``.

    Returns:
        None.

    Raises:
        ValueError: If ``train.resume_from_checkpoint`` is not provided.
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

    logger_cfg = OmegaConf.select(cfg, 'logger')
    if logger_cfg is None:
        logger = False
    else:
        os.makedirs(os.path.join(cfg.hydra.run.dir, 'logs'), exist_ok=True)
        logger = hydra.utils.instantiate(cfg.logger)(
            save_dir=os.path.join(cfg.hydra.run.dir, 'logs'),
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
        logger=logger,
        accelerator=accelerator,
        precision=cfg.general.trainer_precision
    )

    module = hydra.utils.instantiate(cfg.module)(cfg)

    if cfg.train.resume_from_checkpoint is None:
        raise ValueError(
            "Provide a checkpoint path for testing."
            )

    datamodule = hydra.utils.instantiate(cfg.datamodule)(
        **transforms,
    )

    trainer.test(
        module,
        datamodule=datamodule,
        ckpt_path=cfg.train.resume_from_checkpoint,
    )


@hydra.main(
        config_path="../conf",
        config_name='gan_test.yaml',
        version_base=None
        )
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for GAN testing.

    This function executes ``run`` and reports total wall-clock runtime.

    Args:
        cfg (DictConfig): Hydra-composed test configuration.

    Returns:
        None.
    """
    tic = time()
    run(cfg)
    toc = time()
    print(f"Execution time: {(toc - tic) / 3600:.2f} hours.")

    return


if __name__ == "__main__":
    main()
