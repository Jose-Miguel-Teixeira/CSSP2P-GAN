import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import DictConfig, OmegaConf
from time import time
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

# Project Imports
from utils import (
    get_callbacks,
    get_transforms,
    setup,
    )


def run(cfg: DictConfig) -> float:
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

    augmentations = get_transforms(
        cfg=cfg,
        transforms_key=['HE_augmentations', 'IHC_augmentations'],
        verbose=cfg.general.verbose
    )

    logger_cfg = OmegaConf.select(cfg, 'logger')
    if logger_cfg is None:
        logger = None
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

    resume_training: bool = False
    if cfg.train.resume_from_checkpoint:
        if os.path.isfile(cfg.train.resume_from_checkpoint):
            if cfg.train.resume_from_checkpoint.endswith('.ckpt'):
                print(f"Resuming from checkpoint: {cfg.train.resume_from_checkpoint}")
                # module = module.load_from_checkpoint(cfg.train.resume_from_checkpoint)
                resume_training = True
            else:
                raise ValueError(f"The file {cfg.train.resume_from_checkpoint} is not a checkpoint file.")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {cfg.train.resume_from_checkpoint}.")

    datamodule = hydra.utils.instantiate(cfg.datamodule)(
        **transforms,
        **augmentations,
    )

    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=cfg.train.resume_from_checkpoint if resume_training else None
    )

    wandb.finish()

    sweep_metric = OmegaConf.select(
        cfg,
        'sweep_metric',
        default=None
        )

    if sweep_metric is not None:
        device = module.device
        checkpoint_callbacks = [
            cb for cb in callbacks if isinstance(cb, ModelCheckpoint)
        ]
        for cb in checkpoint_callbacks:
            if sweep_metric in cb.monitor:
                return cb.best_model_score.to(device)

    return None


@hydra.main(
        config_path="./conf/",
        config_name='gan_config.yaml',
        version_base=None
        )
def main(cfg: DictConfig) -> None:
    """
    Main function to run the training of the GAN.

    Args:
        cfg (DictConfig): Configuration object.
    """
    tic = time()
    print(f"Current configuration: {cfg}")
    best_model_score = run(cfg)
    print(f"Best model score: {best_model_score}")
    print(f"Run directory: {HydraConfig.get().run.dir}")
    toc = time()
    print(f"Execution time: {(toc - tic) / 3600:.2f} hours.")

    return best_model_score


if __name__ == "__main__":
    main()
