import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Literal

# Project Imports
from module.GAN_base_module import HydraBCEGAN
from criterion.image_loss import CSSLoss


class HydraP2PGAN(HydraBCEGAN):
    """
    Conditional GAN module with adversarial plus pixel-to-pixel pyramid loss.

    This module extends ``HydraBCEGAN`` by adding a reconstruction objective
    implemented through a Hydra-instantiated ``PyramidLoss`` wrapper around the
    configured base criterion. The generator objective is:

    ``L_G = L_adv + L_p2p``

    During training, per-step adversarial and p2p losses are accumulated and
    averaged at epoch end for logging.

    Args:
        cfg (DictConfig): Hydra configuration containing at least
            ``criterion`` and ``wrapper`` entries. The wrapper target must be
            ``PyramidLoss``.

    Raises:
        ValueError: If the configured wrapper is not ``PyramidLoss``.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        wrapper_cfg = OmegaConf.select(cfg, 'wrapper')
        if wrapper_cfg._target_.split('.')[-1] != 'PyramidLoss':
            raise ValueError(
                f'Wrapper must be PyramidLoss, but got {wrapper_cfg._target_}'
            )

        loss_fn = hydra.utils.instantiate(cfg.criterion)
        self.p2p_loss = hydra.utils.instantiate(
            wrapper_cfg,
            loss_fn=loss_fn,
        )

        # * Loss Values
        # ** Train
        self.adversarial_step_loss = []
        self.p2p_step_loss = []

        self.avg_adversarial_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_p2p_loss: torch.Tensor = torch.tensor(float('inf'))

    def generator_loss_fn(
        self,
        HE: torch.Tensor,
        IHC: torch.Tensor,
        IHC_hat: torch.Tensor,
        phase: Literal['train', 'val', 'test'],
        **kwargs
    ) -> torch.Tensor:
        # * Compute the Adversarial Loss
        adversarial_loss = super().generator_loss_fn(
            HE=HE,
            IHC=IHC,
            IHC_hat=IHC_hat,
            phase=phase,
            **kwargs
            )

        # * Compute the Generator Specific Loss
        p2p_loss = self.p2p_loss(
            input=IHC_hat,
            target=IHC,
        )

        # ** Compute the Global Loss
        if phase == 'train':
            self.adversarial_step_loss.append(adversarial_loss.detach())
            self.p2p_step_loss.append(p2p_loss.detach())

        return adversarial_loss + p2p_loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        with torch.no_grad():
            # * Compute the Average Losses
            self.avg_adversarial_loss = torch.stack(
                self.adversarial_step_loss
                ).nanmean()
            self.avg_p2p_loss = torch.stack(
                self.p2p_step_loss
                ).nanmean()

            # * Log the Average Losses
            self.log(
                'adversarial_train_loss',
                self.avg_adversarial_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )
            self.log(
                'p2p_train_loss',
                self.avg_p2p_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )

        # * Reset the Losses and Weights
        self.adversarial_step_loss.clear()
        self.p2p_step_loss.clear()


class HydraCSSP2PGAN(HydraBCEGAN):
    """
    Conditional GAN module with adversarial, CSS, and pyramid p2p losses.

    This module extends ``HydraBCEGAN`` by combining three generator-loss
    components:

    ``L_G = L_adv + L_css + L_p2p``

    where ``L_css`` is computed by ``CSSLoss`` and ``L_p2p`` is a
    Hydra-instantiated ``PyramidLoss`` wrapper around the configured criterion.
    During training, per-step loss terms are accumulated and epoch means are
    logged separately.

    Args:
        cfg (DictConfig): Hydra configuration containing at least
            ``criterion`` and ``wrapper`` entries. The wrapper target must be
            ``PyramidLoss``.

    Raises:
        ValueError: If the configured wrapper is not ``PyramidLoss``.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        # * Instantiate Generator Loss Function
        # ** CSS Loss
        self.css_loss = CSSLoss()

        # ** Pyramid Loss
        wrapper_cfg = OmegaConf.select(cfg, 'wrapper')
        if wrapper_cfg._target_.split('.')[-1] != 'PyramidLoss':
            raise ValueError(
                f'Wrapper must be PyramidLoss, but got {wrapper_cfg._target_}'
            )

        loss_fn = hydra.utils.instantiate(cfg.criterion)
        self.loss_fn = hydra.utils.instantiate(
            wrapper_cfg,
            loss_fn=loss_fn,
        )

        # * Loss Values
        # ** Train
        self.adversarial_step_loss = []
        self.css_step_loss = []
        self.p2p_step_loss = []

        self.avg_adversarial_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_css_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_p2p_loss: torch.Tensor = torch.tensor(float('inf'))

    def generator_loss_fn(
            self,
            HE: torch.Tensor,
            IHC: torch.Tensor,
            IHC_hat: torch.Tensor,
            phase: Literal['train', 'val', 'test'],
            **kwargs
    ) -> torch.Tensor:
        # * Compute the Adversarial Loss
        adversarial_loss = super().generator_loss_fn(
            HE=HE,
            IHC=IHC,
            IHC_hat=IHC_hat,
            phase=phase,
            **kwargs
            )

        # * Compute the Generator Specific Loss
        # ** CSS Loss
        css_loss = self.css_loss(
            preds=IHC_hat,
            target=HE
        )

        # ** P2P Loss
        p2p_loss = self.loss_fn(
            input=IHC_hat,
            target=IHC,
        )

        # ** Compute the Global Loss
        if phase == 'train':
            self.adversarial_step_loss.append(adversarial_loss.detach())
            self.css_step_loss.append(css_loss.detach())
            self.p2p_step_loss.append(p2p_loss.detach())

        return adversarial_loss + css_loss + p2p_loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        with torch.no_grad():
            # * Compute the Average Losses
            self.avg_adversarial_loss = torch.stack(
                self.adversarial_step_loss
                ).nanmean()
            self.avg_css_loss = torch.stack(
                self.css_step_loss
                ).nanmean()
            self.avg_p2p_loss = torch.stack(
                self.p2p_step_loss
                ).nanmean()

            # * Log the Average Losses
            self.log(
                'adversarial_train_loss',
                self.avg_adversarial_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )
            self.log(
                'css_train_loss',
                self.avg_css_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )
            self.log(
                'p2p_train_loss',
                self.avg_p2p_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )

        # * Reset the Losses and Weights
        self.adversarial_step_loss.clear()
        self.css_step_loss.clear()
        self.p2p_step_loss.clear()
