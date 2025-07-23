import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from monai.losses import MaskedLoss
from typing import Literal

# Project Imports
from module.GAN_base_module import HydraBCEGAN
from criterion.image_loss import CSSLoss
from utils import (
    HED,
    GaussianFilter,
)


class HydraP2PGAN(HydraBCEGAN):
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

        # * Instantiate Color Space Conversion Functions
        self.ToHED = HED()

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
            # mask=self._generate_mask(IHC),
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
