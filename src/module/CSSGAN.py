import torch
from omegaconf import DictConfig, OmegaConf
from typing import Literal

# Project Imports
from module.GAN_base_module import HydraBCEGAN
from criterion.image_loss import CSSLoss


class HydraCSSGAN(HydraBCEGAN):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        # * Initialize the CSSLoss and SoftAdapt modules
        self.css_loss = CSSLoss()

        # * Loss Values
        self.adversarial_step_loss = []
        self.css_step_loss = []

        self.avg_adversarial_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_css_loss: torch.Tensor = torch.tensor(float('inf'))

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

        # * Compute the CSS Loss
        css_loss = self.css_loss(
            preds=IHC_hat,
            target=HE
            )

        # * Compute the Global Loss
        if phase == 'train':
            self.adversarial_step_loss.append(adversarial_loss.detach())
            self.css_step_loss.append(css_loss.detach())

        return adversarial_loss + css_loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        # * Compute the Average Losses and Weights
        with torch.no_grad():
            self.avg_adversarial_loss = torch.stack(
                self.adversarial_step_loss
                ).nanmean()
            self.avg_css_loss = torch.stack(
                self.css_step_loss
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

        # * Reset the Losses and Weights
        self.adversarial_step_loss.clear()
        self.css_step_loss.clear()
