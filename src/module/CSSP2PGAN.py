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
        loss_fn = hydra.utils.instantiate(
            wrapper_cfg,
            loss_fn=loss_fn,
        )

        # ** Masked Loss
        # The loss function is wrapped with the MaskedLoss class.
        # This class allows for additional weighting masks to be applied to
        # both input and target.

        self.loss_fn = MaskedLoss(loss_fn)

        self.apply_mask = OmegaConf.select(
            cfg.train.Mask,
            'apply_mask',
            default=True,
        )

        self.register_buffer(
            'mask_brown_thresholds',
            torch.tensor(
                OmegaConf.select(
                        cfg.train.Mask,
                        'brown_thresholds',
                        default=[6e-2, 0.3]
                    )
                ),
            persistent=False
        )

        self.register_buffer(
            'mask_blue_thresholds',
            torch.tensor(
                OmegaConf.select(
                        cfg.train.Mask,
                        'blue_thresholds',
                        default=[-1e-3]
                    )
                ),
            persistent=False
        )

        self.register_buffer(
            'mask_brown_weights',
            torch.tensor(
                OmegaConf.select(
                    cfg.train.Mask,
                    'brown_weights',
                    default=[0.9, 0.1]
                )
            ),
            persistent=False
        )

        self.register_buffer(
            'mask_blue_weights',
            torch.tensor(
                OmegaConf.select(
                    cfg.train.Mask,
                    'blue_weights',
                    default=[0.05]
                )
            ),
            persistent=False
        )

        self.gauss_filter = GaussianFilter(
            kernel_size=OmegaConf.select(
                cfg.train.Mask,
                'kernel_size',
                default=9
            ),
            sigma=OmegaConf.select(
                cfg.train.Mask,
                'sigma',
                default=5.0
            ),
            channels=1
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

    @torch.no_grad()
    def _generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_mask:
            # Get the DAB Channel
            DAB = self.ToHED(x)[:, 2, ...]  # [B, H, W]

            DAB_mask = torch.zeros_like(
                DAB.unsqueeze(1),
                dtype=x.dtype,
                device=x.device
                )
            for threshold, weight in zip(self.mask_brown_thresholds, self.mask_brown_weights):
                DAB_mask += (DAB > threshold).to(x.dtype).unsqueeze(1) * weight # [B, 1, H, W]

            for threshold, weight in zip(self.mask_blue_thresholds, self.mask_blue_weights):
                DAB_mask += (DAB < threshold).to(x.dtype).unsqueeze(1) * weight # [B, 1, H, W]

            DAB_mask = self.gauss_filter(DAB_mask)
            DAB_mask = DAB_mask / DAB_mask.max()
        else:
            B, _, H, W = x.shape
            DAB_mask = torch.ones(
                size=(B, 1, H, W),
                dtype=x.dtype,
                device=x.device
                )

        return DAB_mask

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
            mask=self._generate_mask(IHC),
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
