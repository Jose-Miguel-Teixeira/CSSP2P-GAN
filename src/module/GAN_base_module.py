import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import pytorch_lightning as pl
from pytorch_lightning.loggers import (
    TensorBoardLogger,
    WandbLogger
)
from torchvision.utils import make_grid
from torchmetrics.utilities.compute import normalize_logits_if_needed
import hydra
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from typing import Literal, List
from collections import namedtuple
import warnings
import gc
from time import time
import os

GeneratorOutput = namedtuple('GeneratorOutput', ['IHC_hat'])


class HydraBCEGAN(pl.LightningModule):

    def __init__(self, cfg: DictConfig) -> None:
        super(HydraBCEGAN, self).__init__()

        # * Save Hyperparameters
        self.cfg = cfg
        self.save_hyperparameters(
            OmegaConf.to_container(cfg=cfg, resolve=True)
            )

        # * Define example input array
        self.example_input_array = torch.randn(1, 3, 512, 512)

        # * Model
        # ** Generator
        self.generator: nn.Module = hydra.utils.instantiate(
            cfg.generator_model
        )

        if cfg.general.verbose:
            print('\nGENERATOR SUMMARY:')
            summary(
                model=self.generator,
                input_size=(1, 3, 512, 512)
                )

        if not isinstance(self.generator.activation, nn.Sigmoid):
            raise ValueError(
                "The activation function of the generator must be nn.Sigmoid."
                "Othwerwise the discriminator will receive two different distributions."
                "Currently it expects the output to be in the range [0, 1] which is the range of the input IHC."
                )

        if cfg.generator.compilation.enabled:
            self.generator = torch.compile(
                model=self.generator,
                backend='inductor',
                fullgraph=cfg.generator.compilation.fullgraph,
            )

        # ** Discriminator
        self.discriminator: nn.Module = hydra.utils.instantiate(
            cfg.discriminator_model
        )

        if cfg.general.verbose:
            print('\nDISCRIMINATOR SUMMARY:')
            summary(
                model=self.discriminator,
                input_size=(1, self.cfg.discriminator_model.input_nc, 512, 512)
                )

        if cfg.discriminator.compilation.enabled:
            self.discriminator = torch.compile(
                model=self.discriminator,
                backend='inductor',
                fullgraph=cfg.discriminator.compilation.fullgraph,
            )

        # * Optimizers and Schedulers
        # ** Instantiate Optimizers and Schedulers

        self.generator_optimizer: torch.optim = hydra.utils.instantiate(
            cfg.generator_optimizer,
            params=self.generator.parameters()
        )

        self.discriminator_optimizer: torch.optim = hydra.utils.instantiate(
            cfg.discriminator_optimizer,
            params=self.discriminator.parameters()
        )

        self.generator_scheduler: torch.optim = hydra.utils.instantiate(
            cfg.generator_scheduler,
            optimizer=self.generator_optimizer
        )

        self.discriminator_scheduler: torch.optim = hydra.utils.instantiate(
            cfg.discriminator_scheduler,
            optimizer=self.discriminator_optimizer
        )

        # ** Disable automatic optimization
        self.automatic_optimization = False

        # * Loss Functions
        self.adversarial_loss: nn.Module = hydra.utils.instantiate(
            cfg.adversarial_loss
        )

        # * Loss Values
        self.generator_training_step = []
        self.discriminator_training_step = []
        self.generator_validation_step = []
        self.discriminator_validation_step = []
        self.generator_test_step = []
        self.discriminator_test_step = []

        self.avg_disc_train_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_gen_train_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_disc_val_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_gen_val_loss: torch.Tensor = torch.tensor(float('inf'))

        # * Gradient Values
        self.gradient_stats = {
            'generator': {},
            'discriminator': {}
        }

        # * Metrics
        # ** Train Metrics
        train_metrics_cfg = OmegaConf.select(cfg, 'train_metrics')
        if train_metrics_cfg is None:
            warnings.warn("No training metrics found in the configuration.")
            self.train_metrics = None
        else:
            self.train_metrics = hydra.utils.instantiate(train_metrics_cfg)

        # ** Validation Metrics
        val_metrics_cfg = OmegaConf.select(cfg, 'val_metrics')
        if val_metrics_cfg is None:
            warnings.warn("No validation metrics found in the configuration.")
            self.val_metrics = None
        else:
            self.val_metrics = hydra.utils.instantiate(val_metrics_cfg)

        # ** Discriminator Accuracy Metrics
        self.train_discriminator_fake_accuracy = []
        self.train_discriminator_real_accuracy = []
        self.val_discriminator_fake_accuracy = []
        self.val_discriminator_real_accuracy = []
        self.test_discriminator_fake_accuracy = []
        self.test_discriminator_real_accuracy = []

        self.avg_train_discriminator_fake_accuracy: float = 0.0
        self.avg_train_discriminator_real_accuracy: float = 0.0
        self.avg_val_discriminator_fake_accuracy: float = 0.0
        self.avg_val_discriminator_real_accuracy: float = 0.0
        self.avg_test_discriminator_fake_accuracy: float = 0.0
        self.avg_test_discriminator_real_accuracy: float = 0.0
        self.train_disc_acc: float = 0.0
        self.val_disc_acc: float = 0.0
        self.test_disc_acc: float = 0.0

        # * Profiler Variables
        if self.cfg.general.profiler is not None:
            self.running_training_forward_time = 0.0
            self.running_training_metrics = 0.0
            self.running_val_forward_time = 0.0
            self.running_val_metrics = 0.0
            self.logging_training_time = 0.0
            self.logging_val_time = 0.0

        # * Predicting Step Variables
        self.dimension_flag = False

        # * Training Strategy
        self.alternate_training = cfg.train.alternate_training
        if self.alternate_training:
            self.generator_counter = 0
            self.discriminator_counter = 0
        else:
            self.generator_counter = None
            self.discriminator_counter = None
        self.train_generator_every_n_epochs = cfg.train.train_generator_every_n_epochs
        self.train_discriminator_every_n_epochs = cfg.train.train_discriminator_every_n_epochs
        self.start_training_generator_epoch = cfg.train.start_training_generator_epoch
        self.start_training_discriminator_epoch = cfg.train.start_training_discriminator_epoch

        # List containing the names of the models that will be optimized at each epoch
        self.models_to_optimize = []

        self.register_buffer(
            'tic',
            torch.tensor(0.0, dtype=torch.float32)
        )
        self.register_buffer(
            'toc',
            torch.tensor(0.0, dtype=torch.float32)
        )

    def setup(self, stage):
        # * Send Metrics to Device
        if self.cfg.train.compute_metrics_on_gpu:
            if self.train_metrics is not None:
                self.train_metrics = self.train_metrics.to(self.device)
            if self.val_metrics is not None:
                self.val_metrics = self.val_metrics.to(self.device)

        # * Example Images
        self.example_images_train = {
            'HE': torch.tensor([], device=self.device),
            'IHC': torch.tensor([], device=self.device),
            'IHC_hat': torch.tensor([], device=self.device)
        }
        self.example_images_val = {
            'HE': torch.tensor([], device=self.device),
            'IHC': torch.tensor([], device=self.device),
            'IHC_hat': torch.tensor([], device=self.device)
        }
        self.example_images_test = {
            'HE': torch.tensor([], device=self.device),
            'IHC': torch.tensor([], device=self.device),
            'IHC_hat': torch.tensor([], device=self.device)
        }

    def configure_optimizers(self):
        return [self.discriminator_optimizer, self.generator_optimizer], [
            self.discriminator_scheduler, self.generator_scheduler]

    # ! To be implemented in the child class
    def forward(self, x):
        return GeneratorOutput(
            IHC_hat=self.generator(x)
        )

    def _compute_binary_accuracy(
        self,
        preds: torch.Tensor,
        real: bool,
        threshold: int = 0.5,
            ) -> torch.Tensor:
        """
        Compute the binary accuracy of the discriminator.

        Each pixel in the feature map votes independently. The image is real
        only if enough regions agree on the prediction.

        Args:
            preds (torch.Tensor): Predictions of the discriminator of size [B, C, H, W].
            real (bool): Whether the input images are real or fake.
            threshold (int): Threshold to use for the predictions.

        """

        with torch.no_grad():
            preds = normalize_logits_if_needed(tensor=preds, normalization='sigmoid')
            votes_per_region = (preds > threshold).float()
            mean_vote_per_image = torch.nanmean(
                votes_per_region,
                dim=(1, 2, 3)
                )
            votes = (mean_vote_per_image > threshold).float()
            target = torch.ones_like(votes) if real else torch.zeros_like(votes)
            accuracy = torch.nanmean((votes == target).float()).detach()

            return accuracy

    def _append_discriminator_accuracy(
        self,
        preds: torch.Tensor,
        real: bool,
        phase: Literal['train', 'val', 'test'],
        threshold: int = 0.5,
            ) -> None:

        match phase:
            case 'train':
                if real:
                    self.train_discriminator_real_accuracy.append(
                        self._compute_binary_accuracy(
                            preds=preds,
                            real=True,
                            threshold=threshold
                        )
                    )
                else:
                    self.train_discriminator_fake_accuracy.append(
                        self._compute_binary_accuracy(
                            preds=preds,
                            real=False,
                            threshold=threshold
                        )
                    )
            case 'val':
                if real:
                    self.val_discriminator_real_accuracy.append(
                        self._compute_binary_accuracy(
                            preds=preds,
                            real=True,
                            threshold=threshold
                        )
                    )
                else:
                    self.val_discriminator_fake_accuracy.append(
                        self._compute_binary_accuracy(
                            preds=preds,
                            real=False,
                            threshold=threshold
                        )
                    )
            case 'test':
                if real:
                    self.test_discriminator_real_accuracy.append(
                        self._compute_binary_accuracy(
                            preds=preds,
                            real=True,
                            threshold=threshold
                        )
                    )
                else:
                    self.test_discriminator_fake_accuracy.append(
                        self._compute_binary_accuracy(
                            preds=preds,
                            real=False,
                            threshold=threshold
                        )
                    )
            case _:
                raise ValueError(
                    "Invalid phase. Must be 'train', 'val', or 'test'."
                    )

    def discriminator_adv_loss_fn(
            self,
            disc_fake_input: torch.Tensor,
            disc_real_input: torch.Tensor,
            phase: Literal['train', 'val', 'test'],
            ) -> torch.Tensor:

        if phase not in ['train', 'val', 'test']:
            raise ValueError(
                "Invalid phase. Must be 'train', 'val', or 'test'."
                )

        # * Fake Loss
        # ** Compute Loss
        disc_fake_hat = self.discriminator(disc_fake_input)
        if isinstance(self.adversarial_loss, nn.MSELoss):
            disc_fake_hat = normalize_logits_if_needed(tensor=disc_fake_hat, normalization='sigmoid')
        disc_fake_loss = self.adversarial_loss(
            disc_fake_hat,
            torch.zeros_like(disc_fake_hat)
            )
        # ** Append Accuracy
        self._append_discriminator_accuracy(
            preds=disc_fake_hat,
            real=False,
            phase=phase
            )

        # * Real Loss
        # ** Compute Loss
        disc_real_hat = self.discriminator(disc_real_input)
        if isinstance(self.adversarial_loss, nn.MSELoss):
            disc_real_hat = normalize_logits_if_needed(tensor=disc_real_hat, normalization='sigmoid')
        disc_real_loss = self.adversarial_loss(
            disc_real_hat,
            torch.ones_like(disc_real_hat)
            )

        # ** Append Accuracy
        self._append_discriminator_accuracy(
            preds=disc_real_hat,
            real=True,
            phase=phase
            )

        # * Compute Total Loss
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        return disc_loss

    # ! To be implemented in the child class
    def discriminator_loss_fn(
            self,
            HE: torch.Tensor,  # condition
            IHC: torch.Tensor,  # target
            IHC_hat: torch.Tensor,  # prediction
            phase: Literal['train', 'val', 'test'],
            **kwargs
            ) -> torch.Tensor:

        # Detach the generator output to avoid backpropagating
        # through the generator
        if self.cfg.train.condition_discriminator:
            disc_fake_input = torch.cat((IHC_hat.detach(), HE), dim=1)
            disc_real_input = torch.cat((IHC, HE), dim=1)
        else:
            disc_fake_input = IHC_hat.detach()
            disc_real_input = IHC

        return self.discriminator_adv_loss_fn(
            disc_fake_input=disc_fake_input,
            disc_real_input=disc_real_input,
            phase=phase
            )

    def generator_adv_loss_fn(
            self,
            disc_input: torch.Tensor,
            ) -> torch.Tensor:

        disc_preds = self.discriminator(disc_input)
        if isinstance(self.adversarial_loss, nn.MSELoss):
            disc_preds = normalize_logits_if_needed(tensor=disc_preds, normalization='sigmoid')

        adversarial_loss = self.adversarial_loss(
            disc_preds,
            torch.ones_like(disc_preds)
            )

        return adversarial_loss

    # ! To be implemented in the child class
    def generator_loss_fn(
            self,
            HE: torch.Tensor,  # condition
            IHC: torch.Tensor,  # target
            IHC_hat: torch.Tensor,  # prediction
            phase: Literal['train', 'val', 'test'],
            **kwargs
            ) -> torch.Tensor:

        # Don't detach the generator output to backpropagate
        if self.cfg.train.condition_discriminator:
            disc_input = torch.cat((IHC_hat, HE), dim=1)
        else:
            disc_input = IHC_hat

        return self.generator_adv_loss_fn(
            disc_input=disc_input
            )

    def _optimization_scheduler(self) -> List[str]:

        if self.cfg.general.verbose:
            print(f"\nEpoch {self.current_epoch}: Deciding which model to train...")

        if self.alternate_training:

            if self.current_epoch < self.start_training_discriminator_epoch and self.current_epoch < self.start_training_generator_epoch:
                if self.cfg.general.verbose:
                    print("  ❌ No training yet. Both models start training later.")
                return []  # No model trains yet

            if self.discriminator_counter < self.train_discriminator_every_n_epochs:
                self.discriminator_counter += 1
                if self.cfg.general.verbose:
                    print(f"  ✅ Training Discriminator ({self.discriminator_counter}/{self.train_discriminator_every_n_epochs}).")
                if self.discriminator_counter == self.train_discriminator_every_n_epochs:
                    if self.cfg.general.verbose:
                        print("  🔄 Switching phase: Discriminator → Generator")
                return ["discriminator"]

            if self.generator_counter < self.train_generator_every_n_epochs:
                self.generator_counter += 1
                if self.cfg.general.verbose:
                    print(f"  ✅ Training Generator ({self.generator_counter}/{self.train_generator_every_n_epochs}).")
                if self.discriminator_counter == self.train_discriminator_every_n_epochs and \
                        self.generator_counter == self.train_generator_every_n_epochs:
                    if self.cfg.general.verbose:
                        print("  🔄 Switching phase: Generator → Discriminator")
                    self.generator_counter = 0
                    self.discriminator_counter = 0
                return ["generator"]

        else:
            models_to_train = []

            if self.current_epoch >= self.start_training_generator_epoch and \
                    self.current_epoch % self.train_generator_every_n_epochs == 0:
                models_to_train.append("generator")

            if self.current_epoch >= self.start_training_discriminator_epoch and \
                    self.current_epoch % self.train_discriminator_every_n_epochs == 0:
                models_to_train.append("discriminator")

            if models_to_train:
                if self.cfg.general.verbose:
                    print(f"  ✅ Training models: {models_to_train}")
                return models_to_train

            print("  ❌ No training this epoch.")
            return []

    @torch.no_grad()
    def _store_images(
            self,
            HE: torch.Tensor,
            IHC: torch.Tensor,
            IHC_hat: torch.Tensor,
            phase: Literal['Train', 'Val', 'Test']
            ) -> None:

        IHC_hat = IHC_hat.clone().detach()

        match phase:
            case 'Train':
                remaining = (
                    self.cfg.general.num_example_images
                    -
                    len(self.example_images_train['HE'])
                    )
                if remaining > 0:
                    self.example_images_train['HE'] = torch.cat(
                        (self.example_images_train['HE'], HE[:remaining]),
                        dim=0,
                    )
                    self.example_images_train['IHC'] = torch.cat(
                        (self.example_images_train['IHC'], IHC[:remaining]),
                        dim=0,
                    )
                    self.example_images_train['IHC_hat'] = torch.cat(
                        (self.example_images_train['IHC_hat'], IHC_hat[:remaining]),
                        dim=0,
                    )

            case 'Val':
                remaining = (
                    self.cfg.general.num_example_images
                    -
                    len(self.example_images_val['HE'])
                    )
                if remaining > 0:
                    self.example_images_val['HE'] = torch.cat(
                        (self.example_images_val['HE'], HE[:remaining]),
                        dim=0,
                    )
                    self.example_images_val['IHC'] = torch.cat(
                        (self.example_images_val['IHC'], IHC[:remaining]),
                        dim=0,
                    )
                    self.example_images_val['IHC_hat'] = torch.cat(
                        (self.example_images_val['IHC_hat'], IHC_hat[:remaining]),
                        dim=0,
                    )

            case 'Test':
                remaining = (
                    self.cfg.general.num_example_images
                    -
                    len(self.example_images_test['HE'])
                    )
                if remaining > 0:
                    self.example_images_test['HE'] = torch.cat(
                        (self.example_images_test['HE'], HE[:remaining]),
                        dim=0,
                    )
                    self.example_images_test['IHC'] = torch.cat(
                        (self.example_images_test['IHC'], IHC[:remaining]),
                        dim=0,
                    )
                    self.example_images_test['IHC_hat'] = torch.cat(
                        (self.example_images_test['IHC_hat'], IHC_hat[:remaining]),
                        dim=0,
                    )

            case _:
                raise ValueError(
                    "Invalid phase. Must be 'Train', 'Val', or 'Test'."
                    )
        return None

    @torch.no_grad()
    def _log_images(
            self,
            HE: torch.Tensor,
            IHC: torch.Tensor,
            IHC_hat: torch.Tensor,
            epoch: int,
            phase: Literal['Train', 'Val', 'Test']
            ) -> None:
        images = torch.cat((HE, IHC, IHC_hat), dim=0)
        if images.size(0) != 3 * self.cfg.general.num_example_images:
            raise ValueError(
                f"Mismatch in the number of examples: Expected {3 * self.cfg.general.num_example_images}, but got {images.size(0)}"
            )
        images = images.view(3, self.cfg.general.num_example_images, *images.shape[1:])  # Shape: (3, num_examples, C, H, W)
        images = images.transpose(0, 1).contiguous()  # Shape: (num_examples, 3, C, H, W)
        images = images.view(-1, *images.shape[2:])

        # Resize images to reduce the size of the grid
        target_size = (images.shape[2] // 2, images.shape[3] // 2)  # Example: reduce size by half
        images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=True)

        grid = make_grid(
            tensor=images,
            nrow=self.cfg.general.num_example_images,
            pad_value=1
        )
        # title = f'H&E Vs. IHC Vs. Fake - {phase}: {epoch}'
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'H&E Vs. IHC Vs. Fake - {phase}',
                grid,
                global_step=epoch
            )
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=f'H&E Vs. IHC Vs. Fake - {phase}',
                images=[grid],
                step=epoch
            )
        else:
            raise NotImplementedError(
                f"Logger must be one of 'TensorBoardLogger' or 'WandbLogger', got {self.logger.type}."
            )

    def _verify_training_setup(self) -> None:
        settings = {
            "Seed": torch.initial_seed(),
            "Default dtype": torch.get_default_dtype(),
            "Device": self.device,
            "Deterministic": torch.are_deterministic_algorithms_enabled(),
            "Matmul Precision": torch.get_float32_matmul_precision(),
            "CUDNN Benchmarking": torch.backends.cudnn.benchmark if self.device.type == 'cuda' else "N/A",
            "Trainer Precision": self.trainer.precision,
        }

        # Check if benchmarking is enabled for CUDA
        if self.device.type == 'cuda' and not torch.backends.cudnn.benchmark:
            warnings.warn(
                "CUDNN Benchmarking is disabled on CUDA. Performance may be suboptimal. "
                "You can enable it by setting `torch.backends.cudnn.benchmark = True`."
            )

        if self.cfg.general.verbose:
            print("\n=== Training Setup Verification ===")
            for key, value in settings.items():
                print(f"{key}: {value}")
            print("====================================\n")

    def on_train_start(self):
        self._verify_training_setup()

    def on_train_epoch_start(self):
        self.models_to_optimize = self._optimization_scheduler()
        if self.current_epoch > 0:
            self.toc = torch.tensor(time(), dtype=self.dtype, device=self.device)
            print(f"Epoch {self.current_epoch - 1} took {(self.toc - self.tic)/3600:.4f} hours.")
        self.tic = torch.tensor(time(), dtype=self.dtype, device=self.device)

    def _optimize_discriminator(
            self,
            discriminator_optimizer: torch.optim.Optimizer,
            batch_idx: int,
            HE: torch.Tensor,
            IHC: torch.Tensor,
            generator_output: GeneratorOutput
            ) -> torch.Tensor:

        # ** Compute Discriminator Loss
        disc_train_loss = self.discriminator_loss_fn(
            HE=HE,
            IHC=IHC,
            phase='train',
            **generator_output._asdict()
            )
        disc_train_loss = disc_train_loss / self.cfg.train.accumulate_grad_batches

        # ** Backward Pass
        self.manual_backward(
            loss=disc_train_loss,
            model_name='discriminator',
            retain_graph=True
            )

        # ** Gradient Clipping
        if self.cfg.train.clip_grad_value is not None \
                and self.cfg.train.clip_grad_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.cfg.train.clip_grad_value
                )

        # ** Update Discriminator
        if (batch_idx + 1) % self.cfg.train.accumulate_grad_batches == 0 \
                or (batch_idx + 1) == (len(self.trainer.train_dataloader)):
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()

        return disc_train_loss

    def _optimize_generator(
            self,
            generator_optimizer: torch.optim.Optimizer,
            batch_idx: int,
            HE: torch.Tensor,
            IHC: torch.Tensor,
            generator_output: GeneratorOutput
            ) -> torch.Tensor:

        # ** Compute Generator Loss
        gen_train_loss = self.generator_loss_fn(
            HE=HE,
            IHC=IHC,
            phase='train',
            **generator_output._asdict()
            )
        gen_train_loss = gen_train_loss / self.cfg.train.accumulate_grad_batches

        # ** Backward Pass
        self.manual_backward(loss=gen_train_loss, model_name='generator')

        # ** Gradient Clipping
        if self.cfg.train.clip_grad_value is not None \
                and self.cfg.train.clip_grad_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.cfg.train.clip_grad_value
                )

        # ** Update Generator
        if (batch_idx + 1) % self.cfg.train.accumulate_grad_batches == 0 \
                or (batch_idx + 1) == (len(self.trainer.train_dataloader)):
            generator_optimizer.step()
            generator_optimizer.zero_grad()

        return gen_train_loss

    def training_step(self, batch, batch_idx):
        HE, IHC = batch

        if HE.dim() != IHC.dim():
            raise ValueError(
                "HE and IHC must have the same number of dimensions."
                f"HE: {HE.dim()}, IHC: {IHC.dim()}"
                )
        else:
            dim = HE.dim()

        if dim == 5:
            HE = HE.view(-1, *HE.shape[2:])
            IHC = IHC.view(-1, *IHC.shape[2:])

        discriminator_optimizer, generator_optimizer = self.optimizers()

        # Track gradients in forward pass only if we are training the generator
        if 'generator' in self.models_to_optimize:
            out = self.forward(HE)
        else:
            with torch.no_grad():
                out = self.forward(HE)

        # with torch.no_grad():
        #     HE[:, 0, :, :] = HE[:, 0, :, :] * 0.229 + 0.485
        #     HE[:, 1, :, :] = HE[:, 1, :, :] * 0.224 + 0.456
        #     HE[:, 2, :, :] = HE[:, 2, :, :] * 0.225 + 0.406

        # * Update Metrics
        if self.cfg.general.profiler is not None:
            tic = time()
        if self.train_metrics is not None:
            if self.cfg.train.compute_metrics_on_gpu:
                self.train_metrics.update(
                    y_pred=out.IHC_hat,
                    y=IHC
                    )
            else:
                self.train_metrics.update(
                    y_pred=out.IHC_hat.clone().detach().cpu(),
                    y=IHC.cpu()
                    )
        if self.cfg.general.num_example_images > 0:
            with torch.no_grad():
                # Unnormalize the ImageNet normalizaton
                log_HE = HE.clone().detach()
                log_HE[:, 0, :, :] = log_HE[:, 0, :, :] * 0.229 + 0.485
                log_HE[:, 1, :, :] = log_HE[:, 1, :, :] * 0.224 + 0.456
                log_HE[:, 2, :, :] = log_HE[:, 2, :, :] * 0.225 + 0.406

                self._store_images(log_HE, IHC, out.IHC_hat, phase='Train')

        if self.cfg.general.profiler is not None:
            toc = time()
            self.running_training_metrics += toc - tic

        # * Update Discriminator
        if self.cfg.general.profiler is not None:
            tic = time()
        if 'discriminator' in self.models_to_optimize:
            self.toggle_optimizer(optimizer=discriminator_optimizer)
            disc_train_loss = self._optimize_discriminator(
                discriminator_optimizer=discriminator_optimizer,
                batch_idx=batch_idx,
                HE=HE,
                IHC=IHC,
                generator_output=out
            )
            self.discriminator_training_step.append(disc_train_loss.detach())
            self.untoggle_optimizer(optimizer=discriminator_optimizer)
        else:
            with torch.no_grad():
                disc_train_loss = self.discriminator_loss_fn(
                    HE=HE,
                    IHC=IHC,
                    phase='train',
                    **out._asdict()
                    )
                self.discriminator_training_step.append(disc_train_loss.detach())

        # * Update Generator
        if 'generator' in self.models_to_optimize:
            self.toggle_optimizer(optimizer=generator_optimizer)
            gen_train_loss = self._optimize_generator(
                generator_optimizer=generator_optimizer,
                batch_idx=batch_idx,
                HE=HE,
                IHC=IHC,
                generator_output=out
            )
            self.generator_training_step.append(gen_train_loss.detach())
            self.untoggle_optimizer(optimizer=generator_optimizer)
        else:
            with torch.no_grad():
                gen_train_loss = self.generator_loss_fn(
                    HE=HE,
                    IHC=IHC,
                    **out._asdict()
                    )
                self.generator_training_step.append(gen_train_loss.detach())

        if self.cfg.general.profiler is not None:
            toc = time()
            self.running_training_forward_time += toc - tic

        # # * Clear Memory
        # del out
        # gc.collect()
        # if self.device.type == 'cuda':
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()

        return {
            'discriminator_train_loss': disc_train_loss,
            'generator_train_loss': gen_train_loss
            }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        HE, IHC = batch

        if HE.dim() != IHC.dim():
            raise ValueError(
                "HE and IHC must have the same number of dimensions."
                f"HE: {HE.dim()}, IHC: {IHC.dim()}"
                )
        else:
            dim = HE.dim()

        if dim == 5:
            HE = HE.view(-1, *HE.shape[2:])
            IHC = IHC.view(-1, *IHC.shape[2:])

        out = self.forward(HE)

        # with torch.no_grad():
        #     HE[:, 0, :, :] = HE[:, 0, :, :] * 0.229 + 0.485
        #     HE[:, 1, :, :] = HE[:, 1, :, :] * 0.224 + 0.456
        #     HE[:, 2, :, :] = HE[:, 2, :, :] * 0.225 + 0.406

        # * Update Metrics
        if self.cfg.general.profiler is not None:
            tic = time()
        if self.val_metrics is not None:
            if self.cfg.train.compute_metrics_on_gpu:
                self.val_metrics.update(
                    y_pred=out.IHC_hat,
                    y=IHC
                    )
            else:
                self.val_metrics.update(
                    y_pred=out.IHC_hat.clone().detach().cpu(),
                    y=IHC.cpu()
                    )
        if self.cfg.general.num_example_images > 0:
            # # Unnormalize the ImageNet normalizaton
            log_HE = HE.clone().detach()
            log_HE[:, 0, :, :] = log_HE[:, 0, :, :] * 0.229 + 0.485
            log_HE[:, 1, :, :] = log_HE[:, 1, :, :] * 0.224 + 0.456
            log_HE[:, 2, :, :] = log_HE[:, 2, :, :] * 0.225 + 0.406
            self._store_images(log_HE, IHC, out.IHC_hat, phase='Val')

        if self.cfg.general.profiler is not None:
            toc = time()
            self.running_val_metrics += toc - tic

        # * Compute Discriminator Loss
        if self.cfg.general.profiler is not None:
            tic = time()

        disc_val_loss = self.discriminator_loss_fn(
            HE=HE,
            IHC=IHC,
            phase='val',
            **out._asdict()
            )
        self.discriminator_validation_step.append(disc_val_loss.detach())

        # * Compute Generator Loss
        gen_val_loss = self.generator_loss_fn(
            HE=HE,
            IHC=IHC,
            phase='val',
            **out._asdict()
            )
        self.generator_validation_step.append(gen_val_loss.detach())

        if self.cfg.general.profiler is not None:
            toc = time()
            self.running_val_forward_time += toc - tic

        # # * Clear Memory
        # del out
        # gc.collect()
        # if self.device.type == 'cuda':
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()

        return {
            'discriminator_val_loss': disc_val_loss,
            'generator_val_loss': gen_val_loss
            }

    def on_test_start(self):
        self.ToPIL = v2.ToPILImage()
        os.makedirs(os.path.join(self.cfg.hydra.run.dir, 'predictions_test'), exist_ok=True)
        test_metrics_cfg = OmegaConf.select(self.cfg, 'test_metrics')
        if test_metrics_cfg is None:
            warnings.warn("No test metrics found in the configuration.")
        else:
            self.test_metrics = hydra.utils.instantiate(test_metrics_cfg)
            os.makedirs(os.path.join(self.cfg.hydra.run.dir, 'predictions_test', 'metrics'), exist_ok=True)
            if self.cfg.train.compute_metrics_on_gpu:
                self.test_metrics = self.test_metrics.to(self.device)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        HE, IHC, base_name = batch

        if HE.dim() != IHC.dim():
            raise ValueError(
                "HE and IHC must have the same number of dimensions."
                f"HE: {HE.dim()}, IHC: {IHC.dim()}"
                )
        else:
            dim = HE.dim()

        if dim == 5:
            HE = HE.view(-1, *HE.shape[2:])
            IHC = IHC.view(-1, *IHC.shape[2:])
        
        out = self.forward(HE)

        # with torch.no_grad():
        #     HE[:, 0, :, :] = HE[:, 0, :, :] * 0.229 + 0.485
        #     HE[:, 1, :, :] = HE[:, 1, :, :] * 0.224 + 0.456
        #     HE[:, 2, :, :] = HE[:, 2, :, :] * 0.225 + 0.406

        # * Update Metrics
        if self.test_metrics is not None:
            if self.cfg.train.compute_metrics_on_gpu:
                self.test_metrics.update(
                    y_pred=out.IHC_hat,
                    y=IHC
                    )
            else:
                self.test_metrics.update(
                    y_pred=out.IHC_hat.clone().detach().cpu(),
                    y=IHC.cpu()
                    )
        if self.cfg.general.num_example_images > 0:
            log_HE = HE.clone().detach()
            log_HE[:, 0, :, :] = log_HE[:, 0, :, :] * 0.229 + 0.485
            log_HE[:, 1, :, :] = log_HE[:, 1, :, :] * 0.224 + 0.456
            log_HE[:, 2, :, :] = log_HE[:, 2, :, :] * 0.225 + 0.406

            self._store_images(log_HE, IHC, out.IHC_hat, phase='Test')

        # Compute Discriminator Loss
        disc_loss = self.discriminator_loss_fn(
            HE=HE,
            IHC=IHC,
            phase='test',
            **out._asdict()
            )
        self.discriminator_test_step.append(disc_loss.detach())

        # Compute Generator Loss
        gen_loss = self.generator_loss_fn(
            HE=HE,
            IHC=IHC,
            phase='test',
            **out._asdict()
            )
        self.generator_test_step.append(gen_loss.detach())

        for pred in out.IHC_hat:
            pred_image = self.ToPIL(pred)
            pred_image.save(
                os.path.join(
                    self.cfg.hydra.run.dir,
                    'predictions_test',
                    f'{base_name[0]}.jpg'
                    )
                )

        # # * Clear Memory
        # del out
        # gc.collect()
        # if self.device.type == 'cuda':
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()

        return {
            'discriminator_test_loss': disc_loss,
            'generator_test_loss': gen_loss
            }

    def on_predict_epoch_start(self):
        self.ToPIL = v2.ToPILImage()
        os.makedirs(os.path.join(self.cfg.hydra.run.dir, 'predictions'), exist_ok=True)

        predict_metrics_cfg = OmegaConf.select(self.cfg, 'predict_metrics')
        if predict_metrics_cfg is None:
            warnings.warn("No predict metrics found in the configuration.")
            self.predict_metrics = None
        else:
            self.predict_metrics = hydra.utils.instantiate(predict_metrics_cfg)
            os.makedirs(os.path.join(self.cfg.hydra.run.dir, 'predictions', 'metrics'), exist_ok=True)
            if self.cfg.train.compute_metrics_on_gpu:
                self.predict_metrics = self.predict_metrics.to(self.device)

    @torch.no_grad()
    def predict_step(self, *args, **kwargs):
        """
        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)
        """
        HE, IHC, base_name = kwargs.get("batch", args[0])

        if HE.dim() != IHC.dim():
            raise ValueError(
                "HE and IHC must have the same number of dimensions."
                f"HE: {HE.dim()}, IHC: {IHC.dim()}"
                )
        else:
            dim = HE.dim()

        if dim == 5:
            HE = HE.view(-1, *HE.shape[2:])
            IHC = IHC.view(-1, *IHC.shape[2:])
            self.dimension_flag = True

        if self.dimension_flag:
            raise NotImplementedError(
                "The predict_step method has not been implemented for 5D tensors."
            )

        IHC_hat = self.forward(HE).IHC_hat

        if self.predict_metrics is not None:
            if self.cfg.train.compute_metrics_on_gpu:
                self.predict_metrics.update(
                    y_pred=IHC_hat,
                    y=IHC
                    )
            else:
                self.predict_metrics.update(
                    y_pred=IHC_hat.clone().detach().cpu(),
                    y=IHC.cpu()
                    )

        for pred in IHC_hat:
            pred_image = self.ToPIL(pred)
            pred_image.save(
                os.path.join(
                    self.cfg.hydra.run.dir,
                    'predictions',
                    f'{base_name[0]}.jpg'
                    )
                )

    def on_predict_epoch_end(self):
        if self.predict_metrics is not None:
            metrics_result = self.predict_metrics.compute()
            with open(
                os.path.join(self.cfg.hydra.run.dir, 'predictions', 'metrics', 'metrics.txt'),
                'w',
            ) as f:
                for key, value in metrics_result.items():
                    if 'kid' in key:
                        f.write(f"{key}: {value[0]:.4f} ± {value[1]:.4f}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")

    def on_train_epoch_end(self) -> None:
        # ! Stepping the scheduler should be after computing the average losses
        # TODO: Correct this error
        if 'discriminator' in self.models_to_optimize:
            self.log(
                'disc_train_stage',
                1,
                on_epoch=True,
                on_step=False,
                logger=True,
            )

            # * Step Scheduler
            if isinstance(
                self.discriminator_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                self.discriminator_scheduler.step(self.avg_disc_train_loss)
            else:
                self.discriminator_scheduler.step()
        else:
            self.log(
                'disc_train_stage',
                0,
                on_epoch=True,
                on_step=False,
                logger=True,
            )

        if 'generator' in self.models_to_optimize:
            self.log(
                'gen_train_stage',
                1,
                on_epoch=True,
                on_step=False,
                logger=True,
            )

            # ** Step Scheduler
            if isinstance(
                self.generator_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                self.generator_scheduler.step(self.avg_gen_train_loss)
            else:
                self.generator_scheduler.step()
        else:
            self.log(
                'gen_train_stage',
                0,
                on_epoch=True,
                on_step=False,
                logger=True,
            )

        # * Loss Averages
        # ** Discriminator
        # Compute Average Loss
        with torch.no_grad():
            self.avg_disc_train_loss = torch.stack(
                self.discriminator_training_step
                ).nanmean()

            # Reset Losses
            self.discriminator_training_step.clear()

            # ** Generator
            # Compute Average Loss
            self.avg_gen_train_loss = torch.stack(
                self.generator_training_step
                ).nanmean()

            # Reset Losses
            self.generator_training_step.clear()

            self.log(
                'discriminator_train_loss',
                self.avg_disc_train_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )

            self.log(
                'generator_train_loss',
                self.avg_gen_train_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )

        if self.cfg.general.verbose:
            print("\n\n === Train Losses ===")
            print(f"discriminator_train_loss: {self.avg_disc_train_loss}")
            print(f"generator_train_loss: {self.avg_gen_train_loss}")

        # * Discriminator Accuracy Averages
        # ** Compute Average Accuracy
        with torch.no_grad():
            self.avg_train_discriminator_fake_accuracy = torch.stack(
                self.train_discriminator_fake_accuracy
                ).nanmean()
            self.avg_train_discriminator_real_accuracy = torch.stack(
                self.train_discriminator_real_accuracy
                ).nanmean()
            self.train_disc_acc = (
                self.avg_train_discriminator_fake_accuracy
                +
                self.avg_train_discriminator_real_accuracy) / 2

            # ** Log Average Accuracy
            self.log(
                'train_disc_fake_acc',
                self.avg_train_discriminator_fake_accuracy,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )

            self.log(
                'train_disc_real_acc',
                self.avg_train_discriminator_real_accuracy,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )

            self.log(
                'train_disc_acc',
                self.train_disc_acc,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True
            )

            # ** Reset Accuracies
            self.train_discriminator_fake_accuracy.clear()
            self.train_discriminator_real_accuracy.clear()

        # * Log Metrics
        if self.cfg.general.profiler is not None:
            tic = time()
        if self.train_metrics is not None:
            with torch.no_grad():
                metrics_result = self.train_metrics.compute()
                for key, value in metrics_result.items():
                    if 'kid' in key:
                        self.log(
                            name=f"{key}_train",
                            value=value[0],
                            on_epoch=True,
                            on_step=False,
                            sync_dist=True
                        )
                    else:
                        self.log(
                            name=f"{key}_train",
                            value=value,
                            on_epoch=True,
                            on_step=False,
                            sync_dist=True
                        )

            if self.cfg.general.verbose:
                print("\n\n === Train Metrics ===")
                for key, value in metrics_result.items():
                    print(f"{key}_train: {value}")
                print("\n\n")

            self.train_metrics.reset()

        # * Log Images
        if self.cfg.general.num_example_images > 0:
            self._log_images(
                HE=self.example_images_train['HE'],
                IHC=self.example_images_train['IHC'],
                IHC_hat=self.example_images_train['IHC_hat'],
                epoch=self.current_epoch,
                phase='Train'
            )
            for key in self.example_images_train.keys():
                self.example_images_train[key] = torch.tensor([], device=self.device)

        if self.cfg.general.profiler is not None:
            toc = time()
            self.logging_training_time += toc - tic

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        # * Loss Averages
        # ** Discriminator
        # Compute Average Loss
        self.avg_disc_val_loss = torch.stack(
            self.discriminator_validation_step
            ).nanmean()

        # Reset Losses
        self.discriminator_validation_step.clear()

        # ** Generator
        # Compute Average Loss
        self.avg_gen_val_loss = torch.stack(
            self.generator_validation_step
            ).nanmean()

        # Reset Losses
        self.generator_validation_step.clear()

        # ** Log Average Loss
        self.log(
            'discriminator_val_loss',
            self.avg_disc_val_loss,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        self.log(
            'generator_val_loss',
            self.avg_gen_val_loss,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        if self.cfg.general.verbose:
            print("\n\n === Validation Losses ===")
            print(f"discriminator_val_loss: {self.avg_disc_val_loss}")
            print(f"generator_val_loss: {self.avg_gen_val_loss}")

        # * Discriminator Accuracy Averages
        # ** Compute Average Accuracy
        self.avg_val_discriminator_fake_accuracy = torch.stack(
            self.val_discriminator_fake_accuracy
            ).nanmean()
        self.avg_val_discriminator_real_accuracy = torch.stack(
            self.val_discriminator_real_accuracy
            ).nanmean()
        self.val_disc_acc = (
            self.avg_val_discriminator_fake_accuracy
            +
            self.avg_val_discriminator_real_accuracy) / 2

        # ** Log Average Accuracy
        self.log(
            'val_disc_fake_acc',
            self.avg_val_discriminator_fake_accuracy,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        self.log(
            'val_disc_real_acc',
            self.avg_val_discriminator_real_accuracy,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        self.log(
            'val_disc_acc',
            self.val_disc_acc,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        # ** Reset Accuracies
        self.val_discriminator_fake_accuracy.clear()
        self.val_discriminator_real_accuracy.clear()

        # * Log Metrics
        if self.cfg.general.profiler is not None:
            tic = time()
        if self.val_metrics is not None:
            metrics_result = self.val_metrics.compute()
            for key, value in metrics_result.items():
                if 'kid' in key:
                    self.log(
                        name=f"{key}_val",
                        value=value[0],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True
                    )
                else:
                    self.log(
                        name=f"{key}_val",
                        value=value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True
                    )

            if self.cfg.general.verbose:
                print("\n\n === Validation Metrics ===")
                for key, value in metrics_result.items():
                    print(f"{key}_val: {value}")
                print("\n\n")

            self.val_metrics.reset()

        # * Log Images
        if self.cfg.general.num_example_images > 0:
            self._log_images(
                HE=self.example_images_val['HE'],
                IHC=self.example_images_val['IHC'],
                IHC_hat=self.example_images_val['IHC_hat'],
                epoch=self.current_epoch,
                phase='Val'
            )
            for key in self.example_images_val.keys():
                self.example_images_val[key] = torch.tensor([], device=self.device)

        if self.cfg.general.profiler is not None:
            toc = time()
            self.logging_val_time += toc - tic

    @torch.no_grad()
    def on_test_epoch_end(self):
        # * Loss Averages
        # ** Compute Average Loss
        avg_disc_loss = torch.stack(self.discriminator_test_step).nanmean()
        avg_gen_loss = torch.stack(self.generator_test_step).nanmean()

        # ** Log Average Loss
        self.log(
            'discriminator_test_loss',
            avg_disc_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True
            )
        self.log(
            'generator_test_loss',
            avg_gen_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True
            )

        # ** Reset Losses
        self.discriminator_test_step.clear()
        self.generator_test_step.clear()

        # * Discriminator Accuracy Averages
        # ** Compute Average Accuracy
        self.avg_test_discriminator_fake_accuracy = torch.stack(
            self.test_discriminator_fake_accuracy
            ).nanmean()
        self.avg_test_discriminator_real_accuracy = torch.stack(
            self.test_discriminator_real_accuracy
            ).nanmean()
        self.test_disc_acc = (
            self.avg_test_discriminator_fake_accuracy
            +
            self.avg_test_discriminator_real_accuracy) / 2

        # ** Log Average Accuracy
        self.log(
            'test_disc_fake_acc',
            self.avg_test_discriminator_fake_accuracy,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        self.log(
            'test_disc_real_acc',
            self.avg_test_discriminator_real_accuracy,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        self.log(
            'test_disc_acc',
            self.test_disc_acc,
            on_epoch=True,
            logger=True,
            on_step=False,
            sync_dist=True
        )

        # ** Reset Accuracies
        self.test_discriminator_fake_accuracy.clear()
        self.test_discriminator_real_accuracy.clear()

        # * Log Metrics
        if self.test_metrics is not None:
            metrics_result = self.test_metrics.compute()
            with open(
                os.path.join(self.cfg.hydra.run.dir, 'predictions_test', 'metrics', 'metrics.txt'),
                'w',
            ) as f:
                for key, value in metrics_result.items():
                    if 'kid' in key:
                        self.log(
                            name=f"{key}_test",
                            value=value[0],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True
                        )
                        f.write(f"{key}: {value[0]:.4f} ± {value[1]:.4f}\n")
                    else:
                        self.log(
                            name=f"{key}_test",
                            value=value,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True
                        )
                        f.write(f"{key}: {value:.4f}\n")

            if self.cfg.general.verbose:
                print("\n\n === Test Metrics ===")
                for key, value in metrics_result.items():
                    print(f"{key}_test: {value}\n")

            self.test_metrics.reset()

        # * Log Images
        if self.cfg.general.num_example_images > 0:
            self._log_images(
                HE=self.example_images_test['HE'],
                IHC=self.example_images_test['IHC'],
                IHC_hat=self.example_images_test['IHC_hat'],
                epoch=self.current_epoch,
                phase='Test'
            )
            for key in self.example_images_test.keys():
                self.example_images_test[key] = torch.tensor([], device=self.device)

    def on_train_end(self):
        # Sort and print running times
        if self.cfg.general.profiler is not None:
            running_times = {
                'running_training_forward_time': self.running_training_forward_time,
                'running_training_metrics': self.running_training_metrics,
                'running_val_forward_time': self.running_val_forward_time,
                'running_val_metrics': self.running_val_metrics,
                'logging_training_time': self.logging_training_time,
                'logging_val_time': self.logging_val_time
            }

            sorted_times = sorted(running_times.items(), key=lambda item: item[1], reverse=True)

            print("\n\n=== Running Times (sorted) ===")
            for name, times in sorted_times:
                print(f"{name}: {times:.4f} seconds")
            print("=============================\n")

    def manual_backward(self, loss, model_name=None, *args, **kwargs):
        super().manual_backward(loss, *args, **kwargs)
        if self.cfg.general.log_gradients:
            if model_name is not None:
                model = getattr(self, model_name, None)
                if model is None:
                    raise ValueError(
                        f"Invalid model_name '{model_name}'. Must be 'generator' or 'discriminator'."
                    )

                with torch.no_grad():
                    grads = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]

                    if grads:  # Avoid computing if no gradients are present
                        overall_grad = torch.cat(grads)  # Concatenate all gradients
                        grad_norm = torch.norm(overall_grad).item()  # Compute the norm
                    else:
                        grad_norm = 0.0  # Default to 0 if no gradients exist

                    # Store the overall gradient norm for the model
                    self.gradient_stats[model_name] = grad_norm

                    self.log(
                        f'{model_name}_grad_norm',
                        grad_norm,
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True
                        )

