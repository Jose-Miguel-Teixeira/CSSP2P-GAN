import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import pytorch_lightning as pl
from pytorch_lightning.loggers import (
    TensorBoardLogger,
    WandbLogger
)
import hydra
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from torchvision.utils import make_grid
from collections import namedtuple
from time import time
import os
from typing import Literal, List
import warnings
import gc

GeneratorOutput = namedtuple('GeneratorOutput', ['IHC_hat'])


class HydraUNet(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(HydraUNet, self).__init__()

        # * Save Hyperparameters
        self.cfg = cfg
        self.save_hyperparameters(
            OmegaConf.to_container(cfg, resolve=True)
            )

        # * Define Example Array
        self.example_input_array = torch.randn(
            (1, 3, *self.cfg.generator.input_size)
        )

        # * Model
        self.generator: nn.Module = hydra.utils.instantiate(
            cfg.generator_model,
        )

        if cfg.general.verbose:
            print('\n GENERATOR SUMMARY:')
            summary(
                model=self.generator,
                input_size=(1, 3, *self.cfg.generator.input_size),
            )

        if not isinstance(self.generator.activation, nn.Sigmoid):
            raise ValueError(
                f"Generator activation function must be Sigmoid, but got {self.generator.activation.__class__.__name__}"
                "Currently it is expected that the output to be in the range [0, 1] which is the range of the input IHC."
            )

        if cfg.generator.compilation.enabled:
            self.generator = torch.compile(
                model=self.generator,
                backend='inductor',
                fullgraph=cfg.generator.compilation.fullgraph,
            )

        # * Optimizer and Scheduler
        self.optimizer: torch.optim = hydra.utils.instantiate(
            cfg.generator_optimizer,
            params=self.generator.parameters(),
        )

        self.scheduler: torch.optim = hydra.utils.instantiate(
            cfg.generator_scheduler,
            optimizer=self.optimizer,
        )

        # ** Disable automatic optimization
        self.automatic_optimization = False

        # * Loss Function
        self.loss_fn: nn.Module = hydra.utils.instantiate(
            cfg.criterion
        )

        # * Loss Values
        self.generator_training_step: List[torch.Tensor] = []
        self.generator_validation_step: List[torch.Tensor] = []
        self.generator_test_step: List[torch.Tensor] = []

        self.avg_gen_train_loss: torch.Tensor = torch.tensor(float('inf'))
        self.avg_gen_val_loss: torch.Tensor = torch.tensor(float('inf'))

        # * Metrics
        train_metrics_cfg = OmegaConf.select(cfg, 'train_metrics')
        if train_metrics_cfg is None:
            warnings.warn("No training metrics found in the configuration.")
            self.train_metrics = None
        else:
            self.train_metrics = hydra.utils.instantiate(train_metrics_cfg)

        val_metrics_cfg = OmegaConf.select(cfg, 'val_metrics')
        if val_metrics_cfg is None:
            warnings.warn("No validation metrics found in the configuration.")
            self.val_metrics = None
        else:
            self.val_metrics = hydra.utils.instantiate(val_metrics_cfg)

        # * Profiler Variables
        if self.cfg.general.profiler is not None:
            self.running_training_forward_time = 0.0
            self.running_training_metrics = 0.0
            self.running_val_forward_time = 0.0
            self.running_val_metrics = 0.0
            self.logging_training_time = 0.0
            self.logging_val_time = 0.0
        
        self.dimension_flag = False

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
        return [self.optimizer], [self.scheduler]

    def forward(self, x):
        return GeneratorOutput(
            IHC_hat=self.generator(x)
        )

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
        if self.current_epoch > 0:
            self.toc = torch.tensor(time(), dtype=self.dtype, device=self.device)
            print(f"Epoch {self.current_epoch - 1} took {(self.toc - self.tic)/3600:.4f} hours.")
        self.tic = torch.tensor(time(), dtype=self.dtype, device=self.device)

    def _compute_loss(
            self,
            HE: torch.Tensor,
            IHC: torch.Tensor,
            IHC_hat: torch.Tensor,
            phase: Literal['train', 'val', 'test'],
            **kwargs
        ) -> torch.Tensor:

        return self.loss_fn(
            input=IHC_hat,
            target=IHC
            )

    def optimize_generator(
            self,
            optimizer: torch.optim.Optimizer,
            batch_idx: int,
            HE: torch.Tensor,
            IHC: torch.Tensor,
            generator_output: GeneratorOutput,
            **kwargs
        ) -> torch.Tensor:

        loss = self._compute_loss(
            HE=HE,
            IHC=IHC,
            phase='train',
            **generator_output._asdict()
        )

        loss = loss / self.cfg.train.accumulate_grad_batches

        self.manual_backward(loss)

        # ** Gradient Clipping
        if self.cfg.train.clip_grad_value is not None \
                and self.cfg.train.clip_grad_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.cfg.train.clip_grad_value
                )

        # ** Optimizer Step
        if (batch_idx + 1) % self.cfg.train.accumulate_grad_batches == 0 \
                or (batch_idx + 1) == (len(self.trainer.train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
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

        # * Compute Loss
        if self.cfg.general.profiler is not None:
            tic = time()

        optimizer = self.optimizers()
        self.toggle_optimizer(optimizer)

        loss = self.optimize_generator(
            optimizer=optimizer,
            batch_idx=batch_idx,
            HE=HE,
            IHC=IHC,
            generator_output=out,
        )

        self.generator_training_step.append(loss.detach())
        self.untoggle_optimizer(optimizer)

        if self.cfg.general.profiler is not None:
            toc = time()
            self.running_training_forward_time += toc - tic

        # # * Clear Memory
        # del out
        # gc.collect()
        # if self.device.type == 'cuda':
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
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
            # Unnormalize the ImageNet normalizaton
            log_HE = HE.clone().detach()
            log_HE[:, 0, :, :] = log_HE[:, 0, :, :] * 0.229 + 0.485
            log_HE[:, 1, :, :] = log_HE[:, 1, :, :] * 0.224 + 0.456
            log_HE[:, 2, :, :] = log_HE[:, 2, :, :] * 0.225 + 0.406
            self._store_images(log_HE, IHC, out.IHC_hat, phase='Val')

        if self.cfg.general.profiler is not None:
            toc = time()
            self.running_val_metrics += toc - tic

        # * Compute Loss
        if self.cfg.general.profiler is not None:
            tic = time()
        loss = self._compute_loss(
            HE=HE,
            IHC=IHC,
            phase='val',
            **out._asdict()
        )
        self.generator_validation_step.append(loss.clone().detach())
        if self.cfg.general.profiler is not None:
            toc = time()
            self.running_val_forward_time += toc - tic

        return loss
    
    def on_test_start(self):
        self.ToPIL = v2.ToPILImage()
        os.makedirs(os.path.join(self.cfg.hydra.run.dir, 'predictions_test'), exist_ok=True)
        test_metrics_cfg = OmegaConf.select(self.cfg, 'test_metrics')
        if test_metrics_cfg is None:
            warnings.warn("No test metrics found in the configuration.")
        else:
            self.test_metrics = hydra.utils.instantiate(test_metrics_cfg)
            if self.cfg.train.compute_metrics_on_gpu:
                self.test_metrics = self.test_metrics.to(self.device)

    @torch.no_grad()
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        HE, IHC, base_name = batch

        out = self.forward(HE)

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

        # * Compute Loss
        loss = self._compute_loss(
            HE=HE,
            IHC=IHC,
            phase='test',
            **out._asdict()
        )
        self.generator_test_step.append(loss.clone().detach())

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

        return loss

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

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        with torch.no_grad():
            self.avg_gen_train_loss = torch.stack(
                self.generator_training_step
            ).nanmean()
            self.generator_test_step.clear()

            self.log(
                'generator_train_loss',
                self.avg_gen_train_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True,
            )

        if self.cfg.general.verbose:
            print('\n\n === Training Loss ===')
            print(f"Generator Training Loss: {self.avg_gen_train_loss:.4f}")

        # * Step Scheduler
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
            self.scheduler.step(self.avg_gen_train_loss)
        else:
            self.scheduler.step()

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
        super().on_validation_epoch_end()

        with torch.no_grad():
            self.avg_gen_val_loss = torch.stack(
                self.generator_validation_step
            ).nanmean()
            self.generator_validation_step.clear()

            self.log(
                'generator_val_loss',
                self.avg_gen_val_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True,
            )

        if self.cfg.general.verbose:
            print('\n\n === Validation Loss ===')
            print(f"Generator Validation Loss: {self.avg_gen_val_loss:.4f}")

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
        super().on_test_epoch_end()

        with torch.no_grad():
            self.avg_gen_test_loss = torch.stack(
                self.generator_test_step
            ).nanmean()
            self.generator_test_step.clear()

            self.log(
                'generator_test_loss',
                self.avg_gen_test_loss,
                on_epoch=True,
                logger=True,
                on_step=False,
                sync_dist=True,
            )

        if self.cfg.general.verbose:
            print('\n\n === Test Loss ===')
            print(f"Generator Test Loss: {self.avg_gen_test_loss:.4f}")

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