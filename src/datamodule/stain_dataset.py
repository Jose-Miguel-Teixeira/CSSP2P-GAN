import torch
import os
import glob
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Tuple, Literal
# import psutil


class StainDataset(Dataset):

    IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.tiff']

    def __init__(
            self,
            dataroot: str,
            phase: Literal['Train', 'Val', 'Test'],
            HE_transforms=None,
            IHC_transforms=None,
            max_dataset_size: int = float('inf'),
            pre_load_dataset: bool = False,
            return_base_name: bool = False
    ) -> None:
        super(StainDataset, self).__init__()

        self.dataroot = dataroot
        self.phase = phase
        self.pre_load_dataset = pre_load_dataset
        self.return_base_name = return_base_name

        self.to_tensor_image = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(
                dtype=torch.float32,
                scale=True,
                ),
        ])

        # * Load Images
        # ** Load the dataset images paths
        def get_image_paths(dataroot, subfolder, phase):
            paths = []
            for ext in self.IMAGE_EXTENSIONS:
                paths.extend(
                    glob.glob(os.path.join(dataroot, subfolder, phase, ext))
                    )
            return sorted(paths)

        match phase:
            case 'Train':  # Train dataset
                self.HE_path = get_image_paths(self.dataroot, 'HE', 'train')
                self.IHC_path = get_image_paths(self.dataroot, 'IHC', 'train')
            case 'Val':  # Val dataset
                self.HE_path = get_image_paths(self.dataroot, 'HE', 'val')
                self.IHC_path = get_image_paths(self.dataroot, 'IHC', 'val')
            case 'Test':  # Test dataset
                self.HE_path = get_image_paths(self.dataroot, 'HE', 'test')
                self.IHC_path = get_image_paths(self.dataroot, 'IHC', 'test')
            case _:
                raise ValueError(
                    f"Invalid phase: {phase}."
                    "'phase' must be one of ['Train', 'Val', 'Test']"
                    )

        if len(self.HE_path) != len(self.IHC_path):
            raise ValueError(
                'The number of H&E and IHC images are not equal. H&E: {}, IHC: {}'.format(
                    len(self.HE_path), len(self.IHC_path)
                    )
                )

        # ** Limit the dataset size
        self.max_dataset_size = max_dataset_size
        if max_dataset_size < len(self.HE_path):
            self.HE_path = self.HE_path[:max_dataset_size]
            self.IHC_path = self.IHC_path[:max_dataset_size]

        # ** Load the images
        if self.pre_load_dataset:
            self.images = {
                'HE_image': [None] * len(self.HE_path),
                'IHC_image': [None] * len(self.IHC_path),
                'HE_path': self.HE_path,
                'IHC_path': self.IHC_path
            }
        else:
            self.images = None

        if self.pre_load_dataset:
            for i in len(self.HE_path):
                HE_image, IHC_image = self._get_image(i)
                self.images['HE_image'][i] = HE_image
                self.images['IHC_image'][i] = IHC_image

                # Print occupied RAM
                # process = psutil.Process(os.getpid())
                # print(f"Memory usage after loading image {i}: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Set the transform
        if HE_transforms is not None:
            self.HE_transforms = HE_transforms
        else:
            self.HE_transforms = v2.Identity()

        if IHC_transforms is not None:
            self.IHC_transforms = IHC_transforms
        else:
            self.IHC_transforms = v2.Identity()

    def _get_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        HE_path, IHC_path = self.HE_path[index], self.IHC_path[index]
        if os.path.basename(HE_path) != os.path.basename(IHC_path):
            raise ValueError(
                'The H&E and IHC images are not aligned. H&E: {}, IHC: {}'.format(
                    os.path.basename(HE_path), os.path.basename(IHC_path)
                    )
                )
        HE_image = self.to_tensor_image(Image.open(HE_path).convert('RGB'))
        IHC_image = self.to_tensor_image(Image.open(IHC_path).convert('RGB'))

        return HE_image, IHC_image

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_load_dataset:
            HE_image = self.images['HE_image'][index]
            IHC_image = self.images['IHC_image'][index]
        else:
            HE_image, IHC_image = self._get_image(index=index)

        # Apply the transform
        # Save the random state and set it back after applying the transform
        # so to have the same random state for both images
        state = torch.get_rng_state()
        HE_image = self.HE_transforms(HE_image)
        torch.set_rng_state(state)
        IHC_image = self.IHC_transforms(IHC_image)

        if type(HE_image) is type(IHC_image):
            if isinstance(HE_image, (tuple, list)):
                HE_image = torch.stack(HE_image, dim=0)
                IHC_image = torch.stack(IHC_image, dim=0)
        else:
            raise TypeError(
                "HE_image and IHC_image must be of the same type."
                f"HE_image: {type(HE_image)}, IHC_image: {type(IHC_image)}"
                )

        # Debugging
        # print(f"{index}: {self.HE_path[index]} loaded. Allocated cuda memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Allocated RAM: {psutil.virtual_memory().percent}, Allocated cpu%: {psutil.cpu_percent()}")

        if self.return_base_name:
            return HE_image, IHC_image, os.path.basename(self.HE_path[index]).split('.')[0]

        return HE_image, IHC_image

    def __len__(self) -> int:
        return len(self.HE_path)


class StainDataModule(pl.LightningDataModule):
    def __init__(
            self,
            # Dataset Variables
            train_dataroot: str,
            test_dataroot: str,
            HE_augmentations=None,
            IHC_augmentations=None,
            HE_transforms=None,
            IHC_transforms=None,
            max_dataset_size: int = float('inf'),
            pre_load_dataset: bool = False,

            # DataModule Variables
            apply_augmentation_on_val: bool = False,

            # DataLoader Variables
            batch_size: int = 1,
            num_workers: int = 3,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            prefetch_factor: int = 2,
            shuffle: bool = True,
    ) -> None:
        super(StainDataModule, self).__init__()

        # Dataset Variables
        self.train_dataroot = train_dataroot
        self.test_dataroot = test_dataroot
        self.HE_augmentations = HE_augmentations
        self.IHC_augmentations = IHC_augmentations
        self.HE_transforms = HE_transforms
        self.IHC_transforms = IHC_transforms
        self.max_dataset_size = max_dataset_size
        self.pre_load_dataset = pre_load_dataset

        # Dataloader Variables
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.max_dataset_size = max_dataset_size

        # DataModule Variables
        self.apply_augmentation_on_val = apply_augmentation_on_val

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:  # Train and Val
            self.train_dataset = StainDataset(
                dataroot=self.train_dataroot,
                HE_transforms=self.HE_augmentations,
                IHC_transforms=self.IHC_augmentations,
                phase='Train',
                max_dataset_size=self.max_dataset_size,
                pre_load_dataset=self.pre_load_dataset
            )

            if self.apply_augmentation_on_val:
                self.val_dataset = StainDataset(
                    dataroot=self.train_dataroot,
                    HE_transforms=self.HE_augmentations,
                    IHC_transforms=self.IHC_augmentations,
                    phase='Val',
                    max_dataset_size=self.max_dataset_size,
                    pre_load_dataset=self.pre_load_dataset
                )
            else:
                self.val_dataset = StainDataset(
                    dataroot=self.train_dataroot,
                    HE_transforms=self.HE_transforms,
                    IHC_transforms=self.IHC_transforms,
                    phase='Val',
                    max_dataset_size=self.max_dataset_size,
                    pre_load_dataset=self.pre_load_dataset
                )

        elif stage == 'predict':
            self.predict_dataset = StainDataset(
                    dataroot=self.train_dataroot,
                    HE_transforms=self.HE_transforms,
                    IHC_transforms=self.IHC_transforms,
                    phase='Val',
                    # phase='Train',
                    # phase='Test',
                    max_dataset_size=self.max_dataset_size,
                    pre_load_dataset=self.pre_load_dataset,
                    return_base_name=True
                )

        elif stage == 'test':
            self.test_dataset = StainDataset(
                dataroot=self.train_dataroot,
                HE_transforms=self.HE_transforms,
                IHC_transforms=self.IHC_transforms,
                phase='Test',
                max_dataset_size=self.max_dataset_size,
                pre_load_dataset=self.pre_load_dataset,
                return_base_name=True
            )

        else:
            raise ValueError(
                f"Invalid stage: {stage}."
                "'stage' must be one of ['fit', 'predict', 'test']"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.shuffle,
            persistent_workers=True,
            # collate_fn=custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,  # No need to shuffle the validation dataset
            persistent_workers=True,
            # collate_fn=custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            # collate_fn=custom_collate,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
            persistent_workers=True,
        )
