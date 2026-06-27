from pathlib import Path
from typing import Optional

import albumentations as A
import lightning as pl
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from dataset import KenyanFood13Dataset


def build_transforms(aug_configs: list) -> A.Compose:
    transforms = []
    for cfg in aug_configs:
        cfg = cfg.copy()
        name = cfg.pop("name")
        if name == "ToTensorV2":
            transforms.append(ToTensorV2(**cfg))
        else:
            transforms.append(getattr(A, name)(**cfg))
    return A.Compose(transforms)


def denormalize(tensors):
    """
    Denormalize image tensors back to range [0.0, 1.0]

    From: Deep Learning with PyTorch - OpenCV University
    """

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    tensors = tensors.clone()
    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors.cpu(), 0.0, 1.0)


class KenyaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_folder: str,
        batch_size: int = 32,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        target_size: int = 256,
        train_split: float = 0.75,
        train_augmentations: Optional[list] = None,
        eval_augmentations: Optional[list] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_folder = root_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_size = target_size
        self.train_split = train_split

        self.train_transforms = (
            build_transforms(train_augmentations)
            if train_augmentations is not None
            else None
        )
        self.valid_transforms = (
            build_transforms(eval_augmentations)
            if eval_augmentations is not None
            else None
        )
        self.test_transforms = (
            build_transforms(eval_augmentations)
            if eval_augmentations is not None
            else None
        )

    def prepare_data(self):
        """
        This function handles downloads and any data processing.
        This function makes sure that when you use multiple GPUs,
        you don't download multiple datasets or apply double manipulations to the data.
        """
        pass

    def setup(self, stage=None):
        train_csv_path = Path(self.root_folder) / "train.csv"
        test_csv_path = Path(self.root_folder) / "test.csv"

        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)

        self.whole_dataset = KenyanFood13Dataset(
            self.root_folder,
            "images/images",
            train_df.id,
        )

        # Encode classes as integers
        self.le = LabelEncoder()
        y = self.le.fit_transform(train_df["class"])
        y = torch.Tensor(y).to(torch.int64)

        # Image ids
        X = train_df.id

        # Stratified split into training and validation sets
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=self.train_split, random_state=0
        )
        for train_index, valid_index in sss.split(X, y):
            train_ids, valid_ids, train_y, valid_y = (
                X[train_index],
                X[valid_index],
                y[train_index],
                y[valid_index],
            )

        test_ids = test_df.id

        self.train_dataset = KenyanFood13Dataset(
            self.root_folder,
            "images/images",
            train_ids.tolist(),
            labels=train_y,
            transforms=self.train_transforms,
        )

        self.valid_dataset = KenyanFood13Dataset(
            self.root_folder,
            "images/images",
            valid_ids.tolist(),
            labels=valid_y,
            transforms=self.valid_transforms,
        )

        self.test_dataset = KenyanFood13Dataset(
            self.root_folder,
            "images/images",
            test_ids.tolist(),
            transforms=self.test_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
        )
