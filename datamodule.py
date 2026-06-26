from pathlib import Path

import albumentations as A
import cv2
import lightning as pl
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from dataset import KenyanFood13Dataset


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
        target_size: int = 256,
        train_split: float = 0.75,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_folder = root_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.train_split = train_split

        self.common_transforms = A.Compose(
            [
                A.Resize(height=self.target_size, width=self.target_size, p=1),
                A.CenterCrop(height=224, width=224, p=1),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        self.train_transforms = A.Compose(
            [
                # A.RandomResizedCrop(data_config.TARGET_SIZE, data_config.TARGET_SIZE),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                A.ToGray(p=0.1),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.Affine(
                    border_mode=cv2.BORDER_CONSTANT,
                    rotate_limit=10,
                    scale_limit=0.0,
                    value=0,
                    p=0.75,
                ),
                A.ElasticTransform(
                    alpha=120,
                    sigma=6,
                    alpha_affine=8,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.5,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.05,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.5,
                ),
                self.common_transforms,
            ]
        )

        self.valid_transforms = A.Compose([self.common_transforms])
        self.test_transforms = A.Compose([self.common_transforms])

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

        self.predict_dataset = KenyanFood13Dataset(
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
