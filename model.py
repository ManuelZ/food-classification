# External imports
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision import models


class FineTuningWithResNet(pl.LightningModule):
    def __init__(
        self,
        resnet_model_name: str = "resnet18",
        weights: str = "DEFAULT",
        num_classes=10,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ):
        """
        Modified from: Deep Learning with PyTorch - OpenCV University
        """

        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler

        # Init the backbone of a pretrained Resnet
        resnet = getattr(models, resnet_model_name)(weights=weights)
        layers = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*layers)

        # Add a classifier head
        self.classifier = nn.Linear(
            in_features=resnet.fc.in_features, out_features=num_classes
        )

        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_acc = MulticlassAccuracy(
            num_classes=num_classes, average="micro"
        )
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_acc = MulticlassAccuracy(
            num_classes=num_classes, average="micro"
        )

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        """ """
        x = self.backbone(x).flatten(1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, *args, **kwargs):
        """ """

        data, target = batch

        # Get prediction
        output = self(data)

        # Calculate batch loss
        loss = F.cross_entropy(output, target)

        # Batch Predictions
        pred_batch = output.detach().argmax(dim=1)

        self.mean_train_loss.update(loss, weight=data.shape[0])
        self.mean_train_acc.update(pred_batch, target)

        # self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=True)
        # self.log("train/batch_acc", self.mean_train_acc, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        """Calculate epoch level metrics for the train set"""

        self.log("train/loss", self.mean_train_loss, prog_bar=True, logger=True)
        self.log("train/acc", self.mean_train_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)

    def validation_step(self, batch, *args, **kwargs):
        """ """

        data, target = batch

        # Predict
        output = self(data)

        loss = F.cross_entropy(output, target)

        # Batch Predictions
        pred_batch = output.argmax(dim=1)

        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_acc.update(pred_batch, target)
        self.confusion_matrix.update(pred_batch, target)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, imageid = batch
        output = self(data)
        pred = output.detach().argmax(dim=1)
        return imageid, pred

    def on_validation_epoch_end(self):
        """Calculate epoch level metrics for the validation set"""

        self.log("valid/loss", self.mean_valid_loss, prog_bar=True, logger=True)
        self.log("valid/acc", self.mean_valid_acc, prog_bar=True, logger=True)
        self.log("step", self.current_epoch, logger=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#multiple-optimizers-and-schedulers
        """
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
