from typing import Callable

import torch
import torchmetrics
from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader


class LitModel(LightningModule):
    def __init__(
        self,
        model,
        loss_function: Callable,
        unwrap_data: Callable,
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        batch_size: int = 128,
        **kwargs,
    ):
        super(LitModel, self).__init__()
        self.save_hyperparameters(ignore=["model"])

        # optimizer params
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.model = model
        self.loss_function = loss_function
        self.unwrap_data = unwrap_data

        self.metrics = {
            "accuracy": torchmetrics.Accuracy("binary").to(self.device),
            "mcc": torchmetrics.MatthewsCorrCoef("binary").to(self.device),
        }

    def forward(self, x):
        return self.model(x)

    def embedding(self, x):
        return self.model.embedding(x)

    def _common_step(self, data):
        x, y = self.unwrap_data(data)
        pred = self.model(x).flatten()
        assert pred.isnan().sum() == 0, "There are nans"
        return self.loss_function(pred, y)

    def training_step(self, data):
        loss = self._common_step(data)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size
        )
        return loss

    def validation_step(self, data):
        loss = self._common_step(data)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size
        )
        return loss

    def predict_step(self, data):
        x, y = self.unwrap_data(data)
        return self.model(x).flatten()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [{"optimizer": optimizer}]


class LitDataModule(LightningDataModule):
    def __init__(self, train, val, batch_size):
        super().__init__()
        self.train = train
        self.val = val
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=20, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=len(self.val), num_workers=20, shuffle=False
        )
