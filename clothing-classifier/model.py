#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import lightly
import pytorch_lightning as pl
from PIL import Image
import numpy as np
from typing import Union, Any, List, Dict
from torchmetrics import Accuracy, AveragePrecision, ConfusionMatrix, F1Score



# import the checkpoint API 
import torch.utils.checkpoint as checkpoint

from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss


class SimCLRModel(pl.LightningModule):
    def __init__(self, output_dim: int = 128, epochs: int = 20,
                 loss: Union[nn.Module,pl.LightningModule] = NTXentLoss(),
                 optimizer = None, optimizer_params: dict = None,
                 lr_scheduler = None, lr_scheduler_params: dict = None,
            ):
        super().__init__()
        self.epochs = epochs
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, output_dim)

        self.criterion = loss
        self.optim = optimizer
        self.optim_conf = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_conf = lr_scheduler_params

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = self.optim(**self.optim_conf)
        if self.self.optim_conf is None:
            scheduler = self.lr_scheduler(
                optim, T_max=self.epochs
            )
        else:
            raise NotImplementedError
        return [optim], [scheduler]

class PytorchLightningModule(pl.LightningModule):
    def __init__(self, 
                backbone_model: nn.Module,
                weights_dir: Path = "pretrained_models/mobilenet_v3/mobilenet_v3_small.pth",
                output_dim: int = None, batch_size: int = 8, lr: float = 0.01,
                loss: Union[nn.Module,pl.LightningModule] = None,
                loss_params: dict = None, amp:bool = False,
                optimizer = None, optimizer_params: dict = None,
                lr_scheduler = None, lr_scheduler_params: dict = None,
        ):
        super().__init__()
        # log hyperparameters
        self.save_hyperparameters()
        # create mobilnetv3 and load weights
        self.output_dim = output_dim
        self.model = backbone_model()
        weights = torch.load(weights_dir, map_location='cpu')
        self.model.load_state_dict(weights)
        del weights
        # remove old head
        try:
            # mobilenet_v3
            self.model.classifier[-1] = torch.nn.Linear(in_features=self.model.classifier[0].out_features, out_features=output_dim)
        except:
            # resnet50
            self.model.fc = torch.nn.Linear(in_features=2048, out_features=output_dim)

        if loss_params is None:
            self.criterion = loss()
        else:
            self.criterion = loss(**loss_params)

        self.train_acc = Accuracy(task="multiclass", num_classes=self.output_dim, threshold=0.5, average="micro").to(self.device)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.output_dim, threshold=0.5, average="micro").to(self.device)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.output_dim, threshold=0.5, average="micro").to(self.device)
        self.averageprecision = AveragePrecision(task="multiclass", num_classes=self.output_dim, threshold=None, average="weighted").to(self.device)
        # self.confussion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.output_dim, normalize=None)
        self.optim = optimizer
        self.optim_conf = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_conf = lr_scheduler_params

        
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, Y, P = batch
        if self.hparams.amp:
            with torch.autocast(device_type=self.device.type):
                _, _, loss, acc = self._get_logits_preds_loss_accuracy(X, Y, self.train_acc)
        else:
            _, _, loss, acc = self._get_logits_preds_loss_accuracy(X, Y, self.train_acc)
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("train_acc", acc, on_epoch=True, reduce_fx="mean", batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, Y, P = batch
        _, preds, loss, acc = self._get_logits_preds_loss_accuracy(X, Y, self.val_acc)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)
        
        self.log("val_acc", acc, on_epoch=True, reduce_fx="mean", batch_size=self.hparams.batch_size)
        
        return preds

    def test_step(self, batch, batch_idx):
        X, Y, P = batch
        logits, preds, loss, acc = self._get_logits_preds_loss_accuracy(X, Y, self.test_acc)
        self.log("test_loss", loss, batch_size=self.hparams.batch_size)
        self.log("test_acc", acc, on_epoch=True, reduce_fx="mean", batch_size=self.hparams.batch_size)
        ap = self.averageprecision(logits, Y)
        self.log("test_ap", ap, on_epoch=True, reduce_fx="mean", batch_size=self.hparams.batch_size)
        # matrix = self.confussion_matrix(logits, Y)
        # self.log("test_confussion_matrix", matrix, on_epoch=True, reduce_fx="mean", batch_size=self.hparams.batch_size)
        # self.log_dict({"wav": P, "preds": preds, "labels": Y})
        return {'loss': loss, 'preds': preds}

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        optim = self.optim(self.model.parameters(), **self.optim_conf)
        if self.lr_conf is not None:
            scheduler = self.lr_scheduler(
                optim, **self.lr_conf
            )
        else:
            raise NotImplementedError
        return [optim], [scheduler]

    def _get_logits_preds_loss_accuracy(self, X, Y, acc_fn):
        logits = self.forward(X)
        preds = torch.argmax(logits, dim=-1)
        loss = self.criterion(logits, Y)
        acc = acc_fn(preds, Y)
        return logits, preds, loss, acc

    def on_test_end(self) -> None:
        return super().on_test_end()