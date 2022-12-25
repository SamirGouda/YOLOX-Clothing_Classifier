#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import sys
import datetime
import shutil
import hashlib
import math

import torch
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
from util.calc_metrics import get_metrics, plotConfussionMatrix, create_table, read_output_file
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import torchvision
from torchvision import transforms
from pathlib import Path
import wandb

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def load_pytorch_model(model: nn.Module, ckpt: Path):
    try:
        model = model()
    except:
        pass
    checkpoint = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    return model

def predict(test_dataloader: DataLoader, model: nn.Module):
    with torch.no_grad():
        model.eval()
        for batch in test_dataloader:
            X, Y = batch
            raise NotImplementedError



class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, batches: list = list(range(100)), n: int = 3):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx in batches:
            try:
                x, y = batch
            except ValueError:
                x, y, _ = batch
            images = [img for img in x[:n]]
            y = [(trainer.datamodule.train_ds.idx_to_class[y_i.item()]) for y_i in y]
            outputs = [(trainer.datamodule.train_ds.idx_to_class[output.item()]) for output in outputs]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            # log images with `WandbLogger.log_image`
            trainer.wandb_logger.log_image(key='sample_images', images=images, caption=captions)
            # log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(images, y[:n], outputs[:n]))]
            trainer.wandb_logger.log_table(key='sample_table', columns=columns, data=data)
        
            return y, outputs
        
    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        y, outputs = self.on_validation_batch_end(trainer, pl_module, outputs['preds'], batch, batch_idx, dataloader_idx)
        file = os.path.join(trainer.params.exp_dir, "test.txt")
        with open(file, 'a') as fd:
            for label, pred in zip(y, outputs):
                fd.write('{} {}\n'.format(label, pred))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_end(trainer, pl_module)
        classes = trainer.datamodule.test_ds.classes
        file = os.path.join(trainer.params.exp_dir, "test.txt")
        y_true, y_pred = read_output_file(file, classes)
        metric = get_metrics(y_true, y_pred, classes)
        table = create_table(metric, 'test')
        conf_matrix = plotConfussionMatrix(y_true, y_pred, classes=classes)  
        out = os.path.join(trainer.params.exp_dir, "test.log")
        with open(out, 'w') as fd:
            fd.write('{}\n\n'.format(table))
            fd.write('{}\n\n'.format(conf_matrix))