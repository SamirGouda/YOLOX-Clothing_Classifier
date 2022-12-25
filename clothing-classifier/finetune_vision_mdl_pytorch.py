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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from util.logconf import logging, formatter
from util.utils import read_yaml_conf
from util.pytorch_utils import LogPredictionsCallback
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import wandb
wandb.login()

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from dataset import ClothingSmallDataModule
# these imports are necessary for yaml to load
import model
import dataset
import loss


WANDB_USERNAME= "samir-gouda"

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        help='path to images',
                        default='clothing-dataset-small',
                        )
    parser.add_argument('--num-workers',
                        help='Number of worker processes for background data loading',
                        default=4,  # 4 * torch.cuda.device_count()
                        type=int
                        )
    parser.add_argument('--batch-size',
                        help='Batch size used for training',
                        default=256,
                        type=int
                        )
    parser.add_argument('--epochs',
                        help='Number of epochs to train for',
                        default=40,
                        type=int
                        )
    parser.add_argument('--conf',
                        help="""path to conf dir, contains conf.yaml""",
                        default='conf/mobilenet.yaml',
                        )
    parser.add_argument('--use-gpu',
                        help='use cuda in training',
                        default=False,
                        action='store_true'
                        )
    parser.add_argument('--exp-dir',
                        help='export directory',
                        default="exp/finetuneing_mobilenet_v3_small")
    parser.add_argument('--skip-train',
                        action='store_true',
                        default=False)
    parser.add_argument('--tune',
                        action='store_true',
                        default=False)
    parser.add_argument('--ckpt',
                        help='checkpoint',
                        default=None)
    parser.add_argument('--run-id',
                        help='wandb run id',
                        default='v1')

    params = parser.parse_args()
    return params

class Trainer(pl.Trainer):
    def __init__(self, params: argparse.Namespace, conf: dict) -> None:
        self.params = params
        self.conf = conf

        self.checkpoint_dir = os.path.join(self.params.exp_dir, 'checkpoints')
        pl.seed_everything(self.conf['seed'])
        if self.params.use_gpu:
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = False
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.ngpus = torch.cuda.device_count() if self.use_cuda else 0

        self.project_id = os.path.basename(self.params.exp_dir)
        self.wandb_logger = WandbLogger(project=self.project_id, save_dir=self.params.exp_dir,
                        id=self.params.run_id,
                        log_model=True, # log all new checkpoints during training
                        )
        # self.tensorboard_logger = TensorBoardLogger(
        #                             self.params.exp_dir, log_graph=False
        #                             )
        
        checkpoint_callback = ModelCheckpoint(**self.conf['model_checkpoint_params'])

        callbacks=[checkpoint_callback, LogPredictionsCallback()]
        if self.conf['use_swa']:
            callbacks.append(StochasticWeightAveraging(**self.conf['swa_params']))
        
        super().__init__(
            logger=self.wandb_logger,
            callbacks=callbacks,
            default_root_dir=self.params.exp_dir,
            accelerator='gpu' if self.use_cuda else 'cpu',
            devices=self.ngpus,
            max_epochs=self.params.epochs,
            **self.conf['pl_params']
        )

        self.model = self.initLightningModule()
        # log gradients and model topology.
        self.wandb_logger.watch(self.model)

        # initalizes DataModule
        self.datamodule = self.initDataModule(self.params.input_dir)
        # check number of classes in dataset equals to given output dim
        # assert len(self.datamodule.train_ds.classes) == self.conf['model_params']['output_dim']
        
        
    def initLightningModule(self):
        batch_size = self.conf['dataloader_params']['batch_size']
        lr = self.conf['model_params']['optimizer_params']['lr']
        try:
            model = self.conf['model'](**self.conf['model_params'], batch_size=batch_size, lr=lr, amp=self.conf['use_amp'])
        except:
            raise ValueError
        if self.use_cuda:
            # detect multiple GPUs
            if self.ngpus > 1:
                # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
                model = nn.DataParallel(model)  # wraps model with data parallelism
            # move model parameters to device for calculations
            # must be done before constructing the optimizer
            model = model.to(self.device)
        
        return model

    def initDataModule(self, input_dir: Path):
        batch_size = self.conf['dataloader_params']['batch_size']
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        # dataloaders batches examples and also provide parallel loading using
        # separate processes and shared memory by specifying num_workers
        conf = {"batch_size": batch_size,
                "num_workers": self.params.num_workers, # pinned memory transfer to GPU quickly
                "pin_memory": self.use_cuda,
                "collate_fn":None,
        }
        lightningdatamodule = ClothingSmallDataModule(input_dir, conf)
    
        return lightningdatamodule

    def download_best_mdl_from_wandb(self, run_id: str = None, tag: str = "best_k"):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        run_id = run_id if run_id is not None else self.logger.experiment.id
        checkpoint_reference = f'{WANDB_USERNAME}/{self.project_id}/model-{run_id}:{tag}'
        run = wandb.init(project=self.project_id)
        artifact = run.use_artifact(checkpoint_reference, type='model')
        artifact.download(root=self.checkpoint_dir)
        shutil.move(f'{self.checkpoint_dir}/model.ckpt', f'{self.checkpoint_dir}/model-best.ckpt')
        print(f"Best model saved to {self.checkpoint_dir}/model-best.ckpt")

        

if __name__ == "__main__":
    print(' '.join(sys.argv))   # print command line for logging
    params = parser_arguments()
    # read conf files
    os.makedirs(params.exp_dir, exist_ok=True)
    conf = read_yaml_conf(params.conf)
    try:
        trainer = Trainer(params, conf)
    except Exception as e:
        print(e)
        sys.exit(1)
    if params.tune:
        trainer.tune(trainer.model, datamodule=trainer.datamodule)

    if not params.skip_train:
        trainer.fit(trainer.model, datamodule=trainer.datamodule, ckpt_path=trainer.params.ckpt)
        # trainer.download_best_mdl_from_wandb(run_id=params.run_id)
    
    # trainer.model.load_from_checkpoint(f"{trainer.checkpoint_dir}/model-best.ckpt")
    trainer.test(trainer.model, datamodule=trainer.datamodule)

    # trainer.predict(trainer.model, dataloaders=trainer.test_dataloader)