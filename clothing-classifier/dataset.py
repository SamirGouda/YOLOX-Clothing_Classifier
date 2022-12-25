#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import functools
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms, datasets
from skimage import io, transform 
from tqdm import tqdm
import math
import sys
import os
import pandas as pd
import pytorch_lightning as pl

from typing import List, Union
from pathlib import Path
from processing.image_transformations import Rescale, RandomCrop, ToTensor
import csv

DATA_TRANSFORMS = {
    'NORM_AUG': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomGrayscale(),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'NORM': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


@functools.lru_cache(1)
def read_csv_file(csv_file: Path):
    data = pd.read_csv(csv_file)
    raise NotImplementedError


class ClothingSmallDataset(datasets.ImageFolder):
    def __init__(self, images_dir: Path, 
                 apply_augmentation: bool = False) -> None:
        super().__init__(root=images_dir, 
            transform=DATA_TRANSFORMS['NORM_AUG'] if apply_augmentation else DATA_TRANSFORMS['NORM'])
        self.idx_to_class = {v: k for k,v in self.class_to_idx.items()}
        
    # custom get_item that additionally returns image path 
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path
        
    def to_csv(self, file: Path):
        header = ['image', 'label']
        list_ = [(os.path.splitext(os.path.basename(image))[0] , self.idx_to_class[idx]) for image, idx in self.samples]
        df = pd.DataFrame(list_, columns=header)
        df.to_csv(file)

class ClothingSmallDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: Path, dataloader_conf: dict):
        super().__init__()
        self.data_dir = data_dir
        self.dataloader_conf = dataloader_conf

    def find_sampler_to_handle_data_imbalance(self):
        targets = np.array(self.train_ds.targets)
        class_sample_count = np.array(
            [len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in targets])

        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit, validate, test or predict step'''
        # we set up only relevant datasets when stage is specified (automatically set by Lightning)
        if stage in [None, 'fit', 'validate']:
            self.train_ds = ClothingSmallDataset(os.path.join(self.data_dir, 'train'), True)
            self.val_ds = ClothingSmallDataset(os.path.join(self.data_dir, 'validation'), False)
        if stage == 'test' or stage is None:
            self.test_ds = ClothingSmallDataset(os.path.join(self.data_dir, 'test'), False)

    def train_dataloader(self):
        sampler = self.find_sampler_to_handle_data_imbalance()

        train_dl = DataLoader(self.train_ds, **self.dataloader_conf,
                                sampler=sampler,
                                shuffle= False,
                                drop_last= True,
        )
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val_ds, **self.dataloader_conf,
                                shuffle= False,
                                drop_last= False,
        )
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test_ds, **self.dataloader_conf,
                                shuffle= False,
                                drop_last= False,
        )
        return test_dl