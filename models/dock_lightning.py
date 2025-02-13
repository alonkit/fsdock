from collections import defaultdict
import copy
from datetime import datetime
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.custom_distributed_sampler import CustomDistributedSampler
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from datasets.process_chem.process_sidechains import (
    calc_tani_sim,
    get_fp,
    reconstruct_from_core_and_chains,
)
from models.cfom_dock import CfomDock
from models.graph_encoder import GraphEncoder
from models.tasks.task import AtomNumberTask
from utils.logging_utils import configure_logger, get_logger


class DockLightning(pl.LightningModule):
    def __init__(
        self,
        graph_encoder_model: GraphEncoder,
        lr,
        weight_decay,
        name=None,
        smol=True,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.graph_encoder_model = graph_encoder_model
        self.name = name or f'{datetime.today().strftime("%Y-%m-%d-%H_%M_%S")}'
        self.name = f'cfom_dock_{self.name}'
        self.smol = smol

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sub_proteins.open()
    
    def train_dataloader(self):
        if self.smol:
            dst = FsDockDatasetPartitioned('data/fsdock/valid','../docking_cfom/valid_tasks.csv', num_workers=torch.get_num_threads())
        else:
            dst = FsDockDatasetPartitioned("data/fsdock/train", "data/fsdock/train_tasks.csv")
        dlt = DataLoader(dst, batch_size=2 if self.smol else 24 , sampler=CustomDistributedSampler(dst, shuffle=True), num_workers=torch.get_num_threads(), 
                        worker_init_fn=self.worker_init_fn)
        return dlt
    
    def val_dataloader(self):
        dsv = FsDockClfDataset("data/fsdock/clfs/valid", "data/fsdock/valid_tasks.csv")
        dlv = DataLoader(dsv, batch_size=32, 
                num_workers=torch.get_num_threads()//2, 
                worker_init_fn=self.worker_init_fn)
        return dlv
    
    def t_to_sigma(self, t):
        return 0.05 ** (1-min(t,1)) * 0.2 ** t
    
    def training_step(self, data, batch_idx):
        poses = data['ligand'].pos
        sigma = self.t_to_sigma(self.current_epoch / self.trainer.max_epochs)
        pos_noise = torch.normal(0,sigma, poses.shape,device=poses.device)
        data['ligand'].pos = poses + pos_noise
        pred_noise = self.graph_encoder_model.noise_forward(data)
        loss = ((pred_noise - (pos_noise / sigma **2)) ** 2 * sigma ** 2).mean()
        self.log("train_noise_loss", loss, sync_dist=True)
        return loss
            
            
    def validation_step(self, data, batch_idx):
        poses = data['ligand'].pos
        sigma = self.t_to_sigma(self.current_epoch / self.trainer.max_epochs)
        pos_noise = torch.normal(0,sigma, poses.shape,device=poses.device)
        data['ligand'].pos = poses + pos_noise
        pred_noise = self.graph_encoder_model.noise_forward(data)
        loss = ((pred_noise - (pos_noise / sigma **2)) ** 2 * sigma ** 2).mean()
        self.log("val_noise_loss", loss, sync_dist=True, batch_size=len(data))
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
