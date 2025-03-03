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
        max_noise_scale=5.
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.graph_encoder_model = graph_encoder_model
        edge_c = graph_encoder_model.edge_channels
        g_out = graph_encoder_model.out_channels
        self.distances_layer = nn.Sequential(nn.Linear(g_out*2+edge_c, g_out*2+edge_c),nn.Dropout(0.1), nn.ReLU(), nn.Linear(g_out*2+edge_c, 1))
        self.name = name or f'{datetime.today().strftime("%Y-%m-%d-%H_%M_%S")}'
        self.name = f'dock_{self.name}'
        self.smol = smol
        self.max_noise_scale = max_noise_scale

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
        dlv = DataLoader(dsv, batch_size=24, 
                num_workers=torch.get_num_threads()//2, 
                worker_init_fn=self.worker_init_fn)
        return dlv
    
    def t_to_sigma(self, t):
        return 0.05 ** (1-min(t,1)) * self.max_noise_scale ** t
    
    def pred_distances(self, data, orig_lig_poses):
        data = self.graph_encoder_model(data, keep_hetrograph=True)
        ll = data['ligand','ligand'].edge_index
        ligand_edges_no_self = ll[:, ll[0]!=ll[1]]
        ll_i, ll_j = data['ligand'].x[ligand_edges_no_self]
        lr_i, lr_j = data['ligand'].x[data['ligand', 'receptor'].edge_index[0]], data['receptor'].x[data['ligand', 'receptor'].edge_index[1]]
        la_i, la_j = data['ligand'].x[data['ligand', 'atom'].edge_index[0]], data['atom'].x[data['ligand', 'atom'].edge_index[1]]
        ll = torch.concat([ll_i, ll_j, data['ligand','ligand'].edge_attr],dim=-1)
        lr = torch.concat([lr_i, lr_j, data['ligand','receptor'].edge_attr],dim=-1)
        la = torch.concat([la_i, la_j, data['ligand','atom'].edge_attr],dim=-1)
        edges = torch.concat([ll,lr,la], dim=0)
        pred_dists = self.distances_layer(edges).squeeze(-1)
        orig_dists = self.get_distances(data, orig_lig_poses)
        return pred_dists, orig_dists

    def get_distances(self,data, lig_poses):
        ll_i, ll_j = lig_poses[data['ligand','ligand'].edge_index]
        lr_i, lr_j = lig_poses[data['ligand', 'receptor'].edge_index[0]], data['receptor'].pos[data['ligand', 'receptor'].edge_index[1]]
        la_i, la_j = lig_poses[data['ligand', 'atom'].edge_index[0]], data['atom'].pos[data['ligand', 'atom'].edge_index[1]]
        ll = (ll_i - ll_j).norm(dim=-1)
        lr = (lr_i - lr_j).norm(dim=-1)
        la = (la_i - la_j).norm(dim=-1)
        return torch.concat([ll,lr,la], dim=0)

    
    def training_step(self, data, batch_idx):
        poses = data['ligand'].pos
        sigma = self.t_to_sigma(self.current_epoch / self.trainer.max_epochs)
        pos_noise = torch.normal(0,sigma, poses.shape,device=poses.device)
        data['ligand'].pos = poses + pos_noise
        pred_dists, orig_dists = self.pred_distances(data, poses)
        loss = ((orig_dists - pred_dists)**2 ) * (1/(orig_dists+1))
        loss = loss.mean()
        self.log("train_noise_loss", loss, sync_dist=True, prog_bar=True)
        return loss
            
            
    def validation_step(self, data, batch_idx):
        poses = data['ligand'].pos
        sigma = self.t_to_sigma(self.current_epoch / self.trainer.max_epochs)
        pos_noise = torch.normal(0,sigma, poses.shape,device=poses.device)
        data['ligand'].pos = poses + pos_noise
        pred_dists, orig_dists = self.pred_distances(data, poses)
        loss = ((orig_dists - pred_dists)**2 ) * (1/(orig_dists+1))
        loss = loss.mean()
        self.log("val_noise_loss", loss, batch_size=len(data), sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
