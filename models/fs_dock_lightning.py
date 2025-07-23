from collections import defaultdict
import copy
from datetime import datetime
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score, average_precision_score


from datasets.custom_distributed_sampler import CustomDistributedSampler, CustomTaskDistributedSampler
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
from utils.logging_utils import configure_logger, get_logger
from rdkit import Chem
from torchmetrics import ROC, AUROC
from models.protonet.protonet import PrototypicalNetwork

class FSDockLightning(pl.LightningModule):
    def __init__(
        self,
        graph_encoder_model: GraphEncoder,
        protonet: PrototypicalNetwork,
        lr,
        weight_decay,
        name=None,
        smol=True,
        num_examples=10,
        support_size=5
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.graph_encoder_model = graph_encoder_model
        self.protonet = protonet
        edge_c = graph_encoder_model.edge_channels
        g_out = graph_encoder_model.out_channels
        self.name = name or f'{datetime.today().strftime("%Y-%m-%d-%H_%M_%S")}'
        self.name = f'fs_dock_{self.name}'
        self.smol = smol
        self.num_examples = num_examples
        self.support_size = support_size
        self.freeze_layers = self.graph_encoder_model.freeze_layers
        self.unfreeze_start = 0
        self.unfreeze_step = 2
        
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sub_proteins.open()

    
    @staticmethod
    def collator_fix(collate_fn):
        def fix(batch):
            if isinstance(batch,list) and len(batch) == 1:
                return collate_fn(batch[0])
            return collate_fn(batch)
        return fix
    
    def train_dataloader(self):
        if self.smol:
            dst = FsDockDatasetPartitioned(
                'data/fsdock/valid',
                '../docking_cfom/valid_tasks.csv')
        else:
            dst = FsDockDatasetPartitioned(
                "data/fsdock/train", 
                "data/fsdock/train_tasks.csv",
                )
        dlt = DataLoader(dst, 
                         sampler=CustomTaskDistributedSampler(dst, shuffle=True,
                                           support_size=32, query_size=16))
        dlt.collate_fn = lambda x: x[0]
        # dlt.collate_fn = self.collator_fix(dlt.collate_fn)
        return dlt
    
    def val_dataloader(self):
        dsv = FsDockDatasetPartitioned(
                'data/fsdock/valid',
                '../docking_cfom/valid_tasks.csv',
                              )
        dlv = DataLoader(dsv, 
                         sampler=CustomTaskDistributedSampler(dsv, shuffle=True,
                                           support_size=32, query_size=20), 
                worker_init_fn=self.worker_init_fn)
        dlv.collate_fn = lambda x: x[0]
        # dlv.collate_fn = self.collator_fix(dlv.collate_fn)
        return dlv

    def test_dataloader(self):
        dsv = FsDockDatasetPartitioned(
                'data/fsdock/test',
                'data/fsdock/test_tasks.csv',
                              )
        dlv = DataLoader(dsv, 
                         sampler=CustomTaskDistributedSampler(dsv, shuffle=True,
                                           support_size=30, query_size=15), 
                worker_init_fn=self.worker_init_fn)
        dlv.collate_fn = lambda x: x[0]
        # dlv.collate_fn = self.collator_fix(dlv.collate_fn)
        return dlv
    

    
    def on_train_epoch_start(self):
        return
        if self.current_epoch == 0:
            for layers in self.freeze_layers:
                if not isinstance(layers,list):
                    layers = [layers]
                for layer in layers:
                    if layer is None:
                        continue
                    for param in layer.parameters():
                        param.requires_grad=False
        if self.current_epoch < self.unfreeze_start:
            return
        elif (self.current_epoch - self.unfreeze_start) % self.unfreeze_step == 0:
            layer_idx = len(self.freeze_layers) - (self.current_epoch - self.unfreeze_start) // self.unfreeze_step - 1
            if layer_idx < 0:
                return
            layers = self.freeze_layers[layer_idx]

            if not isinstance(layers,list):
                layers = [layers]
            for layer in layers:
                if layer is None:
                    continue
                for param in layer.parameters():
                    param.requires_grad=True
    
    
    def get_stats(self,graph):
        support_graph = Batch.from_data_list(graph[0])
        query_graph = Batch.from_data_list(graph[1])
        support_graph = self.graph_encoder_model(support_graph, keep_hetrograph=True)
        query_graph = self.graph_encoder_model(query_graph, keep_hetrograph=True)
        logits = self.protonet(support_graph, query_graph)
        loss = self.protonet.compute_loss(logits, query_graph.label.to(logits.device))
        
        pred_labels = torch.nn.functional.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        labels = query_graph.label.numpy()
        roc_auc = roc_auc_score(labels, pred_labels)
        auprc = average_precision_score(labels, pred_labels)
        d_auprc = auprc - labels.sum() / len(labels)
        return roc_auc, d_auprc, loss
    
    def training_step(self, graph, batch_idx):
        roc_auc, d_auprc, loss = self.get_stats(graph)
        
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_roc_auc", roc_auc,batch_size=len(graph),  sync_dist=True)
        self.log("train_delta_auprc", d_auprc,batch_size=len(graph),  sync_dist=True)
        return loss

    def validation_step(self, graph, batch_idx):
        roc_auc, d_auprc, loss = self.get_stats(graph)

        self.log("val_loss", loss,batch_size=len(graph),  sync_dist=True)
        self.log("val_roc_auc", roc_auc,batch_size=len(graph),  sync_dist=True)
        self.log("val_delta_auprc", d_auprc,batch_size=len(graph),  sync_dist=True)
        return loss

    def test_step(self, graph, batch_idx):
        roc_auc, d_auprc, loss = self.get_stats(graph)

        self.log("test_loss", loss, batch_size=len(graph), sync_dist=True)
        self.log("test_roc_auc", roc_auc,batch_size=len(graph),  sync_dist=True)
        self.log("test_delta_auprc", d_auprc,batch_size=len(graph),  sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=self.lr / 100)
        return {
                        "optimizer": optimizer,
                        "lr_scheduler": sched,
                        "monitor": "train_loss"
                    }
