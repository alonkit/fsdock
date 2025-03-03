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
from models.tasks.task import AtomNumberTask
from utils.logging_utils import configure_logger, get_logger
from rdkit import Chem
from torchmetrics import ROC, AUROC


class FSDockLightning(pl.LightningModule):
    def __init__(
        self,
        graph_encoder_model: GraphEncoder,
        lr,
        weight_decay,
        name=None,
        smol=True,
        num_examples=10,
        k_nearest=5
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.graph_encoder_model = graph_encoder_model
        edge_c = graph_encoder_model.edge_channels
        g_out = graph_encoder_model.out_channels
        self.attention_layer = nn.MultiheadAttention(g_out, 8,batch_first=True)
        self.name = name or f'{datetime.today().strftime("%Y-%m-%d-%H_%M_%S")}'
        self.name = f'fs_dock_{self.name}'
        self.smol = smol
        self.num_examples = num_examples
        self.k_nearest = k_nearest
        self.freeze_layers = self.graph_encoder_model.freeze_layers
        self.unfreeze_start = 2
        self.unfreeze_step = 2
        
        self.auroc = AUROC('binary')
        self.roc = ROC('binary')


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
                                           task_size=10))
        dlt.collate_fn = self.collator_fix(dlt.collate_fn)
        return dlt
    
    def val_dataloader(self):
        dsv = FsDockDatasetPartitioned(
                'data/fsdock/valid',
                '../docking_cfom/valid_tasks.csv',
                              )
        dlv = DataLoader(dsv, 
                         sampler=CustomTaskDistributedSampler(dsv, shuffle=True,
                                           task_size=18), 
                worker_init_fn=self.worker_init_fn)
        dlv.collate_fn = self.collator_fix(dlv.collate_fn)
        return dlv
    
    
    def on_train_epoch_start(self):
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
    
    def get_embedding(self,data):
        memory = self.graph_encoder_model(data) # (N,L,E)
        graph_padding_mask = self.graph_encoder_model.create_memory_key_padding_mask(
            data
        )
        attn_output, attn_output_weights = self.attention_layer(memory,memory,memory, key_padding_mask=graph_padding_mask)
        embeddings = attn_output.max(1).values
        return embeddings / embeddings.norm(2,dim=-1,keepdim=True)
        
    def get_distances(self,embd, embd2=None):
        '''
        embeddings: (N, E)
        '''
        embd2 = embd if embd2 is None else embd2
        diff = embd.unsqueeze(1) - embd2.unsqueeze(0)
        distances = diff.norm(2,dim=2)
        return distances
    
    def training_step(self, data, batch_idx):
        embeddings = self.get_embedding(data)
        distances = self.get_distances(embeddings)
        labels = data.label
        
        same_class_mask = labels.unsqueeze(0) == labels.unsqueeze(-1)
        dist2 = distances
        loss = dist2[same_class_mask].mean() - dist2[~same_class_mask].mean()

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def classify(self,data):
        embeddings = self.get_embedding(data)
        labels = data.label
        example_indices = torch.randperm(labels.shape[0])[:self.num_examples]
        example_mask = torch.fill(torch.zeros(len(labels)).bool(),False)
        example_mask[example_indices] = True
        query_mask = ~example_mask
        
        example_labels = labels[example_mask]
        query_labels = labels[query_mask]
        query_example_mask = query_mask.unsqueeze(-1) & example_mask.unsqueeze(0)
        query_example_distances = self.get_distances(embeddings[query_mask], embeddings[example_mask]) # (queries, examples)
        query_example_topk = query_example_distances.topk(self.k_nearest,largest=False).indices
        pred_labels = []
        for topk_examples_idx in query_example_topk:
            topk_labels = example_labels[topk_examples_idx]
            pred_label = (topk_labels.sum() / self.k_nearest )
            pred_labels.append(pred_label)
        pred_labels = torch.tensor(pred_labels, device=query_labels.device)
        return query_labels,pred_labels
        

    def validation_step(self, graph, batch_idx):
        query_labels,pred_labels = self.classify(graph)
        roc_auc = self.auroc(pred_labels, query_labels)
        fpr, tpr, thresholds = self.roc(pred_labels, query_labels)
        self.log("val_roc_auc", roc_auc,batch_size=len(graph),  sync_dist=True)
        return roc_auc

    def test_step(self, graph, batch_idx):
        query_labels,pred_labels = self.classify(graph)
        roc_auc = self.auroc(pred_labels, query_labels)
        fpr, tpr, thresholds = self.roc(pred_labels, query_labels)

        self.log("test_roc_auc", roc_auc,batch_size=len(graph),  sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=self.lr / 100)
        return {
                        "optimizer": optimizer,
                        "lr_scheduler": sched,
                        "monitor": "train_loss"
                    }
