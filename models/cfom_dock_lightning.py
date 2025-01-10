import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

from models.cfom_dock import CfomDock

class CfomDockLightning(pl.LightningModule):
    def __init__(self, cfom_dock_model:CfomDock, lr, weight_decay, loss=None):
        super().__init__()
        self.cfom_dock_model = cfom_dock_model
        if loss is None:
            loss = nn.CrossEntropyLoss()
        self.loss = loss
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.save_hyperparameters(ignore=['cfom_dock_model', 'loss'])

    def training_step(self, data, batch_idx):
        logits = self.cfom_dock_model(data.core_tokens, data.sidechain_tokens, data, (data.activity_type, data.label))
        logits = logits.reshape(-1, logits.shape[-1])
        tgt = data.sidechain_tokens.reshape(-1)
        loss = self.loss(logits, tgt)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, batch_idx):
        logits = self.cfom_dock_model(data.core_tokens, data.sidechain_tokens, data, (data.activity_type, data.label))
        logits = logits.reshape(-1, logits.shape[-1])
        tgt = data.sidechain_tokens.reshape(-1)
        loss = self.loss(logits, tgt)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr , weight_decay=self.weight_decay)
        return optimizer

