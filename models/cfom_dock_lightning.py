import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets.process_sidechains import get_fp, reconstruct_from_core_and_chains
from models.cfom_dock import CfomDock

class CfomDockLightning(pl.LightningModule):
    def __init__(self, cfom_dock_model:CfomDock,tokenizer, lr, weight_decay, num_gen_samples, loss=None):
        super().__init__()
        self.cfom_dock_model = cfom_dock_model
        self.num_gen_samples = num_gen_samples
        self.tokenizer = tokenizer
        if loss is None:
            loss = nn.CrossEntropyLoss()
        self.loss = loss
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.save_hyperparameters(ignore=['cfom_dock_model', 'loss','tokenizer'])

    def generate_samples(self, data):
        samples = self.cfom_dock_model.generate_samples(self.num_gen_samples, data.core_tokens, data.core_smiles, data, (data.activity_type, 1))
        # we want to genenerate good samples so we give label=1
        samples = self.tokenizer.decode_batch(samples, skip_special_tokens=True)
        new_mols = []
        for core, chains in samples.items():
            new_mol = reconstruct_from_core_and_chains(core, chains)
            new_mols.append((new_mol, get_fp(new_mol)))
        return new_mols          

    def training_step(self, data, batch_idx):
        data = data['graphs'][0]
        logits = self.cfom_dock_model(data.core_tokens, data.sidechain_tokens, data, (data.activity_type, data.label))
        logits = logits.reshape(-1, logits.shape[-1])
        tgt = data.sidechain_tokens.reshape(-1)
        loss = self.loss(logits, tgt)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['graphs'][0]
        new_mols = self.generate_samples(data)
        
        
        classifiers = batch['clf']
        logits = self.cfom_dock_model(data.core_tokens, data.sidechain_tokens, data, (data.activity_type, data.label))
        logits = logits.reshape(-1, logits.shape[-1])
        tgt = data.sidechain_tokens.reshape(-1)
        loss = self.loss(logits, tgt)
        self.log('validation_loss', loss, batch_size=len(data.label), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr , weight_decay=self.weight_decay)
        return optimizer

