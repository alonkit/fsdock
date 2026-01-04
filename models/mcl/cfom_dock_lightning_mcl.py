

import torch
import torch.nn.functional as F
from models.cfom_dock_lightning import CfomDockLightning
from models.mcl.multi_contrastive_loss import MultiContrastiveLoss

class CfomDockLightningMCL(CfomDockLightning):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss = MultiContrastiveLoss()
    
    def get_loss(self,data):
        flag = torch.zeros_like(data.label,device=data.label.device)+1

        logits, graph_proj, smiles_proj = self.model(
            None,
            data['ligand'].frag_tokens[:, :-1],
            data,
            (data.activity_type, flag), 
            molecule_sidechain_mask_idx=1
        )
        logits = logits.transpose(1, -1)
        tgt = data['ligand'].frag_tokens[:, 1:]
        recon_loss = self.loss(logits, tgt).mean()
        
        labels = data.label.repeat_interleave(data['ligand'].num_frags)
        
        contrastive_loss = self.contrastive_loss(graph_proj, smiles_proj, labels)
        return recon_loss + contrastive_loss, {'recon':recon_loss, 'contrastive_loss': contrastive_loss}
    
    def training_step(self, data, batch_idx):
        loss, loss_dict = self.get_loss(data)

        self.log("train_loss", loss,prog_bar=True, sync_dist=True)
        for k,v in loss_dict.items():
            self.log(f"train_loss_{k}",v, sync_dist=True)

        return loss