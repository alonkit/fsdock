

import torch
import torch.nn.functional as F
from models.cfom_dock_lightning import CfomDockLightning
from datasets.process_chem.process_sidechains import (
    calc_tani_sim,
    get_fp,
    reconstruct_from_core_and_chains,
)

class CfomDockLightningFull(CfomDockLightning):
    def get_loss(self,data):
        if self.handle_inactive == 'hide':
            if data.label.any(): # exist active and inactive
                # filter actives
                gs = data.to_data_list()
                gs = list(filter(lambda x: x.label.item(), gs))
                data = type(data).from_data_list(gs)
            else: # all inactive
                return torch.tensor(0.0, device=data.label.device,), {}
            
        if self.handle_inactive == 'flag':
            flag = data.label
        else:
            # flag is not interesting so..
            flag = torch.zeros_like(data.label,device=data.label.device)+1

        logits = self.model(
            None,
            data['ligand'].mol_tokens[:, :-1],
            data,
            (data.activity_type, flag), 
            molecule_sidechain_mask_idx=1
        )
        logits = logits.transpose(1, -1)
        tgt = data['ligand'].mol_tokens[:, 1:]
        losses = self.loss(logits, tgt)
        if self.handle_inactive in ('penalty','hide'):
            good_spots = torch.nonzero(data.label==1, as_tuple=True)[0]
            bad_spots = torch.nonzero(data.label==0, as_tuple=True)[0]
            
            recon_loss = losses.mean()
            if len(good_spots) == 0 or len(bad_spots) == 0:
                active_loss = losses[good_spots].mean().nan_to_num(0)
                inactive_loss = losses[bad_spots].mean().nan_to_num(0)
                return recon_loss, {'recon':recon_loss, 'active':active_loss, 'inactive': inactive_loss}
            good_spots, bad_spots = self.extend_tensors(good_spots,bad_spots)
            # maybe use distances
            active_loss = losses[good_spots].mean(1)
            inactive_loss = losses[bad_spots].mean(1)
            
            margin = 0.05
            margin_loss = F.relu(margin + active_loss - inactive_loss).mean()
            
            loss = recon_loss + 2 * margin_loss
            return loss, {'recon':recon_loss, 'margin': margin_loss, 'active':active_loss.mean(), 'inactive': inactive_loss.mean()}
        if self.handle_inactive == 'flag':
            return losses.mean() , {}
    

    def generate_samples(self, data):
        mols_batches = self.model.optimized_generate_samples(
            self.num_gen_samples,
            data,
            (data.activity_type, [1] * len(data)),
            self.tokenizer,
            **self.gen_meta_params
        )
        # we want to genenerate good samples so we give label=1
        new_mols = []
        for old_smile, task, gen_mol_variations in zip(data['ligand'].smiles, data.task, mols_batches):
            for new_smile in gen_mol_variations:
                # chains = self.tokenizer.decode_batch(chains, skip_special_tokens=True)
                new_smile = self.removeChirality(new_smile)
                old_smile = self.removeChirality(old_smile)
                if new_smile is None:
                    new_mols.append((task, None, old_smile, None))
                else:
                    fp = get_fp(new_smile)
                    if fp is None:
                        new_mols.append((task, None, old_smile, None))
                    else:
                        new_mols.append((task, new_smile, old_smile, get_fp(new_smile)))
        return new_mols
    