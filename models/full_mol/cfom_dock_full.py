

import torch
from models.cfom_dock import CfomDock
from models.mcl.multi_contrastive_loss import AttnProjection
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

class CfomDockFull(CfomDock):
    
    def _create_graph_memory(self, graph_data, molecule_sidechain_mask_idx):
        if self.graph_encoder is None:
            return None, None

        if not self.use_receptors:
            del graph_data['receptor']
            del graph_data['receptor','receptor']
            del graph_data['atom','receptor']
            del graph_data['ligand','receptor']

        masked_graph_data = self.prep_graph(graph_data)
        x = self.graph_encoder(masked_graph_data, just_x=True)
        graph_memory, graph_padding_mask = self.stack_ligand_memory(x, graph_data)
        return graph_memory, graph_padding_mask.to(graph_memory.device)
    
    def stack_ligand_memory(self, x, graph_data):
        mems = []
        lengths = []
        batch = graph_data['ligand'].batch
        batch[~graph_data['ligand'].core_mask] = -1
        for i in batch.unique():
            if i == -1:
                continue
            
            mems.append(x[batch==i])
            lengths.append(mems[-1].shape[0])
        mems = pad_sequence(mems, batch_first=True, padding_value=0.0) # T, B, D
        lengths = torch.tensor(lengths)
        arange_T = torch.arange(mems.shape[1])
        valid_data_mask = arange_T.unsqueeze(0) < lengths.unsqueeze(1)
        return mems, ~valid_data_mask


    def _create_memory(
        self,
        smiles_tokens_src,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
    ):

        smiles_memory, smiles_padding_mask = self._create_text_memory(smiles_tokens_src)
        graph_memory, graph_padding_mask = self._create_graph_memory(
            graph_data, molecule_sidechain_mask_idx
        )
        interaction_memory, interaction_padding_mask = self._create_interaction_memory(
            interaction_data, torch.tensor([1] * len(graph_data['ligand'].smiles))
        )

        # Concatenate encoder output with GNN output
        combined_memory = torch.cat(
            self.remove_nones([smiles_memory, graph_memory, interaction_memory]), dim=1
        )
        memory_padding_mask = torch.cat(
            self.remove_nones([smiles_padding_mask, graph_padding_mask, interaction_padding_mask]), dim=1
        )
        return combined_memory, memory_padding_mask

    def optimized_generate_samples(
        self,
        num_samples,
        graph_data,
        interaction_data,
        tokenizer,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        combined_memory, memory_padding_mask = self._create_memory(
            None, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        gen_mol_batches = []
        error_rate = 0
        for i in range(combined_memory.shape[0]):
            gen_mols = []
            num_retries_left=5
            while len(gen_mols) < num_samples:
                if num_retries_left<=0:
                    break
                num_retries_left = num_retries_left -1
                n = (num_samples - len(gen_mols)) * (1 // (1.001 - error_rate))
                n = max(1,int(n * 1.2)) # increase the number of samples to account for errors
                memory = combined_memory[i].repeat(num_samples, 1,1)
                mask = memory_padding_mask[i].repeat(num_samples, 1)
                batch_samples = self.decoder.generate(memory, mask, **kwargs).cpu().numpy()
                gen = tokenizer.decode_batch(batch_samples, skip_special_tokens=True)
                good_gen = [m for m in gen if Chem.MolFromSmiles(m) is not None]
                error_rate = error_rate*0.9+  (1- len(good_gen) / len(gen)) *0.1
                gen_mols.extend(good_gen)
                if len(good_gen) == 0:
                    break
            gen_mol_batches.append(gen_mols[:num_samples]) 
        return gen_mol_batches


    def generate_samples(
        self,
        num_samples,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        raise ValueError("wat")