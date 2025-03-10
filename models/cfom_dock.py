from collections import defaultdict
import torch
from torch_geometric.nn import GCNConv

import torch.nn as nn
import torch.nn.functional as F


class CfomDock(nn.Module):
    def __init__(
        self,
        transformer_encoder,
        transformer_decoder,
        interaction_encoder,
        graph_encoder,
    ):
        super(CfomDock, self).__init__()
        self.text_encoder = transformer_encoder
        self.graph_encoder = graph_encoder
        self.decoder = transformer_decoder
        self.interaction_encoder = interaction_encoder
        # self.linear = nn.Linear(num_gnn_features, d_model)
        if self.graph_encoder is not None:
            self.freeze_layers = [*self.graph_encoder.freeze_layers, [self.text_encoder, self.interaction_encoder]]
        else:
            self.freeze_layers = []

    def _train_decode(self, tgt, memory, memory_key_padding_mask):
        target_mask, target_padding_mask = self.decoder.create_target_masks(tgt)
        output = self.decoder(
            tgt,
            memory,
            target_mask,
            target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output
    
    def _create_text_memory(self, smiles_tokens_src):
        if self.text_encoder is None:
            return None, None
        smiles_padding_mask = self.text_encoder.create_src_key_padding_mask(
            smiles_tokens_src
        )
        smiles_memory = self.text_encoder(smiles_tokens_src, smiles_padding_mask)
        if (
            smiles_memory.shape[1] == smiles_padding_mask.shape[-1]
        ):  # transformer encoder did not take the fast path
            smiles_memory = smiles_memory[:, ~smiles_padding_mask.all(0)]

        smiles_padding_mask = smiles_padding_mask[:, ~smiles_padding_mask.all(0)]
        return smiles_memory, smiles_padding_mask

    def _create_graph_memory(self, graph_data, molecule_sidechain_mask_idx):
        if self.graph_encoder is None:
            return None, None
        graph_data = self.graph_encoder.mask_graph_sidechains(
            graph_data, molecule_sidechain_mask_idx
        )
        graph_memory = self.graph_encoder(graph_data)
        graph_memory = graph_memory  # need to get mask and do all(0) to get the idx where all are padding
        graph_padding_mask = self.graph_encoder.create_memory_key_padding_mask(
            graph_data
        )
        graph_memory = graph_memory[:, ~graph_padding_mask.all(0)]
        graph_padding_mask = graph_padding_mask[:, ~graph_padding_mask.all(0)]
        return graph_memory, graph_padding_mask
    
    def _create_interaction_memory(self, interaction_data):
        if self.interaction_encoder is None:
            return None, None
        interaction_memory = self.interaction_encoder(*interaction_data).unsqueeze(1)
        interaction_padding_mask = (
            torch.zeros(*interaction_memory.shape[0:2])
            .bool()
            .to(interaction_memory.device)
        )
        return interaction_memory, interaction_padding_mask
    
    def remove_nones(self, l):
        return [x for x in l if x is not None]
    
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
            interaction_data
        )
        
        # Concatenate encoder output with GNN output
        combined_memory = torch.cat(
            self.remove_nones([smiles_memory, graph_memory, interaction_memory]), dim=1
        )
        memory_padding_mask = torch.cat(
            self.remove_nones([smiles_padding_mask, graph_padding_mask, interaction_padding_mask]), dim=1
        )
        return combined_memory, memory_padding_mask

    def generate_samples(
        self,
        num_samples,
        smiles_tokens_src,
        smiles_src,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        combined_memory, memory_padding_mask = self._create_memory(
            smiles_tokens_src, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        sidechains_lists = []
        for i in range(num_samples):
            batch_samples = self.decoder.generate(combined_memory, memory_padding_mask, **kwargs)
            batch_samples = batch_samples.cpu().numpy()
            sidechains_list = []
            for sidechains, core in zip(batch_samples, smiles_src):
                sidechains_list.append(sidechains)
            sidechains_lists.append(sidechains_list)
        return sidechains_lists

    def forward(
        self,
        smiles_tokens_src,
        smiles_tokens_tgt,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
    ):
        combined_memory, memory_padding_mask = self._create_memory(
            smiles_tokens_src, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        # Transformer Decoder
        # if self.training:
        output = self._train_decode(
            smiles_tokens_tgt, combined_memory, memory_padding_mask
        )
        # output = self.decoder(smiles_tokens_tgt, combined_memory)
        return output
