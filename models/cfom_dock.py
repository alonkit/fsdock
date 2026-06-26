import torch
from rdkit import Chem
import torch.nn as nn
import torch.nn.functional as F

from models.graph_encoder import GraphEncoder


class CfomDock(nn.Module):
    def __init__(
        self,
        transformer_encoder,
        transformer_decoder,
        interaction_encoder,
        graph_encoder: GraphEncoder,
        add_hole_heighbours=False,
        use_receptors=True
    ):
        super(CfomDock, self).__init__()
        self.add_hole_heighbours = add_hole_heighbours
        self.text_encoder = transformer_encoder
        self.graph_encoder = graph_encoder
        self.decoder = transformer_decoder
        self.interaction_encoder = interaction_encoder
        self.use_receptors = use_receptors
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

        if not self.use_receptors:
            del graph_data['receptor']
            del graph_data['receptor','receptor']
            del graph_data['atom','receptor']
            del graph_data['ligand','receptor']

        masked_graph_data = self.prep_graph(graph_data)
        encoded_graph = self.graph_encoder(masked_graph_data, keep_hetrograph=True)
        graph_memory = encoded_graph['ligand'].x[graph_data['ligand'].frag_hole][:,None,:]
        return graph_memory, torch.zeros(graph_memory.shape[0], graph_memory.shape[1]).bool().to(graph_memory.device)

    def prep_graph(self, graph):
        graph = graph.clone()
        self.fix_graph_batch_idxs(graph)
        
        self.mask_fragments(graph)
        graph = self.graph_encoder.embed_graph(graph)
        graph = self.graph_encoder.undirect_graph(graph)
        return graph
    
    def mask_edges_between(self, graph, src_key,dst_key):
        edge_index = graph[src_key, dst_key].edge_index
        frag_mask = ~graph['ligand'].core_mask
        if src_key == dst_key and src_key == "ligand":
            mask = frag_mask[edge_index]
            mask = (mask[0] | mask[1]) # one of the nodes is in fragment
        else:
            # mask edges that start at sidechains and connect to the protein
            mask = frag_mask[edge_index[0]]
        edge_index = edge_index[:,~mask]
        graph[src_key, dst_key].edge_index = edge_index
        if "edge_attr" in graph[src_key, dst_key]:
            edge_attr = graph[src_key, dst_key].edge_attr
            edge_attr = edge_attr[~mask]
            graph[src_key, dst_key].edge_attr = edge_attr
    
    def mask_fragments(self, graph):
        self.mask_edges_between(graph, "ligand", "ligand")
        if self.use_receptors:
            self.mask_edges_between(graph, "ligand", "receptor")
        # else: there are no receptor nodes
        self.mask_edges_between(graph, "ligand", "atom")

    def _create_interaction_memory(self, interaction_data, num_sidechains):
        if self.interaction_encoder is None:
            return None, None
        interaction_data = list(interaction_data)
        interaction_data[0] = sum([[d]*n.item() for d,n in zip(interaction_data[0], num_sidechains)], [])
        interaction_data[1] = sum([[d]*n.item() for d,n in zip(interaction_data[1], num_sidechains)], [])
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
            interaction_data, graph_data['ligand'].num_frags
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
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        core_tokens = graph_data.core_tokens
        combined_memory, memory_padding_mask = self._create_memory(
            core_tokens, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        sidechains_lists = []
        for i in range(num_samples):
            batch_samples = self.decoder.generate(combined_memory, memory_padding_mask, **kwargs)
            batch_samples = batch_samples.cpu().numpy()
            sidechains_list = []
            splits = torch.cumsum(graph_data.num_sidechains, dim=0)
            for src, dst in zip([0,*splits[:-1]], [*splits]):
                sidechains_list.append(batch_samples[src:dst])
            sidechains_lists.append(sidechains_list)
        return sidechains_lists

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
        sidechains_batches = []
        error_rate = 0
        for i in range(combined_memory.shape[0]):
            chains = []
            while len(chains) < num_samples:
                n = (num_samples - len(chains)) * (1 // (1.001 - error_rate))
                n = max(1,int(n * 1.2)) # increase the number of samples to account for errors
                memory = combined_memory[i].repeat(num_samples, 1,1)
                mask = memory_padding_mask[i].repeat(num_samples, 1)
                batch_samples = self.decoder.generate(memory, mask, **kwargs).cpu().numpy()
                gen_chains = tokenizer.decode_batch(batch_samples, skip_special_tokens=True)
                good_chains = [c for c in gen_chains if Chem.MolFromSmiles(f'[1*]{c}') is not None]
                error_rate = error_rate*0.9+  (1- len(good_chains) / len(gen_chains)) *0.1
                chains.extend(good_chains)
                if len(good_chains) == 0:
                    break
            sidechains_batches.append(chains[:num_samples]) 
        
        mols_sidechains_batches = []
        splits = torch.cumsum(graph_data['ligand'].num_frags, dim=0)
        for src, dst in zip([0,*splits[:-1]], [*splits]):
            mols_sidechains_batches.append(sidechains_batches[src:dst])
        return mols_sidechains_batches        

    def fix_graph_batch_idxs(self,graph):
        device = graph['ligand'].frag_idxs.device
        new_frag_idxs = graph['ligand'].frag_idxs + (
            torch.concat([
                torch.tensor([0], device=device), 
                graph['ligand'].num_frags[:-1]
            ], 0).cumsum(0).repeat_interleave(graph['ligand'].ptr[1:] - graph['ligand'].ptr[:-1])
        )
        new_frag_idxs[graph['ligand'].core_mask] = -1
        graph['ligand'].frag_idxs = new_frag_idxs
        new_frag_hole = graph['ligand'].frag_hole + graph['ligand'].ptr[:-1].repeat_interleave(graph['ligand'].num_frags)
        graph['ligand'].frag_hole = new_frag_hole
        
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
