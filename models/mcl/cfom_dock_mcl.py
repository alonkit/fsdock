

import torch
from models.cfom_dock import CfomDock
from models.mcl.multi_contrastive_loss import AttnProjection
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

class CfomDockMCL(CfomDock):
    def __init__(self,fragment_projection_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_attn_proj =  AttnProjection(self.graph_encoder.out_channels, fragment_projection_dim)
        self.smiles_attn_proj =  AttnProjection(self.decoder.embedding_dim, fragment_projection_dim)
    
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
        return graph_memory, torch.zeros(graph_memory.shape[0], graph_memory.shape[1]).bool().to(graph_memory.device) , self.make_projections(encoded_graph)
    
    def make_projections(self, encoded_graph):
        vecs= []
        frag_idxs = encoded_graph['ligand'].frag_idxs
        for v in frag_idxs[~encoded_graph['ligand'].core_mask].unique():
            frag_vecs = encoded_graph['ligand'].x[frag_idxs==v]
            vecs.append(frag_vecs)
        lengths = [len(v) for v in vecs]
        vecs = pad_sequence(vecs, batch_first=False, padding_value=0.0) # T, B, D
        return self.graph_attn_proj(vecs, lengths=lengths)
    
    def prep_graph(self, graph):
        graph = graph.clone()
        self.fix_graph_batch_idxs(graph)
        graph = self.graph_encoder.embed_graph(graph)
        graph = self.graph_encoder.undirect_graph(graph)
        self.mask_fragments(graph)
        return graph

    def mask_edges_between(self, graph, src_key,dst_key):
        assert src_key == 'ligand'
        edge_index = graph[src_key, dst_key].edge_index
        frag_mask = ~graph['ligand'].core_mask
        if src_key == dst_key and src_key == "ligand":
            mask = frag_mask[edge_index]
            mask = mask[0] & ~mask[1] # start is in a fragment, end is in a core
            frag_idxs = graph['ligand'].frag_idxs
            different_fragments_mask = (frag_idxs[edge_index[0]] != frag_idxs[edge_index[1]]) & (frag_mask[edge_index[0]] & frag_mask[edge_index[1]])  # start and end are in different fragments
            mask = mask | different_fragments_mask  #  start is in a fragment, end is in a core  OR start and end are in different fragments
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
        self.mask_edges_between(graph, "ligand", "receptor")
        self.mask_edges_between(graph, "ligand", "atom")



    def _create_memory(
        self,
        smiles_tokens_src,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
    ):

        smiles_memory, smiles_padding_mask = self._create_text_memory(smiles_tokens_src)
        graph_memory, graph_padding_mask, fragment_projections = self._create_graph_memory(
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
        return combined_memory, memory_padding_mask, fragment_projections

    def optimized_generate_samples(
        self,
        num_samples,
        graph_data,
        interaction_data,
        tokenizer,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        combined_memory, memory_padding_mask, _ = self._create_memory(
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


    def generate_samples(
        self,
        num_samples,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
        **kwargs
    ):
        core_tokens = graph_data.core_tokens
        combined_memory, memory_padding_mask, _  = self._create_memory(
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

        
    def forward(
        self,
        smiles_tokens_src,
        smiles_tokens_tgt,
        graph_data,
        interaction_data,
        molecule_sidechain_mask_idx=1,
    ):

        
        combined_memory, memory_padding_mask, graph_proj = self._create_memory(
            smiles_tokens_src, graph_data, interaction_data, molecule_sidechain_mask_idx
        )
        # Transformer Decoder
        # if self.training:
        output, smiles_proj = self._train_decode(
            smiles_tokens_tgt, combined_memory, memory_padding_mask
        )
        # output = self.decoder(smiles_tokens_tgt, combined_memory)
        return output, graph_proj, smiles_proj
    
    def _train_decode(self, tgt, memory, memory_key_padding_mask):
        target_mask, target_padding_mask = self.decoder.create_target_masks(tgt)
        logits, output = self.decoder(
            tgt,
            memory,
            target_mask,
            target_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        projections = self.smiles_attn_proj(output, target_padding_mask)
        return logits, projections
