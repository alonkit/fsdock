


import torch

from models.cfom_dock import CfomDock


class CfomDockAblation(CfomDock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prot_dim_fit = torch.nn.Linear(self.graph_encoder.out_channels, self.graph_encoder.out_channels*2+self.graph_encoder.edge_channels)    

    def _create_graph_memory(self, graph_data, molecule_sidechain_mask_idx):
        if self.graph_encoder is None:
            return None, None
        del graph_data["atom"]
        del graph_data["ligand","receptor"]
        del graph_data["ligand","atom"]
        del graph_data["atom","receptor"]
        del graph_data["atom","atom"]

        neighbor_idxs = graph_data.hole_neighbors + graph_data["ligand"].ptr[
            :-1
        ].repeat_interleave(graph_data.num_sidechains)
        neighbor_idxs = self.graph_encoder.get_new_indexes_after_masking(
            graph_data, neighbor_idxs, molecule_sidechain_mask_idx
        )
        masked_graph_data = self.graph_encoder.mask_graph_sidechains(
            graph_data, molecule_sidechain_mask_idx
        )
        encoded_graph = self.graph_encoder(masked_graph_data, keep_hetrograph=True)
        graph_memory = self._collect_local_clusters(encoded_graph, neighbor_idxs)

        prot_xs = encoded_graph["receptor"].x
        prot_xs = self.prot_dim_fit(torch.mean(prot_xs, dim=0))
        graph_memory = torch.cat([graph_memory,prot_xs.repeat(len(neighbor_idxs),1,1)],dim=1)

        return graph_memory, torch.zeros(graph_memory.shape[0], graph_memory.shape[1]).bool().to(graph_memory.device)

    # def _create_graph_memory(self, graph_data, molecule_sidechain_mask_idx):
    #     if self.graph_encoder is None:
    #         return None, None 
    #     graph_embedder = self.graph_encoder.graph_embedder
    #     prot_xs = graph_embedder.node_embedders['receptor'](graph_data["receptor"].x)
    #     prot_xs = torch.mean(prot_xs, dim=0).unsqueeze(1)
    #     del graph_data["receptor"]
    #     del graph_data["atom"]
    #     del graph_data["ligand","receptor"]
    #     del graph_data["ligand","atom"]
    #     del graph_data["receptor","receptor"]
    #     del graph_data["atom","receptor"]
    #     del graph_data["atom","atom"]
        
    #     graph_memory, mask = super()._create_graph_memory(
    #         graph_data, molecule_sidechain_mask_idx
    #     )
    #     graph_memory = torch.cat([graph_memory, prot_xs], dim=-1)
    #     mask = torch.cat([mask, torch.zeros_like(prot_xs).bool().to(mask.device)], dim=-1)
    #     return graph_memory, mask

    def _collect_local_clusters(self, graph_data, cluster_centers_idxs):
        lig_clusters = self._get_local_clusters_vecs(graph_data, cluster_centers_idxs, 'ligand', 10)
        clusters = []
        for l in lig_clusters:
            clusters.append(l.unsqueeze(0))
        return torch.cat(clusters)
        
