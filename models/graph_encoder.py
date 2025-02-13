from typing import List, Tuple, Union
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


from models.layers.point_graph_transformer_conv import PGHTConv


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        hidden_channels: Union[int, List[int]],
        out_channels: int,
        attention_groups: int,
        graph_embedder: torch.nn.Module,
        dropout: float=0.1,
        nodes: List[str]=None,
        edges: List[Tuple[str, str, str]]=None,
        num_layers: int = None,
        max_length=128,
        hidden_noise_channel:int = None
    ):

        assert isinstance(hidden_channels, list) or num_layers, "Either hidden_channels is a list or num_layers must be provided"
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (num_layers - 1)
        nodes = nodes or ['ligand', 'receptor', 'atom']
        edges = edges or [('ligand', 'lig_bond', 'ligand'), ('receptor', 'to', 'receptor'), ('atom', 'to', 'atom'), ('atom', 'to', 'receptor'), ('ligand', 'to', 'receptor'), ('ligand', 'to', 'atom'), ('receptor', 'rev_to', 'atom'), ('receptor', 'rev_to', 'ligand'), ('atom', 'rev_to', 'ligand')]
        super(GraphEncoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        num_channels = [in_channels,*hidden_channels,out_channels]
        for i, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            self.convs.append(
                PGHTConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    edge_in_channels=edge_channels,
                    num_attn_groups=attention_groups,
                    dropout=dropout,
                    metadata=(nodes, edges),
                )
            )
        hidden_noise_channel = hidden_noise_channel or num_channels[-2]
        self.noise_conv = PGHTConv(
                    in_channels=num_channels[-2],
                    out_channels=hidden_noise_channel,
                    edge_in_channels=edge_channels,
                    num_attn_groups=attention_groups,
                    dropout=dropout,
                    metadata=(nodes, edges),
                )
        self.noise_final_layer = nn.Sequential(nn.Linear(hidden_noise_channel, hidden_noise_channel),nn.Dropout(dropout), nn.ReLU(), nn.Linear(hidden_noise_channel, 3))
        self.graph_embedder = graph_embedder
        self.max_length = max_length

    def mask_graph_sidechains(self, graph, molecule_sidechain_mask_idx):
        device = graph['ligand'].x.device
        masks = {
            node_t:(
                (graph.sidechains_mask < molecule_sidechain_mask_idx).to(device)
                if node_t == "ligand"
                else torch.arange(graph[node_t].num_nodes, device=device)
            )
            for node_t in graph.metadata()[0]
        }
        graph=  graph.subgraph(masks)
        batch = graph['ligand'].batch
        ptr = torch.arange(batch.shape[0]-1, device=batch.device) + 1
        change = batch[:-1] != batch[1:]
        ptr = torch.tensor([0, *ptr[change], batch.shape[0]], device=batch.device)
        graph['ligand'].ptr = ptr
        return graph

    def noise_forward(self, hdata: HeteroData):
        convs = self.convs[:-1] + [self.noise_conv]
        hdata = self.forward(hdata, convs = convs, keep_graph=True)
        noise_pred = self.noise_final_layer(hdata['ligand'].x)
        return noise_pred

    def forward(self, hdata: HeteroData, keep_graph=False,convs=None):
        hdata = self.graph_embedder(hdata)
        hdata = ToUndirected()(hdata)
        data = hdata.to_homogeneous()
        x = data.x
        for conv in (convs or self.convs):
            x = conv(x, data.edge_index, data.edge_attr, data.pos)
        data.x = x
        data = data.to_heterogeneous()
        if keep_graph:
            return data
        output = data['ligand'].x
        batch_indices = data['ligand'].batch
        batch_size = batch_indices.max().item() + 1
        emb_dim = output.size(1)

        res = torch.zeros(batch_size,self.max_length,emb_dim).to(output.device)
        res[self._graph_batch_indices_to_sequence(batch_indices)] = output
        return res

    def _graph_batch_indices_to_sequence(self, batch_indices: torch.Tensor):
        change_indices = torch.nonzero(batch_indices[1:] != batch_indices[:-1]).flatten() + 1
        dist_between_change = change_indices.clone()
        dist_between_change[1:] = change_indices[:-1] - change_indices[1:]
        dist_between_change[0] = -dist_between_change[0]
        dist_between_change = dist_between_change + 1
        jumps = torch.ones_like(batch_indices)
        jumps[change_indices] = dist_between_change
        batch_range = torch.cumsum(jumps,0) - 1
        return batch_indices,batch_range

    def create_memory_key_padding_mask(self,data:HeteroData):
        batch_indices = data['ligand'].batch
        batch_size = batch_indices.max().item() + 1
        mask = torch.zeros(batch_size,self.max_length).to(batch_indices.device) + 1
        mask[self._graph_batch_indices_to_sequence(batch_indices)] = 0
        return mask.bool()
