
from torch import Tensor, nn
from torch.nn import functional as F
import torch
from datasets.process_chem.features import allowable_features, allowable_features_index
from models.common import MultiLayerPerceptron


class Task(nn.Module):

    def __init__(self,task_name, mlp, model, is_per_node=True, embed_dim=128, num_heads=1):
        super().__init__()
        self.mlp = mlp
        self.name = task_name
        self.model = model
        self.is_per_node = is_per_node
        if not is_per_node:
            self.grouping_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
    
    @staticmethod
    def choose_indices(ptr: Tensor, mask_rate):
        sizes = ptr[1:] - ptr[:-1]
        num_hidden = (sizes*mask_rate).long().clamp(1)
        hidden_to_batch = torch.repeat_interleave(num_hidden)
        chosen_mask = (torch.rand(num_hidden.sum(), device=ptr.device) * sizes[hidden_to_batch]).long()
        chosen_mask = chosen_mask + ptr[:-1][hidden_to_batch]
        return chosen_mask

    
    def forward(self, data):
        '''
        get the graph, return the loss
        '''
        pass
    
class LabelTask(Task):
    def __init__(self, embed_dim, hidden_dims, model):
        assert hidden_dims[-1] == 1 ,'because is boolean'
        mlp = MultiLayerPerceptron(embed_dim, hidden_dims,)
        super().__init__('activity', mlp, model, is_per_node=False)
        self.loss = nn.BCELoss() 
    
    def forward(self, data):
        labels = data.label
        graph_memory = self.model(data) # (N L, E)
        graph_padding_mask = self.model.create_memory_key_padding_mask(
            data
        )
        graph_memory = graph_memory[:, ~graph_padding_mask.all(0)]
        graph_padding_mask = graph_padding_mask[:, ~graph_padding_mask.all(0)]
        
        # graph_embed, weights = self.grouping_attn(graph_memory,graph_memory,graph_memory, key_padding_mask=graph_padding_mask)
        graph_embed = graph_memory.mean(1)
        score = self.mlp(graph_embed)
        prob = F.sigmoid(score)
        return self.loss(prob.squeeze(-1),labels.float())
        

class AtomNumberTask(Task):
    def __init__(self, embed_dim, hidden_dims, model):
        self.feature = 'possible_atomic_num_list'
        self.task_feature_idx = allowable_features_index[self.feature]
        num_classes = len(allowable_features[self.feature])
        assert hidden_dims[-1] == num_classes ,'because is class num'
        mlp = MultiLayerPerceptron(embed_dim, hidden_dims,)
        super().__init__(f'feature {self.feature}', mlp, model, is_per_node=True)
        self.loss = nn.CrossEntropyLoss() 
    
    def forward(self, data):
        mask = self.choose_indices(data['ligand'].ptr, 0.3)
        labels = data['ligand'].x[mask, self.task_feature_idx].clone()
        data['ligand'].x[mask] = 0
        data = self.model(data, keep_graph=True)
        data['ligand'].x
        score = self.mlp(data['ligand'].x[mask]) # (N, C)
        return self.loss(score,labels)
        


