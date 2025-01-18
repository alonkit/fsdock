import numpy as np
from tqdm import tqdm
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_grouped import GFsDockDataset

from visualize import make_fig
import torch
import os.path as osp
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch

# # ds = GFsDockDataset('data/fsdock/smol','data/tasks_smol.csv', num_workers=2)
# ds = GFsDockDataset('data/fsdock/single','data/single.csv', num_workers=2)
# for t in tqdm(ds):
#     g = t['graphs'][0]
#     #g.subgraph({'ligand': torch.from_numpy(g.sidechains_mask == 0)})
#     print(t['name'])
# exit()
ds = GFsDockDataset('data/fsdock/valid','../docking_cfom/valid_tasks.csv', num_workers=20)
ds = GFsDockDataset('data/fsdock/test','../docking_cfom/test_tasks.csv', num_workers=20)
ds = GFsDockDataset('data/fsdock/train','../docking_cfom/train_tasks.csv', num_workers=20)



