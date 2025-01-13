import numpy as np
from tqdm import tqdm
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock_grouped import GFsDockDataset
from visualize import make_fig
import torch
import os.path as osp
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch

ds = GFsDockDataset('data/fsdock/smol','data/tasks_smol.csv', num_workers=2,hide_sidechains=True)

for t in tqdm(ds):
    continue
    print(t['name'])

# ds = GFsDockDataset('data/fsdock/full','../docking_cfom/csv_tasks.csv', num_workers=2)



