import numpy as np
from tqdm import tqdm
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock import FsDockDataset
from torch_geometric.loader import DataLoader

from visualize import make_fig
import torch
import os.path as osp
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch

# ds = FsDockDataset('data/fsdock/smol','data/fsdock/tasks_smol.csv', num_workers=2)
# # # ds = FsDockDataset('data/fsdock/single','data/single.csv', num_workers=2)
ds = FsDockDataset('data/fsdock/valid','../docking_cfom/valid_tasks.csv', num_workers=torch.get_num_threads())

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()

dl = DataLoader(ds, batch_size=64, 
                shuffle=True, 
                num_workers=torch.get_num_threads(), 
                worker_init_fn=worker_init_fn)



for t in tqdm(dl):
    g = t['graphs'][0]
for t in tqdm(ds):
    g = t['graphs'][0]
    #g.subgraph({'ligand': torch.from_numpy(g.sidechains_mask == 0)})
exit()



ds = FsDockDataset('data/fsdock/valid','../docking_cfom/valid_tasks.csv', num_workers=torch.get_num_threads())
ds = FsDockDataset('data/fsdock/test','../docking_cfom/test_tasks.csv', num_workers=20)
ds = FsDockDataset('data/fsdock/train','../docking_cfom/train_tasks.csv', num_workers=20)



