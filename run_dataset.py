from datetime import datetime
import numpy as np
from tqdm import tqdm
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock import FsDockDataset
from torch_geometric.loader import DataLoader

import torch
import os.path as osp

from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.samplers import TaskRandomSampler, TaskSequentialSampler
from datasets.task_data_loader import TaskDataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch

# ds = FsDockDataset('data/fsdock/smol','data/fsdock/tasks_smol.csv', num_workers=2)
# # # # ds = FsDockDataset('data/fsdock/single','data/single.csv', num_workers=2)
# # ds = FsDockDataset('data/fsdock/valid','../docking_cfom/valid_tasks.csv', num_workers=torch.get_num_threads())

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()

# # dl = DataLoader(ds, batch_size=64, 
# #                 shuffle=True, 
# #                 num_workers=torch.get_num_threads(), 
# #                 worker_init_fn=worker_init_fn)

# dl = TaskDataLoader(ds, batch_sampler=TaskRandomSampler(ds.task_sizes, 64),
#                 num_workers=torch.get_num_threads(), 
#                 worker_init_fn=worker_init_fn)


# for t in tqdm(dl):
#     pass
# for t in tqdm(ds):
#     pass     
# exit()
dsv = FsDockClfDataset("data/fsdock/valid", "data/fsdock/valid_tasks.csv")
dlv = DataLoader(dsv, batch_size=64, 
                        num_workers=torch.get_num_threads()//2, 
                    worker_init_fn=worker_init_fn)

for t in tqdm(dsv):
    pass
for t in tqdm(dlv):
    pass     
exit()
ds = FsDockDataset('data/fsdock/valid','../docking_cfom/valid_tasks.csv', num_workers=torch.get_num_threads())
ds = FsDockDataset('data/fsdock/test','../docking_cfom/test_tasks.csv', num_workers=20)
ds = FsDockDataset('data/fsdock/train','../docking_cfom/train_tasks.csv', num_workers=20)



