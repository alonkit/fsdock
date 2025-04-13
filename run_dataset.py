
import scipy.spatial # very important, does not work without it, i don't know why
from datetime import datetime
import numpy as np
from tqdm import tqdm
from datasets.custom_distributed_sampler import CustomDistributedSampler
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock import FsDockDataset
from torch_geometric.loader import DataLoader

import torch
import os.path as osp

from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from datasets.samplers import TaskRandomSampler, TaskSequentialSampler
from datasets.task_data_loader import TaskDataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch

ds = FsDockDatasetPartitioned('data/fsdock/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads())

exit()
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()

    
ds = FsDockClfDataset('data/fsdock/clfs/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads(), min_roc_auc=0.7)
dl = DataLoader(ds, batch_size=2, 
                shuffle=False,   
                # num_workers=torch.get_num_threads(), 
                worker_init_fn=worker_init_fn)
for i,t in enumerate(tqdm(dl)):
    print(i)
exit()
    

ds = FsDockDatasetPartitioned('data/fsdock/train','data/fsdock/train_tasks.csv', num_workers=torch.get_num_threads())

# sampler = CustomDistributedSampler(ds, 3, 1, True)
# dlv = DataLoader(ds, batch_size=64, sampler=sampler)
# print(3)
# for t in tqdm(dlv):
#     pass
# print(4)

# exit()

ds = FsDockDatasetPartitioned('data/fsdock/smol','data/fsdock/smol_tasks.csv', num_workers=torch.get_num_threads())
ds = FsDockDatasetPartitioned('data/fsdock/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads())
# exit()

# # dl = DataLoader(ds, batch_size=64, 
# #                 shuffle=True,   
# #                 num_workers=torch.get_num_threads(), 
# #                 worker_init_fn=worker_init_fn)
#  srun -c 20 python ./run_dataset.py
# dl = TaskDataLoader(ds, batch_sampler=TaskRandomSampler(ds.task_sizes, 64),
#                 num_workers=torch.get_num_threads(), 
#                 worker_init_fn=worker_init_fn)


# for t in tqdm(dl):
#     pass
# for t in tqdm(ds):
#     pass     
# exit()
# dsv = FsDockClfDataset("data/fsdock/valid", "data/fsdock/valid_tasks.csv", num_workers=torch.get_num_threads())
# dsv = FsDockClfDataset("data/fsdock/test", "data/fsdock/test_tasks.csv", num_workers=torch.get_num_threads())
# dlv = DataLoader(dsv, batch_size=64, 
#                         num_workers=torch.get_num_threads(), 
#                     worker_init_fn=worker_init_fn)

# for t in tqdm(dsv):
#     pass
# for t in tqdm(dlv):
#     pass     
# exit()
ds = FsDockClfDataset('data/fsdock/clfs/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads(), min_roc_auc=0.7)
# exit()


# ds = FsDockDataset('data/fsdock/train','data/fsdock/train_tasks.csv', num_workers=torch.get_num_threads())
# ds = FsDockClfDataset('data/fsdock/clfs/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads())
ds = FsDockClfDataset('data/fsdock/clfs/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads(), min_roc_auc=0.7)



