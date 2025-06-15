
import scipy.spatial # very important, does not work without it, i don't know why
from datetime import datetime
import numpy as np
from tqdm import tqdm
from datasets.custom_distributed_sampler import CustomDistributedSampler, CustomTaskDistributedSampler
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
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()

    
ds = FsDockDatasetPartitioned('data/fsdock/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads())
dl = DataLoader(ds, batch_sampler=CustomTaskDistributedSampler(ds, 20, num_replicas=1, rank=0), 
                worker_init_fn=worker_init_fn)
for i,t in enumerate(tqdm(dl)):
    print(i)
exit()



