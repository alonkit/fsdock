
import sys
import scipy.spatial # very important, does not work without it, i don't know why
from datetime import datetime
import numpy as np
import tokenizers
from tqdm import tqdm
from datasets.cross_partitioned import CrossPartitionedFsDockDataset
from datasets.custom_distributed_sampler import CustomDistributedSampler
from datasets.fsmol_dock import FsDockDataset
from datasets.fsmol_dock import FsDockDataset
from torch_geometric.loader import DataLoader

import torch
import os.path as osp

from datasets.fsmol_dock_clf import FsDockClfDataset
from datasets.fsmol_dock_custom_scafs import FsDockCustomDataset
from datasets.partitioned_fsmol_dock import FsDockDatasetPartitioned
from datasets.samplers import TaskRandomSampler, TaskSequentialSampler
from datasets.task_data_loader import TaskDataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.data import Dataset, HeteroData, makedirs, Batch

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.sub_proteins.open()

def make_datasets(core_weight):
    # ds = FsDockDatasetPartitioned('data/cross/train','data/cross/train_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    # ds = FsDockDataset('data/cross/valid','data/cross/valid_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    ds = FsDockDataset('data/cross/test2','data/cross/test_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
   
    
    # ds = FsDockDatasetPartitioned('data/fsdock/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    # ds = FsDockDatasetPartitioned('data/fsdock/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    
    # ds = FsDockDataset('data/fsdock/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    # ds = FsDockDataset('data/fsdock/test','data/fsdock/actual_test_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)
    
    # ds = FsDockClfDataset('data/fsdock/clfs/valid','data/fsdock/valid_tasks.csv', num_workers=torch.get_num_threads(), min_roc_auc=0.7, core_weight=core_weight)
    # ds = FsDockClfDataset('data/fsdock/clfs/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads(), min_roc_auc=0.7, core_weight=core_weight)
    # ds = FsDockDatasetPartitioned('data/fsdock/train','data/fsdock/train_tasks.csv', num_workers=torch.get_num_threads(), core_weight=core_weight)



if __name__ == "__main__":
    # ds = FsDockCustomDataset('data/cross_reinvent/test','/home/alon.kitin/DiffDec/scaffolds/cross_reinvent_for_fsdock.csv', num_workers=torch.get_num_threads(), core_weight=0.7)
    ds = FsDockCustomDataset('data/pose_reinvent/test','/home/alon.kitin/DiffDec/scaffolds/pose_reinvent_for_fsdock.csv', num_workers=torch.get_num_threads(), core_weight=0.7)
    # ds = FsDockDataset('data/posebuster/test','data/posebuster/test_tasks.csv', num_workers=torch.get_num_threads(), core_weight=0.7)
    exit()
    # ds = CrossPartitionedFsDockDataset(
    #     'data/cross/train','data/cross/train_tasks.csv', 
    #     num_workers=torch.get_num_threads(), core_weight=0.7,
    #     tokenizer=tokenizers.Tokenizer.from_file('models/configs/smiles_tokenizer.json'))
    # # ds.load()
    # ds = DataLoader(ds, batch_size=16, 
                         
    #                     sampler=CustomDistributedSampler(ds, 1, 0, True)
    #                     )
    # # ds = FsDockDataset(
    # #     'data/cross/valid','data/cross/valid_tasks.csv', 
    # #     num_workers=torch.get_num_threads(), core_weight=0.7
    # # )
    # # ds = DataLoader(ds, batch_size=16)
    # for t in tqdm(ds):
    #     pass
    # exit()
    
    make_datasets(0.7)
    # play()
    exit()
    frac = float(sys.argv[1])
    make_datasets(frac)


# sampler = CustomDistributedSampler(ds, 3, 1, True)
# dlv = DataLoader(ds, batch_size=64, sampler=sampler)
# print(3)
# for t in tqdm(dlv):
#     pass
# print(4)

# exit()
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
# exit()


# ds = FsDockDataset('data/fsdock/train','data/fsdock/train_tasks.csv', num_workers=torch.get_num_threads())
# ds = FsDockClfDataset('data/fsdock/clfs/test','data/fsdock/test_tasks.csv', num_workers=torch.get_num_threads())



